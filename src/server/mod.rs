// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ergonomic safe Rust wrapper around the NVIDIA cuObj server API.
//!
//! This module sits on top of `cuobj-sys` and provides:
//!
//! * [`CuObjServer`] — the main entry point.  Thread-safe.  Wraps a
//!   `cuObjServer` C++ object, which listens on an RDMA-capable NIC and
//!   services RDMA_READ (PUT) and RDMA_WRITE (GET) requests from clients.
//! * [`Channel`] — a RAII guard that returns a channel ID to the server's pool
//!   on drop.  Each concurrent RDMA operation must use a distinct channel.
//! * [`AllocatedBuffer`] — a RAII guard for a 4 KiB-aligned host buffer that
//!   is registered with the RDMA fabric.  Deregisters and frees memory on
//!   drop.
//!
//! # High-level vs. low-level APIs
//!
//! [`CuObjServer::receive_put`] and [`CuObjServer::send_get`] handle
//! channel and buffer lifecycle automatically and return/accept `Vec<u8>`.
//! Use [`CuObjServer::handle_put`] / [`CuObjServer::handle_get`] when
//! you need finer control (e.g., reusing pre-allocated buffers).
//!
//! # Quick start
//!
//! ```no_run
//! use cuobj::CuObjServer;
//!
//! let server = CuObjServer::new("192.168.100.1", 4421)
//!     .expect("cuObjServer init failed");
//! assert!(server.is_connected());
//!
//! // High-level PUT handler — call from your HTTP server thread.
//! // The RDMA token carries the client's buffer address; offset is usually 0
//! // for a full-buffer transfer, or N bytes for a chunked streaming write.
//! let rdma_token = "<descriptor from x-amz-rdma-token header>";
//! let data: Vec<u8> = server.receive_put("bucket/key", 0, 4096, rdma_token)
//!     .expect("RDMA PUT failed");
//! println!("received {} bytes", data.len());
//! ```

use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::sync::Arc;

use cuobj_sys::{
    cuobj_srv_alloc_channel, cuobj_srv_alloc_register, cuobj_srv_buf_data, cuobj_srv_buf_size,
    cuobj_srv_create, cuobj_srv_deregister_free, cuobj_srv_destroy, cuobj_srv_free_channel,
    cuobj_srv_handle_get, cuobj_srv_handle_put, cuobj_srv_is_connected,
};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from cuObj server operations.
#[derive(Debug, thiserror::Error)]
pub enum CuObjServerError {
    /// `cuObjServer` could not be created (RDMA hardware unavailable, bad IP,
    /// or library not installed).
    #[error("failed to create cuObjServer at {ip}:{port}")]
    CreateFailed { ip: String, port: u16 },

    /// Buffer allocation / RDMA registration failed.
    #[error("failed to allocate/register RDMA buffer of size {size}")]
    BufferAllocFailed { size: usize },

    /// A generic allocation or registration error.
    #[error("allocation error: {0}")]
    AllocError(String),

    /// An RDMA PUT (RDMA_READ from client) failed.
    #[error("RDMA PUT failed for key '{key}': server returned {transferred} (expected >0)")]
    PutFailed { key: String, transferred: isize },

    /// An RDMA GET (RDMA_WRITE to client) failed.
    #[error("RDMA GET failed for key '{key}': server returned {transferred} (expected >0)")]
    GetFailed { key: String, transferred: isize },

    /// A NUL byte appeared in a string argument.
    #[error("argument contains interior NUL byte: {0}")]
    NulError(#[from] std::ffi::NulError),
}

// ─────────────────────────────────────────────────────────────────────────────
// CuObjServer
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper around the `cuObjServer` C++ object.
///
/// # Thread safety
///
/// `CuObjServer` is `Send + Sync`.  Concurrent operations must each hold
/// a distinct [`Channel`] obtained via [`alloc_channel`][Self::alloc_channel].
///
/// # Lifecycle
///
/// ```text
///   CuObjServer::new(ip, port)
///         │
///         ├── alloc_channel()           → Channel (auto-freed on drop)
///         │
///         ├── alloc_buffer(size)        → AllocatedBuffer (auto-freed on drop)
///         │       │
///         │       ├── handle_put / handle_get  ← low-level RDMA
///         │       └── (drop)           → deregister + free
///         │
///         ├── receive_put(key, offset, size, token)  ← high-level: alloc + PUT + return Vec
///         ├── send_get(key, offset, data, token)     ← high-level: alloc + copy + GET
///         │
///         └── Drop                     → cuobj_srv_destroy
/// ```

/// Parse the client's buffer virtual address from the first colon-delimited
/// hex field of an RDMA token.
///
/// Token format: `<remote_vaddr_hex>:<size_hex>:<rkey_hex>:…`
///
/// This address must be used as `remote_start` in the underlying C API.
/// Adding a byte offset to it points to the correct chunk position within
/// the client's registered buffer.
fn token_remote_addr(token: &str) -> usize {
    token
        .split(':')
        .next()
        .and_then(|s| usize::from_str_radix(s, 16).ok())
        .unwrap_or(0)
}

fn do_handle_put(
    server_handle: *mut c_void,
    key:           &str,
    buf:           *mut c_void,
    remote_start:  usize,
    size:          usize,
    rdma_token:    &str,
    channel_id:    u16,
) -> Result<usize, CuObjServerError> {
    let key_cs = CString::new(key)?;
    let tok_cs = CString::new(rdma_token)?;
    let n = unsafe {
        cuobj_srv_handle_put(
            server_handle,
            key_cs.as_ptr(),
            buf,
            remote_start as u64,
            size,
            tok_cs.as_ptr(),
            channel_id,
        )
    };
    if n < 0 {
        return Err(CuObjServerError::PutFailed { key: key.to_owned(), transferred: n });
    }
    Ok(n as usize)
}

/// Internal helper: call `cuobj_srv_handle_get` given raw handles.
fn do_handle_get(
    server_handle: *mut c_void,
    key:           &str,
    buf:           *mut c_void,
    remote_start:  usize,
    size:          usize,
    rdma_token:    &str,
    channel_id:    u16,
) -> Result<usize, CuObjServerError> {
    let key_cs = CString::new(key)?;
    let tok_cs = CString::new(rdma_token)?;
    let n = unsafe {
        cuobj_srv_handle_get(
            server_handle,
            key_cs.as_ptr(),
            buf,
            remote_start as u64,
            size,
            tok_cs.as_ptr(),
            channel_id,
        )
    };
    if n < 0 {
        return Err(CuObjServerError::GetFailed { key: key.to_owned(), transferred: n });
    }
    Ok(n as usize)
}

pub struct CuObjServer {
    pub(crate) handle: *mut c_void,
}

// SAFETY: cuObjServer uses channel IDs for internal thread isolation.
unsafe impl Send for CuObjServer {}
unsafe impl Sync for CuObjServer {}

impl CuObjServer {
    /// Create a `cuObjServer` listening on `ip:port` of the RDMA-capable NIC.
    ///
    /// # Arguments
    ///
    /// * `ip`   — IP address of the RDMA NIC (e.g. `"192.168.100.1"`).
    /// * `port` — TCP/RDMA port (e.g. `4421`).
    ///
    /// # Errors
    ///
    /// Returns [`CuObjServerError::CreateFailed`] if the underlying
    /// `cuObjServer` constructor fails.
    pub fn new(ip: &str, port: u16) -> Result<Self, CuObjServerError> {
        let ip_cstr = CString::new(ip)?;
        let handle = unsafe { cuobj_srv_create(ip_cstr.as_ptr(), port) };
        if handle.is_null() {
            return Err(CuObjServerError::CreateFailed {
                ip: ip.to_owned(),
                port,
            });
        }
        Ok(Self { handle })
    }

    /// Returns `true` if the server is connected to the RDMA fabric.
    pub fn is_connected(&self) -> bool {
        unsafe { cuobj_srv_is_connected(self.handle) != 0 }
    }

    // ── Channel management ───────────────────────────────────────────────────

    /// Allocate a unique channel ID for one concurrent RDMA operation.
    ///
    /// The channel is returned to the pool when the returned [`Channel`] is
    /// dropped.  Every concurrent RDMA transfer (across threads) must hold a
    /// distinct channel.
    pub fn alloc_channel(&self) -> Channel<'_> {
        let id = unsafe { cuobj_srv_alloc_channel(self.handle) };
        Channel {
            server_handle: self.handle,
            id,
            _server: PhantomData,
        }
    }

    // ── Buffer management ────────────────────────────────────────────────────

    /// Allocate a 4 KiB-aligned host buffer of `size` bytes and register it
    /// with the RDMA fabric.
    ///
    /// The buffer is automatically deregistered and freed when the returned
    /// [`AllocatedBuffer`] is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`CuObjServerError::BufferAllocFailed`] if either the
    /// allocation or the RDMA registration fails.
    pub fn alloc_buffer(&self, size: usize) -> Result<AllocatedBuffer<'_>, CuObjServerError> {
        let buf = unsafe { cuobj_srv_alloc_register(self.handle, size) };
        if buf.is_null() {
            return Err(CuObjServerError::BufferAllocFailed { size });
        }
        Ok(AllocatedBuffer {
            server_handle: self.handle,
            buf,
            _server: PhantomData,
        })
    }

    // ── Low-level RDMA operations ────────────────────────────────────────────

    /// Perform an RDMA PUT: RDMA_READ from the client's buffer (described by
    /// `rdma_token`) into `buf`.
    ///
    /// `buf` must have been allocated with [`alloc_buffer`][Self::alloc_buffer].
    /// After a successful call, `buf.as_slice()[..size]` contains the data from
    /// the client's registered buffer starting at `offset`.
    ///
    /// # Arguments
    ///
    /// * `key`       — S3 object key (informational, passed to `handlePutObject`).
    /// * `buf`       — server-side RDMA-registered buffer.
    /// * `offset`    — byte offset within the client's registered buffer to read
    ///                 from.  Use `0` for a full-buffer transfer; use `N` for
    ///                 chunked streaming to advance the read window by chunk size.
    /// * `size`      — number of bytes to transfer.
    /// * `rdma_token` — RDMA descriptor from the `x-amz-rdma-token` header.
    /// * `channel`   — channel allocated via [`alloc_channel`][Self::alloc_channel].
    ///
    /// # Errors
    ///
    /// Returns [`CuObjServerError::PutFailed`] if `handlePutObject` returns
    /// a negative value.
    pub fn handle_put(
        &self,
        key: &str,
        buf: &AllocatedBuffer<'_>,
        offset: u64,
        size: usize,
        rdma_token: &str,
        channel: &Channel<'_>,
    ) -> Result<usize, CuObjServerError> {
        let remote_start = token_remote_addr(rdma_token) + offset as usize;
        do_handle_put(self.handle, key, buf.buf, remote_start, size, rdma_token, channel.id)
    }

    /// Perform an RDMA GET: RDMA_WRITE from `buf` into the client's buffer
    /// (described by `rdma_token`).
    ///
    /// Populate `buf` with the object data **before** calling this function.
    ///
    /// # Arguments
    ///
    /// Same as [`handle_put`][Self::handle_put], except the data flows in the
    /// opposite direction (server → client) and `offset` is the write position
    /// within the client's buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CuObjServerError::GetFailed`] if `handleGetObject` returns
    /// a negative value.
    pub fn handle_get(
        &self,
        key: &str,
        buf: &AllocatedBuffer<'_>,
        offset: u64,
        size: usize,
        rdma_token: &str,
        channel: &Channel<'_>,
    ) -> Result<usize, CuObjServerError> {
        let remote_start = token_remote_addr(rdma_token) + offset as usize;
        do_handle_get(self.handle, key, buf.buf, remote_start, size, rdma_token, channel.id)
    }

    // ── High-level convenience ───────────────────────────────────────────────

    /// Receive a PUT object via RDMA.
    ///
    /// Allocates a server-side RDMA buffer, calls `handlePutObject`
    /// (RDMA_READ from client), copies the received bytes into a `Vec<u8>`,
    /// then frees the buffer.
    ///
    /// # Arguments
    ///
    /// * `key`        — S3 object key.
    /// * `offset`     — byte offset within the client's registered buffer to
    ///                  read from.  Use `0` for a full-buffer (non-chunked)
    ///                  transfer.
    /// * `size`       — expected transfer size in bytes.
    /// * `rdma_token` — RDMA descriptor from the `x-amz-rdma-token` HTTP
    ///                  request header.  The client's buffer base address is
    ///                  parsed internally from this token.
    ///
    /// # Returns
    ///
    /// The received bytes as an owned `Vec<u8>`.
    pub fn receive_put(
        &self,
        key: &str,
        offset: u64,
        size: usize,
        rdma_token: &str,
    ) -> Result<Vec<u8>, CuObjServerError> {
        let buf     = self.alloc_buffer(size)?;
        let channel = self.alloc_channel();
        let n       = self.handle_put(key, &buf, offset, size, rdma_token, &channel)?;
        // Copy out of the RDMA-registered buffer before it is freed on drop.
        let data = buf.as_slice()[..n].to_vec();
        Ok(data)
        // `channel` and `buf` drop here → free channel ID and deregister buffer.
    }

    /// Send a GET object via RDMA.
    ///
    /// Allocates a server-side RDMA buffer, copies `data` into it, calls
    /// `handleGetObject` (RDMA_WRITE to client), then frees the buffer.
    ///
    /// # Arguments
    ///
    /// * `key`        — S3 object key.
    /// * `offset`     — byte offset within the client's registered buffer to
    ///                  write into.  Use `0` for a full-buffer transfer.
    /// * `data`       — object bytes to transfer to the client.
    /// * `rdma_token` — RDMA descriptor from the `x-amz-rdma-token` HTTP
    ///                  request header.  The client's buffer base address is
    ///                  parsed internally from this token.
    ///
    /// # Returns
    ///
    /// Number of bytes transferred.
    pub fn send_get(
        &self,
        key: &str,
        offset: u64,
        data: &[u8],
        rdma_token: &str,
    ) -> Result<usize, CuObjServerError> {
        let mut buf = self.alloc_buffer(data.len())?;
        // Fill the RDMA-registered buffer with the object data.
        buf.as_mut_slice().copy_from_slice(data);
        let channel = self.alloc_channel();
        let n       = self.handle_get(key, &buf, offset, data.len(), rdma_token, &channel)?;
        Ok(n)
        // `channel` and `buf` drop here → free channel ID and deregister buffer.
    }
}

impl Drop for CuObjServer {
    fn drop(&mut self) {
        unsafe { cuobj_srv_destroy(self.handle) };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard for a cuObjServer channel ID.
///
/// Obtained via [`CuObjServer::alloc_channel`].  The channel ID is returned
/// to the server's pool when this guard is dropped.
///
/// Each concurrent RDMA operation must use a distinct `Channel`.
pub struct Channel<'a> {
    server_handle: *mut c_void,
    id:            u16,
    _server:       PhantomData<&'a CuObjServer>,
}

// SAFETY: the server handle is thread-safe; channel IDs are per-operation.
unsafe impl Send for Channel<'_> {}
unsafe impl Sync for Channel<'_> {}

impl Channel<'_> {
    /// The raw channel ID, for use with lower-level FFI if needed.
    #[inline]
    pub fn id(&self) -> u16 {
        self.id
    }
}

impl Drop for Channel<'_> {
    fn drop(&mut self) {
        unsafe { cuobj_srv_free_channel(self.server_handle, self.id) };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AllocatedBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard for a 4 KiB-aligned RDMA-registered host buffer.
///
/// Obtained via [`CuObjServer::alloc_buffer`].  The buffer is deregistered
/// from the RDMA fabric and freed when this guard is dropped.
pub struct AllocatedBuffer<'a> {
    server_handle: *mut c_void,
    buf:           *mut c_void,
    _server:       PhantomData<&'a CuObjServer>,
}

// SAFETY: the server handle is thread-safe; each AllocatedBuffer owns its
// distinct allocation.
unsafe impl Send for AllocatedBuffer<'_> {}
unsafe impl Sync for AllocatedBuffer<'_> {}

impl AllocatedBuffer<'_> {
    /// Length of the buffer in bytes.
    pub fn len(&self) -> usize {
        unsafe { cuobj_srv_buf_size(self.buf) }
    }

    /// Returns `true` if the buffer has zero size.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw pointer to the host data region.
    ///
    /// Use [`as_slice`][Self::as_slice] / [`as_mut_slice`][Self::as_mut_slice]
    /// for slice access.
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { cuobj_srv_buf_data(self.buf) as *const u8 }
    }

    /// Raw mutable pointer to the host data region.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { cuobj_srv_buf_data(self.buf) as *mut u8 }
    }

    /// View the buffer contents as a byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent mutable access exists for the
    /// duration of the returned reference.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let ptr = cuobj_srv_buf_data(self.buf) as *const u8;
            let len = cuobj_srv_buf_size(self.buf);
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// View the buffer contents as a mutable byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure no other references (mutable or shared) to this
    /// buffer exist for the duration of the returned reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            let ptr = cuobj_srv_buf_data(self.buf) as *mut u8;
            let len = cuobj_srv_buf_size(self.buf);
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }
}

impl Drop for AllocatedBuffer<'_> {
    fn drop(&mut self) {
        unsafe { cuobj_srv_deregister_free(self.server_handle, self.buf) };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PersistentBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A 4 KiB-aligned RDMA-registered host buffer whose lifetime is tied to an
/// [`Arc<CuObjServer>`] rather than a borrow of one.
///
/// Unlike [`AllocatedBuffer`], which uses a borrowed `&CuObjServer` (and
/// therefore cannot outlive the server reference), `PersistentBuffer` holds an
/// `Arc<CuObjServer>` and can therefore be stored in long-lived structures
/// such as buffer pools without lifetime parameters.
///
/// The buffer is allocated and registered with the RDMA fabric once, on
/// construction, and deregistered + freed only when `PersistentBuffer` is
/// dropped.
///
/// # Usage
///
/// ```no_run
/// use std::sync::Arc;
/// use cuobj::{CuObjServer, PersistentBuffer};
///
/// let server = Arc::new(CuObjServer::new("192.168.100.1", 4421).unwrap());
/// let mut buf = PersistentBuffer::new(Arc::clone(&server), 4 * 1024 * 1024).unwrap();
/// buf.as_mut_slice().fill(0xAB_u8);
/// ```
pub struct PersistentBuffer {
    server: Arc<CuObjServer>,
    buf:    *mut c_void,
}

// SAFETY: the server handle is internally thread-safe; each `PersistentBuffer`
// owns its distinct allocation.
unsafe impl Send for PersistentBuffer {}
unsafe impl Sync for PersistentBuffer {}

impl PersistentBuffer {
    /// Allocate a 4 KiB-aligned host buffer of `size` bytes and register it
    /// with the RDMA fabric.
    pub fn new(server: Arc<CuObjServer>, size: usize) -> Result<Self, CuObjServerError> {
        let buf = unsafe { cuobj_srv_alloc_register(server.handle, size) };
        if buf.is_null() {
            return Err(CuObjServerError::AllocError(
                "cuobj_srv_alloc_register returned null".into(),
            ));
        }
        Ok(Self { server, buf })
    }

    /// Length of the buffer in bytes.
    pub fn len(&self) -> usize {
        unsafe { cuobj_srv_buf_size(self.buf) }
    }

    /// Returns `true` if the buffer has zero size.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw const pointer to the host data region.
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { cuobj_srv_buf_data(self.buf) as *const u8 }
    }

    /// Raw mutable pointer to the host data region.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { cuobj_srv_buf_data(self.buf) as *mut u8 }
    }

    /// View the buffer as a byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent mutable access exists.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let ptr = cuobj_srv_buf_data(self.buf) as *const u8;
            std::slice::from_raw_parts(ptr, self.len())
        }
    }

    /// View the buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            let ptr = cuobj_srv_buf_data(self.buf) as *mut u8;
            std::slice::from_raw_parts_mut(ptr, self.len())
        }
    }

    /// Perform an RDMA_READ from the remote client buffer into this buffer.
    ///
    /// A fresh channel is allocated internally and freed when the operation
    /// completes.  `offset` is the byte offset within the client's registered
    /// buffer; use `0` for a full-buffer transfer.
    pub fn handle_put(
        &self,
        key:    &str,
        offset: usize,
        size:   usize,
        token:  &str,
    ) -> Result<usize, CuObjServerError> {
        let remote_start = token_remote_addr(token) + offset;
        let ch_id = unsafe { cuobj_srv_alloc_channel(self.server.handle) };
        let result = do_handle_put(self.server.handle, key, self.buf, remote_start, size, token, ch_id);
        unsafe { cuobj_srv_free_channel(self.server.handle, ch_id) };
        result
    }

    /// Perform an RDMA_WRITE from this buffer into the remote client buffer.
    ///
    /// A fresh channel is allocated internally and freed when the operation
    /// completes.  `offset` is the byte offset within the client's registered
    /// buffer to write into; use `0` for a full-buffer transfer.
    pub fn handle_get(
        &self,
        key:    &str,
        offset: usize,
        size:   usize,
        token:  &str,
    ) -> Result<usize, CuObjServerError> {
        let remote_start = token_remote_addr(token) + offset;
        let ch_id = unsafe { cuobj_srv_alloc_channel(self.server.handle) };
        let result = do_handle_get(self.server.handle, key, self.buf, remote_start, size, token, ch_id);
        unsafe { cuobj_srv_free_channel(self.server.handle, ch_id) };
        result
    }

    /// Raw pointer to the internal `CuObjSrvBuffer*` handle.
    ///
    /// Intended for advanced callers that need to call sys-level functions
    /// directly (e.g., to pair with a pre-allocated channel).
    pub fn raw_buf_handle(&self) -> *mut c_void {
        self.buf
    }
}

impl Drop for PersistentBuffer {
    fn drop(&mut self) {
        unsafe { cuobj_srv_deregister_free(self.server.handle, self.buf) };
    }
}
