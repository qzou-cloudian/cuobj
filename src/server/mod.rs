// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ergonomic safe Rust wrapper around the NVIDIA cuObj server API.
//!
//! This module sits on top of `cuobj-sys` and provides:
//!
//! * [`CuObjServer`] — thread-safe main entry point.  Wraps a `cuObjServer`
//!   C++ object that listens on an RDMA-capable NIC and services RDMA_READ
//!   (PUT) and RDMA_WRITE (GET) requests from clients.
//! * [`Channel`] — RAII guard that returns a channel ID to the server's pool
//!   on drop.  Each concurrent RDMA operation must hold a distinct channel.
//! * [`AllocatedBuffer`] — RAII guard for a 4 KiB-aligned RDMA-registered
//!   host buffer.  Deregisters and frees on drop.
//! * [`PersistentBuffer`] — like [`AllocatedBuffer`] but constructed from
//!   [`Arc<CuObjServer>`], suitable for long-lived buffer pools.
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

use std::ffi::{c_void, CString};
use std::sync::Arc;

pub use cuobj_sys::CuObjAsyncEvent;
use cuobj_sys::{
    cuobj_srv_alloc_channel, cuobj_srv_alloc_register, cuobj_srv_buf_data, cuobj_srv_buf_size,
    cuobj_srv_create, cuobj_srv_deregister_buffer, cuobj_srv_deregister_free, cuobj_srv_destroy,
    cuobj_srv_free_channel, cuobj_srv_handle_get, cuobj_srv_handle_get_async, cuobj_srv_handle_put,
    cuobj_srv_handle_put_async, cuobj_srv_is_connected, cuobj_srv_poll, cuobj_srv_register_buffer,
    cuobj_srv_set_telem_flags, cuobj_srv_setup_telemetry, cuobj_srv_shutdown_telemetry,
    ibv_wc_status, wc_status,
};

// ─────────────────────────────────────────────────────────────────────────────
// WC status display helper
// ─────────────────────────────────────────────────────────────────────────────

/// Human-readable wrapper around [`ibv_wc_status`].
#[derive(Clone, Copy, Debug)]
pub struct WcStatus(pub ibv_wc_status);

impl std::fmt::Display for WcStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self.0 {
            wc_status::SUCCESS => "SUCCESS",
            wc_status::LOC_LEN_ERR => "LOC_LEN_ERR",
            wc_status::LOC_QP_OP_ERR => "LOC_QP_OP_ERR",
            wc_status::LOC_EEC_OP_ERR => "LOC_EEC_OP_ERR",
            wc_status::LOC_PROT_ERR => "LOC_PROT_ERR",
            wc_status::WR_FLUSH_ERR => "WR_FLUSH_ERR",
            wc_status::MW_BIND_ERR => "MW_BIND_ERR",
            wc_status::BAD_RESP_ERR => "BAD_RESP_ERR",
            wc_status::LOC_ACCESS_ERR => "LOC_ACCESS_ERR",
            wc_status::REM_INV_REQ_ERR => "REM_INV_REQ_ERR",
            wc_status::REM_ACCESS_ERR => "REM_ACCESS_ERR",
            wc_status::REM_OP_ERR => "REM_OP_ERR",
            wc_status::RETRY_EXC_ERR => "RETRY_EXC_ERR",
            wc_status::RNR_RETRY_EXC_ERR => "RNR_RETRY_EXC_ERR",
            wc_status::LOC_RDD_VIOL_ERR => "LOC_RDD_VIOL_ERR",
            wc_status::REM_INV_RD_REQ_ERR => "REM_INV_RD_REQ_ERR",
            wc_status::REM_ABORT_ERR => "REM_ABORT_ERR",
            wc_status::INV_EECN_ERR => "INV_EECN_ERR",
            wc_status::INV_EEC_STATE_ERR => "INV_EEC_STATE_ERR",
            wc_status::FATAL_ERR => "FATAL_ERR",
            wc_status::RESP_TIMEOUT_ERR => "RESP_TIMEOUT_ERR",
            wc_status::GENERAL_ERR => "GENERAL_ERR",
            _ => return write!(f, "{}", self.0),
        };
        write!(f, "{name} ({})", self.0)
    }
}

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

    /// Polling the completion queue failed.
    #[error("poll channel (id={channel_id}) failed: cuobj_srv_poll returned {raw_return}")]
    PollFailed { channel_id: u16, raw_return: isize },

    /// Async operation submission failed (before any RDMA transfer started).
    #[error("async submission of key '{key}' failed: server returned {raw_return}")]
    SubmitFailed { key: String, raw_return: isize },

    /// An RDMA PUT (RDMA_READ from client) failed.
    #[error(
        "RDMA PUT failed for key '{key}': server returned {transferred}, WC status: {wc_status}"
    )]
    PutFailed {
        key: String,
        transferred: isize,
        wc_status: WcStatus,
    },

    /// An RDMA GET (RDMA_WRITE to client) failed.
    #[error(
        "RDMA GET failed for key '{key}': server returned {transferred}, WC status: {wc_status}"
    )]
    GetFailed {
        key: String,
        transferred: isize,
        wc_status: WcStatus,
    },

    /// `allocateChannelId` returned `INVALID_CHANNEL_ID` — no channels available.
    #[error(
        "channel allocation failed: no channels available (server returned INVALID_CHANNEL_ID)"
    )]
    ChannelAllocFailed,

    /// The RDMA token could not be parsed.
    #[error("failed to parse RDMA token: {0}")]
    InvalidToken(String),

    /// A NUL byte appeared in a string argument.
    #[error("argument contains interior NUL byte: {0}")]
    NulError(#[from] std::ffi::NulError),
}

// ─────────────────────────────────────────────────────────────────────────────
// ServerHandle — shared ownership of the C++ cuObjServer
// ─────────────────────────────────────────────────────────────────────────────

/// Shared handle to the C++ `cuObjServer` object.
///
/// [`CuObjServer`], [`AllocatedBuffer`], [`Channel`], and [`PersistentBuffer`]
/// all hold an `Arc` to this handle so the C++ object is not destroyed until
/// all of them have been dropped.
struct ServerHandle(*mut c_void);

// SAFETY: cuObjServer uses channel IDs for internal thread isolation.
unsafe impl Send for ServerHandle {}
unsafe impl Sync for ServerHandle {}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        unsafe { cuobj_srv_destroy(self.0) };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal token helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the client's buffer virtual address from the first colon-delimited
/// hex field of an RDMA token.
///
/// Token format: `<remote_vaddr_hex>:<size_hex>:<rkey_hex>:…`
///
/// This address must be used as `remote_start` in the underlying C API.
/// Adding a byte offset to it points to the correct chunk position within
/// the client's registered buffer.
fn token_remote_addr(token: &str) -> Result<usize, CuObjServerError> {
    token
        .split(':')
        .next()
        .and_then(|s| usize::from_str_radix(s, 16).ok())
        .ok_or_else(|| {
            CuObjServerError::InvalidToken(format!(
                "remote address field missing or unparseable in token: {token}"
            ))
        })
}

/// Extract the client buffer size (second colon-separated hex field) from a
/// cuObj RDMA token.
fn token_client_size(token: &str) -> Result<usize, CuObjServerError> {
    token
        .split(':')
        .nth(1)
        .and_then(|s| usize::from_str_radix(s, 16).ok())
        .ok_or_else(|| {
            CuObjServerError::InvalidToken(format!(
                "client size field missing or unparseable in token: {token}"
            ))
        })
}

/// Resolve the remote buffer start address and clamp the transfer size to
/// the client buffer limit encoded in the token. Returns `(remote_start, clamped_size)`.
fn token_resolve_clamp(
    token: &str,
    offset: usize,
    size: usize,
) -> Result<(usize, usize), CuObjServerError> {
    let remote_start = token_remote_addr(token)? + offset;
    let max_size = token_client_size(token)?.saturating_sub(offset);
    Ok((remote_start, size.min(max_size)))
}

/// Internal helper: call `cuobj_srv_handle_put` given raw handles.
fn do_handle_put(
    server_handle: *mut c_void,
    key: &str,
    buf: *mut c_void,
    remote_start: usize,
    size: usize,
    rdma_token: &str,
    channel_id: u16,
) -> Result<usize, CuObjServerError> {
    let key_cs = CString::new(key)?;
    let tok_cs = CString::new(rdma_token)?;
    let mut wc_status: ibv_wc_status = 0;
    let n = unsafe {
        cuobj_srv_handle_put(
            server_handle,
            key_cs.as_ptr(),
            buf,
            remote_start as u64,
            size,
            tok_cs.as_ptr(),
            channel_id,
            &mut wc_status,
        )
    };
    if n < 0 {
        return Err(CuObjServerError::PutFailed {
            key: key.to_owned(),
            transferred: n,
            wc_status: WcStatus(wc_status),
        });
    }
    Ok(n as usize)
}

/// On-drop guard that frees a channel allocated for a single-operation scope.
///
/// Used in `PersistentBuffer` when `channel_id` is `None` to ensure the
/// auto-allocated channel is freed even if the inner operation panics.
struct ScopedChannel {
    server_handle: *mut c_void,
    id: u16,
}

impl Drop for ScopedChannel {
    fn drop(&mut self) {
        unsafe { cuobj_srv_free_channel(self.server_handle, self.id) };
    }
}

/// Internal helper: call `cuobj_srv_handle_get` given raw handles.
fn do_handle_get(
    server_handle: *mut c_void,
    key: &str,
    buf: *mut c_void,
    remote_start: usize,
    size: usize,
    rdma_token: &str,
    channel_id: u16,
) -> Result<usize, CuObjServerError> {
    let key_cs = CString::new(key)?;
    let tok_cs = CString::new(rdma_token)?;
    let mut wc_status: ibv_wc_status = 0;
    let n = unsafe {
        cuobj_srv_handle_get(
            server_handle,
            key_cs.as_ptr(),
            buf,
            remote_start as u64,
            size,
            tok_cs.as_ptr(),
            channel_id,
            &mut wc_status,
        )
    };
    if n < 0 {
        return Err(CuObjServerError::GetFailed {
            key: key.to_owned(),
            transferred: n,
            wc_status: WcStatus(wc_status),
        });
    }
    Ok(n as usize)
}

// ─────────────────────────────────────────────────────────────────────────────
// CuObjServer
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper around the `cuObjServer` C++ object.
///
/// This type is not [`Clone`] — cloning would duplicate ownership of the
/// underlying C++ handle and lead to a double-free.
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
///         └── Drop                     → cuobj_srv_destroy (when last Arc released)
/// ```
pub struct CuObjServer {
    srv: Arc<ServerHandle>,
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
    /// * `port` — RDMA port (e.g. `4421`).
    ///
    /// # Errors
    ///
    /// Returns [`CuObjServerError::CreateFailed`] if the underlying
    /// `cuObjServer` constructor fails.
    pub fn new(ip: &str, port: u16) -> Result<Self, CuObjServerError> {
        let ip_cstr = CString::new(ip)?;
        let raw = unsafe { cuobj_srv_create(ip_cstr.as_ptr(), port) };
        if raw.is_null() {
            return Err(CuObjServerError::CreateFailed {
                ip: ip.to_owned(),
                port,
            });
        }
        Ok(Self {
            srv: Arc::new(ServerHandle(raw)),
        })
    }

    /// Returns `true` if the server is connected to the RDMA fabric.
    pub fn is_connected(&self) -> bool {
        unsafe { cuobj_srv_is_connected(self.srv.0) != 0 }
    }

    /// Raw C++ server handle for direct FFI use.
    pub fn handle(&self) -> *mut c_void {
        self.srv.0
    }

    // ── Telemetry (static / library-wide) ────────────────────────────────────

    /// Set up the telemetry output stream for all `cuObjServer` instances in
    /// this process.
    ///
    /// * `use_otel` — `true` for OpenTelemetry backend, `false` for ostream.
    /// * `log_file` — optional file path for ostream mode; `None` uses stdout.
    pub fn setup_telemetry(use_otel: bool, log_file: Option<&str>) {
        // Panics if `log_file` contains an interior NUL byte (path strings
        // never should, so this is acceptable in practice).
        let file_path = log_file.map(|s| std::ffi::CString::new(s).unwrap());
        let file_ptr = file_path
            .as_ref()
            .map(|cs| cs.as_ptr())
            .unwrap_or(std::ptr::null());
        unsafe {
            cuobj_srv_setup_telemetry(use_otel as i32, file_ptr);
        }
    }

    /// Shut down telemetry for all `cuObjServer` instances (reset to default
    /// output stream).
    pub fn shutdown_telemetry() {
        unsafe { cuobj_srv_shutdown_telemetry() };
    }

    /// Set telemetry logging flags for all `cuObjServer` instances.
    pub fn set_telem_flags(flags: u32) {
        unsafe { cuobj_srv_set_telem_flags(flags) };
    }

    // ── Channel management ───────────────────────────────────────────────────

    /// Allocate a unique channel ID for one concurrent RDMA operation.
    ///
    /// The channel is returned to the pool when the returned [`Channel`] is
    /// dropped.  Every concurrent RDMA transfer (across threads) must hold a
    /// distinct channel.
    ///
    /// Returns [`CuObjServerError::ChannelAllocFailed`] when no channels are
    /// available (`INVALID_CHANNEL_ID` returned by the C API).
    pub fn alloc_channel(&self) -> Result<Channel, CuObjServerError> {
        let id = unsafe { cuobj_srv_alloc_channel(self.srv.0) };
        if id == u16::MAX {
            return Err(CuObjServerError::ChannelAllocFailed);
        }
        Ok(Channel {
            srv: Arc::clone(&self.srv),
            id,
        })
    }

    /// Free a channel by its raw `u16` ID, bypassing the [`Channel`] RAII guard.
    ///
    /// Use this when the channel was allocated via [`alloc_channel`][Self::alloc_channel]
    /// and the [`Channel`] guard was leaked (e.g., via [`std::mem::forget`]) to
    /// decouple the channel lifetime from the server borrow.  This is necessary
    /// for worker threads that outlive a single `&CuObjServer` reference.
    ///
    /// Passing `u16::MAX` (`INVALID_CHANNEL_ID`) is a no-op.
    pub fn free_channel_by_id(&self, channel_id: u16) {
        if channel_id != u16::MAX {
            unsafe { cuobj_srv_free_channel(self.srv.0, channel_id) };
        }
    }

    /// Poll for completed async operations on `channel`.
    ///
    /// Fills up to `events.len()` entries and returns the number completed.
    /// Non-blocking — returns 0 if no completions are ready yet.
    ///
    /// # Errors
    ///
    /// Returns [`CuObjServerError::PollFailed`] on unexpected C-level errors.
    pub fn poll_channel(
        &self,
        events: &mut [CuObjAsyncEvent],
        channel_id: u16,
    ) -> Result<usize, CuObjServerError> {
        let n = unsafe {
            cuobj_srv_poll(
                self.srv.0,
                events.as_mut_ptr(),
                events.len() as std::os::raw::c_int,
                channel_id,
            )
        };
        if n < 0 {
            return Err(CuObjServerError::PollFailed {
                channel_id,
                raw_return: n as isize,
            });
        }
        Ok(n as usize)
    }

    // ── Buffer management ────────────────────────────────────────────────────

    /// Register an existing host buffer for RDMA.
    ///
    /// Returns an opaque `*mut c_void` handle for deregistration.
    /// Does NOT free the host memory on deregister — caller owns that.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live allocation of at least `size` bytes that
    /// remains valid until the returned handle is passed to
    /// [`deregister_buffer`][Self::deregister_buffer].
    pub unsafe fn register_existing_buffer(
        &self,
        ptr: *mut u8,
        size: usize,
    ) -> Option<*mut c_void> {
        let rbuf = unsafe { cuobj_srv_register_buffer(self.srv.0, ptr as *mut c_void, size) };
        if rbuf.is_null() {
            None
        } else {
            Some(rbuf)
        }
    }

    /// Deregister a buffer registered via [`register_existing_buffer`](Self::register_existing_buffer).
    /// Does NOT free the host memory.
    ///
    /// # Safety
    ///
    /// `rbuf` must be a handle previously returned by
    /// [`register_existing_buffer`][Self::register_existing_buffer] on the same
    /// server instance and must not be used again after this call. The caller is
    /// responsible for freeing the underlying host memory separately.
    pub unsafe fn deregister_buffer(&self, rbuf: *mut c_void) {
        unsafe { cuobj_srv_deregister_buffer(self.srv.0, rbuf) };
    }

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
    pub fn alloc_buffer(&self, size: usize) -> Result<AllocatedBuffer, CuObjServerError> {
        let buf = unsafe { cuobj_srv_alloc_register(self.srv.0, size) };
        if buf.is_null() {
            return Err(CuObjServerError::BufferAllocFailed { size });
        }
        let data_ptr = unsafe { cuobj_srv_buf_data(buf) as *mut u8 };
        let buf_size = unsafe { cuobj_srv_buf_size(buf) };
        Ok(AllocatedBuffer {
            srv: Arc::clone(&self.srv),
            buf,
            data_ptr,
            size: buf_size,
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
        buf: &AllocatedBuffer,
        offset: usize,
        size: usize,
        rdma_token: &str,
        channel: &Channel,
    ) -> Result<usize, CuObjServerError> {
        let (remote_start, size) = token_resolve_clamp(rdma_token, offset, size)?;
        do_handle_put(
            self.srv.0,
            key,
            buf.buf,
            remote_start,
            size,
            rdma_token,
            channel.id,
        )
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
        buf: &AllocatedBuffer,
        offset: usize,
        size: usize,
        rdma_token: &str,
        channel: &Channel,
    ) -> Result<usize, CuObjServerError> {
        let (remote_start, size) = token_resolve_clamp(rdma_token, offset, size)?;
        do_handle_get(
            self.srv.0,
            key,
            buf.buf,
            remote_start,
            size,
            rdma_token,
            channel.id,
        )
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
        offset: usize,
        size: usize,
        rdma_token: &str,
    ) -> Result<Vec<u8>, CuObjServerError> {
        let buf = self.alloc_buffer(size)?;
        let channel = self.alloc_channel()?;
        let n = self.handle_put(key, &buf, offset, size, rdma_token, &channel)?;
        // Clamp: guard against the C++ layer reporting more bytes than the buffer holds.
        let len = n.min(buf.len());
        Ok(buf[..len].to_vec())
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
        offset: usize,
        data: &[u8],
        rdma_token: &str,
    ) -> Result<usize, CuObjServerError> {
        let mut buf = self.alloc_buffer(data.len())?;
        buf[..data.len()].copy_from_slice(data);
        let channel = self.alloc_channel()?;
        self.handle_get(key, &buf, offset, data.len(), rdma_token, &channel)
    }
}

impl std::fmt::Debug for CuObjServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CuObjServer")
            .field("handle", &self.srv.0)
            .finish()
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
///
/// `Channel` holds an `Arc` to the underlying server handle, so it has no
/// lifetime parameter and can be stored in structs or held across `async`
/// await points.
pub struct Channel {
    srv: Arc<ServerHandle>,
    id: u16,
}

// SAFETY: the server handle is thread-safe; channel IDs are per-operation.
unsafe impl Send for Channel {}
unsafe impl Sync for Channel {}

impl Channel {
    /// The raw channel ID.
    #[inline]
    pub fn id(&self) -> u16 {
        self.id
    }

    /// Consume this guard and return the raw channel ID **without** returning
    /// it to the pool.
    ///
    /// Use this when the channel must outlive the current scope — for example
    /// when handing ownership to a worker thread.  The caller is then fully
    /// responsible for eventually passing the ID to
    /// [`CuObjServer::free_channel_by_id`].
    #[inline]
    pub fn into_id(self) -> u16 {
        let id = self.id;
        std::mem::forget(self);
        id
    }
}

impl Drop for Channel {
    fn drop(&mut self) {
        // u16::MAX == INVALID_CHANNEL_ID — freeing it is undefined behaviour.
        if self.id != u16::MAX {
            unsafe { cuobj_srv_free_channel(self.srv.0, self.id) };
        }
    }
}

impl std::fmt::Debug for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Channel").field("id", &self.id).finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AllocatedBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard for a 4 KiB-aligned RDMA-registered host buffer.
///
/// Obtained via [`CuObjServer::alloc_buffer`].  The buffer is deregistered
/// from the RDMA fabric and freed when this guard is dropped.
///
/// `AllocatedBuffer` holds an `Arc` to the underlying server handle, so it
/// has no lifetime parameter and can be stored in structs or held across
/// `async` await points.
pub struct AllocatedBuffer {
    srv: Arc<ServerHandle>,
    buf: *mut c_void,
    data_ptr: *mut u8,
    size: usize,
}

// SAFETY: the server handle is thread-safe; each AllocatedBuffer owns its
// distinct allocation.
unsafe impl Send for AllocatedBuffer {}
unsafe impl Sync for AllocatedBuffer {}

impl AllocatedBuffer {
    /// Length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the buffer has zero size.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Raw pointer to the host data region.
    ///
    /// Use [`as_slice`][Self::as_slice] / [`as_mut_slice`][Self::as_mut_slice]
    /// for slice access.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data_ptr as *const u8
    }

    /// Raw mutable pointer to the host data region.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data_ptr
    }

    /// View the buffer contents as a byte slice.
    ///
    /// # Concurrency
    ///
    /// The caller must ensure no concurrent mutable access exists for the
    /// duration of the returned reference.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data_ptr as *const u8, self.size) }
    }

    /// View the buffer contents as a mutable byte slice.
    ///
    /// # Concurrency
    ///
    /// The caller must ensure no other references (mutable or shared) to this
    /// buffer exist for the duration of the returned reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr, self.size) }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe { cuobj_srv_deregister_free(self.srv.0, self.buf) };
    }
}

impl std::fmt::Debug for AllocatedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocatedBuffer")
            .field("buf", &self.buf)
            .finish()
    }
}

impl std::ops::Deref for AllocatedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AllocatedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PersistentBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A 4 KiB-aligned RDMA-registered host buffer with no lifetime parameter.
///
/// Like [`AllocatedBuffer`], but constructed from an [`Arc<CuObjServer>`]
/// instead of a borrow, making it suitable for long-lived structures such as
/// buffer pools.  Internally it holds an `Arc` to the shared server handle so
/// the C++ object remains alive for as long as any `PersistentBuffer` is live.
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
    srv: Arc<ServerHandle>,
    buf: *mut c_void,
    data_ptr: *mut u8,
    size: usize,
}

// SAFETY: the server handle is internally thread-safe; each `PersistentBuffer`
// owns its distinct allocation.
unsafe impl Send for PersistentBuffer {}
unsafe impl Sync for PersistentBuffer {}

impl PersistentBuffer {
    /// Allocate a 4 KiB-aligned host buffer of `size` bytes and register it
    /// with the RDMA fabric.
    pub fn new(server: Arc<CuObjServer>, size: usize) -> Result<Self, CuObjServerError> {
        let srv = Arc::clone(&server.srv);
        let buf = unsafe { cuobj_srv_alloc_register(srv.0, size) };
        if buf.is_null() {
            return Err(CuObjServerError::BufferAllocFailed { size });
        }
        let data_ptr = unsafe { cuobj_srv_buf_data(buf) as *mut u8 };
        let buf_size = unsafe { cuobj_srv_buf_size(buf) };
        Ok(Self {
            srv,
            buf,
            data_ptr,
            size: buf_size,
        })
    }

    /// Length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the buffer has zero size.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Raw const pointer to the host data region.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data_ptr as *const u8
    }

    /// Raw mutable pointer to the host data region.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data_ptr
    }

    /// View the buffer as a byte slice.
    ///
    /// # Concurrency
    ///
    /// The caller must ensure no concurrent mutable access exists.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data_ptr as *const u8, self.size) }
    }

    /// View the buffer as a mutable byte slice.
    ///
    /// # Concurrency
    ///
    /// The caller must ensure no other references (mutable or shared) to this
    /// buffer exist for the duration of the returned reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr, self.size) }
    }

    /// Perform an RDMA_READ from the remote client buffer into this buffer.
    ///
    /// `offset` is the byte offset within the client's registered buffer.
    /// When `channel_id` is `None` a fresh channel is allocated and freed
    /// around the call; pass `Some(id)` from a pre-allocated
    /// [`Channel`] to avoid that overhead (required for safe concurrent use
    /// from multiple threads).
    pub fn handle_put(
        &self,
        key: &str,
        offset: usize,
        size: usize,
        token: &str,
        channel_id: Option<u16>,
    ) -> Result<usize, CuObjServerError> {
        let (remote_start, size) = token_resolve_clamp(token, offset, size)?;
        match channel_id {
            Some(ch_id) => {
                do_handle_put(self.srv.0, key, self.buf, remote_start, size, token, ch_id)
            }
            None => {
                let ch_id = unsafe { cuobj_srv_alloc_channel(self.srv.0) };
                if ch_id == u16::MAX {
                    return Err(CuObjServerError::ChannelAllocFailed);
                }
                let _scoped = ScopedChannel {
                    server_handle: self.srv.0,
                    id: ch_id,
                };
                do_handle_put(self.srv.0, key, self.buf, remote_start, size, token, ch_id)
            }
        }
    }

    /// Perform an RDMA_WRITE from this buffer into the remote client buffer.
    ///
    /// `offset` is the byte offset within the client's registered buffer to
    /// write into.  When `channel_id` is `None` a fresh channel is allocated
    /// and freed around the call; pass `Some(id)` from a pre-allocated
    /// [`Channel`] to avoid that overhead (required for safe concurrent use
    /// from multiple threads).
    pub fn handle_get(
        &self,
        key: &str,
        offset: usize,
        size: usize,
        token: &str,
        channel_id: Option<u16>,
    ) -> Result<usize, CuObjServerError> {
        let (remote_start, size) = token_resolve_clamp(token, offset, size)?;
        match channel_id {
            Some(ch_id) => {
                do_handle_get(self.srv.0, key, self.buf, remote_start, size, token, ch_id)
            }
            None => {
                let ch_id = unsafe { cuobj_srv_alloc_channel(self.srv.0) };
                if ch_id == u16::MAX {
                    return Err(CuObjServerError::ChannelAllocFailed);
                }
                let _scoped = ScopedChannel {
                    server_handle: self.srv.0,
                    id: ch_id,
                };
                do_handle_get(self.srv.0, key, self.buf, remote_start, size, token, ch_id)
            }
        }
    }

    /// Raw pointer to the internal `CuObjSrvBuffer*` handle.
    ///
    /// Intended for advanced callers that need to call sys-level functions
    /// directly (e.g., to pair with a pre-allocated channel).
    pub fn raw_buf_handle(&self) -> *mut c_void {
        self.buf
    }

    /// Submit an async RDMA PUT (RDMA_READ from client into this buffer).
    ///
    /// Returns immediately after submission (return value 0 from the C API).
    /// Call [`CuObjServer::poll_channel`] on `channel_id` to retrieve the
    /// completion; `async_handle` will be returned in the event to identify
    /// this operation.
    ///
    /// # Safety
    ///
    /// `async_handle` must remain valid until the corresponding completion is
    /// harvested via [`CuObjServer::poll_channel`].
    pub unsafe fn submit_put_async(
        &self,
        key: &str,
        offset: usize,
        size: usize,
        token: &str,
        channel_id: u16,
        async_handle: *mut c_void,
    ) -> Result<(), CuObjServerError> {
        let (remote_start, size) = token_resolve_clamp(token, offset, size)?;
        let key_cs = CString::new(key)?;
        let tok_cs = CString::new(token)?;
        let rc = unsafe {
            cuobj_srv_handle_put_async(
                self.srv.0,
                key_cs.as_ptr(),
                self.buf,
                remote_start as u64,
                size,
                tok_cs.as_ptr(),
                channel_id,
                async_handle,
            )
        };
        if rc < 0 {
            return Err(CuObjServerError::SubmitFailed {
                key: key.to_owned(),
                raw_return: rc,
            });
        }
        Ok(())
    }

    /// Submit an async RDMA GET (RDMA_WRITE from this buffer to client).
    ///
    /// Populate this buffer with the object data **before** calling.
    /// Returns immediately after submission.  Call [`CuObjServer::poll_channel`]
    /// to retrieve the completion.
    ///
    /// # Safety
    ///
    /// `async_handle` must remain valid until the corresponding completion is
    /// harvested via [`CuObjServer::poll_channel`].
    pub unsafe fn submit_get_async(
        &self,
        key: &str,
        offset: usize,
        size: usize,
        token: &str,
        channel_id: u16,
        async_handle: *mut c_void,
    ) -> Result<(), CuObjServerError> {
        let (remote_start, size) = token_resolve_clamp(token, offset, size)?;
        let key_cs = CString::new(key)?;
        let tok_cs = CString::new(token)?;
        let rc = unsafe {
            cuobj_srv_handle_get_async(
                self.srv.0,
                key_cs.as_ptr(),
                self.buf,
                remote_start as u64,
                size,
                tok_cs.as_ptr(),
                channel_id,
                async_handle,
            )
        };
        if rc < 0 {
            return Err(CuObjServerError::SubmitFailed {
                key: key.to_owned(),
                raw_return: rc,
            });
        }
        Ok(())
    }
}

impl Drop for PersistentBuffer {
    fn drop(&mut self) {
        unsafe { cuobj_srv_deregister_free(self.srv.0, self.buf) };
    }
}

impl std::fmt::Debug for PersistentBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentBuffer")
            .field("buf", &self.buf)
            .finish()
    }
}

impl std::ops::Deref for PersistentBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for PersistentBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WcStatus::Display ──────────────────────────────────────────────────────

    #[test]
    fn wc_status_display_known() {
        let names = [
            (wc_status::SUCCESS, "SUCCESS"),
            (wc_status::LOC_LEN_ERR, "LOC_LEN_ERR"),
            (wc_status::LOC_QP_OP_ERR, "LOC_QP_OP_ERR"),
            (wc_status::LOC_EEC_OP_ERR, "LOC_EEC_OP_ERR"),
            (wc_status::LOC_PROT_ERR, "LOC_PROT_ERR"),
            (wc_status::WR_FLUSH_ERR, "WR_FLUSH_ERR"),
            (wc_status::MW_BIND_ERR, "MW_BIND_ERR"),
            (wc_status::BAD_RESP_ERR, "BAD_RESP_ERR"),
            (wc_status::LOC_ACCESS_ERR, "LOC_ACCESS_ERR"),
            (wc_status::REM_INV_REQ_ERR, "REM_INV_REQ_ERR"),
            (wc_status::REM_ACCESS_ERR, "REM_ACCESS_ERR"),
            (wc_status::REM_OP_ERR, "REM_OP_ERR"),
            (wc_status::RETRY_EXC_ERR, "RETRY_EXC_ERR"),
            (wc_status::RNR_RETRY_EXC_ERR, "RNR_RETRY_EXC_ERR"),
            (wc_status::LOC_RDD_VIOL_ERR, "LOC_RDD_VIOL_ERR"),
            (wc_status::REM_INV_RD_REQ_ERR, "REM_INV_RD_REQ_ERR"),
            (wc_status::REM_ABORT_ERR, "REM_ABORT_ERR"),
            (wc_status::INV_EECN_ERR, "INV_EECN_ERR"),
            (wc_status::INV_EEC_STATE_ERR, "INV_EEC_STATE_ERR"),
            (wc_status::FATAL_ERR, "FATAL_ERR"),
            (wc_status::RESP_TIMEOUT_ERR, "RESP_TIMEOUT_ERR"),
            (wc_status::GENERAL_ERR, "GENERAL_ERR"),
        ];
        for (code, name) in names {
            let s = WcStatus(code).to_string();
            assert!(
                s.contains(name),
                "expected '{name}' in display for {code}, got: {s}"
            );
            assert!(
                s.contains(&code.to_string()),
                "expected code {code} in display: {s}"
            );
        }
    }

    #[test]
    fn wc_status_display_unknown() {
        assert_eq!(WcStatus(99).to_string(), "99");
    }

    // ── token_remote_addr ──────────────────────────────────────────────────────

    #[test]
    fn token_remote_addr_valid() {
        // Simulated token: vaddr=0x7f001000, size=0x100000, rkey=0xDEAD
        let token = "7f001000:100000:DEAD";
        assert_eq!(token_remote_addr(token).unwrap(), 0x7f001000);
    }

    #[test]
    fn token_remote_addr_invalid() {
        assert!(token_remote_addr("").is_err());
        assert!(token_remote_addr("xyz").is_err());
        assert!(token_remote_addr(":size:rkey").is_err());
    }

    // ── token_client_size ──────────────────────────────────────────────────────

    #[test]
    fn token_client_size_valid() {
        let token = "7f001000:100000:DEAD";
        assert_eq!(token_client_size(token).unwrap(), 0x100000);
    }

    #[test]
    fn token_client_size_invalid() {
        assert!(token_client_size("").is_err());
        assert!(token_client_size("addr:").is_err());
        assert!(token_client_size("addr:xyz").is_err());
    }

    // ── token_resolve_clamp ────────────────────────────────────────────────────

    #[test]
    fn token_resolve_clamp_no_clamping() {
        let token = "1000:1000:rkey";
        let (remote_start, clamped) = token_resolve_clamp(token, 0, 0x500).unwrap();
        assert_eq!(remote_start, 0x1000);
        assert_eq!(clamped, 0x500);
    }

    #[test]
    fn token_resolve_clamp_exceeds_client_buf() {
        let token = "1000:1000:rkey";
        let (_, clamped) = token_resolve_clamp(token, 0, 0x2000).unwrap();
        assert_eq!(clamped, 0x1000);
    }

    #[test]
    fn token_resolve_clamp_with_offset() {
        let token = "1000:1000:rkey";
        // Client buffer size 0x1000; offset 0x800 leaves 0x800 available.
        let (remote_start, clamped) = token_resolve_clamp(token, 0x800, 0x1000).unwrap();
        assert_eq!(remote_start, 0x1000 + 0x800);
        assert_eq!(clamped, 0x800);
    }

    #[test]
    fn token_resolve_clamp_offset_exceeds_size() {
        // client size = 0x100, offset = 0x80 → available = 0x80; request 0x100 is clamped.
        let token = "1000:100:rkey";
        let (_, clamped) = token_resolve_clamp(token, 0x80, 0x100).unwrap();
        assert_eq!(clamped, 0x80);
    }
}
