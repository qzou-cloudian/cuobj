// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ergonomic safe Rust wrapper around the NVIDIA cuObj client API.
//!
//! This module sits on top of `cuobj-sys` and provides:
//!
//! * [`MemoryType`] and [`OpType`] — type-safe enums for memory classification
//!   and RDMA operation direction.
//! * [`CuObjRdmaToken`] — owned RDMA descriptor string produced by
//!   `cuMemObjGetRDMAToken`.  Implements [`Display`][std::fmt::Display],
//!   [`AsRef<str>`], and `From<CuObjRdmaToken>` for `String` / `Vec<u8>`.
//! * [`CuObjClient`] — thread-safe main entry point.  Internally tracks
//!   registered memory regions to enforce double-register safety.
//! * [`RegisteredBuffer`] — RAII guard that deregisters on drop.  No lifetime
//!   parameter; holds `Arc` references so it can be stored in structs and held
//!   across `async` await points.
//! * [`memory_type`] / [`is_rdma_suitable`] — free functions that classify a
//!   pointer without requiring a `CuObjClient` instance.
//!
//! # Quick start
//!
//! ```no_run
//! use cuobj::{CuObjClient, OpType};
//!
//! let client = CuObjClient::new().expect("cuObjClient init failed");
//! assert!(client.is_connected());
//!
//! let mut buf = vec![0u8; 4 * 1024 * 1024];
//! let reg = client.register_buffer(buf.as_mut_ptr(), buf.len())
//!     .expect("register failed");
//!
//! // Generate an RDMA PUT descriptor for the full buffer at offset 0.
//! let token = reg.put_token(0, buf.len()).expect("token failed");
//! println!("RDMA descriptor: {token}");
//! // `reg` drops here → automatically deregisters the buffer.
//! ```

use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, Mutex};

use cuobj_sys::{
    cuobj_client_create, cuobj_client_destroy, cuobj_client_get_memory_type,
    cuobj_client_is_connected, cuobj_client_mem_get_descriptor,
    cuobj_client_mem_get_max_request_callback_size, cuobj_client_mem_get_rdma_token,
    cuobj_client_mem_put_descriptor, cuobj_client_mem_put_rdma_token, cuobj_client_set_telem_flags,
    cuobj_client_setup_telemetry, cuobj_client_shutdown_telemetry, CUOBJ_GET,
    CUOBJ_MEMORY_CUDA_DEVICE, CUOBJ_MEMORY_CUDA_MANAGED, CUOBJ_MEMORY_INVALID, CUOBJ_MEMORY_SYSTEM,
    CUOBJ_MEMORY_UNKNOWN, CUOBJ_PUT, CU_OBJ_SUCCESS,
};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from cuObj client operations.
#[derive(Debug, thiserror::Error)]
pub enum CuObjClientError {
    /// `cuObjClient` could not be created (RDMA hardware unavailable or
    /// library not installed).
    #[error(
        "failed to create cuObjClient: RDMA hardware unavailable \
         or libcuobjclient not installed"
    )]
    CreateFailed,

    /// Memory registration failed.
    #[error("memory registration failed for ptr={ptr:#x} size={size}: {reason}")]
    RegistrationFailed {
        ptr: usize,
        size: usize,
        reason: String,
    },

    /// The pointer is already registered.
    #[error("ptr={ptr:#x} is already registered")]
    AlreadyRegistered { ptr: usize },

    /// Memory deregistration failed.
    #[error("memory deregistration failed for ptr={ptr:#x}: {reason}")]
    DeregistrationFailed { ptr: usize, reason: String },

    /// RDMA token generation failed.
    #[error("RDMA token generation failed ({op:?} offset={offset}): {reason}")]
    TokenFailed {
        op: OpType,
        offset: usize,
        reason: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryType
// ─────────────────────────────────────────────────────────────────────────────

/// Classification of a memory pointer by the cuObj runtime.
///
/// Mirrors `cuObjMemoryType_t` from `cuobjclient.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Regular pageable or pinned host (system) memory.
    System,
    /// CUDA unified/managed memory accessible from both CPU and GPU.
    CudaManaged,
    /// CUDA device memory (not accessible from CPU).
    CudaDevice,
    /// Type could not be determined.
    Unknown,
    /// Null or invalid pointer.
    Invalid,
}

impl MemoryType {
    /// Returns `true` if this memory type can participate in cuObj RDMA
    /// transfers (i.e., one of `System`, `CudaManaged`, `CudaDevice`).
    #[inline]
    pub fn is_rdma_suitable(self) -> bool {
        matches!(
            self,
            MemoryType::System | MemoryType::CudaManaged | MemoryType::CudaDevice
        )
    }

    fn from_raw(v: i32) -> Self {
        match v {
            x if x == CUOBJ_MEMORY_SYSTEM => MemoryType::System,
            x if x == CUOBJ_MEMORY_CUDA_MANAGED => MemoryType::CudaManaged,
            x if x == CUOBJ_MEMORY_CUDA_DEVICE => MemoryType::CudaDevice,
            x if x == CUOBJ_MEMORY_UNKNOWN => MemoryType::Unknown,
            x if x == CUOBJ_MEMORY_INVALID => MemoryType::Invalid,
            _ => MemoryType::Unknown,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpType
// ─────────────────────────────────────────────────────────────────────────────

/// Direction of an S3 RDMA operation.
///
/// Mirrors `cuObjOpType_t` from `cuobjclient.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    /// GET: the server will RDMA-write data **into** the client's buffer.
    Get,
    /// PUT: the server will RDMA-read data **from** the client's buffer.
    Put,
}

impl OpType {
    fn to_raw(self) -> i32 {
        match self {
            OpType::Get => CUOBJ_GET,
            OpType::Put => CUOBJ_PUT,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClientHandle — shared ownership of the C++ cuObjClient
// ─────────────────────────────────────────────────────────────────────────────

/// Shared handle to the C++ `cuObjClient` object.
///
/// Both [`CuObjClient`] and any outstanding [`CuObjRdmaToken`]s / [`RegisteredBuffer`]s
/// hold an `Arc` to this handle so the C++ object is not destroyed until all of
/// them have been dropped — even if the `CuObjClient` itself is dropped first.
#[repr(transparent)]
struct ClientHandle(*mut c_void);

// SAFETY: cuObjClient is internally thread-safe.
unsafe impl Send for ClientHandle {}
unsafe impl Sync for ClientHandle {}

impl Drop for ClientHandle {
    fn drop(&mut self) {
        unsafe { cuobj_client_destroy(self.0) };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CuObjRdmaToken
// ─────────────────────────────────────────────────────────────────────────────

/// Owned RDMA descriptor string produced by `cuMemObjGetRDMAToken`.
///
/// Wraps a C string allocated by the cuObj SDK.  The descriptor encodes the
/// client's registered memory region so that an RDMA server can perform
/// one-sided `RDMA_READ` (PUT) or `RDMA_WRITE` (GET) into it.  Send this as
/// the `x-amz-rdma-token` HTTP header value to the S3 server.
///
/// # Lifecycle
///
/// The underlying descriptor remains valid as long as this token is alive.
/// Dropping the token calls `cuMemObjPutRDMAToken` and invalidates the
/// descriptor, so the token **must not be dropped** until the server has
/// completed the RDMA operation.
///
/// The token holds an `Arc` to the shared C++ client handle, preventing the
/// C++ object from being destroyed until all tokens are dropped.  Dropping the
/// [`CuObjClient`] first is safe: the C++ destruction is deferred until the
/// last token finishes.
pub struct CuObjRdmaToken {
    /// SDK-allocated descriptor string; freed (and invalidated) on drop.
    desc_str: *mut c_char,
    /// Keeps the C++ client alive until this token is dropped.
    _handle: Arc<ClientHandle>,
}

// SAFETY: the descriptor string is a plain C string with no thread affinity.
// ClientHandle is Send + Sync.
unsafe impl Send for CuObjRdmaToken {}
unsafe impl Sync for CuObjRdmaToken {}

impl CuObjRdmaToken {
    /// View the raw descriptor bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: desc_str is a valid, NUL-terminated C string for our lifetime.
        unsafe { CStr::from_ptr(self.desc_str) }.to_bytes()
    }

    /// View the descriptor as a UTF-8 string slice.
    ///
    /// cuObj descriptors are always valid ASCII so this never fails in
    /// practice; the fallback returns `""` if the bytes are invalid UTF-8.
    #[inline]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).unwrap_or("")
    }

    /// Consume this token and return the descriptor bytes.
    ///
    /// The returned `Vec` is a copy; the SDK descriptor is freed on `Drop`.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }

    /// Consume this token and return the descriptor as a `String`.
    ///
    /// cuObj descriptors are ASCII; if the bytes happen to be invalid UTF-8
    /// they are replaced with U+FFFD.
    #[inline]
    pub fn into_string(self) -> String {
        String::from_utf8_lossy(self.as_bytes()).into_owned()
    }

    /// Length of the descriptor in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Returns `true` if the descriptor is empty (should not happen in
    /// practice for a valid token).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }
}

impl Drop for CuObjRdmaToken {
    fn drop(&mut self) {
        // Releases the SDK allocation and invalidates the descriptor.
        // _handle.0 is the raw C++ client pointer, kept alive by the Arc.
        unsafe { cuobj_client_mem_put_rdma_token(self._handle.0, self.desc_str) };
    }
}

impl std::fmt::Debug for CuObjRdmaToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CuObjRdmaToken")
            .field(&self.as_str())
            .finish()
    }
}

impl std::fmt::Display for CuObjRdmaToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<[u8]> for CuObjRdmaToken {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl AsRef<str> for CuObjRdmaToken {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<CuObjRdmaToken> for Vec<u8> {
    fn from(t: CuObjRdmaToken) -> Vec<u8> {
        t.into_bytes()
    }
}

impl From<CuObjRdmaToken> for String {
    fn from(t: CuObjRdmaToken) -> String {
        t.into_string()
    }
}

/// Classify the memory type of `ptr` without requiring a [`CuObjClient`] instance.
///
/// Equivalent to `CuObjClient::memory_type(ptr)`.
pub fn memory_type(ptr: *const u8) -> MemoryType {
    CuObjClient::memory_type(ptr)
}

/// Returns `true` if `ptr` points to RDMA-suitable memory, without requiring a
/// [`CuObjClient`] instance.
///
/// Equivalent to `CuObjClient::is_rdma_suitable(ptr)`.
pub fn is_rdma_suitable(ptr: *const u8) -> bool {
    CuObjClient::is_rdma_suitable(ptr)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: token generation helper
// ─────────────────────────────────────────────────────────────────────────────

fn make_rdma_token(
    handle: &Arc<ClientHandle>,
    ptr: *const u8,
    size: usize,
    offset: usize,
    op: OpType,
) -> Result<CuObjRdmaToken, CuObjClientError> {
    let mut desc_str: *mut c_char = std::ptr::null_mut();
    let rc = unsafe {
        cuobj_client_mem_get_rdma_token(
            handle.0,
            ptr as *mut c_void,
            size,
            offset,
            op.to_raw(),
            &mut desc_str,
        )
    };
    if rc != CU_OBJ_SUCCESS || desc_str.is_null() {
        return Err(CuObjClientError::TokenFailed {
            op,
            offset,
            reason: format!("cuMemObjGetRDMAToken returned {rc}"),
        });
    }
    Ok(CuObjRdmaToken {
        desc_str,
        _handle: Arc::clone(handle),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// CuObjClient
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper around the `cuObjClient` C++ object.
///
/// # Thread safety
///
/// `CuObjClient` is `Send + Sync`.  The cuObj library provides its own
/// internal locking; the registration tracking uses a `std::sync::Mutex`.
///
/// This type is not [`Clone`] — cloning would duplicate the ownership of the
/// underlying C++ handle and lead to a double-free.
///
/// # Lifecycle
///
/// ```text
///   CuObjClient::new()          ← cuobj_client_create()
///         │
///         ├── register(ptr, size)  ← cuMemObjGetDescriptor
///         │       │
///         │       └── put_token / get_token  ← cuMemObjGetRDMAToken
///         │
///         ├── deregister(ptr)      ← cuMemObjPutDescriptor
///         │
///         └── Drop                ← deregister outstanding → destroy C++ object
/// ```
///
/// The C++ object is not destroyed until all [`CuObjRdmaToken`]s and
/// [`RegisteredBuffer`]s created by this client have also been dropped.
pub struct CuObjClient {
    handle: Arc<ClientHandle>,
    /// Shared with every [`RegisteredBuffer`] produced by this client so that
    /// buffers can remove themselves from the set on drop without borrowing
    /// `&CuObjClient`.
    registered: Arc<Mutex<HashSet<usize>>>,
}

// SAFETY: cuObjClient is internally thread-safe; the HashSet is behind Mutex.
unsafe impl Send for CuObjClient {}
unsafe impl Sync for CuObjClient {}

impl CuObjClient {
    /// Create a new `cuObjClient` instance.
    ///
    /// # Errors
    ///
    /// Returns [`CuObjClientError::CreateFailed`] if the RDMA subsystem is
    /// unavailable or the library is not installed.
    pub fn new() -> Result<Self, CuObjClientError> {
        let raw = unsafe { cuobj_client_create() };
        if raw.is_null() {
            return Err(CuObjClientError::CreateFailed);
        }
        Ok(Self {
            handle: Arc::new(ClientHandle(raw)),
            registered: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    /// Returns `true` if the client is connected to the RDMA subsystem.
    pub fn is_connected(&self) -> bool {
        unsafe { cuobj_client_is_connected(self.handle.0) != 0 }
    }

    /// Classify the memory type of `ptr` using `cuObjClient::getMemoryType`.
    ///
    /// This is a thin wrapper around a static C++ method — no client instance
    /// is needed, but it is an associated function rather than a free function
    /// for namespace clarity.
    pub fn memory_type(ptr: *const u8) -> MemoryType {
        let v = unsafe { cuobj_client_get_memory_type(ptr as *const c_void) };
        MemoryType::from_raw(v)
    }

    /// Returns `true` if `ptr` points to memory that can participate in
    /// cuObj RDMA transfers (`System`, `CudaManaged`, or `CudaDevice`).
    ///
    /// Equivalent to `CuObjClient::memory_type(ptr).is_rdma_suitable()`.
    pub fn is_rdma_suitable(ptr: *const u8) -> bool {
        Self::memory_type(ptr).is_rdma_suitable()
    }

    // ── Low-level (manual lifetime) ──────────────────────────────────────────

    /// Register `ptr..ptr+size` with the RDMA fabric (`cuMemObjGetDescriptor`).
    ///
    /// A matching call to [`deregister`][Self::deregister] must occur before
    /// the buffer is freed.  Use [`register_buffer`][Self::register_buffer] for
    /// automatic deregistration.
    ///
    /// # Errors
    ///
    /// Returns [`CuObjClientError::AlreadyRegistered`] if the exact same
    /// base address was already registered without an intervening deregister.
    pub fn register(&self, ptr: *mut u8, size: usize) -> Result<(), CuObjClientError> {
        let addr = ptr as usize;
        {
            let mut reg = self.registered.lock().expect("CuObjClient mutex poisoned");
            if reg.contains(&addr) {
                return Err(CuObjClientError::AlreadyRegistered { ptr: addr });
            }
            // Insert before the FFI call: if the call panics (it shouldn't),
            // the entry will prevent a retry that might confuse the library.
            reg.insert(addr);
        }
        let rc =
            unsafe { cuobj_client_mem_get_descriptor(self.handle.0, ptr as *mut c_void, size) };
        if rc != CU_OBJ_SUCCESS {
            self.registered
                .lock()
                .expect("CuObjClient mutex poisoned")
                .remove(&addr);
            return Err(CuObjClientError::RegistrationFailed {
                ptr: addr,
                size,
                reason: format!("cuMemObjGetDescriptor returned {rc}"),
            });
        }
        Ok(())
    }

    /// Deregister a previously registered buffer (`cuMemObjPutDescriptor`).
    ///
    /// If `ptr` was not registered, this is a no-op and returns `Ok(())`.
    pub fn deregister(&self, ptr: *mut u8) -> Result<(), CuObjClientError> {
        let addr = ptr as usize;
        let was_registered = self
            .registered
            .lock()
            .expect("CuObjClient mutex poisoned")
            .remove(&addr);
        if !was_registered {
            return Ok(());
        }
        let rc = unsafe { cuobj_client_mem_put_descriptor(self.handle.0, ptr as *mut c_void) };
        if rc != CU_OBJ_SUCCESS {
            return Err(CuObjClientError::DeregistrationFailed {
                ptr: addr,
                reason: format!("cuMemObjPutDescriptor returned {rc}"),
            });
        }
        Ok(())
    }

    /// Generate an RDMA descriptor for a **PUT** operation
    /// (`cuMemObjGetRDMAToken` with `CUOBJ_PUT`).
    ///
    /// * `ptr`    — base pointer of the **registered** buffer.
    /// * `size`   — number of bytes this descriptor should cover.
    /// * `offset` — byte offset from `ptr` to the start of the region.
    ///
    /// The returned [`CuObjRdmaToken`] contains the descriptor string to send as
    /// the `x-amz-rdma-token` HTTP request header.
    pub fn put_token(
        &self,
        ptr: *const u8,
        size: usize,
        offset: usize,
    ) -> Result<CuObjRdmaToken, CuObjClientError> {
        make_rdma_token(&self.handle, ptr, size, offset, OpType::Put)
    }

    /// Generate an RDMA descriptor for a **GET** operation
    /// (`cuMemObjGetRDMAToken` with `CUOBJ_GET`).
    ///
    /// * `ptr`    — base pointer of the **registered** buffer.
    /// * `size`   — number of bytes this descriptor should cover.
    /// * `offset` — byte offset from `ptr` to the start of the region.
    pub fn get_token(
        &self,
        ptr: *const u8,
        size: usize,
        offset: usize,
    ) -> Result<CuObjRdmaToken, CuObjClientError> {
        make_rdma_token(&self.handle, ptr, size, offset, OpType::Get)
    }

    /// Return the maximum single-callback transfer size for the buffer at `ptr`.
    ///
    /// Returns `None` if the query fails (e.g., `ptr` is not registered).
    pub fn max_transfer_size(&self, ptr: *const u8) -> Option<usize> {
        let v = unsafe {
            cuobj_client_mem_get_max_request_callback_size(self.handle.0, ptr as *mut c_void)
        };
        if v < 0 {
            None
        } else {
            Some(v as usize)
        }
    }

    // ── RAII convenience ─────────────────────────────────────────────────────

    /// Register a buffer and return a [`RegisteredBuffer`] that deregisters
    /// automatically on drop.
    ///
    /// The returned guard owns `Arc` clones of the internal client handle and
    /// registration tracker, so it can be stored in structs, returned from
    /// functions, and used across `async` boundaries without lifetime
    /// annotations.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`register`][Self::register].
    pub fn register_buffer(
        &self,
        ptr: *mut u8,
        size: usize,
    ) -> Result<RegisteredBuffer, CuObjClientError> {
        self.register(ptr, size)?;
        Ok(RegisteredBuffer {
            handle: Arc::clone(&self.handle),
            registered: Arc::clone(&self.registered),
            ptr,
            size,
        })
    }

    // ── Telemetry (static / library-wide) ────────────────────────────────────

    /// Set up the telemetry output stream for all `cuObjClient` instances in
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
            cuobj_client_setup_telemetry(use_otel as i32, file_ptr);
        }
    }

    /// Shut down telemetry for all `cuObjClient` instances (reset to default
    /// output stream).
    pub fn shutdown_telemetry() {
        unsafe { cuobj_client_shutdown_telemetry() };
    }

    /// Set telemetry logging flags for all `cuObjClient` instances.
    ///
    /// See `cuobjtelem.h` for available flags (`CUOBJ_LOG_PATH_INFO`,
    /// `CUOBJ_LOG_PATH_DEBUG`, `CUOBJ_LOG_PATH_ERROR`).
    pub fn set_telem_flags(flags: u32) {
        unsafe { cuobj_client_set_telem_flags(flags) };
    }
}

impl Drop for CuObjClient {
    fn drop(&mut self) {
        // Deregister anything still tracked (best-effort; ignore errors).
        // Only buffers registered via the low-level `register()` call remain
        // here; `RegisteredBuffer` guards remove themselves on their own drop.
        let ptrs: Vec<usize> = self
            .registered
            .lock()
            .expect("CuObjClient mutex poisoned")
            .iter()
            .copied()
            .collect();
        for addr in ptrs {
            unsafe {
                cuobj_client_mem_put_descriptor(self.handle.0, addr as *mut c_void);
            }
        }
        // handle (Arc<ClientHandle>) dropped implicitly here.
        // ClientHandle::drop calls cuobj_client_destroy when the last Arc is
        // gone — safe even if CuObjRdmaToken / RegisteredBuffer clones are
        // still outstanding.
    }
}

impl std::fmt::Debug for CuObjClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CuObjClient")
            .field("handle", &self.handle.0)
            .field(
                "registrations",
                &self
                    .registered
                    .lock()
                    .map(|g| g.len())
                    .unwrap_or(usize::MAX),
            )
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RegisteredBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard for a memory region registered with a [`CuObjClient`].
///
/// The region is automatically deregistered when this guard is dropped.
/// Obtain one via [`CuObjClient::register_buffer`].
///
/// Unlike a borrowed guard, `RegisteredBuffer` holds `Arc` references to the
/// client handle and registration tracker, so it has no lifetime parameter and
/// can be freely stored in structs, returned from functions, or held across
/// `async` await points.
pub struct RegisteredBuffer {
    handle: Arc<ClientHandle>,
    registered: Arc<Mutex<HashSet<usize>>>,
    ptr: *mut u8,
    size: usize,
}

// SAFETY: RegisteredBuffer holds a raw pointer to caller-owned memory.
// The Arc fields are Send + Sync; the pointer has no thread affinity.
unsafe impl Send for RegisteredBuffer {}
unsafe impl Sync for RegisteredBuffer {}

impl RegisteredBuffer {
    /// Raw pointer to the start of the registered buffer.
    #[inline]
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Size of the registered buffer in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// View the buffer contents as a byte slice.
    ///
    /// # Concurrency
    ///
    /// The caller must ensure no concurrent mutable access to this memory
    /// region exists for the duration of the returned reference.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// View the buffer contents as a mutable byte slice.
    ///
    /// # Concurrency
    ///
    /// The caller must ensure no other references (mutable or shared) to this
    /// memory region exist for the duration of the returned reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Generate a PUT RDMA descriptor covering `len` bytes at `offset` within
    /// the registered buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CuObjClientError::TokenFailed`] if `offset + len` exceeds
    /// [`Self::size`].
    pub fn put_token(&self, offset: usize, len: usize) -> Result<CuObjRdmaToken, CuObjClientError> {
        if offset.saturating_add(len) > self.size {
            return Err(CuObjClientError::TokenFailed {
                op: OpType::Put,
                offset,
                reason: format!(
                    "offset({offset}) + len({len}) exceeds buffer size({})",
                    self.size
                ),
            });
        }
        make_rdma_token(&self.handle, self.ptr, len, offset, OpType::Put)
    }

    /// Generate a GET RDMA descriptor covering `len` bytes at `offset` within
    /// the registered buffer.
    pub fn get_token(&self, offset: usize, len: usize) -> Result<CuObjRdmaToken, CuObjClientError> {
        if offset.saturating_add(len) > self.size {
            return Err(CuObjClientError::TokenFailed {
                op: OpType::Get,
                offset,
                reason: format!(
                    "offset({offset}) + len({len}) exceeds buffer size({})",
                    self.size
                ),
            });
        }
        make_rdma_token(&self.handle, self.ptr, len, offset, OpType::Get)
    }

    /// Maximum single-callback RDMA transfer size for this buffer.
    ///
    /// Returns `None` if the query fails.
    pub fn max_transfer_size(&self) -> Option<usize> {
        let v = unsafe {
            cuobj_client_mem_get_max_request_callback_size(self.handle.0, self.ptr as *mut c_void)
        };
        if v < 0 {
            None
        } else {
            Some(v as usize)
        }
    }
}

impl Drop for RegisteredBuffer {
    fn drop(&mut self) {
        // Remove from the shared set first so CuObjClient::drop skips this
        // address if the client is dropped concurrently or later.
        self.registered
            .lock()
            .expect("CuObjClient mutex poisoned")
            .remove(&(self.ptr as usize));
        unsafe { cuobj_client_mem_put_descriptor(self.handle.0, self.ptr as *mut c_void) };
    }
}

impl std::ops::Deref for RegisteredBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for RegisteredBuffer {
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

    // ── MemoryType::from_raw ────────────────────────────────────────────────────

    #[test]
    fn memory_type_from_raw_known() {
        assert_eq!(
            MemoryType::from_raw(CUOBJ_MEMORY_SYSTEM),
            MemoryType::System
        );
        assert_eq!(
            MemoryType::from_raw(CUOBJ_MEMORY_CUDA_MANAGED),
            MemoryType::CudaManaged
        );
        assert_eq!(
            MemoryType::from_raw(CUOBJ_MEMORY_CUDA_DEVICE),
            MemoryType::CudaDevice
        );
        assert_eq!(
            MemoryType::from_raw(CUOBJ_MEMORY_UNKNOWN),
            MemoryType::Unknown
        );
        assert_eq!(
            MemoryType::from_raw(CUOBJ_MEMORY_INVALID),
            MemoryType::Invalid
        );
    }

    #[test]
    fn memory_type_from_raw_unknown_fallback() {
        assert_eq!(MemoryType::from_raw(999), MemoryType::Unknown);
    }

    #[test]
    fn memory_type_is_rdma_suitable() {
        assert!(MemoryType::System.is_rdma_suitable());
        assert!(MemoryType::CudaManaged.is_rdma_suitable());
        assert!(MemoryType::CudaDevice.is_rdma_suitable());
        assert!(!MemoryType::Unknown.is_rdma_suitable());
        assert!(!MemoryType::Invalid.is_rdma_suitable());
    }

    // ── OpType::to_raw ─────────────────────────────────────────────────────────

    #[test]
    fn op_type_to_raw() {
        assert_eq!(OpType::Get.to_raw(), CUOBJ_GET);
        assert_eq!(OpType::Put.to_raw(), CUOBJ_PUT);
    }

    #[test]
    fn op_type_roundtrip() {
        assert_eq!(OpType::Get.to_raw(), 0);
        assert_eq!(OpType::Put.to_raw(), 1);
    }
}
