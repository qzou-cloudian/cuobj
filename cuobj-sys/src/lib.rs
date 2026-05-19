// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw FFI bindings for the NVIDIA cuObj SDK.
//!
//! This crate exposes plain-C entry points implemented by two C++ shims:
//!
//! * `wrapper.cpp` — wraps the `cuObjClient` C++ class (`libcuobjclient.so`)
//! * `server_wrapper.cpp` — wraps the `cuObjServer` C++ class (`libcuobjserver.so`)
//!
//! All symbols are `unsafe`; callers are responsible for upholding the
//! lifetime, threading, and correctness contracts documented on each function.
//!
//! # Client constants
//!
//! These mirror the `cuObjOpType_t` and `cuObjMemoryType_t` enums from
//! `cuobjclient.h`.
//!
//! | Constant                    | Value | Meaning                      |
//! |-----------------------------|-------|------------------------------|
//! | `CUOBJ_GET`                 | 0     | GET operation                |
//! | `CUOBJ_PUT`                 | 1     | PUT operation                |
//! | `CUOBJ_MEMORY_SYSTEM`       | 0     | Regular host memory          |
//! | `CUOBJ_MEMORY_CUDA_MANAGED` | 1     | CUDA managed memory          |
//! | `CUOBJ_MEMORY_CUDA_DEVICE`  | 2     | CUDA device memory           |
//! | `CUOBJ_MEMORY_UNKNOWN`      | 3     | Pointer type undetermined    |
//! | `CUOBJ_MEMORY_INVALID`      | 4     | Invalid / null pointer       |
//! | `CU_OBJ_SUCCESS`            | 0     | Success return code          |
//! | `CUOBJ_PROTO_RDMA_DC_V1`    | 1001  | RDMA DC protocol version 1   |

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};

// ─────────────────────────────────────────────────────────────────────────────
// cuObjOpType_t constants
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
pub const CUOBJ_GET: i32 = 0;
#[cfg(feature = "client")]
pub const CUOBJ_PUT: i32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// cuObjMemoryType_t constants
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
pub const CUOBJ_MEMORY_SYSTEM: i32 = 0;
#[cfg(feature = "client")]
pub const CUOBJ_MEMORY_CUDA_MANAGED: i32 = 1;
#[cfg(feature = "client")]
pub const CUOBJ_MEMORY_CUDA_DEVICE: i32 = 2;
#[cfg(feature = "client")]
pub const CUOBJ_MEMORY_UNKNOWN: i32 = 3;
#[cfg(feature = "client")]
pub const CUOBJ_MEMORY_INVALID: i32 = 4;

// ─────────────────────────────────────────────────────────────────────────────
// cuObjErr_t success value
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
pub const CU_OBJ_SUCCESS: i32 = 0;

// ─────────────────────────────────────────────────────────────────────────────
// cuObjProto_t constant (used by both client and server wrappers)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "client", feature = "server"))]
pub const CUOBJ_PROTO_RDMA_DC_V1: i32 = 1001;

// ─────────────────────────────────────────────────────────────────────────────
// cuObjClient FFI (wrapper.cpp → libcuobjclient.so)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
extern "C" {
    /// Create a `cuObjClient` instance with no-op I/O callbacks.
    ///
    /// Returns an opaque `CuObjClientWrapper*` handle, or `NULL` on failure
    /// (e.g. RDMA hardware unavailable or library not loaded).
    pub fn cuobj_client_create() -> *mut c_void;

    /// Destroy a handle returned by [`cuobj_client_create`].
    pub fn cuobj_client_destroy(handle: *mut c_void);

    /// Register a memory region for RDMA (`cuMemObjGetDescriptor`).
    ///
    /// Returns `CU_OBJ_SUCCESS` (0) on success.
    pub fn cuobj_client_mem_get_descriptor(
        handle: *mut c_void,
        ptr: *mut c_void,
        size: usize,
    ) -> i32;

    /// Deregister a previously registered memory region (`cuMemObjPutDescriptor`).
    ///
    /// Returns `CU_OBJ_SUCCESS` (0) on success.
    pub fn cuobj_client_mem_put_descriptor(handle: *mut c_void, ptr: *mut c_void) -> i32;

    /// Return the maximum single-callback transfer size for the buffer at `ptr`.
    ///
    /// Returns `-1` on error.
    pub fn cuobj_client_mem_get_max_request_callback_size(
        handle: *mut c_void,
        ptr: *mut c_void,
    ) -> isize;

    /// Generate an RDMA descriptor string for a registered buffer slice
    /// (`cuMemObjGetRDMAToken`).
    ///
    /// On success, `*desc_str_out` is set to a heap-allocated NUL-terminated
    /// C string.  The caller **must** free it with
    /// [`cuobj_client_mem_put_rdma_token`].
    ///
    /// * `ptr`        — base pointer of the registered buffer
    /// * `size`       — number of bytes covered by this token
    /// * `buf_offset` — byte offset from `ptr` to start the descriptor region
    /// * `op`         — `CUOBJ_GET` (0) or `CUOBJ_PUT` (1)
    ///
    /// Returns `CU_OBJ_SUCCESS` (0) on success.
    pub fn cuobj_client_mem_get_rdma_token(
        handle: *mut c_void,
        ptr: *mut c_void,
        size: usize,
        buf_offset: usize,
        op: i32,
        desc_str_out: *mut *mut c_char,
    ) -> i32;

    /// Free a descriptor string allocated by [`cuobj_client_mem_get_rdma_token`]
    /// (`cuMemObjPutRDMAToken`).
    ///
    /// Returns `CU_OBJ_SUCCESS` (0) on success.
    pub fn cuobj_client_mem_put_rdma_token(handle: *mut c_void, desc_str: *mut c_char) -> i32;

    /// Classify the memory type of `ptr` (static method `cuObjClient::getMemoryType`).
    ///
    /// Returns one of the `CUOBJ_MEMORY_*` constants.
    pub fn cuobj_client_get_memory_type(ptr: *const c_void) -> i32;

    /// Returns `1` if the cuObjClient is connected to the RDMA subsystem,
    /// `0` otherwise.
    pub fn cuobj_client_is_connected(handle: *mut c_void) -> i32;

    /// Set up telemetry for all cuObjClient instances.
    /// `use_otel`=1 for OpenTelemetry backend; 0 for ostream.
    /// `file_path` may be NULL (uses stdout) or a file path for ostream mode.
    pub fn cuobj_client_setup_telemetry(use_otel: i32, file_path: *const c_char);

    /// Shut down cuObjClient telemetry (reset to default output stream).
    pub fn cuobj_client_shutdown_telemetry();

    /// Set cuObjClient telemetry stream logging level flags.
    pub fn cuobj_client_set_telem_flags(flags: u32);
}

// ─────────────────────────────────────────────────────────────────────────────
// cuObjGet / ops.get callback support
// ─────────────────────────────────────────────────────────────────────────────

/// Rust-compatible mirror of `cufileRDMAInfo_t` from `cufile.h`.
///
/// Passed to the `ops.get` callback by cuObj; carries the RDMA descriptor
/// string that must be forwarded to the server via the control channel.
#[cfg(feature = "client")]
#[repr(C)]
pub struct CuFileRdmaInfo {
    pub version: i32,
    pub desc_len: i32,
    pub desc_str: *const c_char,
}

// SAFETY: desc_str points into cuobj-internal memory that is valid for the
// duration of the ops.get callback invocation.
#[cfg(feature = "client")]
unsafe impl Send for CuFileRdmaInfo {}
#[cfg(feature = "client")]
unsafe impl Sync for CuFileRdmaInfo {}

/// ops.get callback type: receives the RDMA descriptor from cuObjGet.
///
/// Parameters mirror `CUObjOps_t::get` from `cuobjclient.h`:
/// - `handle`    — cuobj internal handle; pass to `cuobj_client_get_ctx_from_handle`
/// - `ptr`       — destination buffer
/// - `size`      — transfer size in bytes
/// - `offset`    — object offset (loff_t = i64)
/// - `rdma_info` — descriptor struct (version, desc_len, desc_str)
///
/// Must return `size` on success or `-1` on failure.
#[cfg(feature = "client")]
pub type OpsGetFn = unsafe extern "C" fn(
    handle: *const c_void,
    ptr: *mut c_char,
    size: usize,
    offset: i64,
    rdma_info: *const CuFileRdmaInfo,
) -> isize;

#[cfg(feature = "client")]
extern "C" {
    /// Retrieve the user context (`ctx`) from a cuobj callback handle.
    ///
    /// Call this inside an `ops.get` callback: `handle` is the opaque argument
    /// passed by cuObj, and the return value is the `ctx` pointer that was
    /// supplied to [`cuobj_client_get`].
    pub fn cuobj_client_get_ctx_from_handle(cuobj_handle: *const c_void) -> *mut c_void;

    /// Create a `cuObjClient` with `ops.get` set to `get_cb`.
    ///
    /// cuObjGet will invoke `get_cb` with the RDMA descriptor; the callback
    /// forwards it to the upstream server and waits for RDMA completion.
    ///
    /// Returns an opaque handle, or `NULL` on failure.
    pub fn cuobj_client_create_with_get_cb(get_cb: OpsGetFn) -> *mut c_void;

    /// Perform a GET via `cuObjClient::cuObjGet`.
    ///
    /// `ctx` is forwarded to the `ops.get` callback via
    /// `cuobj_client_get_ctx_from_handle`.  The buffer at `ptr` must have been
    /// registered with [`cuobj_client_mem_get_descriptor`] beforehand.
    ///
    /// Returns bytes transferred (≥ 0) on success, or -1 on failure.
    pub fn cuobj_client_get(
        handle: *mut c_void,
        ctx: *mut c_void,
        ptr: *mut c_void,
        size: usize,
        offset: i64,
        buf_offset: i64,
    ) -> isize;
}

// ─────────────────────────────────────────────────────────────────────────────
// cuObjServer FFI (server_wrapper.cpp → libcuobjserver.so)
// ─────────────────────────────────────────────────────────────────────────────

/// InfiniBand verbs completion status.
///
/// Mirrors `ibv_wc_status` from `<infiniband/verbs.h>`.  Only the values that
/// can be reported by the cuObj server are listed here; unknown values are
/// preserved as their raw integer.
#[cfg(feature = "server")]
pub type ibv_wc_status = i32;

/// Named `ibv_wc_status` constants.
#[cfg(feature = "server")]
pub mod wc_status {
    use super::ibv_wc_status;
    pub const SUCCESS: ibv_wc_status = 0;
    pub const LOC_LEN_ERR: ibv_wc_status = 1;
    pub const LOC_QP_OP_ERR: ibv_wc_status = 2;
    pub const LOC_EEC_OP_ERR: ibv_wc_status = 3;
    pub const LOC_PROT_ERR: ibv_wc_status = 4;
    pub const WR_FLUSH_ERR: ibv_wc_status = 5;
    pub const MW_BIND_ERR: ibv_wc_status = 6;
    pub const BAD_RESP_ERR: ibv_wc_status = 7;
    pub const LOC_ACCESS_ERR: ibv_wc_status = 8;
    pub const REM_INV_REQ_ERR: ibv_wc_status = 9;
    pub const REM_ACCESS_ERR: ibv_wc_status = 10;
    pub const REM_OP_ERR: ibv_wc_status = 11;
    pub const RETRY_EXC_ERR: ibv_wc_status = 12;
    pub const RNR_RETRY_EXC_ERR: ibv_wc_status = 13;
    pub const LOC_RDD_VIOL_ERR: ibv_wc_status = 14;
    pub const REM_INV_RD_REQ_ERR: ibv_wc_status = 15;
    pub const REM_ABORT_ERR: ibv_wc_status = 16;
    pub const INV_EECN_ERR: ibv_wc_status = 17;
    pub const INV_EEC_STATE_ERR: ibv_wc_status = 18;
    pub const FATAL_ERR: ibv_wc_status = 19;
    pub const RESP_TIMEOUT_ERR: ibv_wc_status = 20;
    pub const GENERAL_ERR: ibv_wc_status = 21;
}

/// Completion event returned by [`cuobj_srv_poll`].
///
/// Mirrors `cuObjAsyncEvent_t` from `cuobjserver.h`.
#[cfg(feature = "server")]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CuObjAsyncEvent {
    /// Opaque handle supplied by the caller when submitting an async operation.
    pub async_handle: *mut c_void,
    /// Bytes transferred on success; negative error code on failure.
    pub status: isize,
}

// SAFETY: `async_handle` is an opaque caller-supplied cookie.  The caller is
// responsible for ensuring the pointed-to data is thread-safe; the struct
// itself is plain data (two words) with no interior mutability.
#[cfg(feature = "server")]
unsafe impl Send for CuObjAsyncEvent {}
#[cfg(feature = "server")]
unsafe impl Sync for CuObjAsyncEvent {}

#[cfg(feature = "server")]
extern "C" {
    /// Create a `cuObjServer` bound to the given RDMA NIC `ip` and `port`.
    ///
    /// Returns an opaque `CuObjSrvHandle*`, or `NULL` on failure.
    pub fn cuobj_srv_create(ip: *const c_char, port: u16) -> *mut c_void;

    /// Destroy a handle returned by [`cuobj_srv_create`].
    pub fn cuobj_srv_destroy(handle: *mut c_void);

    /// Returns `1` if the server is connected to the RDMA fabric, `0` otherwise.
    pub fn cuobj_srv_is_connected(handle: *mut c_void) -> i32;

    /// Allocate a unique channel ID for one concurrent RDMA operation.
    ///
    /// Each thread performing a concurrent RDMA transfer must hold a distinct
    /// channel.  Release when done with [`cuobj_srv_free_channel`].
    pub fn cuobj_srv_alloc_channel(handle: *mut c_void) -> u16;

    /// Release a channel ID obtained with [`cuobj_srv_alloc_channel`].
    pub fn cuobj_srv_free_channel(handle: *mut c_void, channel: u16);

    /// Allocate a 4 KiB-aligned host buffer of `size` bytes and register it
    /// with the RDMA fabric.
    ///
    /// Returns an opaque `CuObjSrvBuffer*`, or `NULL` on failure.  The caller
    /// must release it with [`cuobj_srv_deregister_free`].
    pub fn cuobj_srv_alloc_register(handle: *mut c_void, size: usize) -> *mut c_void;

    /// Deregister and free a buffer created by [`cuobj_srv_alloc_register`].
    pub fn cuobj_srv_deregister_free(handle: *mut c_void, buf: *mut c_void);

    /// Register an existing host buffer for RDMA.
    /// Returns an opaque rdma_buffer* handle, or null on failure.
    pub fn cuobj_srv_register_buffer(
        handle: *mut c_void,
        ptr: *mut c_void,
        size: usize,
    ) -> *mut c_void;

    /// Deregister a buffer registered via [`cuobj_srv_register_buffer`].
    /// Does NOT free the host memory.
    pub fn cuobj_srv_deregister_buffer(handle: *mut c_void, rbuf: *mut c_void);

    /// Return the raw host data pointer inside a `CuObjSrvBuffer`.
    ///
    /// Use this to copy data into or out of the buffer around RDMA operations.
    pub fn cuobj_srv_buf_data(buf: *mut c_void) -> *mut c_void;

    /// Return the allocated size in bytes of a `CuObjSrvBuffer`.
    pub fn cuobj_srv_buf_size(buf: *mut c_void) -> usize;

    /// Handle an S3 PUT: RDMA_READ from the client's buffer into the server's
    /// local `buf` (`cuObjServer::handlePutObject`).
    ///
    /// `rdma_descr` is the RDMA descriptor string received from the client in
    /// the `x-amz-rdma-token` HTTP request header.
    ///
    /// After this call the host memory of `buf` contains the data that was in
    /// the client's GPU/host buffer.
    ///
    /// Returns bytes transferred on success, or `-1` on failure.
    pub fn cuobj_srv_handle_put(
        handle: *mut c_void,
        key: *const c_char,
        buf: *mut c_void,
        remote_buf_start: u64,
        size: usize,
        rdma_descr: *const c_char,
        channel: u16,
        status: *mut ibv_wc_status,
    ) -> isize;

    /// Handle an S3 GET: RDMA_WRITE from the server's local `buf` into the
    /// client's buffer (`cuObjServer::handleGetObject`).
    ///
    /// Populate the host memory of `buf` with the object data **before**
    /// calling this function.
    ///
    /// Returns bytes transferred on success, or `-1` on failure.
    /// On failure, `*status` receives the InfiniBand WC completion status
    /// (pass null if not needed).
    pub fn cuobj_srv_handle_get(
        handle: *mut c_void,
        key: *const c_char,
        buf: *mut c_void,
        remote_buf_start: u64,
        size: usize,
        rdma_descr: *const c_char,
        channel: u16,
        status: *mut ibv_wc_status,
    ) -> isize;

    /// Submit an async S3 PUT (`cuObjServer::handlePutObject` with non-NULL
    /// `async_handle`).  Returns 0 on successful submission or -1 on failure.
    /// Call [`cuobj_srv_poll`] on the same channel to harvest completions.
    pub fn cuobj_srv_handle_put_async(
        handle: *mut c_void,
        key: *const c_char,
        buf: *mut c_void,
        remote_buf_start: u64,
        size: usize,
        rdma_descr: *const c_char,
        channel: u16,
        async_handle: *mut c_void,
    ) -> isize;

    /// Submit an async S3 GET (`cuObjServer::handleGetObject` with non-NULL
    /// `async_handle`).  Returns 0 on successful submission or -1 on failure.
    /// Call [`cuobj_srv_poll`] on the same channel to harvest completions.
    pub fn cuobj_srv_handle_get_async(
        handle: *mut c_void,
        key: *const c_char,
        buf: *mut c_void,
        remote_buf_start: u64,
        size: usize,
        rdma_descr: *const c_char,
        channel: u16,
        async_handle: *mut c_void,
    ) -> isize;

    /// Poll for completions on `channel`.  Fills up to `max_events` entries in
    /// `events`.  Returns the number of events filled (≥ 0) or a negative
    /// error code.
    pub fn cuobj_srv_poll(
        handle: *mut c_void,
        events: *mut CuObjAsyncEvent,
        max_events: std::os::raw::c_int,
        channel: u16,
    ) -> std::os::raw::c_int;

    /// Set up telemetry for all cuObjServer instances.
    pub fn cuobj_srv_setup_telemetry(use_otel: i32, file_path: *const c_char);

    /// Shut down cuObjServer telemetry (reset to default output stream).
    pub fn cuobj_srv_shutdown_telemetry();

    /// Set cuObjServer telemetry stream logging level flags.
    pub fn cuobj_srv_set_telem_flags(flags: u32);
}
