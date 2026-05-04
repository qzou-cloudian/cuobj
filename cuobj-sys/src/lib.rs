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
//! | `CUOBJ_MEMORY_UNKNOWN`      | 3     | Pointer type could not be determined |
//! | `CUOBJ_MEMORY_INVALID`      | 4     | Invalid / null pointer       |
//! | `CU_OBJ_SUCCESS`            | 0     | Success return code          |

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
// cuObjServer FFI (server_wrapper.cpp → libcuobjserver.so)
// ─────────────────────────────────────────────────────────────────────────────

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
    ) -> isize;

    /// Handle an S3 GET: RDMA_WRITE from the server's local `buf` into the
    /// client's buffer (`cuObjServer::handleGetObject`).
    ///
    /// Populate the host memory of `buf` with the object data **before**
    /// calling this function.
    ///
    /// Returns bytes transferred on success, or `-1` on failure.
    pub fn cuobj_srv_handle_get(
        handle: *mut c_void,
        key: *const c_char,
        buf: *mut c_void,
        remote_buf_start: u64,
        size: usize,
        rdma_descr: *const c_char,
        channel: u16,
    ) -> isize;

    /// Submit an async S3 PUT (`cuObjServer::handlePutObject` with non-NULL
    /// `async_handle`).  Returns 0 on successful submission or -1 on failure.
    /// Call [`cuobj_srv_poll`] on the same channel to harvest completions.
    pub fn cuobj_srv_handle_put_async(
        handle:          *mut c_void,
        key:             *const c_char,
        buf:             *mut c_void,
        remote_buf_start: u64,
        size:            usize,
        rdma_descr:      *const c_char,
        channel:         u16,
        async_handle:    *mut c_void,
    ) -> isize;

    /// Submit an async S3 GET (`cuObjServer::handleGetObject` with non-NULL
    /// `async_handle`).  Returns 0 on successful submission or -1 on failure.
    /// Call [`cuobj_srv_poll`] on the same channel to harvest completions.
    pub fn cuobj_srv_handle_get_async(
        handle:          *mut c_void,
        key:             *const c_char,
        buf:             *mut c_void,
        remote_buf_start: u64,
        size:            usize,
        rdma_descr:      *const c_char,
        channel:         u16,
        async_handle:    *mut c_void,
    ) -> isize;

    /// Poll for completions on `channel`.  Fills up to `max_events` entries in
    /// `events`.  Returns the number of events filled (≥ 0) or a negative
    /// error code.
    pub fn cuobj_srv_poll(
        handle:     *mut c_void,
        events:     *mut CuObjAsyncEvent,
        max_events: std::os::raw::c_int,
        channel:    u16,
    ) -> std::os::raw::c_int;

    /// Set up telemetry for all cuObjServer instances.
    pub fn cuobj_srv_setup_telemetry(use_otel: i32, file_path: *const c_char);

    /// Shut down cuObjServer telemetry (reset to default output stream).
    pub fn cuobj_srv_shutdown_telemetry();

    /// Set cuObjServer telemetry stream logging level flags.
    pub fn cuobj_srv_set_telem_flags(flags: u32);
}
