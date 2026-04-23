// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pre-registered RDMA buffer pool for NVIDIA cuObj client and server.
//!
//! This crate provides a fixed-size pool of 4 KiB-aligned host buffers that
//! are pre-registered with the RDMA fabric so that individual acquire/release
//! operations pay no registration overhead.  Each buffer slot may be
//! simultaneously registered with a [`cuobj::CuObjClient`] for
//! sending RDMA tokens to S3 clients, and with a
//! [`cuobj::CuObjServer`] (via
//! [`cuobj::PersistentBuffer`]) for performing server-side
//! RDMA_READ/RDMA_WRITE operations.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use cuobj::BufferPool;
//!
//! # #[cfg(all(feature = "client", feature = "server"))]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use cuobj::CuObjClient;
//! use cuobj::CuObjServer;
//!
//! let client = Arc::new(CuObjClient::new()?);
//! let server = Arc::new(CuObjServer::new("192.168.100.1", 4421)?);
//!
//! let pool = BufferPool::builder()
//!     .count(8)
//!     .size(4 * 1024 * 1024)   // 4 MiB per slot
//!     .with_client(Arc::clone(&client))
//!     .with_server(Arc::clone(&server))
//!     .build()?;
//!
//! // Acquire a buffer — blocks until one is free.
//! let guard = pool.acquire();
//! guard.as_mut_slice().fill(0xAB_u8);
//!
//! // Generate an RDMA token for the client to read from.
//! let token = guard.put_token(0, guard.len())?;
//! # Ok(())
//! # }
//! # #[cfg(not(all(feature = "client", feature = "server")))]
//! # fn main() {}
//! ```

use std::alloc::{Layout, alloc, dealloc};
use std::sync::{Condvar, Mutex};
use std::sync::Arc;
use crossbeam::queue::ArrayQueue;
use tokio::sync::Semaphore;

#[cfg(feature = "client")]
use crate::client::{CuObjClient, CuObjClientError, RdmaToken};

#[cfg(feature = "server")]
use crate::server::{CuObjServer, CuObjServerError, PersistentBuffer};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by buffer-pool operations.
#[derive(Debug, thiserror::Error)]
pub enum BufferPoolError {
    /// Allocation or RDMA registration failed during pool construction.
    #[error("pool construction failed at slot {slot}: {reason}")]
    ConstructionFailed { slot: usize, reason: String },

    /// A token operation was requested but the pool has no client configured.
    #[error("no CuObjClient configured on this pool")]
    NoClient,

    /// A server RDMA operation was requested but the pool has no server
    /// configured.
    #[error("no CuObjServer configured on this pool")]
    NoServer,

    /// The underlying RDMA client reported an error.
    #[cfg(feature = "client")]
    #[error("client RDMA error: {0}")]
    Client(#[from] CuObjClientError),

    /// The underlying RDMA server reported an error.
    #[cfg(feature = "server")]
    #[error("server RDMA error: {0}")]
    Server(#[from] CuObjServerError),
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal slot
// ─────────────────────────────────────────────────────────────────────────────

/// Alignment required by RDMA: 4 KiB.
const RDMA_ALIGN: usize = 4096;

/// A single slot in the pool.
///
/// Slots are created once at pool construction and never reallocated.
struct Slot {
    /// Pointer to the host-memory data region (always 4 KiB-aligned).
    ptr:    *mut u8,
    /// Byte length of the region.
    size:   usize,
    /// Layout used to allocate `ptr` via `std::alloc`. `None` when the memory
    /// is owned by `srv_buf` (a `PersistentBuffer`) instead.
    layout: Option<Layout>,
    /// Server-allocated-and-registered buffer, if a server is configured.
    ///
    /// `PersistentBuffer` owns the allocation; raw ptr (`self.ptr`) is derived
    /// from it.  When `srv_buf` is `Some`, we do NOT separately free `ptr`.
    #[cfg(feature = "server")]
    srv_buf: Option<PersistentBuffer>,
    /// True when `ptr` has been registered with the cuObj client.
    client_registered: bool,
}

// SAFETY: `Slot` contents are only accessed while the slot is not in the free
// queue (i.e. exclusively owned by one guard or the pool's drop path).
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

impl Slot {
    /// Free the heap buffer if it was not allocated by a `PersistentBuffer`.
    ///
    /// Called from `BufferPool::drop`.
    fn free_standalone(&mut self) {
        #[cfg(feature = "server")]
        if self.srv_buf.is_some() {
            // `PersistentBuffer` owns the allocation; just drop it.
            self.srv_buf = None;
            return;
        }
        // Allocated via `std::alloc::alloc`; use the cached layout.
        if let Some(layout) = self.layout.take() {
            if !self.ptr.is_null() {
                unsafe { dealloc(self.ptr, layout) };
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BufferPool
// ─────────────────────────────────────────────────────────────────────────────

/// A thread-safe pool of pre-registered RDMA buffers.
///
/// Create via [`BufferPool::builder()`].  Buffers are acquired with
/// [`acquire`][Self::acquire] (blocking) or
/// [`try_acquire`][Self::try_acquire] (non-blocking) and are returned to the
/// pool when the returned [`BufferGuard`] is dropped.
pub struct BufferPool {
    /// Fixed-size slot array; items are never moved after construction so raw
    /// pointers in guards remain valid.
    slots:      Vec<Slot>,
    /// Lock-free MPMC queue of free slot indices (O(1) push/pop).
    free_queue: ArrayQueue<usize>,
    /// Async-friendly semaphore whose permit count mirrors `free_queue.len()`.
    /// Used by [`acquire_owned`][Self::acquire_owned].
    sem:        Semaphore,
    /// Mutex used only to make `parking_cv` waits race-free.
    parking_mu: Mutex<()>,
    /// Condvar that sync-blocking acquirers sleep on when the pool is empty.
    parking_cv: Condvar,
    /// Client, if configured.
    #[cfg(feature = "client")]
    client: Option<Arc<CuObjClient>>,
    /// Server, if configured (kept alive so `PersistentBuffer`s don't
    /// dangle, though each `PersistentBuffer` already holds an `Arc` clone).
    #[cfg(feature = "server")]
    #[allow(dead_code)]
    server: Option<Arc<CuObjServer>>,
}

// SAFETY: `free_queue` and `sem` provide lock-free / atomic synchronisation;
// raw pointers inside `Slot` are owned by the pool and only dereferenced by a
// live guard that has exclusive logical ownership of the slot.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Return a [`BufferPoolBuilder`] for constructing a pool.
    pub fn builder() -> BufferPoolBuilder {
        BufferPoolBuilder::default()
    }

    // ── Acquire / release ────────────────────────────────────────────────────

    /// Acquire a free buffer, **blocking** until one becomes available.
    ///
    /// The buffer is held until the returned [`BufferGuard`] is dropped.
    pub fn acquire(&self) -> BufferGuard<'_> {
        loop {
            // Fast path: non-blocking attempt.
            if let Ok(permit) = self.sem.try_acquire() {
                permit.forget();
                let idx = self.free_queue.pop().expect("sem/queue invariant violated");
                return self.make_guard(idx);
            }
            // Slow path: sleep on condvar until a release wakes us.
            let lock = self.parking_mu.lock().unwrap();
            // Re-check under the mutex to close the window between the failed
            // try_acquire above and reaching condvar::wait.
            if let Ok(permit) = self.sem.try_acquire() {
                drop(lock);
                permit.forget();
                let idx = self.free_queue.pop().expect("sem/queue invariant violated");
                return self.make_guard(idx);
            }
            drop(self.parking_cv.wait(lock).unwrap());
        }
    }

    /// Try to acquire a free buffer without blocking.
    ///
    /// Returns `None` if all buffers are currently in use.
    pub fn try_acquire(&self) -> Option<BufferGuard<'_>> {
        let permit = self.sem.try_acquire().ok()?;
        permit.forget();
        let idx = self.free_queue.pop().expect("sem/queue invariant violated");
        Some(self.make_guard(idx))
    }

    /// Try to acquire a free buffer, **waiting up to `timeout`** for one to
    /// become available.
    ///
    /// Returns `None` if no buffer became available within the timeout.
    pub fn acquire_timeout(&self, timeout: std::time::Duration) -> Option<BufferGuard<'_>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Ok(permit) = self.sem.try_acquire() {
                permit.forget();
                let idx = self.free_queue.pop().expect("sem/queue invariant violated");
                return Some(self.make_guard(idx));
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let lock = self.parking_mu.lock().unwrap();
            if let Ok(permit) = self.sem.try_acquire() {
                drop(lock);
                permit.forget();
                let idx = self.free_queue.pop().expect("sem/queue invariant violated");
                return Some(self.make_guard(idx));
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let (g, timed_out) = self.parking_cv.wait_timeout(lock, remaining).unwrap();
            drop(g);
            if timed_out.timed_out() {
                // One final attempt after the timeout fires.
                if let Ok(permit) = self.sem.try_acquire() {
                    permit.forget();
                    let idx = self.free_queue.pop().expect("sem/queue invariant violated");
                    return Some(self.make_guard(idx));
                }
                return None;
            }
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Total number of buffer slots.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of buffer slots currently not in use.
    pub fn available(&self) -> usize {
        self.free_queue.len()
    }

    // ── Internals ────────────────────────────────────────────────────────────

    fn make_guard(&self, idx: usize) -> BufferGuard<'_> {
        let slot = &self.slots[idx];
        let ptr             = slot.ptr;
        let size            = slot.size;
        let client_reg      = slot.client_registered;
        #[cfg(feature = "server")]
        let srv_buf_ptr = slot.srv_buf.as_ref().map_or(std::ptr::null(), |b| b as *const PersistentBuffer);
        #[cfg(not(feature = "server"))]
        let srv_buf_ptr = ();
        BufferGuard {
            pool: self,
            idx,
            ptr,
            size,
            client_registered: client_reg,
            srv_buf_ptr,
        }
    }

    /// Release slot `idx` back to the pool.
    ///
    /// Called by the `Drop` impls of [`BufferGuard`] and [`OwnedBufferGuard`].
    pub fn release(&self, idx: usize) {
        // Push the index first (lock-free).
        self.free_queue.push(idx).ok().expect("double-release of buffer slot");
        // Add a semaphore permit so async waiters see the slot immediately.
        self.sem.add_permits(1);
        // Notify exactly one sync-blocking waiter.  Hold the mutex so the
        // notify cannot slip between a waiter's re-check and its condvar::wait.
        let _g = self.parking_mu.lock().unwrap();
        self.parking_cv.notify_one();
    }

    /// Try to acquire a free buffer without blocking, returning an
    /// [`OwnedBufferGuard`] that is `'static` (backed by `Arc<BufferPool>`).
    ///
    /// Returns `None` if all slots are in use.
    pub fn try_acquire_owned(self: &Arc<Self>) -> Option<OwnedBufferGuard> {
        let permit = self.sem.try_acquire().ok()?;
        permit.forget();
        let idx  = self.free_queue.pop().expect("sem/queue invariant violated");
        let ptr  = self.slots[idx].ptr;
        let size = self.slots[idx].size;
        Some(OwnedBufferGuard { pool: Arc::clone(self), idx, ptr, size, len: size })
    }

    /// Acquire a free buffer, **blocking** until one is available.
    ///
    /// Returns an [`OwnedBufferGuard`] that is `'static`.  Prefer
    /// [`acquire_owned`][Self::acquire_owned] in async contexts and
    /// fall back to `spawn_blocking` for this method.
    pub fn acquire_owned_blocking(self: &Arc<Self>) -> OwnedBufferGuard {
        loop {
            if let Ok(permit) = self.sem.try_acquire() {
                permit.forget();
                let idx  = self.free_queue.pop().expect("sem/queue invariant violated");
                let ptr  = self.slots[idx].ptr;
                let size = self.slots[idx].size;
                return OwnedBufferGuard { pool: Arc::clone(self), idx, ptr, size, len: size };
            }
            let lock = self.parking_mu.lock().unwrap();
            if let Ok(permit) = self.sem.try_acquire() {
                drop(lock);
                permit.forget();
                let idx  = self.free_queue.pop().expect("sem/queue invariant violated");
                let ptr  = self.slots[idx].ptr;
                let size = self.slots[idx].size;
                return OwnedBufferGuard { pool: Arc::clone(self), idx, ptr, size, len: size };
            }
            drop(self.parking_cv.wait(lock).unwrap());
        }
    }

    /// Async-friendly acquire returning an [`OwnedBufferGuard`].
    ///
    /// Awaits a semaphore permit (zero cost when slots are available) then
    /// pops a slot index from the lock-free queue.
    pub async fn acquire_owned(self: &Arc<Self>) -> OwnedBufferGuard {
        let permit = self.sem.acquire().await.expect("BufferPool semaphore closed");
        permit.forget();
        let idx  = self.free_queue.pop().expect("sem/queue invariant violated");
        let ptr  = self.slots[idx].ptr;
        let size = self.slots[idx].size;
        OwnedBufferGuard { pool: Arc::clone(self), idx, ptr, size, len: size }
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        // Deregister client slots, then free memory.
        for slot in &mut self.slots {
            #[cfg(feature = "client")]
            if slot.client_registered {
                if let Some(client) = self.client.as_ref() {
                    let _ = client.deregister(slot.ptr);
                }
            }
            slot.free_standalone();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OwnedBufferGuard
// ─────────────────────────────────────────────────────────────────────────────

/// An RAII guard that owns a slot from a [`BufferPool`] via `Arc`.
///
/// Unlike [`BufferGuard`] this type is `'static` and can be held across
/// `.await` points or moved into `spawn_blocking` closures.
///
/// The slot is released back to the pool when this guard is dropped.
///
/// Because this type implements `AsRef<[u8]> + Send + 'static` it can be
/// wrapped in `bytes::Bytes::from_owner(guard)` for a zero-copy transfer.
pub struct OwnedBufferGuard {
    pool: Arc<BufferPool>,
    idx:  usize,
    ptr:  *mut u8,
    size: usize,
    /// Number of valid bytes starting from `ptr`.  Starts equal to `size`;
    /// callers should call [`set_len`][Self::set_len] after a partial write.
    len:  usize,
}

// SAFETY: the guard has exclusive mutable access to the slot bytes.
unsafe impl Send for OwnedBufferGuard {}
unsafe impl Sync for OwnedBufferGuard {}

impl OwnedBufferGuard {
    /// Total capacity of the buffer slot (bytes).
    #[inline]
    pub fn capacity(&self) -> usize { self.size }

    /// Number of valid bytes (set by [`set_len`][Self::set_len]).
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// `true` if `len == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Raw const pointer to the data region.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 { self.ptr }

    /// Raw mutable pointer to the data region.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr }

    /// Immutable view of the valid region (`&ptr[..len]`).
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Mutable view of the **full** slot capacity (`&mut ptr[..size]`).
    ///
    /// Use this for writing; call [`set_len`][Self::set_len] afterwards.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Restrict the valid region to the first `n` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `n > capacity()`.
    #[inline]
    pub fn set_len(&mut self, n: usize) {
        assert!(n <= self.size, "OwnedBufferGuard::set_len({n}) > capacity({})", self.size);
        self.len = n;
    }
}

impl AsRef<[u8]> for OwnedBufferGuard {
    #[inline]
    fn as_ref(&self) -> &[u8] { self.as_slice() }
}

impl AsMut<[u8]> for OwnedBufferGuard {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] { self.as_mut_slice() }
}

impl Drop for OwnedBufferGuard {
    fn drop(&mut self) {
        self.pool.release(self.idx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BufferGuard
// ─────────────────────────────────────────────────────────────────────────────

/// A guard that holds exclusive access to one slot from a [`BufferPool`].
///
/// The slot is released back to the pool when this guard is dropped.
///
/// Use the slice helpers for data access, and the token / server-handle
/// methods for RDMA operations.
pub struct BufferGuard<'pool> {
    pool:              &'pool BufferPool,
    idx:               usize,
    ptr:               *mut u8,
    size:              usize,
    client_registered: bool,
    /// Stable pointer to the `PersistentBuffer` inside the pool slot.
    /// `null` when no server is configured.
    #[cfg(feature = "server")]
    srv_buf_ptr:    *const PersistentBuffer,
    #[cfg(not(feature = "server"))]
    #[allow(dead_code)]
    srv_buf_ptr:    (),
}

// SAFETY: the guard has exclusive access to the slot until it is dropped.
unsafe impl Send for BufferGuard<'_> {}
unsafe impl Sync for BufferGuard<'_> {}

impl<'pool> BufferGuard<'pool> {
    // ── Data access ──────────────────────────────────────────────────────────

    /// Length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the buffer has zero size.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Raw const pointer to the data region.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Raw mutable pointer to the data region.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// View the buffer as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// View the buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    // ── Status queries ───────────────────────────────────────────────────────

    /// `true` if this buffer is registered with the cuObj client.
    pub fn is_client_registered(&self) -> bool {
        self.client_registered
    }

    /// `true` if this buffer is registered with the cuObj server.
    pub fn is_server_registered(&self) -> bool {
        #[cfg(feature = "server")]
        return !self.srv_buf_ptr.is_null();
        #[cfg(not(feature = "server"))]
        false
    }

    // ── Client token helpers ─────────────────────────────────────────────────

    /// Generate an RDMA PUT token (client will RDMA_READ from `[offset, offset+len)`).
    ///
    /// Requires the pool to have been built with a client and this buffer
    /// to be client-registered.
    #[cfg(feature = "client")]
    pub fn put_token(&self, offset: usize, len: usize) -> Result<RdmaToken, BufferPoolError> {
        let client = self
            .pool
            .client
            .as_ref()
            .ok_or(BufferPoolError::NoClient)?;
        Ok(client.put_token(self.ptr as *const u8, len, offset)?)
    }

    /// Generate an RDMA GET token (client will RDMA_WRITE into `[offset, offset+len)`).
    ///
    /// Requires the pool to have been built with a client.
    #[cfg(feature = "client")]
    pub fn get_token(&self, offset: usize, len: usize) -> Result<RdmaToken, BufferPoolError> {
        let client = self
            .pool
            .client
            .as_ref()
            .ok_or(BufferPoolError::NoClient)?;
        Ok(client.get_token(self.ptr as *const u8, len, offset)?)
    }

    // ── Server handle helpers ────────────────────────────────────────────────

    /// Perform an RDMA PUT (RDMA_READ from client into this buffer).
    ///
    /// `offset` is the byte offset within the client's registered buffer.
    /// Use `0` for a full-buffer transfer.  Requires the pool to have been
    /// built with a server.
    #[cfg(feature = "server")]
    pub fn server_handle_put(
        &self,
        key:    &str,
        offset: usize,
        size:   usize,
        token:  &str,
    ) -> Result<usize, BufferPoolError> {
        if self.srv_buf_ptr.is_null() {
            return Err(BufferPoolError::NoServer);
        }
        let srv_buf = unsafe { &*self.srv_buf_ptr };
        Ok(srv_buf.handle_put(key, offset, size, token)?)
    }

    /// Perform an RDMA GET (RDMA_WRITE from this buffer into client).
    ///
    /// `offset` is the byte offset within the client's registered buffer to
    /// write into.  Use `0` for a full-buffer transfer.  Requires the pool
    /// to have been built with a server.
    #[cfg(feature = "server")]
    pub fn server_handle_get(
        &self,
        key:    &str,
        offset: usize,
        size:   usize,
        token:  &str,
    ) -> Result<usize, BufferPoolError> {
        if self.srv_buf_ptr.is_null() {
            return Err(BufferPoolError::NoServer);
        }
        let srv_buf = unsafe { &*self.srv_buf_ptr };
        Ok(srv_buf.handle_get(key, offset, size, token)?)
    }
}

impl Drop for BufferGuard<'_> {
    fn drop(&mut self) {
        self.pool.release(self.idx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BufferPoolBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`BufferPool`].
///
/// Call [`BufferPool::builder()`] to obtain one, configure it, then call
/// [`build`][Self::build].
#[derive(Default)]
pub struct BufferPoolBuilder {
    count: usize,
    size:  usize,
    #[cfg(feature = "client")]
    client: Option<Arc<CuObjClient>>,
    #[cfg(feature = "server")]
    server: Option<Arc<CuObjServer>>,
}

impl BufferPoolBuilder {
    /// Number of buffer slots (must be ≥ 1).
    pub fn count(mut self, n: usize) -> Self {
        self.count = n;
        self
    }

    /// Size of each buffer slot in bytes.
    pub fn size(mut self, bytes: usize) -> Self {
        self.size = bytes;
        self
    }

    /// Configure a cuObj client; buffers will be registered with it at
    /// build time so `put_token`/`get_token` on the guard are always ready.
    #[cfg(feature = "client")]
    pub fn with_client(mut self, client: Arc<CuObjClient>) -> Self {
        self.client = Some(client);
        self
    }

    /// Configure a cuObj server; each buffer slot is allocated and
    /// registered via [`PersistentBuffer`] so `server_handle_put`/
    /// `server_handle_get` on the guard are always ready.
    #[cfg(feature = "server")]
    pub fn with_server(mut self, server: Arc<CuObjServer>) -> Self {
        self.server = Some(server);
        self
    }

    /// Allocate and register all buffer slots.
    ///
    /// # Errors
    ///
    /// Returns [`BufferPoolError::ConstructionFailed`] if any slot cannot be
    /// allocated or registered.
    pub fn build(self) -> Result<BufferPool, BufferPoolError> {
        assert!(self.count >= 1, "pool count must be at least 1");
        assert!(self.size  >= 1, "pool slot size must be at least 1");

        let mut slots = Vec::with_capacity(self.count);
        let free_queue = ArrayQueue::new(self.count);

        for i in 0..self.count {
            let slot = self.make_slot(i)?;
            slots.push(slot);
            free_queue.push(i).ok().expect("ArrayQueue capacity exceeded during build");
        }

        Ok(BufferPool {
            slots,
            free_queue,
            sem:        Semaphore::new(self.count),
            parking_mu: Mutex::new(()),
            parking_cv: Condvar::new(),
            #[cfg(feature = "client")]
            client: self.client,
            #[cfg(feature = "server")]
            server: self.server,
        })
    }

    fn make_slot(&self, i: usize) -> Result<Slot, BufferPoolError> {
        // ── Allocate memory ──────────────────────────────────────────────────
        #[cfg(feature = "server")]
        let (ptr, layout, srv_buf) = if let Some(server) = self.server.as_ref() {
            let mut pb = PersistentBuffer::new(Arc::clone(server), self.size).map_err(|e| {
                BufferPoolError::ConstructionFailed {
                    slot: i,
                    reason: e.to_string(),
                }
            })?;
            let ptr = pb.as_mut_ptr();
            (ptr, None, Some(pb))
        } else {
            let (ptr, layout) = alloc_aligned(self.size, i)?;
            (ptr, Some(layout), None)
        };

        #[cfg(not(feature = "server"))]
        let (ptr, layout): (*mut u8, Option<Layout>) = {
            let (ptr, l) = alloc_aligned(self.size, i)?;
            (ptr, Some(l))
        };

        // ── Register with client ─────────────────────────────────────────────
        #[cfg(feature = "client")]
        let client_registered = if let Some(client) = self.client.as_ref() {
            client.register(ptr, self.size).map_err(|e| {
                BufferPoolError::ConstructionFailed {
                    slot: i,
                    reason: format!("client register: {e}"),
                }
            })?;
            true
        } else {
            false
        };
        #[cfg(not(feature = "client"))]
        let client_registered = false;

        Ok(Slot {
            ptr,
            size: self.size,
            layout,
            #[cfg(feature = "server")]
            srv_buf,
            client_registered,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn alloc_aligned(size: usize, slot_idx: usize) -> Result<(*mut u8, Layout), BufferPoolError> {
    let layout = Layout::from_size_align(size, RDMA_ALIGN).map_err(|_| {
        BufferPoolError::ConstructionFailed {
            slot: slot_idx,
            reason: format!("invalid layout: size={size}, align={RDMA_ALIGN}"),
        }
    })?;
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        return Err(BufferPoolError::ConstructionFailed {
            slot: slot_idx,
            reason: "std::alloc::alloc returned null".into(),
        });
    }
    Ok((ptr, layout))
}
