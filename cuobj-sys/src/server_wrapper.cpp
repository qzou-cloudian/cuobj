// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file server_wrapper.cpp
 * @brief C shim wrapping the cuObjServer C++ class for Rust FFI.
 *
 * Exposes plain C functions so the Rust example can call cuObjServer
 * methods without dealing with C++ name mangling, vtables, or std::string.
 *
 * Lifecycle per RDMA operation:
 *
 *   1. cuobj_srv_alloc_register(handle, size)        // allocate + register
 *   2. cuobj_srv_handle_put / cuobj_srv_handle_get   // RDMA transfer
 *   3. cuobj_srv_deregister_free(handle, buf)         // deregister + free
 */

#include <cuobjserver.h>   // /usr/include (system path)

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

/** Opaque server handle (wraps cuObjServer*). */
struct CuObjSrvHandle {
    cuObjServer *server;
};

/**
 * Opaque registered-buffer handle.
 *
 * Combines the page-aligned host allocation returned by
 * cuObjServer::allocHostBuffer() with the RDMA registration returned by
 * cuObjServer::registerBuffer().
 */
struct CuObjSrvBuffer {
    void              *host_ptr; ///< Aligned host memory (free with ::free())
    struct rdma_buffer *rbuf;    ///< RDMA-registered handle
    size_t             size;     ///< Allocation size in bytes
};

// ─────────────────────────────────────────────────────────────────────────────
// Server lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a cuObjServer instance bound to the given RDMA NIC IP and port.
 *
 * @return Opaque handle, or NULL on allocation failure.
 */
void *cuobj_srv_create(const char *ip, unsigned short port)
{
    CuObjSrvHandle *h = new (std::nothrow) CuObjSrvHandle();
    if (!h) return nullptr;

    h->server = new (std::nothrow) cuObjServer(ip, port, CUOBJ_PROTO_RDMA_DC_V1);
    if (!h->server) {
        delete h;
        return nullptr;
    }
    return static_cast<void *>(h);
}

/** Destroy a handle created by cuobj_srv_create(). */
void cuobj_srv_destroy(void *handle)
{
    if (!handle) return;
    CuObjSrvHandle *h = static_cast<CuObjSrvHandle *>(handle);
    delete h->server;
    delete h;
}

/** Returns 1 if the server is connected to the RDMA fabric, 0 otherwise. */
int cuobj_srv_is_connected(void *handle)
{
    CuObjSrvHandle *h = static_cast<CuObjSrvHandle *>(handle);
    return h->server->isConnected() ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel management (needed for concurrent RDMA operations)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Allocate a unique channel ID for use in one thread.
 * Each concurrent RDMA operation must use a distinct channel.
 */
uint16_t cuobj_srv_alloc_channel(void *handle)
{
    CuObjSrvHandle *h = static_cast<CuObjSrvHandle *>(handle);
    return h->server->allocateChannelId();
}

/** Return a channel ID allocated by cuobj_srv_alloc_channel(). */
void cuobj_srv_free_channel(void *handle, uint16_t channel)
{
    CuObjSrvHandle *h = static_cast<CuObjSrvHandle *>(handle);
    h->server->freeChannelId(channel);
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer management
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Allocate a 4 KiB-aligned host buffer and register it with the RDMA fabric.
 *
 * @return Opaque CuObjSrvBuffer*, or NULL on failure.
 *         Caller must free with cuobj_srv_deregister_free().
 */
void *cuobj_srv_alloc_register(void *handle, size_t size)
{
    CuObjSrvHandle *h = static_cast<CuObjSrvHandle *>(handle);

    CuObjSrvBuffer *buf = new (std::nothrow) CuObjSrvBuffer();
    if (!buf) return nullptr;

    buf->host_ptr = h->server->allocHostBuffer(size);
    if (!buf->host_ptr) {
        delete buf;
        return nullptr;
    }

    buf->rbuf = h->server->registerBuffer(buf->host_ptr, size);
    if (!buf->rbuf) {
        free(buf->host_ptr);
        delete buf;
        return nullptr;
    }

    buf->size = size;
    return static_cast<void *>(buf);
}

/**
 * Deregister and free a buffer previously allocated by cuobj_srv_alloc_register().
 */
void cuobj_srv_deregister_free(void *handle, void *buf_handle)
{
    if (!handle || !buf_handle) return;
    CuObjSrvHandle  *h   = static_cast<CuObjSrvHandle *>(handle);
    CuObjSrvBuffer  *buf = static_cast<CuObjSrvBuffer *>(buf_handle);

    h->server->deRegisterBuffer(buf->rbuf);
    free(buf->host_ptr);
    delete buf;
}

/**
 * Return the raw data pointer for a CuObjSrvBuffer.
 * Use this to copy data into/out of the buffer before/after RDMA operations.
 */
void *cuobj_srv_buf_data(void *buf_handle)
{
    CuObjSrvBuffer *buf = static_cast<CuObjSrvBuffer *>(buf_handle);
    return buf ? buf->host_ptr : nullptr;
}

/** Return the size in bytes of a CuObjSrvBuffer. */
size_t cuobj_srv_buf_size(void *buf_handle)
{
    CuObjSrvBuffer *buf = static_cast<CuObjSrvBuffer *>(buf_handle);
    return buf ? buf->size : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// RDMA transfers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle an S3 PUT: perform an RDMA_READ from the client's buffer into the
 * server's local buffer.
 *
 * The `rdma_descr` is the cuObj RDMA descriptor string received from the
 * client in the `x-amz-rdma-token` HTTP request header.
 *
 * After this call returns, `buf_handle`'s host_ptr contains the data that was
 * in the client's GPU/host buffer.
 *
 * @return Number of bytes transferred on success, or -1 on failure.
 */
ssize_t cuobj_srv_handle_put(
    void       *handle,
    const char *key,
    void       *buf_handle,
    uint64_t    remote_buf_start,
    size_t      size,
    const char *rdma_descr,
    uint16_t    channel)
{
    CuObjSrvHandle *h   = static_cast<CuObjSrvHandle *>(handle);
    CuObjSrvBuffer *buf = static_cast<CuObjSrvBuffer *>(buf_handle);
    ssize_t n = h->server->handlePutObject(
        std::string(key),
        buf->rbuf,
        remote_buf_start,
        size,
        std::string(rdma_descr),
        channel,
        /*local_offset=*/0,
        /*wc_status_out=*/nullptr);
    return n;
}

/**
 * Handle an S3 GET: perform an RDMA_WRITE from the server's local buffer into
 * the client's buffer.
 *
 * Populate `buf_handle`'s host_ptr with the object data **before** calling
 * this function.  The data will be RDMA-written directly into the client's
 * GPU/host memory described by `rdma_descr`.
 *
 * @return Number of bytes transferred on success, or -1 on failure.
 */
ssize_t cuobj_srv_handle_get(
    void       *handle,
    const char *key,
    void       *buf_handle,
    uint64_t    remote_buf_start,
    size_t      size,
    const char *rdma_descr,
    uint16_t    channel)
{
    CuObjSrvHandle *h   = static_cast<CuObjSrvHandle *>(handle);
    CuObjSrvBuffer *buf = static_cast<CuObjSrvBuffer *>(buf_handle);
    return h->server->handleGetObject(
        std::string(key),
        buf->rbuf,
        remote_buf_start,
        size,
        std::string(rdma_descr),
        channel,
        /*local_offset=*/0);
}

} // extern "C"
