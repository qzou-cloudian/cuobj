// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file wrapper.cpp
 * @brief C shim wrapping the cuObjClient C++ class for Rust FFI.
 *
 * Exposes a plain C API so that Rust can call cuObjClient methods without
 * needing to understand C++ name mangling or vtables.
 */

#include "cuobjclient.h"

#include <cstddef>
#include <cstdlib>
#include <new>

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// Internal wrapper struct
// ─────────────────────────────────────────────────────────────────────────────

struct CuObjClientWrapper {
    CUObjOps_t ops;
    cuObjClient *client;
};

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a cuObjClient instance with no-op I/O callbacks.
 *
 * The token-based workflow uses cuMemObjGetRDMAToken / cuMemObjPutRDMAToken
 * directly and does not invoke the CUObjIOOps callbacks, so we supply stubs.
 *
 * @return opaque handle, or NULL on allocation failure.
 */
void *cuobj_client_create(void)
{
    CuObjClientWrapper *w = new (std::nothrow) CuObjClientWrapper();
    if (!w) return nullptr;

    // No-op callbacks – never called on the token-based path.
    w->ops.get = nullptr;
    w->ops.put = nullptr;

    w->client = new (std::nothrow) cuObjClient(w->ops, CUOBJ_PROTO_RDMA_DC_V1);
    if (!w->client) {
        delete w;
        return nullptr;
    }
    return static_cast<void *>(w);
}

/** Destroy a handle created by cuobj_client_create(). */
void cuobj_client_destroy(void *handle)
{
    if (!handle) return;
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    delete w->client;
    delete w;
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory management
// ─────────────────────────────────────────────────────────────────────────────

/** Acquire an RDMA memory descriptor (pin/register the buffer). */
int cuobj_client_mem_get_descriptor(void *handle, void *ptr, size_t size)
{
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    return static_cast<int>(w->client->cuMemObjGetDescriptor(ptr, size));
}

/** Release a previously acquired RDMA memory descriptor. */
int cuobj_client_mem_put_descriptor(void *handle, void *ptr)
{
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    return static_cast<int>(w->client->cuMemObjPutDescriptor(ptr));
}

/**
 * Return the maximum single-callback transfer size for the given buffer.
 * Returns -1 on error.
 */
ssize_t cuobj_client_mem_get_max_request_callback_size(void *handle, void *ptr)
{
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    return w->client->cuMemObjGetMaxRequestCallbackSize(ptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// RDMA token generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get the RDMA descriptor string for a registered buffer slice.
 *
 * @param handle     cuObjClient wrapper handle.
 * @param ptr        Base pointer of the registered buffer.
 * @param size       Number of bytes covered by this token.
 * @param buf_offset Byte offset from ptr to start the descriptor region.
 * @param op         0 = GET, 1 = PUT  (matches cuObjOpType_enum).
 * @param desc_str_out [out] Heap-allocated descriptor string; caller must free
 *                    with cuobj_client_mem_put_rdma_token().
 * @return 0 on success (CU_OBJ_SUCCESS), 1 on failure.
 */
int cuobj_client_mem_get_rdma_token(void *handle, void *ptr, size_t size,
                                    size_t buf_offset, int op,
                                    char **desc_str_out)
{
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    cuObjOpType_t op_type = (op == 1) ? CUOBJ_PUT : CUOBJ_GET;
    return static_cast<int>(
        w->client->cuMemObjGetRDMAToken(ptr, size, buf_offset, op_type, desc_str_out));
}

/**
 * Free a descriptor string allocated by cuobj_client_mem_get_rdma_token().
 */
int cuobj_client_mem_put_rdma_token(void *handle, char *desc_str)
{
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    return static_cast<int>(w->client->cuMemObjPutRDMAToken(desc_str));
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Return the memory type of the given pointer.
 * Maps to cuObjMemoryType_t: 0=SYSTEM, 1=CUDA_MANAGED, 2=CUDA_DEVICE,
 *                             3=UNKNOWN, 4=INVALID.
 */
int cuobj_client_get_memory_type(const void *ptr)
{
    return static_cast<int>(cuObjClient::getMemoryType(ptr));
}

/** Returns 1 if the client is connected to the RDMA subsystem, 0 otherwise. */
int cuobj_client_is_connected(void *handle)
{
    CuObjClientWrapper *w = static_cast<CuObjClientWrapper *>(handle);
    return w->client->isConnected() ? 1 : 0;
}

} // extern "C"
