# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```sh
# Build with both features
cargo build --features client,server

# Build only client or server
cargo build --features client
cargo build --features server

# Run examples
cargo run --example client --features client
RDMA_IP=<rdma-nic-ip> RDMA_PORT=4421 cargo run --example server --features server

# End-to-end pingpong test
cargo build --example pingpong --features client,server
# Terminal 1 (server):  RDMA_IP=<nic-ip> ./target/debug/examples/pingpong server
# Terminal 2 (client): ./target/debug/examples/pingpong client <server-ip>

# Run all tests (none exist yet; cuobj tests belong in tests/ or src/**/tests)
cargo test --features client,server
```

**Runtime library paths**: The build embeds RPATH entries for `libcuobjclient.so`
(CUDA SDK) and `libcuobjserver.so` (system).  If you use a binary built before
RPATH was added, set:
```sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib
```
Do NOT point `LD_LIBRARY_PATH` at `/usr/local/lib` — that directory contains
manually-installed cuObj builds that are incompatible with Ubuntu 24.04's glibc
and will crash with a `malloc` assertion during server init.  The working
`libcuobjserver.so` lives in `/lib/x86_64-linux-gnu` (installed by the system
package).

Build requires the NVIDIA cuObj SDK. Set `CUOBJ_ROOT_DIR` to the SDK root, or `CUOBJ_LIB_DIR` to the directory containing `libcuobjclient.so`. `libcuobjserver.so` is auto-located in `/usr/lib` or `/usr/local/lib`.

## Architecture

**Two-crate workspace**: `cuobj` (safe Rust) → `cuobj-sys` (unsafe FFI + C++ shims).

```
src/client/mod.rs     CuObjClient, RegisteredBuffer (RAII), CuObjRdmaToken, MemoryType, OpType
src/server/mod.rs     CuObjServer, AllocatedBuffer, PersistentBuffer, Channel, async poll
src/lib.rs            Conditional re-exports gated by `client`/`server` features
cuobj-sys/src/lib.rs  extern "C" declarations + constants (CUOBJ_GET/PUT, CUOBJ_MEMORY_*, etc.)
cuobj-sys/src/wrapper.cpp       C shim: cuObjClient C++ class → plain C functions
cuobj-sys/src/server_wrapper.cpp C shim: cuObjServer C++ class → plain C functions
cuobj-sys/build.rs   `cc` crate compiles C++ shims, emits `cargo:rustc-link-lib` directives
```

**Key design patterns**:

- **RAII everywhere**: `RegisteredBuffer` auto-deregisters on drop; `AllocatedBuffer` / `PersistentBuffer` auto-deregister-and-free; `Channel` returns its ID to the pool; `CuObjRdmaToken` calls `cuMemObjPutRDMAToken` on drop. No manual cleanup needed.
- **Shared C++ handle lifetime**: `CuObjClient` holds `Arc<ClientHandle>`. `CuObjRdmaToken` clones the `Arc`, so the C++ `cuObjClient` object survives until all tokens are dropped — even if `CuObjClient` itself is dropped first.
- **Memory registration tracking**: `CuObjClient.registered: Mutex<HashSet<usize>>` prevents double-registration. Insert-before-FFI-call ensures the set remains inconsistent with the library on panic (which shouldn't happen). On drop, all tracked pointers are deregistered before `Arc<ClientHandle>` is released.
- **Token format**: Colon-delimited hex fields — `<remote_vaddr>:<size>:<rkey>:…`. The server parses these with `token_remote_addr()` and `token_client_size()` to compute `remote_start` and clamp transfer sizes.
- **Two API tiers on server**: High-level `receive_put`/`send_get` handle buffer+channel lifecycle internally and return `Vec<u8>`. Low-level `handle_put`/`handle_get` require caller-provided `AllocatedBuffer` and `Channel` for reuse/control.
- **`PersistentBuffer`** owns `Arc<CuObjServer>` (not a borrow), so it can be stored in long-lived structures like buffer pools. Supports `Option<u16>` channel_id: `None` auto-allocates a scoped channel, `Some(id)` uses a pre-allocated one for concurrent use.
- **Feature gates**: `#[cfg(feature = "client")]` and `#[cfg(feature = "server")]` on both pub modules and individual `extern "C"` blocks, so code for an unneeded side is completely excluded.

**Thread safety**: `CuObjClient` and `CuObjServer` are `Send + Sync`. The cuObj C++ library provides internal locking. Concurrent server operations require distinct `Channel` IDs. `CuObjRdmaToken` is `Send + Sync` (the C string is plain data, no thread affinity).

## Changes since `pool` module removal (commit `d05ac06`)

`PersistentBuffer` replaced the removed `BufferPool` — any code that used `BufferPool` should now use `PersistentBuffer::new(Arc<CuObjServer>, size)` directly, with `Option<u16>` channel_id for concurrent control.
