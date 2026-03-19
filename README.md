# cuobj

Safe Rust bindings for the [NVIDIA cuObj SDK](https://developer.nvidia.com/cuobj-sdk) — enabling **GPU-Direct RDMA** data transfers between client host/CUDA memory and a server over InfiniBand or RoCE networks.

The cuObj SDK follows an S3-like PUT/GET object model. RDMA descriptors are exchanged as opaque token strings (compatible with the `x-amz-rdma-token` HTTP header), allowing zero-copy data movement between a client's registered memory and a server.

## Features

- **Client side** — register CPU or CUDA memory buffers with the RDMA fabric and generate PUT/GET token descriptors.
- **Server side** — receive tokens and perform zero-copy RDMA transfers (`RDMA_READ` for PUT, `RDMA_WRITE` for GET).
- **RAII resource management** — registered buffers, channels, and tokens are automatically freed on drop.
- **Two API tiers** — high-level helpers (`receive_put` / `send_get`) for simple use cases, and low-level buffer/channel control for advanced scenarios.
- **Thread safe** — `CuObjClient` and `CuObjServer` are `Send + Sync`.
- **Feature flags** — compile only what you need (`client`, `server`, or both).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    cuobj  (safe Rust)                    │
│   src/client/mod.rs            src/server/mod.rs         │
│   CuObjClient                  CuObjServer               │
│   RegisteredBuffer             AllocatedBuffer           │
│   RdmaToken                    PersistentBuffer          │
│   MemoryType / OpType          Channel                   │
└──────────────────┬─────────────────────┬─────────────────┘
                   │  unsafe FFI calls   │
┌──────────────────▼─────────────────────▼─────────────────┐
│                  cuobj-sys  (unsafe FFI)                  │
│   extern "C" declarations  +  build.rs (cc crate)        │
│   wrapper.cpp (client shim)   server_wrapper.cpp          │
└──────────────────┬─────────────────────┬─────────────────┘
                   │                     │
      libcuobjclient.so          libcuobjserver.so
               │                         │
               └────────── RDMA ─────────┘
                     InfiniBand / RoCE NIC
```

C++ shims bridge Rust's inability to call C++ class methods directly — each shim translates a plain-C opaque handle into the underlying `cuObjClient` / `cuObjServer` C++ object.

## Requirements

- Linux x86_64
- NVIDIA cuObj SDK (`libcuobjclient.so`, `libcuobjserver.so`) — typically installed with CUDA 13.2+
- An RDMA-capable NIC (InfiniBand or RoCE)
- Rust 2021 edition or later

## Installation

Add `cuobj` to your `Cargo.toml` with the features you need:

```toml
[dependencies]
# Client only
cuobj = { version = "0.1", features = ["client"] }

# Server only
cuobj = { version = "0.1", features = ["server"] }

# Both
cuobj = { version = "0.1", features = ["client", "server"] }
```

### Environment variables

The build locates the cuObj headers and shared libraries via:

| Variable | Default | Purpose |
|---|---|---|
| `CUOBJ_ROOT_DIR` | `/usr/local/cuda-13.2/targets/x86_64-linux` | cuObj SDK root (headers + client lib) |
| `CUOBJ_LIB_DIR` | Same as `CUOBJ_ROOT_DIR` | Override library search path |

`libcuobjserver.so` is searched in `/usr/lib/x86_64-linux-gnu`, `/usr/lib`, and `/usr/local/lib`.

## Usage

### Client

```rust
use cuobj::{CuObjClient, MemoryType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CuObjClient::new()?;
    assert!(client.is_connected());

    // Allocate a buffer and register it with the RDMA fabric.
    let mut data = vec![0u8; 4 * 1024 * 1024]; // 4 MiB

    // `register_raii` returns a guard that deregisters on drop.
    let buf = client.register_raii(data.as_mut_ptr(), data.len())?;

    println!("max transfer size: {}", buf.max_transfer_size());

    // Generate token strings to hand to the server (e.g., via HTTP header).
    let put_token = buf.put_token(0, buf.size())?;  // server will READ from us
    let get_token = buf.get_token(0, 1024 * 1024)?; // server will WRITE to us

    println!("PUT token: {put_token}");
    println!("GET token: {get_token}");

    // Tokens and registration are freed automatically here.
    Ok(())
}
```

Run the example:

```sh
cargo run --example client --features client
```

### Server

```rust
use cuobj::CuObjServer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = CuObjServer::new("10.50.132.250", 4421)?;
    assert!(server.is_connected());

    // High-level: receive a PUT (RDMA_READ from client into a new Vec<u8>).
    // `token` was received from the client, e.g., via HTTP header.
    let token: &str = /* ... */ todo!();
    let data: Vec<u8> = server.receive_put("my-object", 0, 4 * 1024 * 1024, token)?;

    // High-level: send a GET (RDMA_WRITE from server buffer into client memory).
    server.send_get("my-object", 0, &data, token)?;

    Ok(())
}
```

Run the example:

```sh
RDMA_IP=10.50.132.250 RDMA_PORT=4421 cargo run --example server --features server
```

### End-to-end ping-pong

The `pingpong` example demonstrates a full bidirectional RDMA round-trip between a client and server process over a real network.

```sh
# Build
cargo build --example pingpong --features client,server

# Terminal 1 — server
RDMA_IP=<nic-ip> RDMA_PORT=4421 ./target/debug/examples/pingpong server

# Terminal 2 — client
./target/debug/examples/pingpong client <server-ip>
```

The client registers two buffers (ping for PUT, pong for GET), sends their RDMA tokens over a TCP control connection, and verifies the pong buffer contains the expected data after the server completes both transfers.

## API Overview

### `CuObjClient` (feature `client`)

| Method | Description |
|---|---|
| `CuObjClient::new()` | Create client; returns error if RDMA is unavailable |
| `is_connected()` | Check RDMA fabric connectivity |
| `register_raii(ptr, size)` | Register buffer; returns `RegisteredBuffer` RAII guard |
| `memory_type(ptr)` | Classify a pointer: `System`, `CudaManaged`, `CudaDevice`, … |
| `is_rdma_suitable(ptr)` | Returns `true` for RDMA-compatible memory types |

**`RegisteredBuffer`** (returned by `register_raii`):

| Method | Description |
|---|---|
| `put_token(offset, len)` | Generate PUT descriptor (server will RDMA_READ) |
| `get_token(offset, len)` | Generate GET descriptor (server will RDMA_WRITE) |
| `max_transfer_size()` | Maximum bytes per RDMA callback |

### `CuObjServer` (feature `server`)

| Method | Description |
|---|---|
| `CuObjServer::new(ip, port)` | Bind RDMA server to NIC IP:port |
| `is_connected()` | Check RDMA fabric connectivity |
| `receive_put(key, start, size, token)` | RDMA_READ from client → `Vec<u8>` |
| `send_get(key, start, data, token)` | RDMA_WRITE `data` → client memory |
| `alloc_buffer(size)` | Allocate RDMA-registered host buffer (low-level) |
| `alloc_channel()` | Allocate a channel ID for one concurrent transfer (low-level) |
| `handle_put(...)` | Low-level RDMA_READ into a pre-allocated buffer |
| `handle_get(...)` | Low-level RDMA_WRITE from a pre-allocated buffer |

## License

Apache-2.0 — see [LICENSE](LICENSE).
