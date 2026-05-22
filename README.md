# cuobj

Safe Rust bindings for the [NVIDIA cuObj SDK](https://developer.nvidia.com/cuobj-sdk) — enabling **GPU-Direct RDMA** data transfers between client host/CUDA memory and a server over InfiniBand or RoCE networks.

The cuObj SDK follows an S3-like PUT/GET object model. RDMA descriptors are exchanged as opaque token strings (compatible with the `x-amz-rdma-token` HTTP header), allowing zero-copy data movement between a client's registered memory and a server.

## Features

- **Client side** — register CPU or CUDA memory buffers with the RDMA fabric and generate PUT/GET token descriptors.
- **Server side** — receive tokens and perform zero-copy RDMA transfers (`RDMA_READ` for PUT, `RDMA_WRITE` for GET).
- **RAII resource management** — registered buffers, channels, and tokens are automatically freed on drop.
- **Lifetime-free buffer guards** — `RegisteredBuffer`, `AllocatedBuffer`, `Channel`, and `PersistentBuffer` hold `Arc` references instead of borrows, so they can be stored in structs and held across `async` await points.
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
│   CuObjRdmaToken               PersistentBuffer          │
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
cuobj = { version = "0.2", features = ["client"] }

# Server only
cuobj = { version = "0.2", features = ["server"] }

# Both
cuobj = { version = "0.2", features = ["client", "server"] }
```

### Environment variables

The build locates the cuObj headers and shared libraries via:

| Variable | Default | Purpose |
|---|---|---|
| `CUOBJ_ROOT_DIR` | See below | Override SDK root (headers + SDK libs) |
| `CUOBJ_LIB_DIR` | — | Override library search path for `libcuobjclient.so` |

When `CUOBJ_ROOT_DIR` is not set, the build searches these paths in order:

1. `/usr/local/cuda-13.2/targets/x86_64-linux`
2. `/usr/local/cuda/targets/x86_64-linux`
3. `/usr/local/cuda`

`libcuobjserver.so` is searched in `/usr/lib/x86_64-linux-gnu`, `/usr/lib`, the CUDA SDK library directories above, and finally `/usr/local/lib`.

### Runtime library paths

Builds produced after the addition of RPATH embedding (see `build.rs`) do not need
`LD_LIBRARY_PATH`.  For builds without an embedded RPATH, set:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib
```

The server may link both `libcuobjserver.so` and `libcuobjclient.so` from the
CUDA SDK library directory.  If your installation provides `libcuobjserver.so`
from a system library path, that path is preferred before the SDK fallback.

**Note:** `/usr/local/lib` may contain manually-installed cuObj builds that are
incompatible with the running glibc (observed as a glibc `malloc` assertion
crash).  Point `LD_LIBRARY_PATH` to the CUDA SDK library directory instead.

## Usage

### Client

```rust
use cuobj::{CuObjClient, MemoryType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CuObjClient::new()?;
    assert!(client.is_connected());

    // Allocate a buffer and register it with the RDMA fabric.
    let mut data = vec![0u8; 4 * 1024 * 1024]; // 4 MiB

    // `register_buffer` returns a guard that deregisters on drop.
    let buf = client.register_buffer(data.as_mut_ptr(), data.len())?;

    println!("max transfer size: {}", buf.max_transfer_size().unwrap_or(0));

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
RDMA_PORT=4421 cargo run --example server --features server
```

If `RDMA_IP` is not set, the example uses the first address in
`/etc/cufile.json` under `properties.rdma_dev_addr_list`.

### End-to-end ping-pong

The `pingpong` example demonstrates a full bidirectional RDMA round-trip between a client and server process over a real network.

```sh
# Build
cargo build --example pingpong --features client,server

# Terminal 1 — server
RDMA_PORT=4421 ./target/debug/examples/pingpong server

# Terminal 2 — client
./target/debug/examples/pingpong client <server-ip>
```

If `RDMA_IP` is not set for the server, `pingpong` uses the first address in
`/etc/cufile.json` under `properties.rdma_dev_addr_list`.

The client registers two buffers (ping for PUT, pong for GET), sends their RDMA tokens over a TCP control connection, and verifies the pong buffer contains the expected data after the server completes both transfers.

If you see an error about missing `libcuobjclient.so.1`, see the
[Runtime library paths](#runtime-library-paths) section above.

## API Overview

### `CuObjClient` (feature `client`)

| Method | Description |
|---|---|
| `CuObjClient::new()` | Create client; returns error if RDMA is unavailable |
| `is_connected()` | Check RDMA fabric connectivity |
| `register_buffer(ptr, size)` | Register buffer; returns `RegisteredBuffer` RAII guard |
| `register(ptr, size)` / `deregister(ptr)` | Low-level manual registration |
| `put_token(ptr, size, offset)` | Generate PUT token for a manually registered buffer |
| `get_token(ptr, size, offset)` | Generate GET token for a manually registered buffer |
| `memory_type(ptr)` | Classify a pointer: `System`, `CudaManaged`, `CudaDevice`, … |
| `is_rdma_suitable(ptr)` | Returns `true` for RDMA-compatible memory types |

**Free functions** (no `CuObjClient` instance required):

| Function | Description |
|---|---|
| `cuobj::memory_type(ptr)` | Classify pointer memory type |
| `cuobj::is_rdma_suitable(ptr)` | Check RDMA suitability |

**`RegisteredBuffer`** (returned by `register_buffer`):

| Method | Description |
|---|---|
| `put_token(offset, len)` | Generate PUT descriptor (server will RDMA_READ) |
| `get_token(offset, len)` | Generate GET descriptor (server will RDMA_WRITE) |
| `max_transfer_size()` | Maximum bytes per RDMA callback |
| `as_slice()` / `as_mut_slice()` | View buffer contents |
| `Deref<[u8]>` / `DerefMut` | Slice indexing (`&buf[..]`, `buf[..n].copy_from_slice(...)`) |

**`CuObjRdmaToken`** (returned by `put_token` / `get_token`):

| Method / Trait | Description |
|---|---|
| `as_str()` / `as_bytes()` | View descriptor without consuming |
| `into_string()` / `into_bytes()` | Consume and copy descriptor |
| `Display` / `AsRef<str>` / `AsRef<[u8]>` | Standard conversions |
| `From<CuObjRdmaToken>` for `String` / `Vec<u8>` | Idiomatic `into()` |

### `CuObjServer` (feature `server`)

| Method | Description |
|---|---|
| `CuObjServer::new(ip, port)` | Bind RDMA server to NIC IP:port |
| `is_connected()` | Check RDMA fabric connectivity |
| `receive_put(key, offset, size, token)` | RDMA_READ from client → `Vec<u8>` |
| `send_get(key, offset, data, token)` | RDMA_WRITE `data` → client memory |
| `alloc_buffer(size)` | Allocate RDMA-registered host buffer (low-level) |
| `alloc_channel()` | Allocate a channel ID for one concurrent transfer (low-level) |
| `scatter_gather_buffer(base, entries)` | Create a derived SG RDMA buffer over ranges in an `AllocatedBuffer` |
| `handle_put(key, buf, offset, size, token, channel)` | Low-level RDMA_READ |
| `handle_get(key, buf, offset, size, token, channel)` | Low-level RDMA_WRITE |
| `free_channel_by_id(id)` | Free a channel whose `Channel` guard was leaked |
| `poll_channel(events, channel_id)` | Harvest async operation completions |

**`PersistentBuffer`** (for buffer pools; constructed from `Arc<CuObjServer>`):

| Method | Description |
|---|---|
| `PersistentBuffer::new(server, size)` | Allocate and register; buffer lives until dropped |
| `scatter_gather(entries)` | Create a derived SG RDMA buffer over ranges in this buffer |
| `handle_put(key, offset, size, token, channel_id)` | RDMA_READ into this buffer |
| `handle_get(key, offset, size, token, channel_id)` | RDMA_WRITE from this buffer |
| `submit_put_async(...)` | Submit async RDMA_READ (`unsafe`) |
| `submit_get_async(...)` | Submit async RDMA_WRITE (`unsafe`) |
| `as_slice()` / `as_mut_slice()` | View buffer contents |
| `Deref<[u8]>` / `DerefMut` | Slice indexing |

**`ScatterGatherBuffer`** (derived from an existing server buffer):

| Method | Description |
|---|---|
| `handle_put(key, remote_offset, token, channel_id)` | RDMA_READ into SG ranges |
| `handle_get(key, remote_offset, token, channel_id)` | RDMA_WRITE from SG ranges |
| `submit_put_async(...)` / `submit_get_async(...)` | Submit async SG transfers (`unsafe`) |

## License

Apache-2.0 — see [LICENSE](LICENSE).
