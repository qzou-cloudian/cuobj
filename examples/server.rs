// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example: handle RDMA PUT and GET operations with cuObjServer.
//!
//! Run with:
//!   cargo run --example server --features server

#[cfg(feature = "server")]
fn main() {
    use cuobj::CuObjServer;

    // Bind to an RDMA-capable NIC. Adjust IP and port for your environment.
    let ip   = std::env::var("RDMA_IP").unwrap_or_else(|_| "10.50.132.250".into());
    let port = std::env::var("RDMA_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(4421u16);

    let server = CuObjServer::new(&ip, port)
        .unwrap_or_else(|e| panic!("cuObjServer init failed ({ip}:{port}): {e}"));

    println!("server created at {ip}:{port}");
    println!("connected: {}", server.is_connected());

    // Low-level: allocate an RDMA-registered buffer manually.
    let buf = server
        .alloc_buffer(4 * 1024 * 1024)
        .expect("alloc_buffer failed");
    println!(
        "allocated {} MiB buffer ({} bytes)",
        buf.len() / (1024 * 1024),
        buf.len()
    );

    // Low-level: allocate a channel for one concurrent operation.
    let ch = server.alloc_channel();
    println!("channel id: {}", ch.id());

    // In a real server you would call:
    //
    //   server.handle_put("bucket/key", &buf, 0, size, rdma_token, &ch)?;
    //   server.handle_get("bucket/key", &buf, 0, size, rdma_token, &ch)?;
    //
    // or use the high-level wrappers that handle everything automatically:
    //
    //   let data = server.receive_put("bucket/key", size, rdma_token)?;
    //   server.send_get("bucket/key", &data, rdma_token)?;

    println!("channel and buffer freed on drop");
    // `ch` and `buf` drop here → channel returned, buffer deregistered and freed.
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("Enable the `server` feature: cargo run --example server --features server");
}
