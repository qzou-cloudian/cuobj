// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example: handle RDMA PUT and GET operations with cuObjServer.
//!
//! Run with:
//!   cargo run --example server --features server
//!
//! If the binary was built without an RPATH, the library search path may be
//! needed for `libcuobjclient.so` (the server links both libraries):
//!   LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib \
//!     RDMA_IP=<nic-ip> RDMA_PORT=4421 \
//!     cargo run --example server --features server

#[cfg(feature = "server")]
fn rdma_ip() -> String {
    if let Ok(ip) = std::env::var("RDMA_IP") {
        return ip;
    }
    if let Some(ip) = rdma_ip_from_cufile() {
        return ip;
    }
    panic!("set RDMA_IP or populate /etc/cufile.json properties.rdma_dev_addr_list");
}

#[cfg(feature = "server")]
fn rdma_ip_from_cufile() -> Option<String> {
    let content = std::fs::read_to_string("/etc/cufile.json").ok()?;
    let after_key = content.split("rdma_dev_addr_list").nth(1)?;
    let after_bracket = after_key.split('[').nth(1)?;
    let ip = after_bracket.split('"').nth(1)?;
    if ip.is_empty() {
        None
    } else {
        Some(ip.to_owned())
    }
}

#[cfg(feature = "server")]
fn main() {
    use cuobj::CuObjServer;

    // Bind to an RDMA-capable NIC. RDMA_IP overrides /etc/cufile.json.
    let ip = rdma_ip();
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
    let ch = server.alloc_channel().expect("alloc_channel failed");
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
