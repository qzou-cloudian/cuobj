// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example: pre-registered RDMA buffer pool for high-throughput server use.
//!
//! Run with:
//!   cargo run --example buffer_pool --features client,server,buffer-pool

#[cfg(all(feature = "client", feature = "server", feature = "buffer-pool"))]
fn main() {
    use std::sync::Arc;
    use cuobj::{BufferPool, CuObjClient, CuObjServer};

    let client = Arc::new(
        CuObjClient::new().expect("cuObjClient init failed — is RDMA available?"),
    );
    let ip   = std::env::var("RDMA_IP").unwrap_or_else(|_| "192.168.100.1".into());
    let port = std::env::var("RDMA_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(4421u16);
    let server = Arc::new(
        CuObjServer::new(&ip, port)
            .unwrap_or_else(|e| panic!("cuObjServer init failed: {e}")),
    );

    // Build a pool of 4 slots × 4 MiB each, registered with both client and server.
    let pool = BufferPool::builder()
        .count(4)
        .size(4 * 1024 * 1024)
        .with_client(Arc::clone(&client))
        .with_server(Arc::clone(&server))
        .build()
        .expect("pool construction failed");

    println!(
        "pool ready: {} slots, {} MiB each ({} available)",
        pool.capacity(),
        4,
        pool.available()
    );

    // Acquire a slot — blocks until one is free.
    {
        let mut guard = pool.acquire();
        println!("acquired slot: {} bytes", guard.len());
        println!(
            "  client registered: {}  server registered: {}",
            guard.is_client_registered(),
            guard.is_server_registered()
        );

        // Fill with dummy data.
        guard.as_mut_slice().fill(0xAB);

        // Generate an RDMA PUT token (server will RDMA_READ from this buffer).
        let token = guard.put_token(0, guard.len()).expect("put_token failed");
        println!(
            "PUT token ({} bytes): {}...",
            token.len(),
            &token.as_str()[..token.len().min(40)]
        );

        // In a real handler you would send `token` as x-amz-rdma-token and call:
        //   guard.server_handle_put("bucket/key", 0, guard.len(), rdma_token_from_client)?;

        println!("available while held: {}", pool.available());
        // guard drops here → slot returned to pool.
    }

    println!("available after release: {}", pool.available());

    // Non-blocking acquire.
    {
        let maybe = pool.try_acquire();
        if let Some(guard) = maybe {
            println!("try_acquire succeeded: {} bytes", guard.len());
        }
    }
}

#[cfg(not(all(feature = "client", feature = "server", feature = "buffer-pool")))]
fn main() {
    eprintln!(
        "Enable required features: \
         cargo run --example buffer_pool --features client,server,buffer-pool"
    );
}
