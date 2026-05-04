// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example: register a host buffer with cuObjClient and generate RDMA tokens.
//!
//! Run with:
//!   cargo run --example client --features client

#[cfg(feature = "client")]
fn main() {
    use cuobj::{CuObjClient, MemoryType};

    let client = CuObjClient::new().expect("cuObjClient init failed — is RDMA available?");

    println!("connected: {}", client.is_connected());

    // Classify a stack pointer (will be System memory).
    let probe: u8 = 0;
    let mem_type = CuObjClient::memory_type(&probe as *const u8);
    println!("stack memory type: {mem_type:?}");
    assert_eq!(mem_type, MemoryType::System);

    // Allocate a 4 MiB buffer and register it.
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let reg = client
        .register_raii(buf.as_mut_ptr(), buf.len())
        .expect("registration failed");

    println!(
        "registered {} MiB at {:p}",
        reg.size() / (1024 * 1024),
        reg.ptr()
    );
    println!("max transfer size: {:?}", reg.max_transfer_size());

    // Generate a PUT token covering the full buffer at offset 0.
    let put_tok = reg.put_token(0, reg.size()).expect("put_token failed");
    println!("PUT token ({} bytes): {}", put_tok.len(), &put_tok.as_str());

    // Generate a GET token for the first 1 MiB.
    let get_tok = reg.get_token(0, 1024 * 1024).expect("get_token failed");
    println!("GET token ({} bytes): {}", get_tok.len(), &get_tok.as_str());

    // `reg` drops here → buffer is automatically deregistered.
    println!("buffer deregistered (on drop)");
}

#[cfg(not(feature = "client"))]
fn main() {
    eprintln!("Enable the `client` feature: cargo run --example client --features client");
}
