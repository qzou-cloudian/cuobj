// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hardware integration tests for the cuObj client and server APIs.
//!
//! All tests are `#[ignore]` by default because they require a live RDMA NIC.
//!
//! # Running
//!
//! Set the NIC address via environment variable or `/etc/cufile.json`:
//!
//! ```sh
//! RDMA_IP=<nic-ip> cargo test --test rdma_integration \
//!     --features client,server -- --include-ignored
//! ```
//!
//! Individual suites can be run by feature:
//!
//! ```sh
//! RDMA_IP=<nic-ip> cargo test --test rdma_integration --features client \
//!     client -- --include-ignored
//! RDMA_IP=<nic-ip> cargo test --test rdma_integration --features server \
//!     server -- --include-ignored
//! ```

// ── helpers ───────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn rdma_ip() -> String {
    if let Ok(ip) = std::env::var("RDMA_IP") {
        return ip;
    }
    if let Some(ip) = rdma_ip_from_cufile() {
        return ip;
    }
    panic!("set RDMA_IP or populate /etc/cufile.json properties.rdma_dev_addr_list");
}

#[allow(dead_code)]
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

// ── client tests ──────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
mod client {
    use cuobj::{CuObjClient, MemoryType};

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn connect() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        assert!(
            client.is_connected(),
            "cuObjClient not connected to RDMA fabric"
        );
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn memory_type_stack() {
        let probe: u8 = 0;
        assert_eq!(
            CuObjClient::memory_type(&probe as *const u8),
            MemoryType::System
        );
        assert!(CuObjClient::is_rdma_suitable(&probe as *const u8));
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn register_and_deregister() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        let mut buf = vec![0u8; 4096];
        let reg = client
            .register_buffer(buf.as_mut_ptr(), buf.len())
            .expect("register_buffer");
        assert_eq!(reg.size(), 4096);
        assert!(!reg.ptr().is_null());
        // reg drops here → automatic deregistration
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn put_token_generation() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        let mut buf = vec![0u8; 4096];
        let reg = client
            .register_buffer(buf.as_mut_ptr(), buf.len())
            .expect("register_buffer");
        let tok = reg.put_token(0, reg.size()).expect("put_token");
        assert!(!tok.is_empty(), "PUT token must not be empty");
        assert!(
            tok.as_str().contains(':'),
            "token should be colon-delimited"
        );
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn get_token_generation() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        let mut buf = vec![0u8; 4096];
        let reg = client
            .register_buffer(buf.as_mut_ptr(), buf.len())
            .expect("register_buffer");
        let tok = reg.get_token(0, reg.size()).expect("get_token");
        assert!(!tok.is_empty());
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn token_bounds_check() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        let mut buf = vec![0u8; 4096];
        let reg = client
            .register_buffer(buf.as_mut_ptr(), buf.len())
            .expect("register_buffer");
        // offset + len > size should return an error, not panic
        assert!(reg.put_token(4000, 200).is_err());
        assert!(reg.get_token(0, 8192).is_err());
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn max_transfer_size() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        let mut buf = vec![0u8; 4096];
        let reg = client
            .register_buffer(buf.as_mut_ptr(), buf.len())
            .expect("register_buffer");
        let mts = reg.max_transfer_size();
        assert!(
            mts.is_some(),
            "max_transfer_size should be available for a registered buffer"
        );
        assert!(mts.unwrap() > 0);
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn token_into_string_and_bytes() {
        let client = CuObjClient::new().expect("CuObjClient::new");
        let mut buf = vec![0u8; 4096];
        let reg = client
            .register_buffer(buf.as_mut_ptr(), buf.len())
            .expect("register_buffer");
        let tok = reg.put_token(0, reg.size()).expect("put_token");
        let s: String = tok.into();
        assert!(!s.is_empty());

        let tok2 = reg.put_token(0, reg.size()).expect("put_token 2");
        let v: Vec<u8> = tok2.into();
        assert!(!v.is_empty());
    }
}

// ── server tests ──────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
mod server {
    use super::rdma_ip;
    use cuobj::{CuObjServer, PersistentBuffer};
    use std::sync::Arc;

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn connect() {
        let srv = CuObjServer::new(&rdma_ip(), 4440).expect("CuObjServer::new");
        assert!(
            srv.is_connected(),
            "cuObjServer not connected to RDMA fabric"
        );
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn channel_lifecycle() {
        let srv = CuObjServer::new(&rdma_ip(), 4441).expect("CuObjServer::new");
        let ch = srv.alloc_channel().expect("alloc_channel");
        let id = ch.id();
        assert_ne!(id, u16::MAX, "channel ID must not be INVALID_CHANNEL_ID");
        drop(ch); // automatically returned to pool
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn channel_into_id() {
        let srv = CuObjServer::new(&rdma_ip(), 4442).expect("CuObjServer::new");
        let id = srv.alloc_channel().expect("alloc_channel").into_id();
        assert_ne!(id, u16::MAX);
        srv.free_channel_by_id(id); // explicit free after into_id
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn multiple_channels_unique() {
        let srv = CuObjServer::new(&rdma_ip(), 4443).expect("CuObjServer::new");
        let channels: Vec<cuobj::Channel> = (0..4)
            .map(|_| srv.alloc_channel().expect("alloc_channel"))
            .collect();
        let ids: Vec<u16> = channels.iter().map(|c| c.id()).collect();
        let unique: std::collections::HashSet<u16> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "channel IDs must be distinct");
        // channels drop here → all returned to pool
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn alloc_buffer_lifecycle() {
        let srv = CuObjServer::new(&rdma_ip(), 4444).expect("CuObjServer::new");
        let buf = srv.alloc_buffer(4096).expect("alloc_buffer");
        assert_eq!(buf.len(), 4096);
        assert!(!buf.as_ptr().is_null());
        // buf drops here → deregistered and freed
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn persistent_buffer_lifecycle() {
        let srv = Arc::new(CuObjServer::new(&rdma_ip(), 4445).expect("CuObjServer::new"));
        let buf = PersistentBuffer::new(Arc::clone(&srv), 4096).expect("PersistentBuffer::new");
        assert_eq!(buf.len(), 4096);
        assert!(!buf.as_ptr().is_null());
        // buf drops before srv → correct order enforced by Arc
    }

    #[test]
    #[ignore = "requires RDMA hardware"]
    fn register_existing_buffer() {
        use std::alloc::{alloc, dealloc, Layout};
        let srv = CuObjServer::new(&rdma_ip(), 4446).expect("CuObjServer::new");
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null());
        let rbuf =
            unsafe { srv.register_existing_buffer(ptr, 4096) }.expect("register_existing_buffer");
        unsafe { srv.deregister_buffer(rbuf) };
        unsafe { dealloc(ptr, layout) };
    }
}

// ── end-to-end roundtrip (requires both features) ────────────────────────────

#[cfg(all(feature = "client", feature = "server"))]
mod roundtrip {
    use super::rdma_ip;
    use cuobj::{CuObjClient, CuObjServer, ScatterGatherEntry};

    #[test]
    #[ignore = "requires RDMA hardware with loopback support"]
    fn put_roundtrip() {
        let ip = rdma_ip();
        let server = CuObjServer::new(&ip, 4447).expect("CuObjServer::new");
        assert!(server.is_connected());

        let client = CuObjClient::new().expect("CuObjClient::new");
        assert!(client.is_connected());

        let mut ping = vec![0u8; 4096];
        ping[..4].copy_from_slice(b"ping");

        let reg = client
            .register_buffer(ping.as_mut_ptr(), ping.len())
            .expect("register_buffer");
        let token = reg.put_token(0, reg.size()).expect("put_token");

        let received = server
            .receive_put("test/put", 0, reg.size(), token.as_str())
            .expect("receive_put");

        assert_eq!(&received[..4], b"ping", "PUT: unexpected payload");
    }

    #[test]
    #[ignore = "requires RDMA hardware with loopback support"]
    fn get_roundtrip() {
        let ip = rdma_ip();
        let server = CuObjServer::new(&ip, 4448).expect("CuObjServer::new");
        assert!(server.is_connected());

        let client = CuObjClient::new().expect("CuObjClient::new");
        assert!(client.is_connected());

        let mut pong = vec![0u8; 4096];

        let reg = client
            .register_buffer(pong.as_mut_ptr(), pong.len())
            .expect("register_buffer");
        let token = reg.get_token(0, reg.size()).expect("get_token");

        server
            .send_get("test/get", 0, b"pong", token.as_str())
            .expect("send_get");

        assert_eq!(&pong[..4], b"pong", "GET: unexpected payload");
    }

    #[test]
    #[ignore = "requires RDMA hardware with loopback support"]
    fn put_scatter_gather_roundtrip() {
        let ip = rdma_ip();
        let server = CuObjServer::new(&ip, 4449).expect("CuObjServer::new");
        assert!(server.is_connected());

        let client = CuObjClient::new().expect("CuObjClient::new");
        assert!(client.is_connected());

        let mut source = vec![0u8; 4096];
        source[..8].copy_from_slice(b"abcdefgh");

        let reg = client
            .register_buffer(source.as_mut_ptr(), source.len())
            .expect("register_buffer");
        let token = reg.put_token(0, 8).expect("put_token");

        let mut local = server.alloc_buffer(4096).expect("alloc_buffer");
        local.as_mut_slice().fill(0xCD);
        let sg = server
            .scatter_gather_buffer(
                &local,
                &[
                    ScatterGatherEntry {
                        offset: 128,
                        size: 4,
                    },
                    ScatterGatherEntry {
                        offset: 512,
                        size: 4,
                    },
                ],
            )
            .expect("scatter_gather_buffer");

        let n = sg
            .handle_put("test/put-sg", 0, token.as_str(), None)
            .expect("handle_put sg");

        assert_eq!(n, 8);
        assert_eq!(&local.as_slice()[128..132], b"abcd");
        assert_eq!(&local.as_slice()[512..516], b"efgh");
        assert_eq!(local.as_slice()[0], 0xCD);
    }

    #[test]
    #[ignore = "requires RDMA hardware with loopback support"]
    fn get_scatter_gather_roundtrip() {
        let ip = rdma_ip();
        let server = CuObjServer::new(&ip, 4450).expect("CuObjServer::new");
        assert!(server.is_connected());

        let client = CuObjClient::new().expect("CuObjClient::new");
        assert!(client.is_connected());

        let mut destination = vec![0u8; 4096];
        let reg = client
            .register_buffer(destination.as_mut_ptr(), destination.len())
            .expect("register_buffer");
        let token = reg.get_token(0, 8).expect("get_token");

        let mut local = server.alloc_buffer(4096).expect("alloc_buffer");
        local.as_mut_slice().fill(0);
        local.as_mut_slice()[64..68].copy_from_slice(b"pong");
        local.as_mut_slice()[256..260].copy_from_slice(b"data");
        let sg = server
            .scatter_gather_buffer(
                &local,
                &[
                    ScatterGatherEntry {
                        offset: 64,
                        size: 4,
                    },
                    ScatterGatherEntry {
                        offset: 256,
                        size: 4,
                    },
                ],
            )
            .expect("scatter_gather_buffer");

        let n = sg
            .handle_get("test/get-sg", 0, token.as_str(), None)
            .expect("handle_get sg");

        assert_eq!(n, 8);
        assert_eq!(&destination[..8], b"pongdata");
    }
}
