// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pingpong example: client sends "ping" to server via RDMA PUT, server replies
//!
//! # Running
//! "pong" via RDMA GET. RDMA tokens are exchanged over a TCP control channel.
//!
//! Build with both features:
//!   cargo build --example pingpong --features client,server
//!
//! On the server machine:
//!   cargo run --example pingpong --features client,server -- server
//!
//! On the client machine:
//!   cargo run --example pingpong --features client,server -- client <server-ip>
//!
//! # Environment variables
//!
//!   RDMA_IP    RDMA NIC address for the server  (default: 10.50.132.250)
//!   RDMA_PORT  RDMA port for the server          (default: 4421)
//!   CTRL_PORT  TCP control-plane port            (default: 7070)

// ── server ───────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
fn run_server() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use cuobj::CuObjServer;

    let rdma_ip = std::env::var("RDMA_IP").unwrap_or_else(|_| "10.50.132.250".into());
    let rdma_port: u16 = std::env::var("RDMA_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(4421);
    let ctrl_port: u16 = std::env::var("CTRL_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7070);

    let server = CuObjServer::new(&rdma_ip, rdma_port)
        .unwrap_or_else(|e| panic!("[server] cuObjServer init failed ({rdma_ip}:{rdma_port}): {e}"));
    println!("[server] RDMA ready at {rdma_ip}:{rdma_port}  connected={}", server.is_connected());

    // Wait for the client to connect on the TCP control plane.
    let listener = TcpListener::bind(format!("0.0.0.0:{ctrl_port}"))
        .unwrap_or_else(|e| panic!("[server] cannot bind ctrl port {ctrl_port}: {e}"));
    println!("[server] listening for client on TCP port {ctrl_port} …");

    let (stream, peer) = listener.accept().expect("[server] accept failed");
    println!("[server] client connected from {peer}");

    let mut reader = BufReader::new(stream.try_clone().expect("stream clone"));
    let mut writer = stream;

    // ── round 1: receive "ping" via RDMA PUT ──────────────────────────────────

    // Read the size and PUT token that the client registered for its ping buffer.
    let mut line = String::new();
    reader.read_line(&mut line).expect("read ping size");
    let ping_size: usize = line.trim().parse().expect("parse ping size");
    line.clear();

    reader.read_line(&mut line).expect("read ping token");
    let ping_token = line.trim().to_owned();
    line.clear();
    println!("[server] receiving {ping_size} bytes via RDMA PUT …");
    let ping_data = server
        .receive_put("pingpong/ping", 0, ping_size, &ping_token)
        .expect("[server] receive_put failed");

    let ping_str = std::str::from_utf8(&ping_data[..4]).unwrap_or("????");
    println!("[server] received: {ping_str:?}");
    assert_eq!(&ping_data[..4], b"ping", "[server] unexpected payload");

    // ── round 2: send "pong" via RDMA GET ────────────────────────────────────

    // Read the size and GET token for the client's pong receive buffer.
    reader.read_line(&mut line).expect("read pong size");
    let _pong_size: usize = line.trim().parse().expect("parse pong size");
    line.clear();

    reader.read_line(&mut line).expect("read pong token");
    let pong_token = line.trim().to_owned();
    let pong_payload: &[u8] = b"pong";
    println!("[server] sending {:?} via RDMA GET …", std::str::from_utf8(pong_payload).unwrap());
    server
        .send_get("pingpong/pong", 0, pong_payload, &pong_token)
        .expect("[server] send_get failed");

    // Signal the client that both RDMA operations are complete.
    writeln!(writer, "done").expect("write done");
    println!("[server] pingpong complete ✓");
}

// ── client ───────────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
fn run_client(server_ip: &str) {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use cuobj::CuObjClient;

    let ctrl_port: u16 = std::env::var("CTRL_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7070);

    let client = CuObjClient::new().expect("[client] cuObjClient init failed");
    println!("[client] connected={}", client.is_connected());

    // ── allocate and register the ping (PUT) buffer ───────────────────────────

    // The client fills this buffer; the server will RDMA-read it.
    let mut ping_buf = vec![0u8; 4096];
    ping_buf[..4].copy_from_slice(b"ping");

    let reg_ping = client
        .register_raii(ping_buf.as_mut_ptr(), ping_buf.len())
        .expect("[client] register ping buffer failed");

    let put_token = reg_ping
        .put_token(0, reg_ping.size())
        .expect("[client] put_token failed");

    // ── allocate and register the pong (GET) buffer ───────────────────────────

    // The server will RDMA-write "pong" into this buffer.
    let mut pong_buf = vec![0u8; 4096];

    let reg_pong = client
        .register_raii(pong_buf.as_mut_ptr(), pong_buf.len())
        .expect("[client] register pong buffer failed");

    let get_token = reg_pong
        .get_token(0, reg_pong.size())
        .expect("[client] get_token failed");

    // ── connect to server control plane ──────────────────────────────────────

    let ctrl_addr = format!("{server_ip}:{ctrl_port}");
    println!("[client] connecting to {ctrl_addr} …");
    let mut stream =
        TcpStream::connect(&ctrl_addr).expect("[client] TCP control connection failed");
    let mut reader = BufReader::new(stream.try_clone().expect("stream clone"));

    // ── send PUT token (ping) ─────────────────────────────────────────────────

    writeln!(stream, "{}", reg_ping.size()).expect("write ping size");
    writeln!(stream, "{}", put_token.as_str()).expect("write put token");
    println!("[client] sent PUT token — server will RDMA-read our ping buffer");

    // ── send GET token (pong) ─────────────────────────────────────────────────

    writeln!(stream, "{}", reg_pong.size()).expect("write pong size");
    writeln!(stream, "{}", get_token.as_str()).expect("write get token");
    println!("[client] sent GET token — server will RDMA-write into our pong buffer");

    // ── wait for server "done" signal ─────────────────────────────────────────

    let mut line = String::new();
    reader.read_line(&mut line).expect("read done");
    assert_eq!(line.trim(), "done", "[client] unexpected control message: {line:?}");

    // ── verify the pong ───────────────────────────────────────────────────────

    // reg_pong.as_slice() and pong_buf share the same underlying memory.
    let received = &reg_pong.as_slice()[..4];
    let pong_str = std::str::from_utf8(received).unwrap_or("????");
    println!("[client] received: {pong_str:?}");
    assert_eq!(received, b"pong", "[client] expected \"pong\", got {pong_str:?}");
    println!("[client] pingpong complete ✓");

    // reg_pong and reg_ping drop here → buffers automatically deregistered.
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("server") => {
            #[cfg(feature = "server")]
            run_server();
            #[cfg(not(feature = "server"))]
            eprintln!("Recompile with `--features server` to enable the server mode.");
        }
        Some("client") => {
            #[cfg(feature = "client")]
            {
                let server_ip = args.get(2).map(String::as_str).unwrap_or("127.0.0.1");
                run_client(server_ip);
            }
            #[cfg(not(feature = "client"))]
            eprintln!("Recompile with `--features client` to enable the client mode.");
        }
        _ => {
            eprintln!(
                "Usage:
  pingpong server
  pingpong client [server-ip]   (default: 127.0.0.1)

Environment variables:
  RDMA_IP    RDMA NIC address (default: 10.50.132.250)
  RDMA_PORT  RDMA port        (default: 4421)
  CTRL_PORT  TCP ctrl port    (default: 7070)"
            );
            std::process::exit(1);
        }
    }
}
