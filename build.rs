// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level build script: re-emits RPATH linker flags for the final binary.
//!
//! `cuobj-sys` discovers the native library paths at build time, but
//! `cargo:rustc-link-arg` emitted by a dependency's build script does not
//! propagate to the final binary linking step (it affects only the rlib
//! creation, which lacks a real linker invocation).  This script ensures the
//! RPATH entries are present in every binary, example, and test that links
//! `cuobj`.
//!
//! The paths mirror the discovery logic in `cuobj-sys/build.rs`.

fn main() {
    let has_client = std::env::var("CARGO_FEATURE_CLIENT").is_ok();
    let has_server = std::env::var("CARGO_FEATURE_SERVER").is_ok();

    if !has_client && !has_server {
        return;
    }

    // ── Client library RPATH ────────────────────────────────────────────────

    if has_client {
        // Same search order as cuobj-sys/build.rs.
        let cuda_candidates: &[&str] = &[
            "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib",
        ];
        let client_lib_dir = std::env::var("CUOBJ_LIB_DIR")
            .ok()
            .or_else(|| {
                cuda_candidates
                    .iter()
                    .find(|p| std::path::Path::new(p).join("libcuobjclient.so").exists())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "/usr/local/cuda-13.2/targets/x86_64-linux/lib".into());

        println!(
            "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
            client_lib_dir
        );
    }

    // ── Server library RPATH ────────────────────────────────────────────────

    if has_server {
        let server_lib_dir = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib",
            "/usr/local/lib",
        ]
        .iter()
        .find(|p| std::path::Path::new(p).join("libcuobjserver.so").exists())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/usr/lib/x86_64-linux-gnu".into());

        println!(
            "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
            server_lib_dir
        );
    }

    // ── Re-run triggers ─────────────────────────────────────────────────────

    println!("cargo:rerun-if-env-changed=CUOBJ_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUOBJ_ROOT_DIR");
}
