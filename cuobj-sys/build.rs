// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script: compiles the cuObj C++ shims and links the native libraries.
//!
//! # Environment variables
//!
//! | Variable         | Description                                              |
//! |------------------|----------------------------------------------------------|
//! | `CUOBJ_ROOT_DIR` | Override the cuObj / CUDA SDK root.  When set, it is     |
//! |                  | tried first before the built-in fallback paths.          |
//! | `CUOBJ_LIB_DIR`  | Override the directory containing `libcuobjclient.so`.   |
//! |                  | Bypasses the normal library search entirely.             |
//!
//! # SDK search order
//!
//! When `CUOBJ_ROOT_DIR` is not set the build script looks for
//! `cuobjclient.h` (and `libcuobjclient.so`) in these paths, in order:
//!
//! 1. `$CUOBJ_ROOT_DIR/include` (if the variable is set)
//! 2. `/usr/local/cuda-13.2/targets/x86_64-linux`
//! 3. `/usr/local/cuda/targets/x86_64-linux`
//! 4. `/usr/local/cuda`

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!("help: Set CUOBJ_ROOT_DIR to the cuObj SDK root directory.");
    eprintln!("help: Set CUOBJ_LIB_DIR to override the library search path.");
    std::process::exit(1);
}

fn main() {
    let has_client = std::env::var("CARGO_FEATURE_CLIENT").is_ok();
    let has_server = std::env::var("CARGO_FEATURE_SERVER").is_ok();

    if !has_client && !has_server {
        return;
    }

    // ── Locate include directory (needed by both shims) ────────────────────────

    // Candidate CUDA / cuObj roots (in priority order).
    let cuda_candidates = [
        std::env::var("CUOBJ_ROOT_DIR").unwrap_or_default(),
        "/usr/local/cuda-13.2/targets/x86_64-linux".into(),
        "/usr/local/cuda/targets/x86_64-linux".into(),
        "/usr/local/cuda".into(),
    ];

    // Find the first root whose include/cuobjclient.h exists.
    let include_dir = cuda_candidates
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| std::path::PathBuf::from(p).join("include"))
        .find(|p| p.join("cuobjclient.h").exists())
        .unwrap_or_else(|| {
            fail("Cannot find cuobjclient.h.");
        });

    // ── Client shim ────────────────────────────────────────────────────────────

    if has_client {
        // Library search path for libcuobjclient.
        let client_lib_dir = if let Ok(d) = std::env::var("CUOBJ_LIB_DIR") {
            std::path::PathBuf::from(d)
        } else {
            cuda_candidates
                .iter()
                .filter(|p| !p.is_empty())
                .map(|p| std::path::PathBuf::from(p).join("lib"))
                .find(|p| p.join("libcuobjclient.so").exists())
                .unwrap_or_else(|| {
                    fail("Cannot find libcuobjclient.so.");
                })
        };

        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("src/wrapper.cpp")
            .include(&include_dir)
            .warnings(false)
            .compile("cuobjclientwrapper");

        println!(
            "cargo:rustc-link-search=native={}",
            client_lib_dir.display()
        );
        println!("cargo:rustc-link-lib=dylib=cuobjclient");
        // Embed the SDK library path in the binary so LD_LIBRARY_PATH isn't
        // needed at runtime (libcuobjclient.so is not in the default ld.so
        // search path on most systems).
        println!(
            "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
            client_lib_dir.display()
        );
        println!("cargo:rerun-if-changed=src/wrapper.cpp");
    }

    // ── Server shim ────────────────────────────────────────────────────────────

    if has_server {
        // Library search path for libcuobjserver.
        // Priority: system multiarch → system → local (last — /usr/local/lib
        // may contain builds that are incompatible with the running glibc).
        let server_lib_dirs: &[&str] = &["/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/local/lib"];
        let server_lib_dir = server_lib_dirs
            .iter()
            .map(|p| std::path::PathBuf::from(p))
            .find(|p| p.join("libcuobjserver.so").exists())
            .unwrap_or_else(|| {
                fail("Cannot find libcuobjserver.so.");
            });

        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("src/server_wrapper.cpp")
            // cuobjserver.h is a system header (/usr/include); no extra -I needed.
            // Include the CUDA path too so cuobjrdma.h can find cufile.h if needed.
            .include(&include_dir)
            .warnings(false)
            .compile("cuobjserverwrapper");

        println!(
            "cargo:rustc-link-search=native={}",
            server_lib_dir.display()
        );
        println!("cargo:rustc-link-lib=dylib=cuobjserver");
        println!(
            "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
            server_lib_dir.display()
        );
        println!("cargo:rerun-if-changed=src/server_wrapper.cpp");
    }

    // ── Re-run triggers ───────────────────────────────────────────────────────

    println!("cargo:rerun-if-env-changed=CUOBJ_ROOT_DIR");
    println!("cargo:rerun-if-env-changed=CUOBJ_LIB_DIR");
}
