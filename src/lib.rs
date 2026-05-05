// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe Rust bindings for the NVIDIA cuObj SDK.
//!
//! Enable features to choose which components to include:
//!
//! | Feature  | What it provides                     |
//! |----------|--------------------------------------|
//! | `client` | [`CuObjClient`], RDMA token generation |
//! | `server` | [`CuObjServer`], RDMA PUT/GET operations |

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "server")]
pub mod server;

// ── Re-exports ────────────────────────────────────────────────────────────────

#[cfg(feature = "client")]
pub use client::{CuObjClient, CuObjClientError, CuObjRdmaToken, MemoryType, OpType, RegisteredBuffer};

#[cfg(feature = "server")]
pub use server::{AllocatedBuffer, Channel, CuObjAsyncEvent, CuObjServer, CuObjServerError, PersistentBuffer, WcStatus};
