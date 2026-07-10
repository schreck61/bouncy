// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Native entry point: a shim over the library so the same modules also
//! back the WebAssembly build (see `lib.rs` and `web.rs`).

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    bouncy::run();
}

/// The wasm build enters through `web::WebHandle`, not `main`; this stub
/// only exists so `cargo build --target wasm32-unknown-unknown` without
/// `--lib` still compiles the workspace.
#[cfg(target_arch = "wasm32")]
fn main() {}
