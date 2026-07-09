// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Native entry point: a shim over the library so the same modules also
//! back the WebAssembly build (see `lib.rs` and `web.rs`).

fn main() {
    bouncy::run();
}
