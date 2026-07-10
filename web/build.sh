#!/bin/sh
# Build the web demo bundles into web/pkg (single-threaded, stable) and
# web/pkg-mt (multi-threaded, nightly). Run from the repository root.
# CI runs the same steps (.github/workflows/docs.yml).
set -eu

echo "== single-threaded bundle (stable) =="
wasm-pack build --target web --release --out-dir web/pkg

echo "== multi-threaded bundle (nightly + build-std) =="
# Shared memory needs explicit link args: rustc does not add
# --shared-memory/--import-memory for you, and wasm-bindgen's thread
# transform needs the TLS/heap symbols exported.
MT_RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals'
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--import-memory"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--shared-memory"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--max-memory=1073741824"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--export=__heap_base"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--export=__tls_base"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--export=__tls_size"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--export=__tls_align"
MT_RUSTFLAGS="$MT_RUSTFLAGS -C link-arg=--export=__wasm_init_tls"
MT_RUSTFLAGS="$MT_RUSTFLAGS --cfg getrandom_backend=\"wasm_js\""

RUSTFLAGS="$MT_RUSTFLAGS" cargo +nightly build --lib \
  --target wasm32-unknown-unknown --release \
  --features web-threads -Z build-std=panic_abort,std

# wasm-bindgen CLI version must match the wasm-bindgen crate version.
BINDGEN_VERSION=$(cargo metadata --format-version 1 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"]=="wasm-bindgen"))')
if ! command -v wasm-bindgen >/dev/null || \
   [ "$(wasm-bindgen --version | cut -d' ' -f2)" != "$BINDGEN_VERSION" ]; then
  cargo install wasm-bindgen-cli --version "$BINDGEN_VERSION" --locked
fi
rm -rf web/pkg-mt
wasm-bindgen --target web --out-dir web/pkg-mt \
  target/wasm32-unknown-unknown/release/bouncy.wasm

echo "== done =="
ls -la web/pkg/bouncy_bg.wasm web/pkg-mt/bouncy_bg.wasm
