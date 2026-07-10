# Bouncy web demo

The same simulation the native binary runs, compiled to WebAssembly and
wrapped in an HTML control panel. Deployed to
`schreck61.github.io/bouncy/demo` by CI on every push to main; this
directory is the page source.

The demo is currently optimized for Chrome (and Chromium-based
browsers). Other browsers run it, but frame rates can be substantially
lower — Firefox in particular pays a much higher per-frame cost for the
thread-pool fan-out and the Canvas2D present path (`?st` may help
there).

## Build (single-threaded)

```sh
wasm-pack build --target web --release --out-dir web/pkg
```

Then serve this directory with any static file server, e.g.:

```sh
python3 -m http.server 8321 --directory web
```

`pkg/` is a build artifact (wasm-pack writes its own `.gitignore`).

## Build (multi-threaded, optional)

Rayon over wasm threads needs SharedArrayBuffer, which needs a
cross-origin-isolated page. GitHub Pages cannot set the COOP/COEP
headers, so `coi.js` (a small service worker of ours) injects them client-side (one reload
on first visit). The build needs nightly and a rebuilt std:

```sh
rustup toolchain install nightly --component rust-src
./web/build.sh    # builds both bundles; see the script for the details
```

Two hard-won details live in that script: rustc does **not** add the
shared-memory link args itself (`--import-memory --shared-memory
--max-memory` plus exported TLS/heap symbols must be passed as
`link-arg`s, or the module builds with ordinary memory and the thread
pool cannot start), and `wasm-bindgen-rayon` needs its `no-bundler`
feature for plain ES-module loading. One web-platform consequence of
shared memory: `ImageData` refuses views into a SharedArrayBuffer, so
the Canvas2D backend blits through a persistent JS-side copy of the
frame.

The loader (`controls.js`) prefers `pkg-mt/` when
`globalThis.crossOriginIsolated` is true and falls back to `pkg/`
otherwise. Both builds produce bit-identical simulations for a given
seed — thread-count invariance is enforced by tests.

## URL parameters

Every CLI option works as a query parameter, resolved by the same
parser with the same validation: `?preset=accretion`,
`?gravity=50&matter&seed=7`, `?width=800&height=600`. Value-less keys
(or `=true`) are boolean flags; `=false` drops the key.

Two loader-only parameters are handled by `controls.js` and stripped
before the query reaches the config parser: `?st` forces the
single-threaded bundle even on an isolated page, and `?cb=<anything>`
is accepted as a cache-buster (ignored by the simulation and removed
from share and restart links).
