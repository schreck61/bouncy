# Roadmap

Feature ideas agreed on but not yet built, in priority order (fun-per-effort).
Interface conventions to preserve: new mechanics get a CLI flag + a single-letter
hotkey + a HUD stats line; presets bundle settings but explicit flags always win;
every mechanic lands with unit tests in the headless `Simulation` core (src/sim.rs);
every bug fix ships with a regression test.

Release-by-release history lives in [CHANGELOG.md](CHANGELOG.md); this
file only tracks what's ahead.

## Shipped so far

- **1.0** completed the original numbered backlog and both display
  debts: fractional-scale window fill, velocity colors respecting
  `--initial-speed`, the screenshot key, scenes (presets carrying
  wells/walls geometry, exported with `E`), and comets.
- **1.1–1.2** scaled the physics: Barnes-Hut self-gravity, parallel
  contact detection (thread-count-invariant results), a terminal
  velocity, and the spawn-pressure explosion trigger with a non-linear
  population cap.
- **1.3** shipped the browser demo: the same simulation compiled to
  WebAssembly with an HTML control panel over the `Command` dispatch,
  WebAudio, live resize, share links, launch options, and an optional
  multi-threaded bundle — deployed to GitHub Pages by CI. This
  delivered the control-panel goal the old "GUI overlay" milestone was
  about, in the browser rather than in the native window.

## Next headline candidates

- **Web demo performance beyond Chrome.** The demo is currently
  optimized for Chrome; Firefox frame rates collapse under the
  per-frame rayon fork-join cost (up to ~8-16 dispatches per frame via
  substepping) and the Canvas2D `putImageData` present path. Options,
  roughly in leverage order: hoist the fan-out so a frame dispatches
  once; raise the parallel thresholds on wasm (the 1024 break-even was
  measured natively); cap the web thread pool below
  `hardwareConcurrency`; longer-term, a WebGL/WebGPU present path.
- **Native GUI overlay (egui), demoted.** An egui panel inside the
  native window (Tab to toggle): the web panel now covers the
  browser, but the native binary is still hotkeys-only. The groundwork
  holds — widgets would issue the same `Command`s as hotkeys, and
  `pixels` exposes `render_with()` for egui's render pass. The
  CPU/softbuffer backend would stay hotkeys-only (documented
  limitation).

## Smaller / opportunistic

- Particle springs/links: click two particles to bind with a spring; molecule
  building. Medium effort, niche payoff. (Particles carry stable ids, so
  the cross-frame identity problem is already solved.)
- Screensaver/attract mode: cycle through presets on a timer
  (`--cycle <secs>`); the sim is a screensaver at heart.

## Tuning debts / watch items

- The `blob` preset can still cool to a stop over very long runs (elastic
  losses via fusion). Mitigations now exist in-app (`F` flow stir or `A`
  self-gravity); accept STOPPED otherwise.
- The demo's multi-threaded bundle needs nightly + `-Zbuild-std`; a
  nightly breakage only degrades the deploy to single-threaded (CI
  builds the bundles in separate steps), but keep an eye on it.
