# Changelog

Notable changes to Bouncy, by release. Version numbers follow
[Semantic Versioning](https://semver.org); each release is tagged
`vX.Y.Z`.

## 1.3.1 — 2026-07-10

- Fixed URL-parameter parsing eating numeric values: the `=1`/`=0`
  truthiness aliases now apply only to boolean flags, so
  `?particle-size=1` (which the demo's launch options produce for
  whole-number sizes) parses as one pixel and `?gravity=0` disables
  gravity instead of being silently dropped.
- The demo header shows the running version, stamped from the wasm
  module itself.
- Documentation accuracy sweep: `accretion` listed in the `--preset`
  help, `J` (comet) listed in `--help`, `libudev-dev` added to the
  Linux prerequisites, loader-only URL parameters documented, this
  changelog added, the roadmap brought up to date, and doc comments on
  the previously undocumented public API surface. New drift tests pin
  the control and preset listings to the code.
- The browser demo is now explicitly recommended for Chrome until the
  frame-rate gap in other browsers is addressed.

## 1.3.0 — 2026-07-10

The browser demo release: the simulation compiled to WebAssembly and
deployed to GitHub Pages.

- Restructured the crate as a library plus binary and ported the
  application shell to WebAssembly (Canvas2D rendering, the same winit
  event loop driven by `requestAnimationFrame`).
- Added the demo page: an HTML control panel over the wasm boundary,
  with sliders, toggles, and buttons issuing the same commands the
  keyboard does.
- Added `Simulation::resize` so the arena tracks a responsive canvas.
- Added the multi-threaded wasm flavor: rayon over wasm threads on
  cross-origin-isolated pages, with a single-threaded fallback bundle.
- Wired the web demo into CI and the Pages deployment.
- Added WebAudio playback (collision pings and explosion rumbles in the
  browser; starts on a user gesture per autoplay policy).
- Added a launch-options section to the demo panel, and share links
  that preserve launch parameters and overlay session changes.

## 1.2.0 — 2026-07-08

- Explosions trigger on spawn pressure instead of successful births, so
  a population pegged at the density cap still explodes.
- Made the population cap non-linear: a window-coverage bound meets a
  flat ceiling, lowered to 12,000 particles.
- Fixed packed-clump regressions in the birth-rate trigger and a
  frame-rate collapse.

## 1.1.0 — 2026-07-08

The physics-scaling release.

- Replaced pairwise self-gravity with an adaptive Barnes-Hut quadtree,
  and parallelized its force pass with rayon.
- Parallelized collision contact detection, keeping resolution serial
  (results are identical for any thread count).
- Clamped particle speed at a terminal velocity so super-elastic
  scenes saturate instead of diverging.
- Kept collision grid cells fine-grained when merged giants appear.
- Manual explosions respect the configured minimum particle count.
- Published rustdoc to GitHub Pages on every push to main.

## 1.0.2 — 2026-07-07

- Fission births count toward the explosion threshold.

## 1.0.1 — 2026-07-07

- Added `J` as the comet hotkey (same action as middle click).

## 1.0.0 — 2026-07-07

First stable release of the native simulation: GPU rendering with CPU
fallback, elastic collisions on a spatial grid with adaptive
substepping, gravity and pinned/held wells, drawable walls, matter
mechanics (fusion/fission), self-gravity with an accretion preset, flow
field, kaleidoscope and trails, synthesized collision pings and
explosion rumbles with a musical mode, presets (built-in and user TOML
files), scene export, screenshots, comets, and bullet time.

## 0.x — 2025–2026

Incremental construction of the above: initial windowed simulation
(0.1–0.3: gravity, elasticity, motion detection), CPU fallback and
HiDPI fixes (0.4), modular restructure and Linux ARM64 builds (0.5),
runtime controls, cursor well, and time scale (0.6), the headless
simulation core (0.7), matter mechanics, presets, and the flow field
(0.8), pinned wells, bullet time, musical mode, kaleidoscope, and
drawable walls (0.9), user preset files (0.10), repo hygiene and
refactors (0.11), and self-gravity with the accretion preset (0.12).
