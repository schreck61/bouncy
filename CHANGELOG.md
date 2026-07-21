# Changelog

Notable changes to Bouncy, by release. Version numbers follow
[Semantic Versioning](https://semver.org); each release is tagged
`vX.Y.Z`.

## Unreleased

- The web demo's "Apply & restart" now starts from the share link's
  parameters rather than the page URL, so settings adjusted during the
  session survive a relaunch even when the session did not begin from
  a parameterized link — the same touched-settings-travel behavior as
  the native panel's relaunch.
- The native window has a control panel: Tab slides in a translucent
  panel with readouts, sliders, toggles, and action buttons matching
  the web demo's, driving the same command dispatch as the hotkeys.
  Sliders snap to meaningful detents — gravity 0 and 100, elasticity
  1.0, time scale 1.0 — with Shift bypassing the snap for fine
  control. Hand-drawn into the frame buffer like the HUD, so it works
  on both the GPU and CPU render backends. `--panel` starts with it
  open.
- The panel's action buttons are one-shot placement tools, exactly
  like the web panel's: click "Pin well" (or burst, comet, explode,
  repeller), then click the arena to place it there — a hint follows
  the cursor while armed, a second press or Esc cancels, and the tool
  survives hiding the panel, so you can arm it and then clear the
  panel away to place beneath where it sat. Toggles are
  capsule switches matching the web panel's look.
- The panel has a launch section mirroring the web demo's launch
  options: preset, particle size, initial speed, and min particles as
  draft values, applied by an "Apply & relaunch" button that rebuilds
  the simulation in place at the current window size — with full
  command-line preset resolution, so a panel relaunch and a CLI launch
  can never disagree. A bad configuration keeps the running simulation.
  Live settings you adjusted during the session survive the relaunch —
  touched values override the new bundle, untouched ones follow it —
  matching the web share link's philosophy.
- The panel also opens without the keyboard: dwelling at the window's
  right edge reveals a thin handle at mid-height — click it to slide
  the panel in or out, or grab and drag it to any position and let
  go. The handle fades with the idle cursor, and quick passes through
  the edge strip (drawing a wall to the edge) never see it.

## 1.4.4 — 2026-07-15

- Fixed collision spawns materializing across a wall when a fast,
  dense clump strikes it (reported with fused blobs full of
  fission-size particles). Integration can carry many particles
  across a wall in the same substep, and contacts between two such
  crossers recorded far-side spawn sites; walls now also resolve
  before the pair pass, so contact detection never sees a cross-wall
  state.
- Pair-separation pushout is now wall-contained immediately, inside
  the pair pass: a massive blob's separation shove could park a light
  particle across a wall transiently, poisoning any spawn site
  recorded from that position before the wall pass corrected it.

## 1.4.3 — 2026-07-14

- A frame-start safety net clamps every particle into the arena
  before physics runs, so no position writer — present or future —
  can park a particle in the out-of-bounds sliver beyond a wall's
  endpoints, the state every wall leak traced back to.
- Fission fragments now clamp into the arena: a shattering impact
  against an arena edge could park a fragment out of bounds, beyond
  the reach of every drawn wall, from where it could round a wall's
  endpoint and re-enter on the far side — the long-run divider leak.
- Fixed a rare pure-physics wall escape: particle-separation pushout
  could park a particle just outside the arena bounds, beyond a wall's
  endpoint, letting it slip around the wall while out of bounds and
  re-enter on the far side. Separation now clamps to the arena, the
  same bounds integration enforces.
- Matter events respect drawn walls: fission picks the fragment
  direction that keeps both fragments on the parent's side (or bounces
  instead of shattering when pressed against a wall), and a fusion's
  merged position never crosses a wall.
- New particles now spawn strictly on their source's side of drawn
  walls: collision-triggered births stay on their collision's side,
  and click bursts stay on the click's side, instead of occasionally
  materializing across a nearby wall.
- Drawn walls now block particle-particle interaction: a contact whose
  center-to-center line crosses a wall is skipped, so particles no
  longer exchange impulses — or, with matter on, fuse — through a
  zero-thickness wall. This closes one of the ways particles could end
  up on the far side of a dividing wall without ever crossing it.

## 1.4.2 — 2026-07-14

- Fixed particles sticking to drawn walls, a 1.4.1 regression: wall
  contact was resolved at the motion's closest approach to the wall,
  which for a particle sliding along a wall is the start of its
  substep motion — so every substep reset the slide and froze the
  particle in place. Contact still triggers on end-position overlap or
  on the motion crossing the wall (the tunneling guard), but it now
  always resolves from the end position, preserving tangential motion.

## 1.4.1 — 2026-07-14

- Fixed fast particles tunneling through drawn walls at low frame
  rates. Wall contact is now swept: the test takes the closest approach
  between the particle's motion over the substep and the wall segment,
  instead of only checking the end position, so a step that would carry
  a particle clear across the zero-thickness wall — possible once the
  8-substep cap saturates on large frame deltas — bounces it instead.
- Wall resolution moved after particle-particle resolution within each
  substep, so a pile of particles pressing one against a wall can no
  longer leave it embedded in (or pushed past) the wall at the end of
  a substep.

## 1.4.0 — 2026-07-12

- The demo's control panel is now state-complete: status chips show
  running/paused/stopped, the launch preset, muted, and
  explosion-in-progress; the spawn/color/HUD cycle buttons display
  their current value; toggles render as switches; the clear buttons
  count their targets; and the particle readout shows the population
  cap. Every indicator renders from the per-frame snapshot, which
  gained `stopped`, `exploding`, and `hud` fields.
- The panel's position-taking buttons (burst, comet, pin well, pin
  repeller, explode) arm a one-shot placement tool — click the button,
  then click the canvas to place the action there. Previously they
  acted at the arena center. Esc or a second press cancels.

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
