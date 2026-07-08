# Roadmap

Feature ideas agreed on but not yet built, in priority order (fun-per-effort).
Interface conventions to preserve: new mechanics get a CLI flag + a single-letter
hotkey + a HUD stats line; presets bundle settings but explicit flags always win;
every mechanic lands with unit tests in the headless `Simulation` core (src/sim.rs);
every bug fix ships with a regression test.

## 1.0 — shipped

The 1.0 release completed the plan: fill the window at fractional display
scales; velocity colors that respect `--initial-speed`; a screenshot key
(`O`); **scenes** — presets that carry wells/walls geometry in
window-fraction coordinates, with the `E` key exporting the current
settings and construction as a shareable preset; and comets on middle
click. With that, the original numbered backlog and both display debts
are fully cleared.

## 1.1 headline: GUI overlay

An egui panel inside the existing window (Tab to toggle, hidden by
default): playback controls, sliders for the runtime parameters, preset
apply, and eventually preset *saving* through the same UI. The
groundwork is already laid — every runtime mutation goes through the
`Command` dispatch in app.rs, so widgets issue the same commands as
hotkeys; `pixels` exposes `render_with()` for egui's render pass. The
CPU/softbuffer backend stays hotkeys-only (documented limitation).
Intended shape: a collapsible side panel via egui-wgpu + egui-winit,
with a staged-restart section for creation-time parameters as a later
phase.

## Smaller / opportunistic

- Particle springs/links: click two particles to bind with a spring; molecule
  building. Medium effort, niche payoff. (Particles carry stable ids, so
  the cross-frame identity problem is already solved.)
- Self-gravity far-field tier: the pairwise force pass is O(n²), intended
  for preset-scale populations. For thousands of self-gravitating
  particles, accumulate per-cell mass and center of mass over the existing
  spatial grid and treat far cells as point masses (one-level Barnes-Hut).
- Screensaver/attract mode: cycle through presets on a timer
  (`--cycle <secs>`); the sim is a screensaver at heart.

## Tuning debts / watch items

- Explosion threshold semantics under `--matter` (fusion/fission don't count
  toward the spawn window; only mid-band spawns do). Fine so far; revisit if
  matter+spawning feels off.
- The `blob` preset can still cool to a stop over very long runs (elastic
  losses via fusion). Mitigations now exist in-app (`F` flow stir or `A`
  self-gravity); accept STOPPED otherwise.
