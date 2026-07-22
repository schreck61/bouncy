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
- **1.5** put the control panel in the native window: a hand-rolled
  frame-buffer GUI (no toolkit — the egui version matrix vetoed itself)
  with web-panel parity, detented magnified sliders, one-shot placement
  tools, an edge-reveal handle, a launch section with in-place
  relaunch, and the touched-settings relaunch semantics shared with
  the web demo (a changed preset takes precedence; adjustments travel
  otherwise). Works on both render backends.
- **1.6** rang the first rung of the emergent instrument: wall
  chimes (walls play length-pitched pentatonic notes on impact, with
  hit flashes), and **1.7** shipped the second — four built-in
  instrument scenes (percussion, marimba, pachinko, harp) on new
  scene-carrying built-in presets, plus silent walls in scene files.
- **1.8–1.10** made the instrument playable live: a dedicated
  ping-volume control, emitters (the free-running sequencer clock,
  demoed by the clockwork polyrhythm preset), and the inspector —
  select an emitter or wall stroke on either shell (panel Select tool
  or hold-D-and-click) to retune its rate, cap, and aim or cycle its
  chime note, or delete it alone instead of clearing everything.
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
- **The emergent instrument (staged).** From user feedback: walls that
  play notes turn the sim into an Otomata-style generative sequencer,
  and the primitives already exist (collision-driven synth, pentatonic
  music mode, per-segment walls, stable particle ids, deterministic
  core). Staged so each rung is independently shippable and the first
  needs no external gear — most people don't have a DAW, so the
  instrument must sound good standing alone:
  1. *Emitters.* (Shipped free-running in 1.9.0; the optional
     quantize snap below remains.) A pinned spawner with direction, rate, and count cap
     (the burst/comet machinery is most of it): the "sequencer clock"
     that makes rhythms reproducible instead of one-shot. An optional
     in-app quantize snap serves the no-DAW majority; purists get
     rubato by leaving it off.
  2. *MIDI out.* Native via `midir`, browser via WebMIDI (Chrome,
     matching the demo's existing Chrome-first stance); per-wall note
     mapping becomes per-wall MIDI note/channel. Strict timing stays
     the DAW's job. A capture/export path (MIDI file or WAV of the
     internal synth) keeps creations for everyone else.
  3. *Filter walls.* Semipermeable walls (every-Nth-particle gates,
     key/mode filters on note-carrying particles) — the full
     generative toolkit. Permeability must be an explicit wall *type*:
     solid walls keep the absolute containment guarantee the 1.4.x
     releases established, and every wall-invariant code path (pair
     filter, spawn side-checks, matter containment) branches on type.
     The divided-arena audit harness re-verifies each new type.

## Smaller / opportunistic

- Demo legibility pass (from user feedback): a short "what am I
  looking at" blurb on the demo page for drive-by visitors (the tour
  PDF is deliberately not public, so the demo currently ships with no
  explanation), plus tooltips on the panel's mechanic toggles that say
  what to *expect* — one line each beats a manual nobody opens.
- Masked-mechanic hints (from user feedback): flow toggled at 100%
  gravity reads as "everything sinks" — the current is real but
  gravity swamps it (`peace` pairs flow with weightlessness for a
  reason). When a toggled mechanic is masked by the current settings,
  say so in the HUD/panel and name the setting to change. Audit the
  other combos for the same trap (self-gravity under full gravity,
  kaleidoscope with trails off).
- HUD/status contrast: the on-screen instructions are easy to miss;
  raise text contrast (or add a subtle backing strip) so first-run
  guidance actually lands.
- Held-tool assists (from user feedback): while `G` is held, enable
  trails presentationally and restore the prior state on release, so
  the well's effect on trajectories is visible without hunting for
  `T`. Same pattern as bullet time: choreography, never physics. Keep
  it to one or two hand-picked assists rather than a general combo
  system — hidden state changes confuse more than they help.

## Tuning debts / watch items

- The `blob` preset can still cool to a stop over very long runs (elastic
  losses via fusion). Mitigations now exist in-app (`F` flow stir or `A`
  self-gravity); accept STOPPED otherwise.
- The demo's multi-threaded bundle needs nightly + `-Zbuild-std`; a
  nightly breakage only degrades the deploy to single-threaded (CI
  builds the bundles in separate steps), but keep an eye on it.
