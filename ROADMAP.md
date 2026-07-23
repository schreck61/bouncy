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
- **1.11** locked it to a pulse: quantize (`--bpm`, `--beat-div`, the
  `L` toggle) snaps due emissions to a beat grid on the simulation
  clock — pause holds the beat, time scale bends the tempo — with a
  Quantize slider and beat-grid button on both panels, and tempo
  riding scenes, share links, and relaunches.
- **1.12** put the instrument on the wire: native MIDI out
  (`--midi-port`) sends wall-chime strikes to a DAW or hardware synth
  — pentatonic degrees as keys from middle C, velocity from impact,
  guaranteed note-offs — independent of the local mute, with the `Y`
  toggle and a panel row while connected.
- **1.13** completed the instrument roadmap with filter walls:
  semipermeable strokes drawn dashed — gates (every Nth striker
  passes silently, blocked strikes chime: an audible escapement) and
  pass-note walls routing particles stamped by noted emitters —
  configured from either inspector or scene files (`gate`/`pass-note`
  wall keys, emitter `note`), with only the bounce branch consulting
  the filter, so spawns, bursts, and matter still treat every wall as
  solid and the divided-arena audit proves a gate leaks exactly its
  grants.
- **1.14** cleared the deferred MIDI backlog: per-wall MIDI
  note/channel mapping (`midi-note`/`midi-channel` scene keys +
  inspector sliders on both shells; the scheduler went channel-aware
  with the default mapping byte-identical to 1.12), self-contained
  capture (`Z`/`--capture` writes a standard `.mid` of the resolved
  stream and a `.wav` bounce of the internal synth — the no-DAW
  path), and browser WebMIDI (Chromium; the demo's Enable MIDI
  button drives the first output port through the same scheduler,
  degrading to a hidden button or a banner everywhere else).
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
- **The emergent instrument: complete.** The staged program that grew
  out of user feedback (walls that play notes turn the sim into an
  Otomata-style generative sequencer) shipped in full across
  1.6–1.14 — chimes, scenes, emitters, the inspector, quantize, MIDI
  out, filter walls, and finally the deferred MIDI rungs (per-wall
  mapping, capture, WebMIDI). Nothing from the program remains ahead;
  possible far-future extensions (MIDI *in*, WAV capture of pings and
  rumble alongside chimes, browser capture downloads) have not been
  scoped and belong to user feedback.

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
