# Roadmap

Feature ideas agreed on but not yet built, in priority order (fun-per-effort).
Interface conventions to preserve: new mechanics get a CLI flag + a single-letter
hotkey + a HUD stats line; presets bundle settings but explicit flags always win;
every mechanic lands with unit tests in the headless `Simulation` core (src/sim.rs);
every bug fix ships with a regression test.

## 1. Musical mode

Quantize ping frequencies to a pentatonic scale (energy picks the scale degree
across ~2 octaves) instead of the continuous 300-1500 Hz map. Flag `--music`,
hotkey maybe `S` (sound mode), HUD flag. Audio-only change in audio.rs
(`ping_frequency(energy) -> f32` becomes mode-aware); the pitch-bucket
pre-generation needs per-note buffers instead of 16 linear buckets.

## 2. Kaleidoscope rendering

Mirror the frame 4- or 6-fold around the screen center as a post-process in
render.rs (pixel-buffer remap after drawing, before HUD). Flag `--kaleidoscope`,
hotkey `K`, HUD flag. Mesmerizing with trails + fireworks.

## 3. Drawable walls

Hold a key (e.g. `V`) and drag to paint static line-segment barriers particles
bounce off; another key erases all. Biggest lift: segment-vs-circle collision in
physics.rs (reflect velocity about segment normal, position correction), a
`Vec<Segment>` in Sim, rendering, and spawn-clearance awareness. Turns the sim
into pachinko/marble-run construction. Consider max segment count and HUD count.

## Smaller / opportunistic

- Screenshot key (`S` if music takes another key): dump the RGBA frame as PNG
  (needs a `png` dependency, or write a PPM with zero deps).
- Comets: occasional fast heavy particle streaking through (matter mode makes
  it shatter things) - could be a timed ambient event or a middle-click.
- Particle springs/links: click two particles to bind with a spring; molecule
  building. Medium effort, niche payoff.
- More presets once #2 exists: `mandala` (kaleidoscope+fireworks).

## Tuning debts / watch items

- Explosion threshold semantics under `--matter` (fusion/fission don't count
  toward the spawn window; only mid-band spawns do). Fine so far; revisit if
  matter+spawning feels off.
- Velocity color mode normalizes hue against the INITIAL_VELOCITY constant
  (600), not `--initial-speed`; slow presets render mostly blue. Cosmetic.
- The `blob` preset can still cool to a stop over very long runs (elastic
  losses via fusion); if it bothers, add a tiny ambient stir or accept STOPPED.
