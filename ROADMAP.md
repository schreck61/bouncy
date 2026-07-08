# Roadmap

Feature ideas agreed on but not yet built, in priority order (fun-per-effort).
Interface conventions to preserve: new mechanics get a CLI flag + a single-letter
hotkey + a HUD stats line; presets bundle settings but explicit flags always win;
every mechanic lands with unit tests in the headless `Simulation` core (src/sim.rs);
every bug fix ships with a regression test.

All numbered items have shipped; what remains is the opportunistic pile.

## Smaller / opportunistic

- Screenshot key: dump the RGBA frame as PNG (needs a `png` dependency, or
  write a PPM with zero deps). Musical mode took `S`, so pick another key
  (e.g. `O`).
- Comets: occasional fast heavy particle streaking through (matter mode makes
  it shatter things) - could be a timed ambient event or a middle-click.
- Particle springs/links: click two particles to bind with a spring; molecule
  building. Medium effort, niche payoff. (Stable particle ids landed in the
  2026-07 review work, so the identity problem is already solved.)
- Self-gravity far-field tier: the pairwise force pass is O(n²), intended
  for preset-scale populations. For thousands of self-gravitating
  particles, accumulate per-cell mass and center of mass over the existing
  spatial grid and treat far cells as point masses (one-level Barnes-Hut).

## Tuning debts / watch items

- Fractional display scales (Windows 125%/150%) letterbox the GPU render:
  the pixels crate scales the frame by an integer factor and centers it, so
  the simulation doesn't fill the window (invisible black bars on a black
  background). Cursor mapping now accounts for it (window_pos_to_sim), but
  the bars remain. Fixes if it bothers anyone: pick the buffer size so the
  integer scale fills exactly (physical / round(scale)), or a custom
  fill-scaling renderer.

- Explosion threshold semantics under `--matter` (fusion/fission don't count
  toward the spawn window; only mid-band spawns do). Fine so far; revisit if
  matter+spawning feels off.
- The `blob` preset can still cool to a stop over very long runs (elastic
  losses via fusion); if it bothers, add a tiny ambient stir or accept STOPPED.
