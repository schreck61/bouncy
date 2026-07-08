# Roadmap

Feature ideas agreed on but not yet built, in priority order (fun-per-effort).
Interface conventions to preserve: new mechanics get a CLI flag + a single-letter
hotkey + a HUD stats line; presets bundle settings but explicit flags always win;
every mechanic lands with unit tests in the headless `Simulation` core (src/sim.rs);
every bug fix ships with a regression test.

All numbered items have shipped; what remains is the opportunistic pile.

## Smaller / opportunistic

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

- Explosion threshold semantics under `--matter` (fusion/fission don't count
  toward the spawn window; only mid-band spawns do). Fine so far; revisit if
  matter+spawning feels off.
- The `blob` preset can still cool to a stop over very long runs (elastic
  losses via fusion); if it bothers, add a tiny ambient stir or accept STOPPED.
