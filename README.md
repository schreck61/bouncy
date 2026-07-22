# Bouncy

A GPU-accelerated particle simulation written in Rust featuring elastic collisions, gravity, dynamic particle spawning, and explosive chain reactions with synthesized audio feedback.

**[Try it in your browser](https://schreck61.github.io/bouncy/demo/)** — the same simulation compiled to WebAssembly, with a control panel for every parameter (multi-threaded where the browser allows; see [web/README.md](web/README.md)). The demo is currently optimized for Chrome; frame rates in other browsers can be substantially lower.

## Features

- **GPU-Accelerated Rendering**: Uses the `pixels` crate with wgpu backend for smooth, hardware-accelerated 2D rendering, with automatic CPU fallback via `softbuffer` when GPU is unavailable
- **Realistic Physics**: Particle-particle collisions with a configurable coefficient of restitution, wall bounces, configurable gravity, adaptive substepping so fast particles never tunnel through each other, and swept wall contact so they never tunnel through drawn walls
- **Fast Collision Detection**: A uniform spatial grid keeps collision detection near-linear, comfortably handling thousands of particles
- **Control Panel**: `Tab` (or a dwell at the window's right edge) slides in a translucent panel — readouts, detented sliders, toggles, one-shot placement tools, and launch options with in-place relaunch — matching the browser demo's controls and driving the same command dispatch as the hotkeys
- **Dynamic Spawning**: New particles spawn on collision, creating organic growth patterns
- **Explosion Mechanics**: When spawn rate exceeds threshold, a dramatic explosion kills 99% of particles
- **Interactive**: Pause, reset, spawn particle bursts with the mouse, trigger explosions with right click, and adjust gravity and elasticity live with the arrow keys
- **HUD Overlay**: Optional on-screen display (H key) with FPS, particle count, and current physics settings
- **Visual Effects**: Optional motion trails (`--trails`), velocity-based coloring (`--color-mode velocity`), and configurable particle size
- **Synthesized Audio**: Real-time audio synthesis for collision pings (pitch based on impact energy, stereo-panned to the collision position) and explosion rumbles; runs silently when no audio device is available
- **Reproducible Runs**: `--seed` fixes the random number generator for repeatable simulations
- **Flexible Display Modes**: Runs in borderless fullscreen by default, or use `--width` and `--height` for fixed-size windowed mode
- **Adaptive Particle Count**: Initial particle count scales based on screen resolution

## Demo

Particles bounce around the screen with realistic physics. Each collision spawns a new particle and plays a ping sound with pitch corresponding to collision energy. When the spawn rate exceeds 30 particles per second, an expanding ring explosion eliminates 99% of particles, resetting the simulation with a deep rumble sound.

## Installation

### Prerequisites

- Rust 1.85 or later
- A GPU with Vulkan, Metal, or DX12 support (optional — CPU rendering is used as a fallback)
- Linux only: ALSA and udev development headers (`libasound2-dev` and `libudev-dev` on Debian/Ubuntu)

### Building

```bash
git clone https://github.com/schreck61/bouncy.git
cd bouncy
cargo build --release
```

### Running

```bash
cargo run --release
```

Or with the collision-point spawning mode:

```bash
cargo run --release -- --spawn-at-collision
```

## Usage

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--preset <NAME>` | Apply a settings bundle: a built-in (`fireworks`, `blob`, `billiards`, `peace`, `orbits`, `mandala`, `accretion`, `percussion`, `marimba`, `pachinko`, `harp`, `clockwork`) or a preset from the user presets file; explicit options override the preset | None |
| `--presets-file <PATH>` | Load user presets from this TOML file instead of the platform default location | Platform config dir |
| `--list-presets` | List built-in and user presets (and where the user file was loaded from), then exit | |
| `--spawn-mode <MODE>` | Where collision spawns appear: `center`, `collision`, or `off` | center |
| `--spawn-at-collision` | Alias for `--spawn-mode collision` (kept for compatibility) | Off |
| `--matter` | Enable matter mechanics: slow contacts fuse particles, hard impacts split them | Off |
| `--flow` | Enable the ambient flow field (drifting wind currents) | Off |
| `--self-gravity` | Give every particle gravity proportional to its mass, so clumps attract and accrete | Off |
| `--wells <N>` | Pin N attracting gravity wells around the screen center at startup (0-16) | 0 |
| `--min-particles <N>` | Override the starting/minimum particle count (2-100) | Screen-based |
| `--gravity <PERCENT>` | Set gravity as percentage of standard (-1000 to 1000); negative values cause upward gravity | 100 |
| `--wall-elasticity <VALUE>` | Set wall bounce elasticity (0.0-1.5); 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy | 1.0 |
| `--particle-elasticity <VALUE>` | Set particle collision restitution (0.0-1.5); 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy | 1.0 |
| `--width <N>` | Set window width in pixels (100-7680); must be used with `--height` | Fullscreen |
| `--height <N>` | Set window height in pixels (100-4320); must be used with `--width` | Fullscreen |
| `--cpu` | Force CPU rendering (softbuffer) instead of GPU | Off |
| `--mute` | Start with audio muted | Off |
| `--music` | Quantize collision pings to a pentatonic scale (energy picks the note) | Off |
| `--wall-chimes` | Walls play pentatonic notes when particles strike them — longer wall, lower note | Off |
| `--ping-volume <PERCENT>` | Particle collision ping volume, 0-100 (0 silences pings; wall chimes and the explosion rumble are unaffected) | 100 |
| `--chime-timbre <VOICE>` | The voice wall chimes play with: `chime`, `marimba`, `pluck`, `drum`, or `bell` (the instrument presets each pick their own; particle pings are unaffected) | chime |
| `--kaleidoscope` | Mirror the frame 4-fold around the screen center | Off |
| `--trails` | Leave motion trails behind particles | Off |
| `--panel` | Start with the native control panel open (Tab toggles; the right-edge handle also reveals it) | Off |
| `--particle-size <R>` | Particle radius in pixels (0.5-10.0) | 1.5 |
| `--initial-speed <V>` | Top speed of newly created particles in px/sec (10-2000); they start at 50-100% of it | 600 |
| `--color-mode <MODE>` | `solid` or `velocity` (hue follows speed) | solid |
| `--explosion-threshold <N>` | Particle births per second — collision spawns plus fission fragments — that trigger an automatic explosion (0-1000); a population pegged at the density cap for a few seconds also explodes; 0 disables automatic explosions (population is then capped at ~20% window coverage or 12,000 particles, whichever is lower) | 30 |
| `--bullet-time` | Slow time briefly (bullet time) whenever an explosion ring starts | Off |
| `--seed <N>` | Seed the random number generator (reproducible starting conditions) | Random |
| `--verbose` | Print per-second FPS statistics to stdout | Off |
| `--help`, `-h` | Display help information | |
| `--version`, `-V` | Display version | |

### Controls

| Input | Action |
|-------|--------|
| `Space`, `Escape`, `Q` | Exit |
| `P` | Pause / resume |
| `N` | Advance one frame while paused |
| `R` | Reset the simulation |
| `M` | Mute / unmute audio |
| `H` | Cycle the HUD overlay: off / stats / stats + key reference |
| `Up` / `Down` | Adjust gravity by 10% |
| `Left` / `Right` | Adjust particle elasticity by 0.05 |
| `[` / `]` | Adjust wall elasticity by 0.05 |
| `,` / `.` | Slow down / speed up time in 0.05 steps (0.1x to 4x) |
| `-` / `=` | Adjust the explosion threshold by 5 spawns/sec (0 = automatic explosions off) |
| `T` | Toggle motion trails |
| `Tab` | Toggle the control panel (sliders, toggles, actions) |
| `C` | Cycle color mode (solid / velocity) |
| `B` | Cycle spawn mode (center / collision points / off) |
| `X` | Toggle matter mechanics (fusion/fission) |
| `F` | Toggle the flow field |
| `A` | Toggle self-gravity (mass attracts mass) |
| `S` | Toggle musical pings (pentatonic scale) |
| `;` / `'` | Adjust particle-ping volume by 10% |
| `I` | Toggle wall chimes (walls play notes on impact) |
| `K` | Toggle kaleidoscope rendering |
| `G` (hold) | Gravity well: attract particles toward the cursor; `Shift+G` repels |
| `W` | Pin a persistent gravity well at the cursor; `Shift+W` pins a repeller |
| `Shift+R` | Clear all pinned wells |
| `V` (hold + drag) | Draw wall segments that particles bounce off |
| `Shift+V` | Clear all drawn walls |
| `U` (hold + drag) | Place an emitter; the drag aims it (a tap aims at the center) |
| `Shift+U` | Clear all emitters |
| `O` | Save a screenshot (PNG in the working directory) |
| `E` | Export settings and scene (wells/walls) as a preset file |
| `J` | Launch a comet from the far edge toward the cursor (same as middle click) |
| Left click | Spawn a burst of particles at the cursor |
| Middle click | Launch a comet from the far edge toward the cursor (heavy and fast — under matter mode it shatters what it hits) |
| Right click | Trigger an explosion centered at the cursor (kills every particle the ring reaches beyond the configured minimum — the base particle count always survives) |

Wall elasticity is the simulation's temperature dial: below 1.0 the walls drain energy on every bounce and the system gradually cools; above 1.0 they pump energy in — speeds then climb until they saturate at a terminal velocity (20,000 px/s) rather than growing without bound.

The mouse cursor hides after 2 seconds of inactivity so it doesn't distract from the simulation. It reappears when moved, stays visible while the gravity well is held, and is always restored when the window loses focus or the program exits.

### Motion Detection

When all particles stop moving (velocity below threshold for ~1 second), a "STOPPED" message is displayed and the simulation halts. This can happen when using low elasticity values where particles eventually lose all energy. Left-click to spawn fresh particles, or press `R` to reset.

## How It Works

### Physics Simulation

The simulation uses a simple but effective physics model:

- **Gravity**: Constant downward (or upward) acceleration applied to all particles
- **Collisions**: Particles exchange momentum along the collision normal using the reduced-mass impulse `j = (1 + e) · dvn · m₁m₂/(m₁ + m₂)`, with mass proportional to area — heavy particles plow through light ones, and unequal-size contacts behave correctly. `e = 1.0` is fully elastic; `e = 0.0` leaves both moving together
- **Wall Bouncing**: Particles reflect off screen boundaries scaled by the wall elasticity
- **Substepping**: Each frame is split into up to 8 physics substeps targeting at most one (smallest) radius of travel per step, keeping particle-particle contacts resolved
- **Swept Wall Contact**: Drawn walls are tested against each particle's entire motion within a substep, not just its end position, so even when the substep cap saturates (a fast particle on a slow frame) nothing can cross a wall — and wall resolution runs after particle-particle resolution, so a crowd pressing a particle wall-ward can't shove it through either

### Matter Mechanics (`--matter` / `X`)

When enabled, collision energy decides each contact's outcome:

- **Fusion** (slow contact): the two particles merge into one, conserving area, momentum, and blending color by mass. Fused giants grow up to 6x the base radius; a blob at the cap absorbs only what fits (partial fusion) and the donor's remainder survives
- **Fission** (hard impact): each participant shatters into two half-area fragments that recede perpendicular to the impact, down to a minimum of half the base radius
- **In between**: an ordinary bounce (and a spawn, if spawning is on)

With spawning off (`--preset blob` uses this), population and size distribution become emergent: slow regions coarsen into heavy blobs, violent regions shatter them back into dust.

### Gravity Wells (`G` / `W`)

Holding `G` creates a temporary gravity well at the cursor (`Shift+G` repels). Pressing `W` pins a persistent well at the cursor instead — `Shift+W` pins a repeller, and `Shift+R` clears them all (up to 16 pinned wells; `R` restores any `--wells` startup layout). Both use a softened (Plummer) force profile — the pull peaks near the softening radius and falls off as 1/d² — with pinned wells at half the held well's strength. Multiple pinned wells make binary systems, slingshots, and, with trails on, orbit painting. Attractors are marked with a cyan ring, repellers with an orange one.

### Self-Gravity (`--self-gravity` / `A`)

Every particle attracts every other with a force proportional to both masses (mass is area, so fused blobs pull harder), softened at close range like the cursor well so near-misses swing by instead of slingshotting. A single pair barely drifts together; a clustered population collapses in seconds — collective attraction is the point.

The magic ingredient is dissipation: perfectly elastic particles fall in and slingshot out forever, but with sub-elastic collisions (and especially matter mode) the energy bleeds off and dust accretes into planetesimals — the `accretion` preset bundles exactly that. Small populations use an exact pairwise force pass (symmetric, so momentum is conserved to the bit); larger ones switch to an adaptive Barnes-Hut quadtree whose force pass fans out across CPU cores, keeping thousands of self-gravitating particles — even collapsed into one dense clump — comfortably real-time.

### The Flow Field (`--flow` / `F`)

A slowly drifting field of currents that particles are *entrained into*: each particle is dragged toward the local current's velocity rather than being pushed by a force, so speeds stay bounded at the current's speed (with gentle gusts) instead of accumulating. Best appreciated with trails enabled or the `peace` preset.

### Drawable Walls (`V`)

Hold `V` and drag to paint static walls: the cursor path becomes a polyline of segments (up to 200) that particles bounce off under the same elasticity rule as the arena walls. Wall contact is swept over each particle's motion and resolved on both sides of particle-particle resolution, so neither raw speed, a low frame rate, nor a crowd pressing from behind can force a particle through a wall — and no collision is ever detected, or spawn site recorded, from a position on the wrong side of one. Walls are full interaction boundaries, not just obstacles: particles never exchange impulses or fuse through a wall, and everything born near one — collision spawns, burst particles, fission fragments — appears on the same side as whatever created it. A full-height wall genuinely divides the arena into independent halves. `Shift+V` erases all walls; `R` (reset) clears them too. Collision-triggered spawns refuse positions inside a wall, so nothing materializes embedded in one. Combine with spawning and gravity for pachinko boards, funnels, and marble runs — this is a purely interactive tool, so it has no CLI flag.

### Kaleidoscope (`--kaleidoscope` / `K`)

A rendering post-process that mirrors the top-left quadrant of the frame 4-fold around the screen center — physics is untouched; only the presented image is symmetric. The HUD and status text draw after the mirror, so they stay readable. Hypnotic with trails and collision sprays; the `mandala` preset bundles exactly that.

### Presets

`--preset` bundles curated settings; any explicit option overrides the preset's value for it:

| Preset | Character |
|--------|-----------|
| `fireworks` | Low gravity, collision sprays, trails, velocity colors, frequent slow-motion explosions |
| `blob` | Slow heavy blobs that merge and drift; matter mechanics, no explosions |
| `billiards` | A fixed rack of large elastic balls; pure collision physics |
| `peace` | Many tiny particles drifting on the flow field with soft walls |
| `orbits` | Weightless particles slung around a binary system of pinned wells; trails paint the orbits |
| `mandala` | The fireworks recipe under a kaleidoscope, minus gravity: symmetric blooms of trails |
| `accretion` | Self-gravitating dust with dissipative collisions: clumps form, fuse, and sweep their orbits clean |
| `percussion` | An instrument scene: silent channels with drumming end caps — weightless particles thump a stereo four-note tom pattern |
| `marimba` | An instrument scene: a stair of eleven pitched bars plays descending pentatonic runs and floor-bounced ascents |
| `pachinko` | An instrument scene: cascades plink down a staggered peg field, pitch falling with depth |
| `harp` | An instrument scene: orbiting particles grace a fan of six pitched strings with slow rolled arpeggios |
| `clockwork` | An instrument scene: three emitter lanes strike pitched bars at different periods — a polyrhythmic mechanism |

### Custom Presets

You can define your own presets in a TOML file. Each top-level table is a preset; its keys are the command-line option names from `--help` (kebab-case; underscores also accepted), and an optional `base` key names a built-in preset to inherit from:

```toml
[pachinko]
description = "Big slow balls under heavy gravity"
base = "billiards"
gravity = 80
particle-size = 4.0

[quiet-fireworks]
description = "Fireworks without the noise, through a kaleidoscope"
base = "fireworks"
mute = true
kaleidoscope = true
```

The optional `description` is shown by `--list-presets`, just like the built-in preset blurbs.

Presets can also carry **scene geometry** — pinned wells and wall segments — in window-fraction coordinates (0.0-1.0 of the window width/height), so a scene lays out identically on any screen:

```toml
[plinko]
description = "Musical peg board"
music = true
gravity = 120
spawn-mode = "off"
walls = [[0.2, 0.3, 0.3, 0.35], [0.45, 0.3, 0.55, 0.35], [0.7, 0.3, 0.8, 0.35]]
wells = [{ x = 0.5, y = 0.9, polarity = "attract" }]
```

(An integer `wells = 3` is still the `--wells` ring count; an array is geometry. Well `polarity` is `"attract"` or `"repel"`, defaulting to attract.)

When wall chimes are on, a wall's pitch normally follows its length, but a scene can pin it or mute it: write the wall as a table with a `note` (pentatonic degree `0`-`10`, low to high) or with `silent = true` for pure geometry that bounces without ever chiming or flashing — the percussion preset's guide rails, for example:

```toml
[drumkit]
wall-chimes = true
walls = [
  { x1 = 0.15, y1 = 0.5, x2 = 0.85, y2 = 0.5, silent = true },
  { x1 = 0.15, y1 = 0.3, x2 = 0.15, y2 = 0.5, note = 3 },
]
```

A wall cannot be both `silent` and note-pinned, and (like booleans on the command line) `silent = false` is rejected rather than ignored. A user preset with a `base` and no `walls`/`wells` of its own inherits the base's scene geometry; defining any geometry replaces the base's scene entirely.

The easiest way to author a scene is in the app itself: pin wells with `W`, draw walls with `V`, tune everything with the hotkeys, then press `E` to export the whole thing — settings and geometry — as a ready-made preset file. Copy the table into your presets file (rename it to taste) and it becomes a `--preset` you can share.

The file is looked up in these locations (first match wins), or wherever `--presets-file` points. The XDG-style `~/.config` path is checked first on every platform, since that's where command-line users expect it; `$XDG_CONFIG_HOME` overrides `~/.config` when set:

| Platform | Locations checked, in order |
|----------|-----------------------------|
| Linux | `~/.config/bouncy/presets.toml` |
| macOS | `~/.config/bouncy/presets.toml`, then `~/Library/Application Support/bouncy/presets.toml` |
| Windows | `~\.config\bouncy\presets.toml`, then `%APPDATA%\bouncy\presets.toml` |

Run `--list-presets` to see every built-in and user preset along with the file they were loaded from — the quickest way to check that your file is being picked up.

Preset values are validated by the same parser as the command line (same ranges, same error messages), and precedence is: explicit command-line flag > user preset value > `base` preset value > default. A few rules keep things predictable: user presets cannot share a name with a built-in (use `base` to build on one), `base` must name a built-in (no chaining user presets), and boolean options can only be enabled — like the command line itself, there is no way to switch one off. A missing default file is silently fine; a malformed file or invalid value is a loud error rather than a silently ignored preset.

### Collision Detection

A uniform spatial grid (rebuilt each substep with zero steady-state allocations) bins particles into cells sized from the *median* particle radius, so a few huge fusion-made blobs among thousands of small particles can't inflate every cell: particles too large for their cell go on a separate oversized list that is swept directly, and if mid-size particles flood that list the cells grow just enough to absorb them. Only particles in the same or adjacent cells are tested pairwise, making collision detection effectively linear in the particle count; contact detection runs across CPU cores (read-only, deterministic for any thread count) while the short list of actual contacts resolves serially.

### Particle Spawning

Each collision between particles spawns a new particle. By default, new particles appear near the screen center with a random velocity. With the `--spawn-at-collision` flag, they spawn beside the collision point — placed perpendicular to the collision axis so the new particle never materializes inside the particles that spawned it — and are ejected outward, away from the collision, within a 45° cone. The ejection speed scales with the collision energy: hard impacts throw off fast fragments, grazing contacts release slow debris. Spawns are only skipped when the surrounding space is genuinely occupied by other particles (dense clusters), which keeps the spawn rate tied to real collisions.

### Explosion Trigger

The simulation tracks the particle birth rate — collision spawns plus matter-mode fission fragments — over a 1-second sliding window. When this rate exceeds the threshold (default 30 births per second), an explosion is triggered. A second, independent valve covers the case the birth rate cannot see: a population pegged at the density cap stops producing births entirely (spawning halts at the cap), so once it has sat there continuously for a few seconds, the explosion fires anyway — maximum congestion ends in a bang rather than gridlock. Either way:

1. An expanding ring emanates from the screen center (or from the collision hotspot in `--spawn-at-collision` mode, or from the cursor on right click)
2. 99% of particles are marked for elimination
3. Particles are killed when the ring reaches them
4. A minimum number of particles survive: the base count from screen size, or `--min-particles` if set

With `--bullet-time`, every explosion ring (automatic or right-click) also triggers a moment of bullet time: the simulation runs at 0.1x the current time scale for the first second (wall-clock), then ramps back up over 0.4s. It's pure presentation — physics is unchanged, just stepped with a smaller dt. The `fireworks` and `mandala` presets enable it.

### Adaptive Particle Count

The initial and minimum particle count scales with logical screen resolution (physical pixels divided by scale factor):

```
particle_count = (logical_width × logical_height) / PIXELS_PER_PARTICLE
```

With `PIXELS_PER_PARTICLE = 375,000`:

| Display | Logical Resolution | Total Pixels | Particles |
|---------|-------------------|--------------|-----------|
| MacBook Pro 14" | 1512×982 | 1,484,784 | 4 |
| MacBook Pro 16" | 1728×1117 | 1,930,176 | 5 |
| 1080p (scale 1.0) | 1920×1080 | 2,073,600 | 6 |
| 1440p (scale 1.0) | 2560×1440 | 3,686,400 | 10 |
| 4K (scale 1.0) | 3840×2160 | 8,294,400 | 22 |

This ensures the simulation feels consistent across different display sizes. Retina/HiDPI displays use their logical resolution, not the physical pixel count. At fractional display scales (Windows 125%/150%) the simulation size is the physical size divided by the *rounded* scale, so the frame always fills the window exactly instead of letterboxing.

Use `--min-particles <N>` to override this with a fixed count between 2 and 100.

### Audio Synthesis

All sounds are generated programmatically:

- **Collision Pings**: Sine wave with exponential decay, frequency mapped to collision energy (300-1500 Hz). Pings are pre-generated into pitch buckets at startup (no per-collision allocation) and stereo-panned to match the on-screen collision position
- **Musical Mode** (`--music` / `S`): pings snap to a major-pentatonic scale instead — collision energy picks the degree across two octaves up from C4 — turning collision showers into wind-chime melodies. Per-note buffers are pre-generated alongside the linear buckets, so toggling is instant
- **Explosion Rumble**: Low-frequency oscillators (40-80 Hz) mixed with noise, shaped with attack/decay envelope

If no audio output device is available (e.g. headless machines), the simulation runs silently instead of failing.

## Configuration

Key constants can be modified in the source:

```rust
// src/physics.rs
const GRAVITY: f64 = 100.0;
const MAX_SPEED: f64 = 20_000.0;             // Terminal velocity (px/s)
const MOTION_VELOCITY_THRESHOLD: f64 = 1.0;  // Minimum velocity to be "moving"
const MOTION_STOPPED_FRAMES: u32 = 60;       // ~1 second at 60fps

// src/explosion.rs
const EXPLOSION_KILL_RATIO: f64 = 0.99;

// src/sim.rs
const PIXELS_PER_PARTICLE: u64 = 375_000;
const CLICK_BURST_SIZE: usize = 10;
const MAX_PARTICLES: usize = 12_000;         // Population ceiling (small sizes)
const SATURATION_EXPLOSION_SECS: f64 = 3.0;  // Time at the cap before the valve fires

// src/audio.rs
const PING_MIN_FREQ: f32 = 300.0;
const PING_MAX_FREQ: f32 = 1500.0;
```

Particle size, initial speed, and the explosion threshold are ordinary
command-line options (`--particle-size`, `--initial-speed`,
`--explosion-threshold`) rather than source constants.

## Dependencies

- [`pixels`](https://crates.io/crates/pixels) - Hardware-accelerated pixel buffer (GPU rendering)
- [`softbuffer`](https://crates.io/crates/softbuffer) - Software pixel buffer (CPU fallback)
- [`winit`](https://crates.io/crates/winit) - Cross-platform window management
- [`rodio`](https://crates.io/crates/rodio) - Audio playback
- [`rand`](https://crates.io/crates/rand) - Random number generation
- [`clap`](https://crates.io/crates/clap) - Command line argument parsing
- [`toml`](https://crates.io/crates/toml) - User presets file parsing
- [`dirs`](https://crates.io/crates/dirs) - Platform config directory discovery
- [`png`](https://crates.io/crates/png) - Screenshot encoding
- [`ouroboros`](https://crates.io/crates/ouroboros) - Safe self-referential struct support
- [`ab_glyph`](https://crates.io/crates/ab_glyph) - Font rendering

## Platform Support

Tested on:
- macOS (Metal backend)

Should work on:
- Windows (DX12/Vulkan backend)
- Linux x86_64 (Vulkan backend)
- Linux ARM64, e.g. Raspberry Pi 4/5 or Asahi Linux (Vulkan backend, with CPU fallback) — prebuilt binaries are provided but untested on real hardware

## Architecture

**API documentation** (rustdoc, including internal items) is published at
[schreck61.github.io/bouncy](https://schreck61.github.io/bouncy/) and rebuilt
on every push to `main`. Generate it locally with:

```bash
cargo doc --no-deps --document-private-items --open
```

The application uses the modern `winit` 0.30 `ApplicationHandler` pattern, split into focused modules:

- `main.rs` - Entry point and event loop setup
- `config.rs` - Command line parsing (clap)
- `presets.rs` - Built-in preset bundles and user-defined presets from a TOML file
- `color.rs` - HSV/RGBA color conversion helpers
- `app.rs` - Application shell: input handling, HUD, audio dispatch, and event loop glue around the simulation core
- `sim.rs` - The headless simulation core: particles, spawning, explosions, and their orchestration. No windowing, rendering, or audio — the `App` layer owns those and drives this struct, which keeps every gameplay rule testable. **Every mechanic lands here, with unit tests.**
- `physics.rs` - Particles, collisions, spatial grid, substepping
- `explosion.rs` - Expanding-ring explosion mechanics
- `render.rs` - `RenderContext` abstraction over GPU (pixels/wgpu) and CPU (softbuffer) backends, plus drawing routines
- `audio.rs` - Synthesized sound with optional output device
- `text.rs` - Bitmap text rendering with the embedded font

Notable implementation details:

- GPU backend uses `ouroboros` for safe self-referential struct (Pixels borrows from Window)
- Automatic GPU-to-CPU fallback when GPU is unavailable; use `--cpu` to force CPU rendering
- The CPU backend scales logical to physical pixels through precomputed nearest-neighbor lookup tables
- Transient render failures (display sleep, mode changes) skip frames instead of crashing
- Physics and rendering run in the main event loop, synchronized to VSync

## Performance

- Targets 120 FPS on modern displays (VSync-locked)
- Spatial-grid collision detection — near-linear in particle count, suitable for thousands of particles
- Adaptive physics substepping only when fast particles require it
- GPU-accelerated rendering via wgpu
- Warmup frames skip physics to allow GPU initialization

## Testing

```bash
cargo test
```

Unit tests cover the collision model (momentum/energy conservation, restitution behavior), the spatial grid (validated against brute-force pair detection), explosion mechanics, argument parsing, text rendering, and audio synthesis.

## License

MIT License

Copyright (c) 2026 James O. Schreckengast

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Compiled binaries (the release downloads and the web demo) statically link
third-party Rust crates under the MIT, Apache-2.0, BSD, ISC, Zlib, MPL-2.0,
and related permissive licenses. Their license texts and copyright notices
are collected in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), which
ships alongside every binary.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for the release history.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute, including our Developer Certificate of Origin (DCO) requirements.

## Acknowledgments

Built with Rust and its excellent ecosystem of graphics and audio crates.

This software embeds the Liberation Sans Bold font (© 2010 Google Corporation, © 2012 Red Hat, Inc.), licensed under the [SIL Open Font License, Version 1.1](https://scripts.sil.org/OFL). The full license text is in [assets/OFL.txt](assets/OFL.txt) and accompanies every binary distribution.
