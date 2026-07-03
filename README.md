# Bouncy

A GPU-accelerated particle simulation written in Rust featuring elastic collisions, gravity, dynamic particle spawning, and explosive chain reactions with synthesized audio feedback.

## Features

- **GPU-Accelerated Rendering**: Uses the `pixels` crate with wgpu backend for smooth, hardware-accelerated 2D rendering, with automatic CPU fallback via `softbuffer` when GPU is unavailable
- **Realistic Physics**: Particle-particle collisions with a configurable coefficient of restitution, wall bounces, configurable gravity, and adaptive substepping so fast particles never tunnel through each other
- **Fast Collision Detection**: A uniform spatial grid keeps collision detection near-linear, comfortably handling thousands of particles
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
- Linux only: ALSA development headers (`libasound2-dev` on Debian/Ubuntu)

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
| `--spawn-at-collision` | Spawn new particles at collision points instead of screen center | Off |
| `--min-particles <N>` | Override the starting/minimum particle count (2-100) | Screen-based |
| `--gravity <PERCENT>` | Set gravity as percentage of standard (-1000 to 1000); negative values cause upward gravity | 100 |
| `--wall-elasticity <VALUE>` | Set wall bounce elasticity (0.0-1.5); 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy | 1.0 |
| `--particle-elasticity <VALUE>` | Set particle collision restitution (0.0-1.5); 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy | 1.0 |
| `--width <N>` | Set window width in pixels (100-7680); must be used with `--height` | Fullscreen |
| `--height <N>` | Set window height in pixels (100-4320); must be used with `--width` | Fullscreen |
| `--cpu` | Force CPU rendering (softbuffer) instead of GPU | Off |
| `--mute` | Start with audio muted | Off |
| `--trails` | Leave motion trails behind particles | Off |
| `--particle-size <R>` | Particle radius in pixels (0.5-10.0) | 1.5 |
| `--color-mode <MODE>` | `solid` or `velocity` (hue follows speed) | solid |
| `--seed <N>` | Seed the random number generator (reproducible starting conditions) | Random |
| `--verbose` | Print per-second FPS statistics to stdout | Off |
| `--help`, `-h` | Display help information | |
| `--version`, `-V` | Display version | |

### Controls

| Input | Action |
|-------|--------|
| `Space`, `Escape`, `Q` | Exit |
| `P` | Pause / resume |
| `R` | Reset the simulation |
| `M` | Mute / unmute audio |
| `H` | Toggle the HUD overlay (FPS, particle count, physics settings) |
| `Up` / `Down` | Adjust gravity by 10% |
| `Left` / `Right` | Adjust particle elasticity by 0.05 |
| Left click | Spawn a burst of particles at the cursor |
| Right click | Trigger an explosion centered at the cursor (kills every particle the ring reaches, down to a minimum of 2 survivors) |

### Motion Detection

When all particles stop moving (velocity below threshold for ~1 second), a "STOPPED" message is displayed and the simulation halts. This can happen when using low elasticity values where particles eventually lose all energy. Left-click to spawn fresh particles, or press `R` to reset.

## How It Works

### Physics Simulation

The simulation uses a simple but effective physics model:

- **Gravity**: Constant downward (or upward) acceleration applied to all particles
- **Collisions**: Equal-mass particles exchange momentum along the collision normal using a coefficient of restitution: the per-particle impulse is `dvn * (1 + e) / 2`, so `e = 1.0` swaps normal velocities and `e = 0.0` leaves both particles moving together
- **Wall Bouncing**: Particles reflect off screen boundaries scaled by the wall elasticity
- **Substepping**: Each frame is split into up to 8 physics substeps so that the fastest particle never travels more than one radius per step, preventing tunneling

### Collision Detection

A uniform spatial grid (rebuilt each substep with zero steady-state allocations) bins particles into cells at least one diameter wide. Only particles in the same or adjacent cells are tested pairwise, making collision detection effectively linear in the particle count.

### Particle Spawning

Each collision between particles spawns a new particle. By default, new particles appear at the screen center. With the `--spawn-at-collision` flag, they spawn at the collision point, creating more localized cluster effects.

### Explosion Trigger

The simulation tracks spawn rate over a 1-second sliding window. When this rate exceeds 30 spawns per second, an explosion is triggered:

1. An expanding ring emanates from the screen center (or from the collision hotspot in `--spawn-at-collision` mode, or from the cursor on right click)
2. 99% of particles are marked for elimination
3. Particles are killed when the ring reaches them
4. A minimum number of particles survive based on screen size

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

This ensures the simulation feels consistent across different display sizes. Retina/HiDPI displays use their logical resolution, not the physical pixel count.

Use `--min-particles <N>` to override this with a fixed count between 2 and 100.

### Audio Synthesis

All sounds are generated programmatically:

- **Collision Pings**: Sine wave with exponential decay, frequency mapped to collision energy (300-1500 Hz). Pings are pre-generated into pitch buckets at startup (no per-collision allocation) and stereo-panned to match the on-screen collision position
- **Explosion Rumble**: Low-frequency oscillators (40-80 Hz) mixed with noise, shaped with attack/decay envelope

If no audio output device is available (e.g. headless machines), the simulation runs silently instead of failing.

## Configuration

Key constants can be modified in the source:

```rust
// src/physics.rs
const GRAVITY: f64 = 100.0;
const DEFAULT_PARTICLE_RADIUS: f64 = 1.5;
const INITIAL_VELOCITY: f64 = 600.0;
const MOTION_VELOCITY_THRESHOLD: f64 = 1.0;  // Minimum velocity to be "moving"
const MOTION_STOPPED_FRAMES: u32 = 60;       // ~1 second at 60fps

// src/explosion.rs
const SPAWN_RATE_THRESHOLD: usize = 30;
const EXPLOSION_KILL_RATIO: f64 = 0.99;

// src/app.rs
const PIXELS_PER_PARTICLE: u64 = 375_000;
const CLICK_BURST_SIZE: usize = 10;

// src/audio.rs
const PING_MIN_FREQ: f32 = 300.0;
const PING_MAX_FREQ: f32 = 1500.0;
```

## Dependencies

- [`pixels`](https://crates.io/crates/pixels) - Hardware-accelerated pixel buffer (GPU rendering)
- [`softbuffer`](https://crates.io/crates/softbuffer) - Software pixel buffer (CPU fallback)
- [`winit`](https://crates.io/crates/winit) - Cross-platform window management
- [`rodio`](https://crates.io/crates/rodio) - Audio playback
- [`rand`](https://crates.io/crates/rand) - Random number generation
- [`clap`](https://crates.io/crates/clap) - Command line argument parsing
- [`pollster`](https://crates.io/crates/pollster) - Minimal async executor
- [`ouroboros`](https://crates.io/crates/ouroboros) - Safe self-referential struct support
- [`ab_glyph`](https://crates.io/crates/ab_glyph) - Font rendering

## Platform Support

Tested on:
- macOS (Metal backend)

Should work on:
- Windows (DX12/Vulkan backend)
- Linux (Vulkan backend)

## Architecture

The application uses the modern `winit` 0.30 `ApplicationHandler` pattern, split into focused modules:

- `main.rs` - Entry point and event loop setup
- `config.rs` - Command line parsing (clap)
- `app.rs` - Application state, input handling, HUD, and event loop glue
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

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute, including our Developer Certificate of Origin (DCO) requirements.

## Acknowledgments

Built with Rust and its excellent ecosystem of graphics and audio crates.

This software includes Liberation Sans Bold font, licensed under the [SIL Open Font License](https://scripts.sil.org/OFL).
