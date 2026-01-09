# Bouncy

A GPU-accelerated particle simulation written in Rust featuring elastic collisions, gravity, dynamic particle spawning, and explosive chain reactions with synthesized audio feedback.

## Features

- **GPU-Accelerated Rendering**: Uses the `pixels` crate with wgpu backend for smooth, hardware-accelerated 2D rendering
- **Realistic Physics**: Elastic particle-particle collisions and wall bounces with configurable gravity
- **Dynamic Spawning**: New particles spawn on collision, creating organic growth patterns
- **Explosion Mechanics**: When spawn rate exceeds threshold, a dramatic explosion kills 99% of particles
- **Synthesized Audio**: Real-time audio synthesis for collision pings (pitch based on impact energy) and explosion rumbles
- **Fullscreen Display**: Runs in borderless fullscreen mode
- **Adaptive Particle Count**: Initial particle count scales based on screen resolution

## Demo

Particles bounce around the screen with realistic physics. Each collision spawns a new particle and plays a ping sound with pitch corresponding to collision energy. When the spawn rate exceeds 30 particles per second, an expanding ring explosion eliminates 99% of particles, resetting the simulation with a deep rumble sound.

## Installation

### Prerequisites

- Rust 1.70 or later
- A GPU with Vulkan, Metal, or DX12 support

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
| `--gravity <PERCENT>` | Set gravity as percentage of standard; negative values cause upward gravity | 100 |
| `--wall-elasticity <VALUE>` | Set wall bounce elasticity (0.0-1.5); 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy | 1.0 |
| `--particle-elasticity <VALUE>` | Set particle collision elasticity (0.0-1.5); 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy | 1.0 |
| `--help`, `-h` | Display help information | |

### Controls

| Key | Action |
|-----|--------|
| `Space` | Exit |
| `Escape` | Exit |
| `Q` | Exit |

### Motion Detection

When all particles stop moving (velocity below threshold for ~1 second), a "STOPPED" message is displayed and the simulation halts. This can happen when using low elasticity values where particles eventually lose all energy.

## How It Works

### Physics Simulation

The simulation uses a simple but effective physics model:

- **Gravity**: Constant downward acceleration applied to all particles
- **Elastic Collisions**: Particles exchange momentum on contact using the relative velocity along the collision normal
- **Wall Bouncing**: Particles reflect off screen boundaries with no energy loss

### Particle Spawning

Each collision between particles spawns a new particle. By default, new particles appear at the screen center. With the `--spawn-at-collision` flag, they spawn at the collision point, creating more localized cluster effects.

### Explosion Trigger

The simulation tracks spawn rate over a 1-second sliding window. When this rate exceeds 30 spawns per second, an explosion is triggered:

1. An expanding ring emanates from the screen center
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

- **Collision Pings**: Sine wave with exponential decay, frequency mapped to collision energy (300-1500 Hz)
- **Explosion Rumble**: Low-frequency oscillators (40-80 Hz) mixed with noise, shaped with attack/decay envelope

## Configuration

Key constants can be modified in `src/main.rs`:

```rust
// Physics constants
const GRAVITY: f64 = 100.0;
const PARTICLE_RADIUS: f64 = 1.5;
const INITIAL_VELOCITY: f64 = 600.0;

// Spawn/explosion constants
const SPAWN_RATE_THRESHOLD: usize = 30;
const EXPLOSION_KILL_RATIO: f64 = 0.99;
const PIXELS_PER_PARTICLE: u64 = 375_000;

// Audio constants
const PING_MIN_FREQ: f32 = 300.0;
const PING_MAX_FREQ: f32 = 1500.0;

// Motion detection constants
const MOTION_VELOCITY_THRESHOLD: f64 = 1.0;  // Minimum velocity to be "moving"
const MOTION_STOPPED_FRAMES: u32 = 60;       // ~1 second at 60fps
```

## Dependencies

- [`pixels`](https://crates.io/crates/pixels) - Hardware-accelerated pixel buffer
- [`winit`](https://crates.io/crates/winit) - Cross-platform window management
- [`rodio`](https://crates.io/crates/rodio) - Audio playback
- [`rand`](https://crates.io/crates/rand) - Random number generation
- [`pollster`](https://crates.io/crates/pollster) - Minimal async executor
- [`ouroboros`](https://crates.io/crates/ouroboros) - Self-referential struct support
- [`rusttype`](https://crates.io/crates/rusttype) - Font rendering

## Platform Support

Tested on:
- macOS (Metal backend)

Should work on:
- Windows (DX12/Vulkan backend)
- Linux (Vulkan backend)

## Architecture

The application uses the modern `winit` 0.30 `ApplicationHandler` pattern:

- `App` struct holds all simulation state
- `RenderContext` uses `ouroboros` for safe self-referential lifetime management (Pixels borrows from Window)
- Physics and rendering run in the main event loop, synchronized to VSync

## Performance

- Targets 120 FPS on modern displays (VSync-locked)
- O(n²) collision detection (suitable for hundreds of particles)
- GPU-accelerated rendering via wgpu
- Warmup frames skip physics to allow GPU initialization

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
