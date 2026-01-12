// Copyright (c) 2026 James O. Schreckengast
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use ouroboros::self_referencing;
use pixels::{Pixels, SurfaceTexture};
use rand::seq::SliceRandom;
use rand::Rng;
use rodio::{OutputStream, OutputStreamHandle, Source};
use rusttype::{Font, Scale};
use std::collections::VecDeque;
use std::env;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

/// Embedded font data (Liberation Sans Bold - SIL Open Font License).
const FONT_DATA: &[u8] = include_bytes!("../assets/LiberationSans-Bold.ttf");

/// Lazily-initialized cached font to avoid loading on every render.
fn get_font() -> &'static Font<'static> {
    static FONT: OnceLock<Font<'static>> = OnceLock::new();
    FONT.get_or_init(|| Font::try_from_bytes(FONT_DATA).expect("Failed to load embedded font"))
}

// Physics constants
const GRAVITY: f64 = 100.0;
const PARTICLE_RADIUS: f64 = 1.5;
const PARTICLE_DIAMETER: f64 = PARTICLE_RADIUS * 2.0;
const PARTICLE_DIAMETER_SQ: f64 = PARTICLE_DIAMETER * PARTICLE_DIAMETER;
const INITIAL_VELOCITY: f64 = 600.0;

// Spawn/explosion constants
const SPAWN_RATE_WINDOW: f64 = 1.0;
const SPAWN_RATE_THRESHOLD: usize = 30;
const EXPLOSION_KILL_RATIO: f64 = 0.99;
const PIXELS_PER_PARTICLE: u64 = 375_000;
const EXPLOSION_SPEED: f64 = 800.0;
const EXPLOSION_RING_WIDTH: f64 = 20.0;

// Audio constants
const AUDIO_SAMPLE_RATE: u32 = 44100;
const PING_DURATION_MS: u64 = 80;
const PING_MIN_FREQ: f32 = 300.0;
const PING_MAX_FREQ: f32 = 1500.0;
const EXPLOSION_DURATION_MS: u64 = 800;

// Collision constants
const COLLISION_ENERGY_NORMALIZER: f64 = 800.0;
const SEPARATION_PADDING: f64 = 0.5;
const PARTICLE_MARGIN: f64 = 10.0;

// Motion detection constants
const MOTION_VELOCITY_THRESHOLD: f64 = 1.0; // Minimum velocity to be considered "moving"
const MOTION_STOPPED_FRAMES: u32 = 60; // Frames of no motion before declaring stopped

// Float comparison constants
const DISTANCE_SQ_EPSILON: f64 = 1e-10; // Minimum distance squared for collision normal

// =============================================================================
// Type conversion helpers for graphics code
// =============================================================================
// These functions document the intent of narrowing conversions that are
// inherent to graphics programming (float coords -> integer pixels, etc.)

/// Convert f64 coordinate to signed pixel position (truncates toward zero).
/// Used for particle rendering where we need signed coords for offset math.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn coord_to_pixel(v: f64) -> i32 {
    v as i32
}

/// Convert f64 to unsigned dimension, clamping negative values to 0.
/// Used for pixel buffer indexing where negative values are invalid.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn coord_to_pixel_unsigned(v: f64) -> u32 {
    v.max(0.0) as u32
}

/// Convert f64 color component (0.0-255.0) to u8.
/// Values are clamped to valid range.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn color_component(v: f64) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Convert f32 color component (0.0-255.0) to u8.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn color_component_f32(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Convert HSV hue sector (0-5) for color calculation.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hue_sector(hue: f64) -> u32 {
    (hue / 60.0) as u32
}

/// Convert physical pixels to logical pixels given a scale factor.
/// Used for display size calculations where we need logical dimensions.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn physical_to_logical(physical: u32, scale_factor: f64) -> u32 {
    (f64::from(physical) / scale_factor) as u32
}

/// Result of a collision between two particles.
struct CollisionResult {
    /// Energy of the collision (relative velocity along collision normal).
    energy: f64,
    /// Midpoint x-coordinate where the collision occurred.
    mid_x: f64,
    /// Midpoint y-coordinate where the collision occurred.
    mid_y: f64,
}

/// Calculate the initial/minimum particle count based on screen size.
fn calculate_particle_count(width: u32, height: u32) -> usize {
    let total_pixels = u64::from(width) * u64::from(height);
    let count = (total_pixels + PIXELS_PER_PARTICLE / 2) / PIXELS_PER_PARTICLE;
    // Safe: count is always small (screen pixels / 375000)
    usize::try_from(count.max(2)).unwrap_or(2)
}

/// A particle in the simulation with position, velocity, and color.
struct Particle {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    color: [u8; 4],
}

impl Particle {
    /// Generate a random velocity vector with speed between 50-100% of `INITIAL_VELOCITY`.
    fn random_velocity() -> (f64, f64) {
        let mut rng = rand::thread_rng();
        let angle: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let speed: f64 = rng.gen_range(INITIAL_VELOCITY * 0.5..INITIAL_VELOCITY);
        (speed * angle.cos(), speed * angle.sin())
    }

    /// Create a particle at a random position within the screen bounds.
    fn new_random(width: u32, height: u32) -> Self {
        let mut rng = rand::thread_rng();
        let (vx, vy) = Self::random_velocity();
        Particle {
            x: rng.gen_range(PARTICLE_MARGIN..(f64::from(width) - PARTICLE_MARGIN)),
            y: rng.gen_range(PARTICLE_MARGIN..(f64::from(height) - PARTICLE_MARGIN)),
            vx,
            vy,
            color: random_bright_color(),
        }
    }

    /// Create a particle at the center of the screen.
    fn new_at_center(width: u32, height: u32) -> Self {
        Self::new_at_position(f64::from(width) / 2.0, f64::from(height) / 2.0)
    }

    /// Create a particle at a specific position.
    fn new_at_position(x: f64, y: f64) -> Self {
        let (vx, vy) = Self::random_velocity();
        Particle {
            x,
            y,
            vx,
            vy,
            color: random_bright_color(),
        }
    }

    /// Update particle position with gravity and wall bouncing.
    /// `gravity_multiplier`: 1.0 = 100% gravity, negative = upward gravity
    /// `wall_elasticity`: 1.0 = fully elastic, 0.0 = completely inelastic (sticks)
    fn update(
        &mut self,
        dt: f64,
        width: u32,
        height: u32,
        gravity_multiplier: f64,
        wall_elasticity: f64,
    ) {
        self.vy += GRAVITY * gravity_multiplier * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        let width_f = f64::from(width);
        let height_f = f64::from(height);

        if self.x <= PARTICLE_RADIUS {
            self.x = PARTICLE_RADIUS;
            self.vx = -self.vx * wall_elasticity;
        } else if self.x >= width_f - PARTICLE_RADIUS {
            self.x = width_f - PARTICLE_RADIUS;
            self.vx = -self.vx * wall_elasticity;
        }

        if self.y <= PARTICLE_RADIUS {
            self.y = PARTICLE_RADIUS;
            self.vy = -self.vy * wall_elasticity;
        } else if self.y >= height_f - PARTICLE_RADIUS {
            self.y = height_f - PARTICLE_RADIUS;
            self.vy = -self.vy * wall_elasticity;
        }
    }

    /// Calculate squared distance from particle to a point.
    fn distance_squared_from(&self, x: f64, y: f64) -> f64 {
        let dx = self.x - x;
        let dy = self.y - y;
        dx * dx + dy * dy
    }
}

/// An expanding ring that kills particles it touches.
struct Explosion {
    x: f64,
    y: f64,
    radius: f64,
    radius_sq: f64,
    max_radius: f64,
    active: bool,
    /// Indices of particles to kill, sorted in descending order for efficient removal.
    indices_to_kill: Vec<usize>,
    killed_count: usize,
}

impl Explosion {
    /// Create a new explosion that will kill a percentage of particles.
    fn new(x: f64, y: f64, max_radius: f64, particle_count: usize, min_survivors: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Calculate kill count: truncation is intentional (we want floor, not round)
        // Precision loss acceptable: particle_count is small relative to f64 mantissa
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let kill_count = ((particle_count as f64 * EXPLOSION_KILL_RATIO) as usize)
            .min(particle_count.saturating_sub(min_survivors));

        let mut all_indices: Vec<usize> = (0..particle_count).collect();
        all_indices.shuffle(&mut rng);
        let mut indices_to_kill: Vec<usize> = all_indices.into_iter().take(kill_count).collect();
        // Sort descending so we can use swap_remove without invalidating indices
        indices_to_kill.sort_unstable_by(|a, b| b.cmp(a));

        println!(
            "Explosion will kill {} of {} particles",
            indices_to_kill.len(),
            particle_count
        );

        Explosion {
            x,
            y,
            radius: 0.0,
            radius_sq: 0.0,
            max_radius,
            active: true,
            indices_to_kill,
            killed_count: 0,
        }
    }

    /// Expand the explosion ring.
    fn update(&mut self, dt: f64) {
        if self.active {
            self.radius += EXPLOSION_SPEED * dt;
            self.radius_sq = self.radius * self.radius;
            if self.radius >= self.max_radius {
                self.active = false;
            }
        }
    }

    /// Kill particles that are within the explosion radius.
    fn process_kills(&mut self, particles: &mut Vec<Particle>) {
        let mut i = 0;
        while i < self.indices_to_kill.len() {
            let target_idx = self.indices_to_kill[i];

            if target_idx >= particles.len() {
                // Index no longer valid
                self.indices_to_kill.swap_remove(i);
                continue;
            }

            let dist_sq = particles[target_idx].distance_squared_from(self.x, self.y);
            if dist_sq <= self.radius_sq {
                // Kill this particle using swap_remove (O(1))
                particles.swap_remove(target_idx);
                self.killed_count += 1;
                self.indices_to_kill.swap_remove(i);

                // The particle that was at particles.len() (the old last position)
                // is now at target_idx. Update any reference to it in our kill list.
                let old_last_idx = particles.len();
                if old_last_idx != target_idx {
                    for idx in &mut self.indices_to_kill {
                        if *idx == old_last_idx {
                            *idx = target_idx;
                            break; // Only one entry can match
                        }
                    }
                }
                // Don't increment i since we swapped a new element into position i
            } else {
                i += 1;
            }
        }
    }
}

/// Generate a random bright color using HSV color space.
fn random_bright_color() -> [u8; 4] {
    let mut rng = rand::thread_rng();
    let hue: f64 = rng.gen_range(0.0..360.0);
    let saturation: f64 = 0.4;
    let value: f64 = 1.0;

    let chroma = value * saturation;
    let secondary = chroma * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let match_value = value - chroma;

    let (red, green, blue) = match hue_sector(hue) {
        0 => (chroma, secondary, 0.0),
        1 => (secondary, chroma, 0.0),
        2 => (0.0, chroma, secondary),
        3 => (0.0, secondary, chroma),
        4 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };

    [
        color_component((red + match_value) * 255.0),
        color_component((green + match_value) * 255.0),
        color_component((blue + match_value) * 255.0),
        255,
    ]
}

/// Attempt elastic collision between two particles.
/// Returns collision result if particles are touching and approaching.
/// `particle_elasticity`: 1.0 = fully elastic, 0.0 = completely inelastic (stick together)
fn try_elastic_collision(
    p1: &mut Particle,
    p2: &mut Particle,
    particle_elasticity: f64,
) -> Option<CollisionResult> {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let dist_sq = dx * dx + dy * dy;

    if !(DISTANCE_SQ_EPSILON..=PARTICLE_DIAMETER_SQ).contains(&dist_sq) {
        return None;
    }

    let mid_x = p1.x.midpoint(p2.x);
    let mid_y = p1.y.midpoint(p2.y);

    let dist = dist_sq.sqrt();
    let nx = dx / dist;
    let ny = dy / dist;

    let dvx = p1.vx - p2.vx;
    let dvy = p1.vy - p2.vy;
    let dvn = dvx * nx + dvy * ny;

    if dvn <= 0.0 {
        return None;
    }

    let energy = dvn;

    // Apply elasticity: 1.0 = full momentum exchange, 0.0 = no bounce
    let impulse = dvn * particle_elasticity;
    p1.vx -= impulse * nx;
    p1.vy -= impulse * ny;
    p2.vx += impulse * nx;
    p2.vy += impulse * ny;

    let overlap = PARTICLE_DIAMETER - dist;
    if overlap > 0.0 {
        let sep = overlap / 2.0 + SEPARATION_PADDING;
        p1.x -= sep * nx;
        p1.y -= sep * ny;
        p2.x += sep * nx;
        p2.y += sep * ny;
    }

    Some(CollisionResult {
        energy,
        mid_x,
        mid_y,
    })
}

/// Generate a ping sound with exponential decay at the given frequency.
fn generate_ping(frequency: f32) -> impl Source<Item = f32> + Send {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * PING_DURATION_MS / 1000) as usize;
    // u32 sample rate fits in f32 mantissa (44100 << 2^23)
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            // Precision loss from usize->f32 is acceptable for audio timing
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let envelope = (-t * 20.0).exp();
            let wave = (2.0 * std::f32::consts::PI * frequency * t).sin();
            wave * envelope * 0.3
        })
        .collect();

    rodio::buffer::SamplesBuffer::new(1, AUDIO_SAMPLE_RATE, samples)
}

/// Generate a low-frequency rumble sound for explosions.
fn generate_explosion() -> impl Source<Item = f32> + Send {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * EXPLOSION_DURATION_MS / 1000) as usize;
    // u32 sample rate fits in f32 mantissa (44100 << 2^23)
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;
    let mut rng = rand::thread_rng();

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            // Precision loss from usize->f32 is acceptable for audio timing
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let envelope = if t < 0.05 {
                t / 0.05
            } else {
                (-((t - 0.05) * 3.0)).exp()
            };

            let rumble = (2.0 * std::f32::consts::PI * 60.0 * t).sin() * 0.4
                + (2.0 * std::f32::consts::PI * 80.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 40.0 * t).sin() * 0.3;

            let noise: f32 = rng.gen_range(-1.0..1.0) * 0.5;

            (rumble + noise) * envelope * 0.6
        })
        .collect();

    rodio::buffer::SamplesBuffer::new(1, AUDIO_SAMPLE_RATE, samples)
}

/// Play a collision sound with pitch based on collision energy.
fn play_collision_sound(stream_handle: &OutputStreamHandle, energy: f64) {
    // f64->f32 truncation is acceptable for audio frequency calculation
    #[allow(clippy::cast_possible_truncation)]
    let energy_normalized = (energy / COLLISION_ENERGY_NORMALIZER).clamp(0.0, 1.0) as f32;
    let frequency = PING_MIN_FREQ + energy_normalized * (PING_MAX_FREQ - PING_MIN_FREQ);
    let _ = stream_handle.play_raw(generate_ping(frequency).convert_samples());
}

/// Play the explosion rumble sound.
fn play_explosion_sound(stream_handle: &OutputStreamHandle) {
    let _ = stream_handle.play_raw(generate_explosion().convert_samples());
}

/// Render all particles as 3x3 pixel squares.
fn render_particles(frame: &mut [u8], particles: &[Particle], width: u32, height: u32) {
    for particle in particles {
        let cx = coord_to_pixel(particle.x);
        let cy = coord_to_pixel(particle.y);

        for dy in -1..=1 {
            for dx in -1..=1 {
                let px = cx + dx;
                let py = cy + dy;

                // Bounds check: px/py are valid pixel coordinates after this check
                #[allow(clippy::cast_sign_loss)]
                if px >= 0 && (px as u32) < width && py >= 0 && (py as u32) < height {
                    let idx = ((py as u32) * width + (px as u32)) as usize * 4;
                    frame[idx..idx + 4].copy_from_slice(&particle.color);
                }
            }
        }
    }
}

/// Render the explosion as an expanding orange ring.
fn render_explosion(frame: &mut [u8], exp: &Explosion, width: u32, height: u32) {
    let inner_radius = (exp.radius - EXPLOSION_RING_WIDTH).max(0.0);
    let outer_radius = exp.radius;

    // Calculate bounding box for the ring (clamped to screen bounds)
    let min_x = coord_to_pixel_unsigned(exp.x - outer_radius);
    let max_x = coord_to_pixel_unsigned((exp.x + outer_radius).ceil()).min(width);
    let min_y = coord_to_pixel_unsigned(exp.y - outer_radius);
    let max_y = coord_to_pixel_unsigned((exp.y + outer_radius).ceil()).min(height);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = f64::from(x) - exp.x;
            let dy = f64::from(y) - exp.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist >= inner_radius && dist <= outer_radius {
                let intensity = 1.0 - ((dist - inner_radius) / EXPLOSION_RING_WIDTH).abs();
                let brightness = color_component(intensity * 255.0);
                let idx = usize::try_from(y * width + x).unwrap_or(0) * 4;
                frame[idx] = brightness;
                frame[idx + 1] = color_component_f32(f32::from(brightness) * 0.6);
                frame[idx + 2] = color_component_f32(f32::from(brightness) * 0.2);
                frame[idx + 3] = 255;
            }
        }
    }
}

/// Check if any particles have significant motion.
fn has_motion(particles: &[Particle]) -> bool {
    const THRESHOLD_SQ: f64 = MOTION_VELOCITY_THRESHOLD * MOTION_VELOCITY_THRESHOLD;
    particles
        .iter()
        .any(|p| p.vx * p.vx + p.vy * p.vy > THRESHOLD_SQ)
}

/// Render centered text on the frame using the cached embedded font.
#[allow(clippy::cast_precision_loss)] // width/height fit in f32 mantissa for reasonable screens
fn render_text(frame: &mut [u8], width: u32, height: u32, text: &str, font_size: f32) {
    let font = get_font();
    let scale = Scale::uniform(font_size);
    let v_metrics = font.v_metrics(scale);
    let glyphs: Vec<_> = font
        .layout(text, scale, rusttype::point(0.0, 0.0))
        .collect();

    // Calculate text dimensions
    let text_width = glyphs.last().map_or(0.0, |g| {
        g.position().x + g.unpositioned().h_metrics().advance_width
    });
    let text_height = v_metrics.ascent - v_metrics.descent;

    // Center the text
    let width_f = width as f32;
    let height_f = height as f32;
    let start_x = ((width_f - text_width) / 2.0).max(0.0);
    let start_y = ((height_f - text_height) / 2.0 + v_metrics.ascent).max(0.0);

    // Re-layout with correct position
    let glyphs: Vec<_> = font
        .layout(text, scale, rusttype::point(start_x, start_y))
        .collect();

    // Render each glyph
    let color = [255u8, 100, 100, 255]; // Light red
    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|glyph_x, glyph_y, coverage| {
                // Font library provides u32 glyph coords, bounding box has i32 positions
                #[allow(clippy::cast_possible_wrap)]
                let px = bounding_box.min.x + glyph_x as i32;
                #[allow(clippy::cast_possible_wrap)]
                let py = bounding_box.min.y + glyph_y as i32;

                // Only render if within bounds and coverage is significant
                #[allow(clippy::cast_sign_loss)]
                if px >= 0
                    && (px as u32) < width
                    && py >= 0
                    && (py as u32) < height
                    && coverage > 0.1
                {
                    let idx = (py as u32 * width + px as u32) as usize * 4;
                    let alpha = color_component_f32(coverage * 255.0);
                    // Blend color with coverage alpha (result always fits in u8: max = 255*255/255 = 255)
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        frame[idx] = (u16::from(color[0]) * u16::from(alpha) / 255) as u8;
                        frame[idx + 1] = (u16::from(color[1]) * u16::from(alpha) / 255) as u8;
                        frame[idx + 2] = (u16::from(color[2]) * u16::from(alpha) / 255) as u8;
                    }
                    frame[idx + 3] = alpha;
                }
            });
        }
    }
}

/// Render a large "STOPPED" message in the center of the screen.
fn render_stopped_message(frame: &mut [u8], width: u32, height: u32) {
    render_text(frame, width, height, "STOPPED", 72.0);
}

/// Update all particle positions with physics simulation.
fn update_physics(
    particles: &mut [Particle],
    dt: f64,
    width: u32,
    height: u32,
    gravity_multiplier: f64,
    wall_elasticity: f64,
) {
    for particle in particles {
        particle.update(dt, width, height, gravity_multiplier, wall_elasticity);
    }
}

/// Process all particle-particle collisions.
/// Returns the maximum collision energy and populates `collision_positions`.
fn handle_collisions(
    particles: &mut [Particle],
    collision_positions: &mut Vec<(f64, f64)>,
    particle_elasticity: f64,
) -> f64 {
    collision_positions.clear();
    let mut max_energy = 0.0f64;
    let n = particles.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let (left, right) = particles.split_at_mut(j);
            let p1 = &mut left[i];
            let p2 = &mut right[0];

            if let Some(result) = try_elastic_collision(p1, p2, particle_elasticity) {
                max_energy = max_energy.max(result.energy);
                collision_positions.push((result.mid_x, result.mid_y));
            }
        }
    }

    max_energy
}

fn print_usage() {
    println!(
        r"Usage: bouncy [OPTIONS]

Options:
  --spawn-at-collision        Spawn new particles at collision points instead of center
  --min-particles <N>         Set starting/minimum particle count (2-100)
  --gravity <PERCENT>         Set gravity as percentage of standard (default: 100)
                              Negative values cause upward gravity
  --wall-elasticity <VALUE>   Set wall bounce elasticity 0.0-1.5 (default: 1.0)
                              0.0 = no bounce (sticks), 1.0 = fully elastic,
                              >1.0 = walls add energy to particles
  --particle-elasticity <VALUE>
                              Set particle collision elasticity 0.0-1.5 (default: 1.0)
                              0.0 = no bounce, 1.0 = fully elastic,
                              >1.0 = collisions add energy to particles
  --width <N>                 Set window width in pixels (100-7680)
  --height <N>                Set window height in pixels (100-4320)
                              Both --width and --height must be specified together.
                              Omit both for fullscreen mode (default).
  --cpu                       Force CPU rendering (softbuffer) instead of GPU
  --help, -h                  Show this help message

Controls:
  Space, Escape, Q            Exit the program

Motion Detection:
  When all particles stop moving, a STOPPED message is displayed.

Rendering:
  Uses GPU rendering (wgpu) by default. Falls back to CPU rendering
  (softbuffer) if GPU is unavailable. Use --cpu to force CPU rendering."
    );
}

/// Configuration parsed from command line arguments.
#[derive(Clone, Copy, Default)]
struct Config {
    spawn_at_collision: bool,
    min_particles: Option<usize>,
    gravity_percent: i32,
    wall_elasticity: f64,
    particle_elasticity: f64,
    width: Option<u32>,
    height: Option<u32>,
    force_cpu: bool,
}

impl Config {
    fn new() -> Self {
        Self {
            gravity_percent: 100,
            wall_elasticity: 1.0,
            particle_elasticity: 1.0,
            ..Default::default()
        }
    }
}

/// Parse a required argument value, returning None with error message if missing or invalid.
fn parse_arg<T: std::str::FromStr>(args: &[String], i: &mut usize, opt_name: &str) -> Option<T> {
    *i += 1;
    if *i >= args.len() {
        eprintln!("Error: {opt_name} requires an argument");
        return None;
    }
    args[*i].parse().ok().or_else(|| {
        eprintln!("Error: {opt_name} requires a valid value");
        None
    })
}

/// Parse a required argument with range validation.
fn parse_arg_range<T>(args: &[String], i: &mut usize, opt_name: &str, min: T, max: T) -> Option<T>
where
    T: std::str::FromStr + PartialOrd + std::fmt::Display + Copy,
{
    let value: T = parse_arg(args, i, opt_name)?;
    if value < min || value > max {
        eprintln!("Error: {opt_name} must be between {min} and {max}");
        return None;
    }
    Some(value)
}

/// Parse command line arguments into a Config, or None if invalid/help requested.
fn parse_args() -> Option<Config> {
    let args: Vec<String> = env::args().collect();
    let mut config = Config::new();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--spawn-at-collision" => config.spawn_at_collision = true,
            "--min-particles" => {
                config.min_particles =
                    Some(parse_arg_range(&args, &mut i, "--min-particles", 2, 100)?);
            }
            "--gravity" => {
                config.gravity_percent = parse_arg(&args, &mut i, "--gravity")?;
            }
            "--wall-elasticity" => {
                config.wall_elasticity =
                    parse_arg_range(&args, &mut i, "--wall-elasticity", 0.0, 1.5)?;
            }
            "--particle-elasticity" => {
                config.particle_elasticity =
                    parse_arg_range(&args, &mut i, "--particle-elasticity", 0.0, 1.5)?;
            }
            "--width" => {
                config.width = Some(parse_arg_range(&args, &mut i, "--width", 100, 7680)?);
            }
            "--height" => {
                config.height = Some(parse_arg_range(&args, &mut i, "--height", 100, 4320)?);
            }
            "--cpu" => config.force_cpu = true,
            "--help" | "-h" => {
                print_usage();
                return None;
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_usage();
                return None;
            }
        }
        i += 1;
    }

    // Validate width/height pairing
    match (config.width, config.height) {
        (Some(_), None) => {
            eprintln!("Error: --width requires --height to also be specified");
            print_usage();
            return None;
        }
        (None, Some(_)) => {
            eprintln!("Error: --height requires --width to also be specified");
            print_usage();
            return None;
        }
        _ => {}
    }

    Some(config)
}

/// GPU render context using ouroboros for safe self-referential struct.
/// Pixels borrows from Window, so they must be in the same struct.
/// Uses Rc<Window> to allow sharing the window with fallback logic.
#[self_referencing]
struct GpuRenderContext {
    window: Rc<Window>,
    width: u32,
    height: u32,
    #[borrows(window)]
    #[covariant]
    pixels: Pixels<'this>,
}

/// CPU render context using softbuffer (no self-reference needed).
struct CpuRenderContext {
    window: Rc<Window>,
    width: u32,           // Logical width (for simulation/rendering)
    height: u32,          // Logical height (for simulation/rendering)
    physical_width: u32,  // Physical width (for softbuffer surface)
    physical_height: u32, // Physical height (for softbuffer surface)
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    buffer: Vec<u8>, // RGBA buffer for rendering functions (logical size)
}

/// Render context abstraction supporting both GPU and CPU backends.
enum RenderContext {
    Gpu(Box<GpuRenderContext>),
    Cpu(CpuRenderContext),
}

impl RenderContext {
    /// Get the window reference for requesting redraws.
    fn window(&self) -> &Window {
        match self {
            RenderContext::Gpu(ctx) => ctx.borrow_window(),
            RenderContext::Cpu(ctx) => &ctx.window,
        }
    }

    /// Get the logical width.
    fn width(&self) -> u32 {
        match self {
            RenderContext::Gpu(ctx) => *ctx.borrow_width(),
            RenderContext::Cpu(ctx) => ctx.width,
        }
    }

    /// Get the logical height.
    fn height(&self) -> u32 {
        match self {
            RenderContext::Gpu(ctx) => *ctx.borrow_height(),
            RenderContext::Cpu(ctx) => ctx.height,
        }
    }

    /// Get a mutable reference to the RGBA frame buffer and call a function with it.
    fn with_frame<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        match self {
            RenderContext::Gpu(ctx) => ctx.with_pixels_mut(|pixels| f(pixels.frame_mut())),
            RenderContext::Cpu(ctx) => f(ctx.buffer.as_mut_slice()),
        }
    }

    /// Present the frame to the screen.
    fn present(&mut self) -> Result<(), String> {
        match self {
            RenderContext::Gpu(ctx) => {
                ctx.with_pixels_mut(|pixels| pixels.render().map_err(|e| e.to_string()))
            }
            RenderContext::Cpu(ctx) => {
                let mut sb_buffer = ctx.surface.buffer_mut().map_err(|e| e.to_string())?;

                // Scale from logical to physical dimensions using nearest-neighbor
                let logical_width = ctx.width as usize;
                let logical_height = ctx.height as usize;
                let physical_width = ctx.physical_width as usize;
                let physical_height = ctx.physical_height as usize;

                for py in 0..physical_height {
                    for px in 0..physical_width {
                        // Map physical pixel to logical pixel (nearest-neighbor)
                        let lx = px * logical_width / physical_width;
                        let ly = py * logical_height / physical_height;

                        let src_idx = (ly * logical_width + lx) * 4;
                        let dst_idx = py * physical_width + px;

                        if src_idx + 2 < ctx.buffer.len() {
                            let r = u32::from(ctx.buffer[src_idx]);
                            let g = u32::from(ctx.buffer[src_idx + 1]);
                            let b = u32::from(ctx.buffer[src_idx + 2]);
                            sb_buffer[dst_idx] = (r << 16) | (g << 8) | b;
                        }
                    }
                }

                sb_buffer.present().map_err(|e| e.to_string())?;
                Ok(())
            }
        }
    }

    /// Resize the surface (for GPU backend only, when window size changes).
    fn resize_surface(&mut self, width: u32, height: u32) {
        if let RenderContext::Gpu(ctx) = self {
            ctx.with_pixels_mut(|pixels| {
                let _ = pixels.resize_surface(width, height);
            });
        }
    }
}

/// Main application state for the particle simulation.
struct App {
    // Configuration
    spawn_at_collision: bool,
    min_particles_override: Option<usize>,
    gravity_percent: i32,
    wall_elasticity: f64,
    particle_elasticity: f64,
    requested_width: Option<u32>,
    requested_height: Option<u32>,
    force_cpu: bool,

    // Audio (must be stored to keep stream alive)
    _audio_stream: OutputStream,
    stream_handle: OutputStreamHandle,

    // Window and rendering (initialized on resume)
    render: Option<RenderContext>,

    // Simulation state
    particles: Vec<Particle>,
    explosion: Option<Explosion>,
    spawn_times: VecDeque<Instant>,
    collision_positions: Vec<(f64, f64)>,

    // Derived values
    base_particle_count: usize,
    center_x: f64,
    center_y: f64,
    max_explosion_radius: f64,

    // Timing
    last_time: Instant,
    frame_count: u64,
    fps_timer: Instant,
    warmup_frames: u32,

    // Motion detection
    stopped: bool,
    frames_without_motion: u32,
}

/// Handle keyboard input for exit keys.
fn handle_key(key_code: KeyCode, event_loop: &ActiveEventLoop) {
    if matches!(key_code, KeyCode::Space | KeyCode::Escape | KeyCode::KeyQ) {
        event_loop.exit();
    }
}

impl App {
    /// Create a new App with the given configuration.
    fn new(config: Config) -> Self {
        let (audio_stream, stream_handle) =
            OutputStream::try_default().expect("Failed to create audio output stream");

        App {
            spawn_at_collision: config.spawn_at_collision,
            min_particles_override: config.min_particles,
            gravity_percent: config.gravity_percent,
            wall_elasticity: config.wall_elasticity,
            particle_elasticity: config.particle_elasticity,
            requested_width: config.width,
            requested_height: config.height,
            force_cpu: config.force_cpu,
            _audio_stream: audio_stream,
            stream_handle,
            render: None,
            particles: Vec::new(),
            explosion: None,
            spawn_times: VecDeque::new(),
            collision_positions: Vec::with_capacity(100),
            base_particle_count: 0,
            center_x: 0.0,
            center_y: 0.0,
            max_explosion_radius: 0.0,
            last_time: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            warmup_frames: 3,
            stopped: false,
            frames_without_motion: 0,
        }
    }

    /// Update explosion state, processing kills and checking completion.
    fn update_explosion(&mut self, dt: f64) {
        if let Some(ref mut exp) = self.explosion {
            exp.update(dt);
            exp.process_kills(&mut self.particles);

            if !exp.active {
                println!(
                    "Explosion complete: killed {}, {} remaining",
                    exp.killed_count,
                    self.particles.len()
                );
                self.explosion = None;
            }
        }
    }

    /// Handle particle spawning from collisions, potentially triggering explosions.
    fn handle_spawning(&mut self, now: Instant, width: u32, height: u32) {
        // Track spawn rate over sliding window
        let spawn_window = std::time::Duration::from_secs_f64(SPAWN_RATE_WINDOW);
        if let Some(cutoff) = now.checked_sub(spawn_window) {
            while self.spawn_times.front().is_some_and(|&t| t < cutoff) {
                self.spawn_times.pop_front();
            }
        }

        // Spawn new particles or trigger explosion
        if !self.collision_positions.is_empty() && self.explosion.is_none() {
            if self.spawn_times.len() >= SPAWN_RATE_THRESHOLD {
                println!(
                    "EXPLOSION! Spawn rate {} per second exceeded threshold, {} total particles",
                    self.spawn_times.len(),
                    self.particles.len()
                );
                self.explosion = Some(Explosion::new(
                    self.center_x,
                    self.center_y,
                    self.max_explosion_radius,
                    self.particles.len(),
                    self.base_particle_count,
                ));
                play_explosion_sound(&self.stream_handle);
                self.spawn_times.clear();
            } else {
                for &(cx, cy) in &self.collision_positions {
                    if self.spawn_at_collision {
                        self.particles.push(Particle::new_at_position(cx, cy));
                    } else {
                        self.particles.push(Particle::new_at_center(width, height));
                    }
                    self.spawn_times.push_back(now);
                }
            }
        }
    }

    /// Check if all particles have stopped moving.
    fn check_motion(&mut self) {
        if self.explosion.is_none() {
            if has_motion(&self.particles) {
                self.frames_without_motion = 0;
            } else {
                self.frames_without_motion += 1;
                if self.frames_without_motion >= MOTION_STOPPED_FRAMES {
                    println!("All motion has stopped");
                    self.stopped = true;
                }
            }
        }
    }

    /// Update FPS counter and print statistics.
    fn update_fps_counter(&mut self) {
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            // Precision loss acceptable: frame_count is small relative to f64 mantissa
            #[allow(clippy::cast_precision_loss)]
            let fps = self.frame_count as f64 / elapsed;
            println!("FPS: {fps:.1}, Particles: {}", self.particles.len());
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }

    fn update_and_render(&mut self) {
        // Get dimensions from render context (if available)
        let Some((width, height)) = self.render.as_ref().map(|r| (r.width(), r.height())) else {
            return;
        };
        let now = Instant::now();

        // Warmup frames for GPU initialization
        if self.warmup_frames > 0 {
            self.warmup_frames -= 1;
            self.last_time = now;
            self.fps_timer = now;
            self.frame_count = 0;

            if let Some(ref mut render) = self.render {
                let particles = &self.particles;
                render.with_frame(|frame| {
                    frame.fill(0);
                    render_particles(frame, particles, width, height);
                });
                render
                    .present()
                    .expect("Failed to render frame during warmup");
                render.window().request_redraw();
            }
            return;
        }

        let dt = now.duration_since(self.last_time).as_secs_f64().min(0.05);
        self.last_time = now;

        // If simulation has stopped, just render the stopped message
        if self.stopped {
            if let Some(ref mut render) = self.render {
                let particles = &self.particles;
                render.with_frame(|frame| {
                    frame.fill(0);
                    render_particles(frame, particles, width, height);
                    render_stopped_message(frame, width, height);
                });
                render.present().expect("Failed to render frame");
                render.window().request_redraw();
            }
            return;
        }

        // Update simulation (no render borrow needed here)
        self.update_explosion(dt);

        let gravity_multiplier = f64::from(self.gravity_percent) / 100.0;
        update_physics(
            &mut self.particles,
            dt,
            width,
            height,
            gravity_multiplier,
            self.wall_elasticity,
        );

        let max_energy = handle_collisions(
            &mut self.particles,
            &mut self.collision_positions,
            self.particle_elasticity,
        );

        if max_energy > 0.0 {
            play_collision_sound(&self.stream_handle, max_energy);
        }

        self.handle_spawning(now, width, height);
        self.check_motion();

        // Render current state (borrow render only for this section)
        if let Some(ref mut render) = self.render {
            let explosion_ref = self.explosion.as_ref();
            let particles = &self.particles;
            let stopped = self.stopped;
            render.with_frame(|frame| {
                frame.fill(0);
                if let Some(exp) = explosion_ref {
                    render_explosion(frame, exp, width, height);
                }
                render_particles(frame, particles, width, height);
                if stopped {
                    render_stopped_message(frame, width, height);
                }
            });
            render.present().expect("Failed to render frame");
            render.window().request_redraw();
        }

        self.update_fps_counter();
    }

    /// Initialize simulation state: particles, geometry, and timers.
    fn init_simulation_state(&mut self, width: u32, height: u32) {
        self.base_particle_count = self
            .min_particles_override
            .unwrap_or_else(|| calculate_particle_count(width, height));
        println!(
            "Base particle count{}: {}",
            if self.min_particles_override.is_some() {
                " (override)"
            } else {
                " for this screen"
            },
            self.base_particle_count
        );

        self.particles = (0..self.base_particle_count)
            .map(|_| Particle::new_random(width, height))
            .collect();

        self.center_x = f64::from(width) / 2.0;
        self.center_y = f64::from(height) / 2.0;
        self.max_explosion_radius = f64::from(width * width + height * height).sqrt();

        self.last_time = Instant::now();
        self.fps_timer = Instant::now();
    }
}

/// Create a GPU render context using ouroboros for safe self-referential struct.
fn try_create_gpu_context(
    window: &Rc<Window>,
    width: u32,
    height: u32,
) -> Result<GpuRenderContext, pixels::Error> {
    GpuRenderContextTryBuilder {
        window: Rc::clone(window),
        width,
        height,
        #[allow(clippy::borrowed_box)]
        pixels_builder: |win: &Rc<Window>| {
            let size = win.inner_size();
            let surface_texture = SurfaceTexture::new(size.width, size.height, win.as_ref());
            Pixels::new(width, height, surface_texture)
        },
    }
    .try_build()
}

/// Create a CPU render context using softbuffer as fallback.
/// `width` and `height` are logical dimensions for the simulation.
/// `physical_width` and `physical_height` are the actual surface dimensions.
fn create_cpu_context(
    window: Rc<Window>,
    width: u32,
    height: u32,
    physical_width: u32,
    physical_height: u32,
) -> CpuRenderContext {
    let context =
        softbuffer::Context::new(Rc::clone(&window)).expect("Failed to create softbuffer context");
    let mut surface = softbuffer::Surface::new(&context, Rc::clone(&window))
        .expect("Failed to create softbuffer surface");
    // Resize to physical dimensions - softbuffer works with actual pixels
    surface
        .resize(
            NonZeroU32::new(physical_width).expect("Width must be > 0"),
            NonZeroU32::new(physical_height).expect("Height must be > 0"),
        )
        .expect("Failed to resize softbuffer surface");
    // Render buffer uses logical dimensions
    let buffer = vec![0u8; (width as usize) * (height as usize) * 4];
    CpuRenderContext {
        window,
        width,
        height,
        physical_width,
        physical_height,
        surface,
        buffer,
    }
}

/// Create render context, trying GPU first with CPU fallback (unless `force_cpu` is set).
/// `width` and `height` are logical dimensions.
/// `physical_width` and `physical_height` are the actual surface dimensions.
fn create_render_context(
    window: &Rc<Window>,
    width: u32,
    height: u32,
    physical_width: u32,
    physical_height: u32,
    force_cpu: bool,
) -> RenderContext {
    if force_cpu {
        println!("Rendering: CPU (softbuffer) [forced]");
        let cpu_ctx = create_cpu_context(
            Rc::clone(window),
            width,
            height,
            physical_width,
            physical_height,
        );
        window.request_redraw();
        return RenderContext::Cpu(cpu_ctx);
    }

    match try_create_gpu_context(window, width, height) {
        Ok(gpu_ctx) => {
            println!("Rendering: GPU (pixels/wgpu)");
            gpu_ctx.borrow_window().request_redraw();
            RenderContext::Gpu(Box::new(gpu_ctx))
        }
        Err(_gpu_error) => {
            println!("GPU unavailable, using CPU rendering");
            let cpu_ctx = create_cpu_context(
                Rc::clone(window),
                width,
                height,
                physical_width,
                physical_height,
            );
            println!("Rendering: CPU (softbuffer)");
            window.request_redraw();
            RenderContext::Cpu(cpu_ctx)
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render.is_some() {
            return; // Already initialized
        }

        // Build window attributes based on requested size or fullscreen
        let window_attrs = Window::default_attributes()
            .with_title("Bouncy Particles - Press SPACE to exit")
            .with_resizable(false);

        let window_attrs = if let (Some(w), Some(h)) = (self.requested_width, self.requested_height)
        {
            // Validate against actual display size
            if let Some(monitor) = event_loop.available_monitors().next() {
                let monitor_size = monitor.size();
                let scale = monitor.scale_factor();
                let max_width = physical_to_logical(monitor_size.width, scale);
                let max_height = physical_to_logical(monitor_size.height, scale);

                if w > max_width || h > max_height {
                    eprintln!(
                        "Error: Requested size {w}x{h} exceeds display size {max_width}x{max_height}"
                    );
                    event_loop.exit();
                    return;
                }
            }
            println!("Window mode: {w}x{h} (fixed size)");
            window_attrs.with_inner_size(LogicalSize::new(w, h))
        } else {
            println!("Window mode: fullscreen");
            window_attrs.with_fullscreen(Some(Fullscreen::Borderless(None)))
        };

        let window = event_loop
            .create_window(window_attrs)
            .expect("Failed to create window");

        let physical_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let width = physical_to_logical(physical_size.width, scale_factor);
        let height = physical_to_logical(physical_size.height, scale_factor);

        println!(
            "Window: {}x{} physical, {}x{} logical, scale={}",
            physical_size.width, physical_size.height, width, height, scale_factor
        );

        self.init_simulation_state(width, height);

        let window = Rc::new(window);
        self.render = Some(create_render_context(
            &window,
            width,
            height,
            physical_size.width,
            physical_size.height,
            self.force_cpu,
        ));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                handle_key(key_code, event_loop);
            }
            WindowEvent::Resized(new_size) => {
                if let Some(ref mut render) = self.render {
                    render.resize_surface(new_size.width, new_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::Key(winit::event::RawKeyEvent {
            physical_key: PhysicalKey::Code(key_code),
            state: ElementState::Pressed,
        }) = event
        {
            handle_key(key_code, event_loop);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref render) = self.render {
            render.window().request_redraw();
        }
    }
}

fn main() {
    let Some(config) = parse_args() else {
        return; // Help was shown or error occurred
    };

    // Print configuration summary
    if config.spawn_at_collision {
        println!("Mode: Spawning particles at collision points");
    } else {
        println!("Mode: Spawning particles at center");
    }
    if config.gravity_percent != 100 {
        println!("Gravity: {}%", config.gravity_percent);
    }
    if (config.wall_elasticity - 1.0).abs() > f64::EPSILON {
        println!("Wall elasticity: {}", config.wall_elasticity);
    }
    if (config.particle_elasticity - 1.0).abs() > f64::EPSILON {
        println!("Particle elasticity: {}", config.particle_elasticity);
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    let _ = event_loop.run_app(&mut app);
}
