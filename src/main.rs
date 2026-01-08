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
use std::collections::VecDeque;
use std::env;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

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
    let total_pixels = width as u64 * height as u64;
    let count = (total_pixels + PIXELS_PER_PARTICLE / 2) / PIXELS_PER_PARTICLE;
    count.max(2) as usize
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
    /// Generate a random velocity vector with speed between 50-100% of INITIAL_VELOCITY.
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
            x: rng.gen_range(PARTICLE_MARGIN..(width as f64 - PARTICLE_MARGIN)),
            y: rng.gen_range(PARTICLE_MARGIN..(height as f64 - PARTICLE_MARGIN)),
            vx,
            vy,
            color: random_bright_color(),
        }
    }

    /// Create a particle at the center of the screen.
    fn new_at_center(width: u32, height: u32) -> Self {
        Self::new_at_position(width as f64 / 2.0, height as f64 / 2.0)
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
    fn update(&mut self, dt: f64, width: u32, height: u32) {
        self.vy += GRAVITY * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        if self.x <= PARTICLE_RADIUS {
            self.x = PARTICLE_RADIUS;
            self.vx = -self.vx;
        } else if self.x >= width as f64 - PARTICLE_RADIUS {
            self.x = width as f64 - PARTICLE_RADIUS;
            self.vx = -self.vx;
        }

        if self.y <= PARTICLE_RADIUS {
            self.y = PARTICLE_RADIUS;
            self.vy = -self.vy;
        } else if self.y >= height as f64 - PARTICLE_RADIUS {
            self.y = height as f64 - PARTICLE_RADIUS;
            self.vy = -self.vy;
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
    let h: f64 = rng.gen_range(0.0..360.0);
    let s: f64 = 0.4;
    let v: f64 = 1.0;

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
        255,
    ]
}

/// Attempt elastic collision between two particles.
/// Returns collision result if particles are touching and approaching.
fn try_elastic_collision(p1: &mut Particle, p2: &mut Particle) -> Option<CollisionResult> {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let dist_sq = dx * dx + dy * dy;

    if dist_sq > PARTICLE_DIAMETER_SQ || dist_sq == 0.0 {
        return None;
    }

    let mid_x = (p1.x + p2.x) / 2.0;
    let mid_y = (p1.y + p2.y) / 2.0;

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

    p1.vx -= dvn * nx;
    p1.vy -= dvn * ny;
    p2.vx += dvn * nx;
    p2.vy += dvn * ny;

    let overlap = PARTICLE_DIAMETER - dist;
    if overlap > 0.0 {
        let sep = overlap / 2.0 + SEPARATION_PADDING;
        p1.x -= sep * nx;
        p1.y -= sep * ny;
        p2.x += sep * nx;
        p2.y += sep * ny;
    }

    Some(CollisionResult { energy, mid_x, mid_y })
}

/// Generate a ping sound with exponential decay at the given frequency.
fn generate_ping(frequency: f32) -> impl Source<Item = f32> + Send {
    let duration_samples = (AUDIO_SAMPLE_RATE as u64 * PING_DURATION_MS / 1000) as usize;

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / AUDIO_SAMPLE_RATE as f32;
            let envelope = (-t * 20.0).exp();
            let wave = (2.0 * std::f32::consts::PI * frequency * t).sin();
            wave * envelope * 0.3
        })
        .collect();

    rodio::buffer::SamplesBuffer::new(1, AUDIO_SAMPLE_RATE, samples)
}

/// Generate a low-frequency rumble sound for explosions.
fn generate_explosion() -> impl Source<Item = f32> + Send {
    let duration_samples = (AUDIO_SAMPLE_RATE as u64 * EXPLOSION_DURATION_MS / 1000) as usize;
    let mut rng = rand::thread_rng();

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / AUDIO_SAMPLE_RATE as f32;
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
        let cx = particle.x as i32;
        let cy = particle.y as i32;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let px = cx + dx;
                let py = cy + dy;

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

    let min_x = (exp.x - outer_radius).max(0.0) as u32;
    let max_x = ((exp.x + outer_radius).ceil() as u32).min(width);
    let min_y = (exp.y - outer_radius).max(0.0) as u32;
    let max_y = ((exp.y + outer_radius).ceil() as u32).min(height);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = x as f64 - exp.x;
            let dy = y as f64 - exp.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist >= inner_radius && dist <= outer_radius {
                let intensity = 1.0 - ((dist - inner_radius) / EXPLOSION_RING_WIDTH).abs();
                let brightness = (intensity * 255.0) as u8;
                let idx = (y * width + x) as usize * 4;
                frame[idx] = brightness;
                frame[idx + 1] = (brightness as f32 * 0.6) as u8;
                frame[idx + 2] = (brightness as f32 * 0.2) as u8;
                frame[idx + 3] = 255;
            }
        }
    }
}

/// Update all particle positions with physics simulation.
fn update_physics(particles: &mut [Particle], dt: f64, width: u32, height: u32) {
    for particle in particles {
        particle.update(dt, width, height);
    }
}

/// Process all particle-particle collisions.
/// Returns the maximum collision energy and populates collision_positions.
fn handle_collisions(
    particles: &mut Vec<Particle>,
    collision_positions: &mut Vec<(f64, f64)>,
) -> f64 {
    collision_positions.clear();
    let mut max_energy = 0.0f64;
    let n = particles.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let (left, right) = particles.split_at_mut(j);
            let p1 = &mut left[i];
            let p2 = &mut right[0];

            if let Some(result) = try_elastic_collision(p1, p2) {
                max_energy = max_energy.max(result.energy);
                collision_positions.push((result.mid_x, result.mid_y));
            }
        }
    }

    max_energy
}

fn print_usage() {
    println!("Usage: bouncy [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --spawn-at-collision  Spawn new particles at collision points instead of center");
    println!("  --help                Show this help message");
    println!();
    println!("Controls:");
    println!("  Space, Escape, Q      Exit the program");
}

/// Self-referential struct: pixels borrows from window
#[self_referencing]
struct RenderContext {
    window: Box<Window>,
    width: u32,
    height: u32,
    #[borrows(window)]
    #[covariant]
    pixels: Pixels<'this>,
}

/// Main application state for the particle simulation.
struct App {
    // Configuration
    spawn_at_collision: bool,

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
}

impl App {
    /// Create a new App with the given configuration.
    fn new(spawn_at_collision: bool) -> Self {
        let (audio_stream, stream_handle) =
            OutputStream::try_default().expect("Failed to create audio output stream");

        App {
            spawn_at_collision,
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
        }
    }

    fn handle_key(&self, key_code: KeyCode, event_loop: &ActiveEventLoop) {
        if matches!(key_code, KeyCode::Space | KeyCode::Escape | KeyCode::KeyQ) {
            event_loop.exit();
        }
    }

    fn update_and_render(&mut self) {
        let Some(ref mut render) = self.render else {
            return;
        };

        let width = *render.borrow_width();
        let height = *render.borrow_height();
        let now = Instant::now();

        // Warmup frames for GPU initialization
        if self.warmup_frames > 0 {
            self.warmup_frames -= 1;
            self.last_time = now;
            self.fps_timer = now;
            self.frame_count = 0;

            render.with_pixels_mut(|pixels| {
                let frame = pixels.frame_mut();
                frame.fill(0);
                render_particles(frame, &self.particles, width, height);
                pixels.render().expect("Failed to render frame during warmup");
            });
            render.borrow_window().request_redraw();
            return;
        }

        let dt = now.duration_since(self.last_time).as_secs_f64().min(0.05);
        self.last_time = now;

        // Update explosion
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

        // Physics
        update_physics(&mut self.particles, dt, width, height);

        // Collisions
        let max_energy = handle_collisions(&mut self.particles, &mut self.collision_positions);

        if max_energy > 0.0 {
            play_collision_sound(&self.stream_handle, max_energy);
        }

        // Spawn rate tracking
        let cutoff = now - std::time::Duration::from_secs_f64(SPAWN_RATE_WINDOW);
        while self.spawn_times.front().map_or(false, |&t| t < cutoff) {
            self.spawn_times.pop_front();
        }

        // Spawning
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

        // Render
        let explosion_ref = self.explosion.as_ref();
        render.with_pixels_mut(|pixels| {
            let frame = pixels.frame_mut();
            frame.fill(0);

            if let Some(exp) = explosion_ref {
                render_explosion(frame, exp, width, height);
            }

            render_particles(frame, &self.particles, width, height);
            pixels.render().expect("Failed to render frame");
        });

        // FPS counter
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            println!(
                "FPS: {:.1}, Particles: {}",
                self.frame_count as f64 / elapsed,
                self.particles.len()
            );
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }

        render.borrow_window().request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render.is_some() {
            return; // Already initialized
        }

        let window = Box::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Bouncy Particles - Press SPACE to exit")
                        .with_fullscreen(Some(Fullscreen::Borderless(None))),
                )
                .expect("Failed to create window"),
        );

        let physical_size = window.inner_size();
        let scale_factor = window.scale_factor();

        let width = (physical_size.width as f64 / scale_factor) as u32;
        let height = (physical_size.height as f64 / scale_factor) as u32;

        println!(
            "Window: {}x{} physical, {}x{} logical, scale={}",
            physical_size.width, physical_size.height, width, height, scale_factor
        );

        self.base_particle_count = calculate_particle_count(width, height);
        println!(
            "Base particle count for this screen: {}",
            self.base_particle_count
        );

        self.particles = (0..self.base_particle_count)
            .map(|_| Particle::new_random(width, height))
            .collect();

        self.center_x = width as f64 / 2.0;
        self.center_y = height as f64 / 2.0;
        self.max_explosion_radius = ((width * width + height * height) as f64).sqrt();

        self.last_time = Instant::now();
        self.fps_timer = Instant::now();

        // Use ouroboros builder to create self-referential struct
        let render = RenderContextBuilder {
            window,
            width,
            height,
            pixels_builder: |window: &Box<Window>| {
                let size = window.inner_size();
                let surface_texture = SurfaceTexture::new(size.width, size.height, window.as_ref());
                Pixels::new(width, height, surface_texture).expect("Failed to create pixels")
            },
        }
        .build();

        render.borrow_window().request_redraw();
        self.render = Some(render);
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
                self.handle_key(key_code, event_loop);
            }
            WindowEvent::Resized(new_size) => {
                if let Some(ref mut render) = self.render {
                    render.with_pixels_mut(|pixels| {
                        pixels
                            .resize_surface(new_size.width, new_size.height)
                            .expect("Failed to resize surface");
                    });
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
            self.handle_key(key_code, event_loop);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref render) = self.render {
            render.borrow_window().request_redraw();
        }
    }
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mut spawn_at_collision = false;

    for arg in &args[1..] {
        match arg.as_str() {
            "--spawn-at-collision" => spawn_at_collision = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                print_usage();
                return;
            }
        }
    }

    if spawn_at_collision {
        println!("Mode: Spawning particles at collision points");
    } else {
        println!("Mode: Spawning particles at center");
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(spawn_at_collision);
    event_loop.run_app(&mut app).expect("Event loop error");
}
