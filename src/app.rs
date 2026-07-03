// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Application state, input handling, and the winit event loop glue.

use crate::audio::Audio;
use crate::config::{ColorMode, Config};
use crate::explosion::{max_radius_from, Explosion, EXPLOSION_KILL_RATIO, SPAWN_RATE_WINDOW};
use crate::physics::{
    apply_attractor, handle_collisions, has_motion, substep_count, update_physics,
    CollisionRecorder, Particle, SpatialGrid, MOTION_STOPPED_FRAMES, WELL_STRENGTH,
};
use crate::render::{
    create_render_context, fade_frame, render_explosion, render_particles, RenderContext,
};
use crate::text::{draw_text, draw_text_centered};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

/// Screen area (in logical pixels) per initial particle.
const PIXELS_PER_PARTICLE: u64 = 375_000;
/// Particles spawned by a left click.
const CLICK_BURST_SIZE: usize = 10;
/// Frames of physics skipped at startup to let the GPU initialize.
const WARMUP_FRAMES: u32 = 3;
/// Runtime gravity adjustment step (percent) for Up/Down arrows.
const GRAVITY_STEP: i32 = 10;
/// Runtime elasticity adjustment step for arrow and bracket keys.
const ELASTICITY_STEP: f64 = 0.05;
/// Runtime explosion-threshold adjustment step for the -/= keys.
const THRESHOLD_STEP: usize = 5;
const THRESHOLD_MAX: usize = 1000;
/// Multiplicative step for time-scale adjustment (comma/period).
const TIME_SCALE_STEP: f64 = 1.25;
const TIME_SCALE_MIN: f64 = 0.1;
const TIME_SCALE_MAX: f64 = 4.0;
/// Simulated time for a single frame-step while paused (N key).
const FRAME_STEP_DT: f64 = 1.0 / 120.0;
/// Consecutive failed presents before giving up (a few seconds at 60 FPS).
const MAX_RENDER_FAILURES: u32 = 300;
/// Seconds of cursor inactivity before the cursor is hidden.
const CURSOR_HIDE_DELAY: f64 = 2.0;
/// Absolute population ceiling. The effective cap is usually lower: spawning
/// stops when particles would occupy ~20% of the window area (a jammed,
/// solid-packed window is neither interesting nor fast). Only reachable with
/// automatic explosions disabled (--explosion-threshold 0).
const MAX_PARTICLES: usize = 100_000;
/// Floor for the density-based cap so small windows still allow bursts.
const MIN_PARTICLE_CAP: usize = 1000;
/// Spawn throttle per frame. In a dense cluster the collision count is
/// quadratic in cluster size, so unthrottled spawning can multiply the
/// population by orders of magnitude in a single frame.
const MAX_SPAWNS_PER_FRAME: usize = 200;
/// Random offset applied to collision-point spawns so new particles never
/// stack at an identical position (identical positions defeat both collision
/// separation and the spatial grid).
const SPAWN_JITTER: f64 = 4.0;
/// Minimum survivors of a cursor-triggered explosion. Unlike automatic
/// explosions (which keep the screen-based base count alive), a manual blast
/// wipes everything it touches down to the simulation's hard minimum.
const MANUAL_EXPLOSION_SURVIVORS: usize = 2;

const HUD_FONT_SIZE: f32 = 16.0;
const HUD_LINE_HEIGHT: f32 = 20.0;
const HUD_MARGIN: f32 = 8.0;
const HUD_COLOR: [u8; 3] = [200, 200, 200];
const MESSAGE_COLOR: [u8; 3] = [255, 100, 100];

/// HUD overlay state, cycled with the H key.
#[derive(Copy, Clone, PartialEq, Eq)]
enum HudMode {
    Hidden,
    Stats,
    StatsAndKeys,
}

impl HudMode {
    fn next(self) -> Self {
        match self {
            HudMode::Hidden => HudMode::Stats,
            HudMode::Stats => HudMode::StatsAndKeys,
            HudMode::StatsAndKeys => HudMode::Hidden,
        }
    }
}

/// Calculate the initial/minimum particle count based on screen size.
fn calculate_particle_count(width: u32, height: u32) -> usize {
    let total_pixels = u64::from(width) * u64::from(height);
    let count = (total_pixels + PIXELS_PER_PARTICLE / 2) / PIXELS_PER_PARTICLE;
    // Safe: count is always small (screen pixels / 375000)
    usize::try_from(count.max(2)).unwrap_or(2)
}

/// Convert physical pixels to logical pixels given a scale factor.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn physical_to_logical(physical: u32, scale_factor: f64) -> u32 {
    (f64::from(physical) / scale_factor) as u32
}

/// Main application state for the particle simulation.
pub struct App {
    // Configuration
    spawn_at_collision: bool,
    min_particles_override: Option<usize>,
    gravity_percent: i32,
    wall_elasticity: f64,
    particle_elasticity: f64,
    requested_width: Option<u32>,
    requested_height: Option<u32>,
    force_cpu: bool,
    trails: bool,
    particle_radius: f64,
    color_mode: ColorMode,
    verbose: bool,

    // Subsystems
    audio: Audio,
    rng: StdRng,

    // Window and rendering (initialized on resume)
    render: Option<RenderContext>,
    scale_factor: f64,

    // Simulation state
    particles: Vec<Particle>,
    explosion: Option<Explosion>,
    spawn_times: VecDeque<Instant>,
    collisions: CollisionRecorder,
    grid: SpatialGrid,

    // Derived values
    base_particle_count: usize,
    /// Density-based population cap for this window size.
    max_particles: usize,
    center_x: f64,
    center_y: f64,

    // Timing
    last_time: Instant,
    frame_count: u64,
    fps_timer: Instant,
    warmup_frames: u32,
    current_fps: f64,

    // Interaction state
    stopped: bool,
    frames_without_motion: u32,
    paused: bool,
    step_once: bool,
    hud_mode: HudMode,
    cursor_x: f64,
    cursor_y: f64,
    /// Cursor gravity well: 0 = off, 1 = attract (G held), -1 = repel (Shift+G).
    well_direction: i8,
    shift_down: bool,
    time_scale: f64,
    /// Cursor auto-hide state: what visibility we last set on the window.
    cursor_visible: bool,
    last_cursor_move: Instant,
    window_focused: bool,
    cursor_inside: bool,
    /// Spawns/sec that trigger an automatic explosion; 0 = never.
    explosion_threshold: usize,

    // Render failure tracking (transient surface loss should not crash)
    consecutive_render_failures: u32,
}

impl App {
    /// Create a new App with the given configuration.
    pub fn new(config: Config) -> Self {
        let rng = config
            .seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);

        App {
            spawn_at_collision: config.spawn_at_collision,
            min_particles_override: config.min_particles.map(|n| n as usize),
            gravity_percent: config.gravity,
            wall_elasticity: config.wall_elasticity,
            particle_elasticity: config.particle_elasticity,
            requested_width: config.width,
            requested_height: config.height,
            force_cpu: config.cpu,
            trails: config.trails,
            particle_radius: config.particle_size,
            color_mode: config.color_mode,
            verbose: config.verbose,
            audio: Audio::new(config.mute),
            rng,
            render: None,
            scale_factor: 1.0,
            particles: Vec::new(),
            explosion: None,
            spawn_times: VecDeque::new(),
            collisions: CollisionRecorder::new(),
            grid: SpatialGrid::new(),
            base_particle_count: 0,
            max_particles: MAX_PARTICLES,
            center_x: 0.0,
            center_y: 0.0,
            last_time: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            warmup_frames: WARMUP_FRAMES,
            current_fps: 0.0,
            stopped: false,
            frames_without_motion: 0,
            paused: false,
            step_once: false,
            hud_mode: HudMode::Hidden,
            cursor_x: 0.0,
            cursor_y: 0.0,
            well_direction: 0,
            shift_down: false,
            time_scale: 1.0,
            cursor_visible: true,
            last_cursor_move: Instant::now(),
            window_focused: true,
            cursor_inside: false,
            explosion_threshold: config.explosion_threshold as usize,
            consecutive_render_failures: 0,
        }
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
            .map(|_| Particle::new_random(&mut self.rng, width, height))
            .collect();

        // Cap the population at ~20% area coverage: one particle per four
        // diameter-squared tiles of window area.
        let diameter = self.particle_radius * 2.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let density_cap =
            (f64::from(width) * f64::from(height) / (4.0 * diameter * diameter)) as usize;
        self.max_particles = density_cap
            .clamp(MIN_PARTICLE_CAP, MAX_PARTICLES)
            .max(self.base_particle_count);

        self.center_x = f64::from(width) / 2.0;
        self.center_y = f64::from(height) / 2.0;

        self.explosion = None;
        self.spawn_times.clear();
        self.collisions.clear();
        self.stopped = false;
        self.frames_without_motion = 0;
        self.last_time = Instant::now();
        self.fps_timer = Instant::now();
        self.frame_count = 0;
    }

    /// Reset the simulation to its initial state (R key).
    fn reset(&mut self) {
        if let Some((width, height)) = self.dimensions() {
            println!("Simulation reset");
            self.init_simulation_state(width, height);
            self.paused = false;
        }
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.render.as_ref().map(|r| (r.width(), r.height()))
    }

    /// Trigger an explosion centered at `(x, y)`, dooming a `kill_ratio`
    /// share of particles while leaving at least `min_survivors` alive.
    fn trigger_explosion(
        &mut self,
        x: f64,
        y: f64,
        width: u32,
        height: u32,
        kill_ratio: f64,
        min_survivors: usize,
    ) {
        let max_radius = max_radius_from(x, y, width, height);
        let explosion = Explosion::new(
            &mut self.rng,
            x,
            y,
            max_radius,
            &mut self.particles,
            kill_ratio,
            min_survivors,
        );
        println!(
            "Explosion will kill {} of {} particles",
            explosion.doomed_count,
            self.particles.len()
        );
        self.explosion = Some(explosion);
        self.audio.play_explosion();
        self.spawn_times.clear();
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
        if !self.collisions.is_empty() && self.explosion.is_none() {
            if self.explosion_threshold > 0 && self.spawn_times.len() >= self.explosion_threshold {
                println!(
                    "EXPLOSION! Spawn rate {} per second exceeded threshold, {} total particles",
                    self.spawn_times.len(),
                    self.particles.len()
                );
                // In collision-spawning mode the action is wherever particles
                // are densest; center the blast on the recent collisions.
                let (ex, ey) = if self.spawn_at_collision {
                    self.collision_centroid()
                        .unwrap_or((self.center_x, self.center_y))
                } else {
                    (self.center_x, self.center_y)
                };
                self.trigger_explosion(
                    ex,
                    ey,
                    width,
                    height,
                    EXPLOSION_KILL_RATIO,
                    self.base_particle_count,
                );
            } else {
                use rand::Rng;
                let spawn_count = self
                    .collisions
                    .positions()
                    .len()
                    .min(MAX_SPAWNS_PER_FRAME)
                    .min(self.max_particles.saturating_sub(self.particles.len()));
                for i in 0..spawn_count {
                    let (cx, cy) = self.collisions.positions()[i];
                    let particle = if self.spawn_at_collision {
                        let jx = self.rng.random_range(-SPAWN_JITTER..SPAWN_JITTER);
                        let jy = self.rng.random_range(-SPAWN_JITTER..SPAWN_JITTER);
                        Particle::new_at_position(&mut self.rng, cx + jx, cy + jy)
                    } else {
                        Particle::new_at_center(&mut self.rng, width, height)
                    };
                    self.particles.push(particle);
                    self.spawn_times.push_back(now);
                }
            }
        }
    }

    /// Average position of this frame's collisions.
    fn collision_centroid(&self) -> Option<(f64, f64)> {
        let positions = self.collisions.positions();
        if positions.is_empty() {
            return None;
        }
        let (sx, sy) = positions
            .iter()
            .fold((0.0, 0.0), |(ax, ay), (x, y)| (ax + x, ay + y));
        #[allow(clippy::cast_precision_loss)]
        let n = positions.len() as f64;
        Some((sx / n, sy / n))
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

    /// Update FPS counter and (in verbose mode) print statistics.
    fn update_fps_counter(&mut self) {
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            // Precision loss acceptable: frame_count is small relative to f64 mantissa
            #[allow(clippy::cast_precision_loss)]
            let fps = self.frame_count as f64 / elapsed;
            self.current_fps = fps;
            if self.verbose {
                println!("FPS: {fps:.1}, Particles: {}", self.particles.len());
            }
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }

    /// Advance the simulation by `dt` seconds.
    fn simulate(&mut self, dt: f64, now: Instant, width: u32, height: u32) {
        self.update_explosion(dt);

        let gravity_multiplier = f64::from(self.gravity_percent) / 100.0;
        self.collisions.clear();
        let substeps = substep_count(&self.particles, dt, self.particle_radius);
        let sub_dt = dt / f64::from(substeps);

        let mut max_energy = 0.0f64;
        for _ in 0..substeps {
            if self.well_direction != 0 {
                apply_attractor(
                    &mut self.particles,
                    self.cursor_x,
                    self.cursor_y,
                    f64::from(self.well_direction) * WELL_STRENGTH,
                    sub_dt,
                );
            }
            update_physics(
                &mut self.particles,
                sub_dt,
                width,
                height,
                self.particle_radius,
                gravity_multiplier,
                self.wall_elasticity,
            );
            let energy = handle_collisions(
                &mut self.particles,
                &mut self.grid,
                &mut self.collisions,
                width,
                height,
                self.particle_radius,
                self.particle_elasticity,
            );
            max_energy = max_energy.max(energy);
        }

        if max_energy > 0.0 {
            // Pan the ping toward where the collisions happened.
            let pan = self
                .collision_centroid()
                .map_or(0.5, |(x, _)| x / f64::from(width));
            #[allow(clippy::cast_possible_truncation)]
            self.audio.play_ping(max_energy, pan as f32);
        }

        self.handle_spawning(now, width, height);
        self.check_motion();
    }

    /// Build the HUD overlay text for the current mode.
    fn hud_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!("FPS: {:.1}", self.current_fps),
            format!("Particles: {}", self.particles.len()),
            format!("Gravity: {}%  (Up/Down)", self.gravity_percent),
            format!(
                "Elasticity: particle {:.2} (Left/Right) / wall {:.2} ([/])",
                self.particle_elasticity, self.wall_elasticity
            ),
            format!("Time scale: {:.2}x  (,/.)", self.time_scale),
            if self.explosion_threshold == 0 {
                "Explosions: off  (-/=)".to_string()
            } else {
                format!(
                    "Explosions: at {}/s spawn rate  (-/=)",
                    self.explosion_threshold
                )
            },
        ];

        let mut flags = Vec::new();
        if self.paused {
            flags.push("PAUSED");
        }
        if self.audio.is_muted() {
            flags.push("MUTED");
        }
        if self.stopped {
            flags.push("STOPPED");
        }
        if self.well_direction > 0 {
            flags.push("WELL: ATTRACT");
        } else if self.well_direction < 0 {
            flags.push("WELL: REPEL");
        }
        if !flags.is_empty() {
            lines.push(flags.join("  "));
        }

        if self.hud_mode == HudMode::StatsAndKeys {
            lines.push(String::new());
            for key_line in [
                "P pause   N step   R reset   M mute",
                "T trails   C colors   B spawn mode",
                "G hold: gravity well (Shift+G repels)",
                "Click: burst   Right-click: explosion",
                "H cycle HUD   Space/Esc/Q quit",
            ] {
                lines.push(key_line.to_string());
            }
        }

        lines
    }

    /// Draw the current frame and present it, tolerating transient failures.
    fn render_frame(&mut self, width: u32, height: u32) {
        let hud_lines = if self.hud_mode == HudMode::Hidden {
            None
        } else {
            Some(self.hud_lines())
        };

        let Some(ref mut render) = self.render else {
            return;
        };

        let explosion_ref = self.explosion.as_ref();
        let particles = &self.particles;
        let trails = self.trails;
        let radius = self.particle_radius;
        let color_mode = self.color_mode;
        let stopped = self.stopped;
        let paused = self.paused;

        render.with_frame(|frame| {
            if trails {
                fade_frame(frame);
            } else {
                frame.fill(0);
            }
            if let Some(exp) = explosion_ref {
                render_explosion(frame, exp, width, height);
            }
            render_particles(frame, particles, width, height, radius, color_mode);
            if stopped {
                draw_text_centered(frame, width, height, "STOPPED", 72.0, MESSAGE_COLOR);
            } else if paused {
                draw_text_centered(frame, width, height, "PAUSED", 72.0, MESSAGE_COLOR);
            }
            if let Some(ref lines) = hud_lines {
                draw_hud(frame, width, height, lines);
            }
        });

        match render.present() {
            Ok(()) => self.consecutive_render_failures = 0,
            Err(e) => {
                // Transient surface loss (display sleep, mode change) should
                // not crash; give up only if it never recovers.
                self.consecutive_render_failures += 1;
                eprintln!("Render failed ({e}); skipping frame");
            }
        }
    }

    /// Apply the cursor auto-hide policy, changing OS cursor state only on
    /// transitions.
    fn update_cursor_visibility(&mut self) {
        let idle = self.last_cursor_move.elapsed().as_secs_f64();
        let desired = cursor_should_be_visible(
            self.window_focused,
            self.cursor_inside,
            self.well_direction != 0,
            idle,
        );
        if desired != self.cursor_visible {
            if let Some(ref render) = self.render {
                render.window().set_cursor_visible(desired);
            }
            self.cursor_visible = desired;
        }
    }

    fn update_and_render(&mut self) {
        self.update_cursor_visibility();
        let Some((width, height)) = self.dimensions() else {
            return;
        };
        let now = Instant::now();

        // Warmup frames for GPU initialization
        if self.warmup_frames > 0 {
            self.warmup_frames -= 1;
            self.last_time = now;
            self.fps_timer = now;
            self.frame_count = 0;
            self.render_frame(width, height);
            return;
        }

        let dt = now.duration_since(self.last_time).as_secs_f64().min(0.05);
        self.last_time = now;

        if !self.stopped && !self.paused {
            self.simulate(dt * self.time_scale, now, width, height);
        } else if self.paused && self.step_once {
            // Frame-step (N while paused): advance one fixed-size step.
            self.simulate(FRAME_STEP_DT * self.time_scale, now, width, height);
        }
        self.step_once = false;

        self.render_frame(width, height);
        self.update_fps_counter();
    }

    /// Spawn a burst of particles at the cursor (left click).
    fn spawn_burst(&mut self, x: f64, y: f64) {
        for _ in 0..CLICK_BURST_SIZE {
            if self.particles.len() >= self.max_particles {
                break;
            }
            self.particles
                .push(Particle::new_at_position(&mut self.rng, x, y));
        }
        // Fresh particles are moving; leave the stopped state if we were in it.
        self.stopped = false;
        self.frames_without_motion = 0;
    }

    /// Handle a key press or release.
    fn handle_key(
        &mut self,
        key_code: KeyCode,
        state: ElementState,
        repeat: bool,
        event_loop: &ActiveEventLoop,
    ) {
        if state == ElementState::Released {
            if key_code == KeyCode::KeyG {
                self.well_direction = 0;
            }
            return;
        }

        match key_code {
            KeyCode::Space | KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
            KeyCode::KeyP if !repeat => {
                self.paused = !self.paused;
                println!("{}", if self.paused { "Paused" } else { "Resumed" });
            }
            KeyCode::KeyN if self.paused => self.step_once = true,
            KeyCode::KeyR if !repeat => self.reset(),
            KeyCode::KeyM if !repeat => {
                let muted = self.audio.toggle_mute();
                println!("Audio {}", if muted { "muted" } else { "unmuted" });
            }
            KeyCode::KeyH if !repeat => self.hud_mode = self.hud_mode.next(),
            KeyCode::KeyT if !repeat => {
                self.trails = !self.trails;
                println!("Trails {}", if self.trails { "on" } else { "off" });
            }
            KeyCode::KeyC if !repeat => {
                self.color_mode = match self.color_mode {
                    ColorMode::Solid => ColorMode::Velocity,
                    ColorMode::Velocity => ColorMode::Solid,
                };
            }
            KeyCode::KeyB if !repeat => {
                self.spawn_at_collision = !self.spawn_at_collision;
                println!(
                    "Spawning at {}",
                    if self.spawn_at_collision {
                        "collision points"
                    } else {
                        "center"
                    }
                );
            }
            KeyCode::KeyG => {
                self.well_direction = if self.shift_down { -1 } else { 1 };
                // The well moves particles; leave the stopped state if set.
                self.stopped = false;
                self.frames_without_motion = 0;
            }
            KeyCode::ArrowUp => {
                self.gravity_percent = (self.gravity_percent + GRAVITY_STEP).min(1000);
                println!("Gravity: {}%", self.gravity_percent);
            }
            KeyCode::ArrowDown => {
                self.gravity_percent = (self.gravity_percent - GRAVITY_STEP).max(-1000);
                println!("Gravity: {}%", self.gravity_percent);
            }
            KeyCode::ArrowRight => {
                self.particle_elasticity = (self.particle_elasticity + ELASTICITY_STEP).min(1.5);
                println!("Particle elasticity: {:.2}", self.particle_elasticity);
            }
            KeyCode::ArrowLeft => {
                self.particle_elasticity = (self.particle_elasticity - ELASTICITY_STEP).max(0.0);
                println!("Particle elasticity: {:.2}", self.particle_elasticity);
            }
            KeyCode::BracketRight => {
                self.wall_elasticity = (self.wall_elasticity + ELASTICITY_STEP).min(1.5);
                println!("Wall elasticity: {:.2}", self.wall_elasticity);
            }
            KeyCode::BracketLeft => {
                self.wall_elasticity = (self.wall_elasticity - ELASTICITY_STEP).max(0.0);
                println!("Wall elasticity: {:.2}", self.wall_elasticity);
            }
            KeyCode::Period => {
                self.time_scale = (self.time_scale * TIME_SCALE_STEP).min(TIME_SCALE_MAX);
                println!("Time scale: {:.2}x", self.time_scale);
            }
            KeyCode::Comma => {
                self.time_scale = (self.time_scale / TIME_SCALE_STEP).max(TIME_SCALE_MIN);
                println!("Time scale: {:.2}x", self.time_scale);
            }
            KeyCode::Equal => {
                self.explosion_threshold =
                    (self.explosion_threshold + THRESHOLD_STEP).min(THRESHOLD_MAX);
                println!("Explosion threshold: {}/s", self.explosion_threshold);
            }
            KeyCode::Minus => {
                self.explosion_threshold = self.explosion_threshold.saturating_sub(THRESHOLD_STEP);
                if self.explosion_threshold == 0 {
                    println!("Explosion threshold: off");
                } else {
                    println!("Explosion threshold: {}/s", self.explosion_threshold);
                }
            }
            _ => {}
        }
    }

    /// Handle a mouse button press at the current cursor position.
    fn handle_mouse(&mut self, button: MouseButton) {
        let Some((width, height)) = self.dimensions() else {
            return;
        };
        let (x, y) = (self.cursor_x, self.cursor_y);
        match button {
            MouseButton::Left => self.spawn_burst(x, y),
            MouseButton::Right if self.explosion.is_none() && !self.particles.is_empty() => {
                println!("Explosion triggered at cursor");
                self.trigger_explosion(x, y, width, height, 1.0, MANUAL_EXPLOSION_SURVIVORS);
            }
            _ => {}
        }
    }
}

/// Cursor auto-hide policy: visible while the window is unfocused, while the
/// cursor is outside the window, while the gravity well is engaged, or until
/// `CURSOR_HIDE_DELAY` seconds have passed since the last movement.
fn cursor_should_be_visible(
    focused: bool,
    inside: bool,
    well_active: bool,
    idle_secs: f64,
) -> bool {
    !focused || !inside || well_active || idle_secs < CURSOR_HIDE_DELAY
}

/// Draw the HUD overlay lines top-left. Free function so it can run inside
/// the frame closure while the render context is mutably borrowed.
fn draw_hud(frame: &mut [u8], width: u32, height: u32, lines: &[String]) {
    for (i, line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let y = HUD_MARGIN + i as f32 * HUD_LINE_HEIGHT;
        draw_text(
            frame,
            width,
            height,
            line,
            HUD_FONT_SIZE,
            (HUD_MARGIN, y),
            HUD_COLOR,
        );
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
        self.scale_factor = scale_factor;

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
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                self.handle_key(key_code, state, repeat, event_loop);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.shift_down = modifiers.state().shift_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x / self.scale_factor;
                self.cursor_y = position.y / self.scale_factor;
                self.cursor_inside = true;
                self.last_cursor_move = Instant::now();
            }
            WindowEvent::CursorEntered { .. } => {
                self.cursor_inside = true;
                self.last_cursor_move = Instant::now();
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_inside = false;
            }
            WindowEvent::Focused(focused) => {
                self.window_focused = focused;
                if focused {
                    // Fresh focus should not instantly hide the cursor.
                    self.last_cursor_move = Instant::now();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                self.handle_mouse(button);
            }
            WindowEvent::Resized(new_size) => {
                if let Some(ref mut render) = self.render {
                    render.resize_surface(new_size.width, new_size.height);
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                // The window is not resizable, but DPI can change when it
                // moves between monitors. The simulation keeps its original
                // logical size; log the change deliberately. (winit also
                // fires this at window creation with the initial factor.)
                if (scale_factor - self.scale_factor).abs() > f64::EPSILON {
                    println!(
                        "Scale factor changed: {} -> {scale_factor} (simulation keeps original size)",
                        self.scale_factor
                    );
                }
                self.scale_factor = scale_factor;
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if self.consecutive_render_failures >= MAX_RENDER_FAILURES {
                    eprintln!("Rendering failed repeatedly; exiting");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref render) = self.render {
            render.window().request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Restore the cursor on every exit path (exit keys, window close,
        // repeated render failures) so it is never left hidden.
        if let Some(ref render) = self.render {
            render.window().set_cursor_visible(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_count_scales_with_screen_size() {
        assert_eq!(calculate_particle_count(100, 100), 2); // minimum enforced
        assert_eq!(calculate_particle_count(1920, 1080), 6);
        assert_eq!(calculate_particle_count(3840, 2160), 22);
    }

    #[test]
    fn physical_to_logical_conversion() {
        assert_eq!(physical_to_logical(3024, 2.0), 1512);
        assert_eq!(physical_to_logical(1920, 1.0), 1920);
    }

    #[test]
    fn cursor_visibility_policy() {
        let recently = CURSOR_HIDE_DELAY - 0.1;
        let idle = CURSOR_HIDE_DELAY + 0.1;

        // Hidden only when focused, inside, well off, and idle.
        assert!(!cursor_should_be_visible(true, true, false, idle));

        // Any of these forces visibility.
        assert!(cursor_should_be_visible(true, true, false, recently)); // moving
        assert!(cursor_should_be_visible(true, true, true, idle)); // well held
        assert!(cursor_should_be_visible(false, true, false, idle)); // unfocused
        assert!(cursor_should_be_visible(true, false, false, idle)); // outside window
    }
}
