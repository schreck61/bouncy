// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! The headless simulation core: particles, spawning, explosions, and their
//! orchestration. No windowing, rendering, or audio — the `App` layer owns
//! those and drives this struct, which keeps every gameplay rule testable.

use crate::config::Config;
use crate::explosion::{max_radius_from, Explosion, EXPLOSION_KILL_RATIO, SPAWN_RATE_WINDOW};
use crate::physics::{
    apply_attractor, handle_collisions, has_motion, substep_count, update_physics,
    CollisionRecorder, Particle, SpatialGrid, MOTION_STOPPED_FRAMES, WELL_STRENGTH,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::time::Instant;

/// Screen area (in logical pixels) per initial particle.
const PIXELS_PER_PARTICLE: u64 = 375_000;
/// Particles spawned by a left click.
const CLICK_BURST_SIZE: usize = 10;
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

/// Calculate the initial/minimum particle count based on screen size.
pub fn calculate_particle_count(width: u32, height: u32) -> usize {
    let total_pixels = u64::from(width) * u64::from(height);
    let count = (total_pixels + PIXELS_PER_PARTICLE / 2) / PIXELS_PER_PARTICLE;
    // Safe: count is always small (screen pixels / 375000)
    usize::try_from(count.max(2)).unwrap_or(2)
}

/// What happened during one simulation step, for the caller to react to
/// (sound effects, logging). The simulation itself has no audio dependency.
#[derive(Default)]
pub struct StepEvents {
    /// Highest collision energy this step (0.0 if no collisions).
    pub max_collision_energy: f64,
    /// Stereo pan (0.0 = left, 1.0 = right) of this step's collisions.
    pub collision_pan: f64,
    /// An automatic explosion was triggered this step.
    pub explosion_started: bool,
}

/// The cursor gravity well as seen by the simulation for one step.
#[derive(Copy, Clone)]
pub struct Well {
    pub x: f64,
    pub y: f64,
    /// 1 = attract, -1 = repel.
    pub direction: i8,
}

/// Headless particle simulation.
pub struct Simulation {
    // Geometry (fixed at creation)
    width: u32,
    height: u32,
    particle_radius: f64,

    // Runtime-tunable parameters
    pub spawn_at_collision: bool,
    pub gravity_percent: i32,
    pub wall_elasticity: f64,
    pub particle_elasticity: f64,
    /// Spawns/sec that trigger an automatic explosion; 0 = never.
    pub explosion_threshold: usize,

    // State
    particles: Vec<Particle>,
    explosion: Option<Explosion>,
    spawn_times: VecDeque<Instant>,
    collisions: CollisionRecorder,
    grid: SpatialGrid,
    rng: StdRng,
    stopped: bool,
    frames_without_motion: u32,

    // Derived values
    base_particle_count: usize,
    /// Density-based population cap for this window size.
    max_particles: usize,
    center_x: f64,
    center_y: f64,
}

impl Simulation {
    /// Create a simulation for a window of `width` x `height` logical pixels.
    pub fn new(config: &Config, width: u32, height: u32) -> Self {
        let rng = config
            .seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);

        let base_particle_count = config
            .min_particles
            .map_or_else(|| calculate_particle_count(width, height), |n| n as usize);
        println!(
            "Base particle count{}: {base_particle_count}",
            if config.min_particles.is_some() {
                " (override)"
            } else {
                " for this screen"
            },
        );

        // Cap the population at ~20% area coverage: one particle per four
        // diameter-squared tiles of window area.
        let diameter = config.particle_size * 2.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let density_cap =
            (f64::from(width) * f64::from(height) / (4.0 * diameter * diameter)) as usize;
        let max_particles = density_cap
            .clamp(MIN_PARTICLE_CAP, MAX_PARTICLES)
            .max(base_particle_count);

        let mut sim = Simulation {
            width,
            height,
            particle_radius: config.particle_size,
            spawn_at_collision: config.spawn_at_collision,
            gravity_percent: config.gravity,
            wall_elasticity: config.wall_elasticity,
            particle_elasticity: config.particle_elasticity,
            explosion_threshold: config.explosion_threshold as usize,
            particles: Vec::new(),
            explosion: None,
            spawn_times: VecDeque::new(),
            collisions: CollisionRecorder::new(),
            grid: SpatialGrid::new(),
            rng,
            stopped: false,
            frames_without_motion: 0,
            base_particle_count,
            max_particles,
            center_x: f64::from(width) / 2.0,
            center_y: f64::from(height) / 2.0,
        };
        sim.populate();
        sim
    }

    fn populate(&mut self) {
        self.particles = (0..self.base_particle_count)
            .map(|_| Particle::new_random(&mut self.rng, self.width, self.height))
            .collect();
    }

    /// Reset to the initial population and clear all transient state.
    pub fn reset(&mut self) {
        self.populate();
        self.explosion = None;
        self.spawn_times.clear();
        self.collisions.clear();
        self.stopped = false;
        self.frames_without_motion = 0;
    }

    // --- Accessors -------------------------------------------------------

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn explosion(&self) -> Option<&Explosion> {
        self.explosion.as_ref()
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn particle_radius(&self) -> f64 {
        self.particle_radius
    }

    // --- Interaction ------------------------------------------------------

    /// Leave the stopped state (something is about to inject motion).
    pub fn wake(&mut self) {
        self.stopped = false;
        self.frames_without_motion = 0;
    }

    /// Spawn a burst of particles around `(x, y)` (left click), spread over
    /// a small disc so they don't materialize inside each other.
    pub fn spawn_burst(&mut self, x: f64, y: f64) {
        let spread = SPAWN_JITTER.max(self.particle_radius * 4.0);
        for _ in 0..CLICK_BURST_SIZE {
            if self.particles.len() >= self.max_particles {
                break;
            }
            let bx = x + self.rng.random_range(-spread..spread);
            let by = y + self.rng.random_range(-spread..spread);
            self.particles
                .push(Particle::new_at_position(&mut self.rng, bx, by));
        }
        // Fresh particles are moving; leave the stopped state if we were in it.
        self.wake();
    }

    /// Trigger a cursor explosion at `(x, y)`: kills everything the ring
    /// reaches, down to the simulation's hard minimum. Returns false if an
    /// explosion is already active or there is nothing to explode.
    pub fn trigger_manual_explosion(&mut self, x: f64, y: f64) -> bool {
        if self.explosion.is_some() || self.particles.is_empty() {
            return false;
        }
        self.trigger_explosion(x, y, 1.0, MANUAL_EXPLOSION_SURVIVORS);
        true
    }

    /// Trigger an explosion centered at `(x, y)`, dooming a `kill_ratio`
    /// share of particles while leaving at least `min_survivors` alive.
    fn trigger_explosion(&mut self, x: f64, y: f64, kill_ratio: f64, min_survivors: usize) {
        let max_radius = max_radius_from(x, y, self.width, self.height);
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
        self.spawn_times.clear();
    }

    // --- Stepping ----------------------------------------------------------

    /// Advance the simulation by `dt` seconds. `now` anchors the sliding
    /// spawn-rate window; `well` is the cursor gravity well, if held.
    pub fn step(&mut self, dt: f64, now: Instant, well: Option<Well>) -> StepEvents {
        let mut events = StepEvents::default();
        if self.stopped {
            return events;
        }

        self.update_explosion(dt);

        let gravity_multiplier = f64::from(self.gravity_percent) / 100.0;
        self.collisions.clear();
        let substeps = substep_count(&self.particles, dt, self.particle_radius);
        let sub_dt = dt / f64::from(substeps);

        let mut max_energy = 0.0f64;
        for _ in 0..substeps {
            if let Some(w) = well {
                apply_attractor(
                    &mut self.particles,
                    w.x,
                    w.y,
                    f64::from(w.direction) * WELL_STRENGTH,
                    sub_dt,
                );
            }
            update_physics(
                &mut self.particles,
                sub_dt,
                self.width,
                self.height,
                self.particle_radius,
                gravity_multiplier,
                self.wall_elasticity,
            );
            let energy = handle_collisions(
                &mut self.particles,
                &mut self.grid,
                &mut self.collisions,
                self.width,
                self.height,
                self.particle_radius,
                self.particle_elasticity,
            );
            max_energy = max_energy.max(energy);
        }

        events.max_collision_energy = max_energy;
        events.collision_pan = self
            .collision_centroid()
            .map_or(0.5, |(x, _)| x / f64::from(self.width));

        events.explosion_started = self.handle_spawning(now);
        self.check_motion(well.is_some());
        events
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

    /// Handle particle spawning from collisions, potentially triggering an
    /// automatic explosion. Returns true if one was triggered.
    fn handle_spawning(&mut self, now: Instant) -> bool {
        // Track spawn rate over sliding window
        let spawn_window = std::time::Duration::from_secs_f64(SPAWN_RATE_WINDOW);
        if let Some(cutoff) = now.checked_sub(spawn_window) {
            while self.spawn_times.front().is_some_and(|&t| t < cutoff) {
                self.spawn_times.pop_front();
            }
        }

        if self.collisions.is_empty() || self.explosion.is_some() {
            return false;
        }

        // Explode instead of spawning once the window fills up.
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
            self.trigger_explosion(ex, ey, EXPLOSION_KILL_RATIO, self.base_particle_count);
            return true;
        }

        let diameter = self.particle_radius * 2.0;
        let jitter = SPAWN_JITTER.max(diameter);
        // Grid state from the last collision pass covers exactly the
        // pre-spawn particles; particles spawned this frame are checked
        // separately below.
        let first_new = self.particles.len();
        let mut spawned = 0;
        for i in 0..self.collisions.positions().len() {
            if spawned >= MAX_SPAWNS_PER_FRAME || self.particles.len() >= self.max_particles {
                break;
            }
            let (bx, by) = if self.spawn_at_collision {
                self.collisions.positions()[i]
            } else {
                (self.center_x, self.center_y)
            };
            let x = bx + self.rng.random_range(-jitter..jitter);
            let y = by + self.rng.random_range(-jitter..jitter);

            // Spawn only into free space. Materializing a particle inside
            // another re-collides instantly and spawns again: a self-feeding
            // cascade that inflates the spawn rate far beyond what the
            // moving particles actually produce.
            let blocked = self
                .grid
                .any_within(&self.particles[..first_new], x, y, diameter)
                || self.particles[first_new..]
                    .iter()
                    .any(|p| p.distance_squared_from(x, y) < diameter * diameter);
            if blocked {
                continue;
            }

            self.particles
                .push(Particle::new_at_position(&mut self.rng, x, y));
            self.spawn_times.push_back(now);
            spawned += 1;
        }
        false
    }

    /// Average position of this step's collisions.
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

    /// Check if all particles have stopped moving. An active gravity well is
    /// about to move them, so it suppresses the check.
    fn check_motion(&mut self, well_active: bool) {
        if self.explosion.is_some() || well_active {
            return;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::time::Duration;

    /// Build a Config from CLI-style args (always sized 800x600, seeded).
    fn config(extra: &[&str]) -> Config {
        let mut args = vec!["bouncy", "--seed", "12345"];
        args.extend_from_slice(extra);
        Config::try_parse_from(args).expect("test config must parse")
    }

    fn sim(extra: &[&str]) -> Simulation {
        Simulation::new(&config(extra), 800, 600)
    }

    /// Freeze all particles and disable outside influences.
    fn freeze(sim: &mut Simulation) {
        sim.gravity_percent = 0;
        for p in &mut sim.particles {
            p.vx = 0.0;
            p.vy = 0.0;
        }
    }

    /// Arrange particles 0 and 1 on a deterministic head-on collision course
    /// at `(x, y)` (they collide within one small step).
    fn arm_collision(sim: &mut Simulation, x: f64, y: f64) {
        assert!(sim.particle_count() >= 2);
        freeze(sim);
        // Park spare particles far away so they never interfere; the arena
        // is 800x600 and the collision site is placed well away from walls.
        for (i, p) in sim.particles.iter_mut().enumerate().skip(2) {
            #[allow(clippy::cast_precision_loss)]
            let offset = i as f64 * 10.0;
            p.x = 790.0;
            p.y = 10.0 + offset;
        }
        sim.particles[0].x = x - 2.0;
        sim.particles[0].y = y;
        sim.particles[0].vx = 100.0;
        sim.particles[1].x = x + 2.0;
        sim.particles[1].y = y;
        sim.particles[1].vx = -100.0;
    }

    #[test]
    fn new_populates_base_count_and_density_cap() {
        let s = sim(&[]);
        assert_eq!(s.particle_count(), calculate_particle_count(800, 600));
        // 800*600 / (4 * 3^2) = 13333 for the default radius 1.5.
        assert_eq!(s.max_particles, 13333);

        let s = sim(&["--min-particles", "30"]);
        assert_eq!(s.particle_count(), 30);
    }

    #[test]
    fn particle_count_scales_with_screen_size() {
        assert_eq!(calculate_particle_count(100, 100), 2); // minimum enforced
        assert_eq!(calculate_particle_count(1920, 1080), 6);
        assert_eq!(calculate_particle_count(3840, 2160), 22);
    }

    /// Re-arm the same collision until a spawn lands. The free-space rule
    /// legitimately rejects spawn positions that fall inside the colliding
    /// pair, so a single collision guarantees an *attempt*, not a spawn.
    /// Returns the events of the step that spawned.
    fn step_until_spawn(s: &mut Simulation, site: (f64, f64), now: Instant) -> StepEvents {
        let before = s.particle_count();
        for _ in 0..50 {
            arm_collision(s, site.0, site.1);
            let events = s.step(0.01, now, None);
            assert!(
                events.max_collision_energy > 0.0,
                "the armed pair must collide every step"
            );
            if s.particle_count() > before {
                return events;
            }
        }
        panic!("a spawn must land within 50 armed collisions");
    }

    #[test]
    fn collision_spawns_a_particle_and_reports_energy() {
        let mut s = sim(&["--min-particles", "2"]);
        let now = Instant::now();
        let events = step_until_spawn(&mut s, (400.0, 300.0), now);
        assert!(events.max_collision_energy > 0.0);
        assert_eq!(s.particle_count(), 3, "one successful spawn");
        assert_eq!(s.spawn_times.len(), 1);
        // Pan reflects the collision site (x=400 of 800 => center).
        assert!((events.collision_pan - 0.5).abs() < 0.05);
    }

    #[test]
    fn spawn_window_prunes_entries_older_than_one_second() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        let now = Instant::now();
        let old = now - Duration::from_millis(1500);
        let recent = now - Duration::from_millis(500);
        s.spawn_times.extend([old, old, recent]);

        s.step(0.001, now, None);
        assert_eq!(
            s.spawn_times.len(),
            1,
            "entries beyond the 1s window must be pruned"
        );
    }

    #[test]
    fn explosion_triggers_at_threshold() {
        let mut s = sim(&["--min-particles", "30", "--explosion-threshold", "5"]);
        arm_collision(&mut s, 400.0, 300.0);
        let now = Instant::now();
        for _ in 0..5 {
            s.spawn_times.push_back(now);
        }

        let events = s.step(0.01, now, None);
        assert!(events.explosion_started);
        assert!(s.explosion().is_some());
        assert!(s.spawn_times.is_empty(), "spawn window resets on explosion");
    }

    #[test]
    fn below_threshold_spawns_instead_of_exploding() {
        let mut s = sim(&["--min-particles", "30", "--explosion-threshold", "5"]);
        let now = Instant::now();
        // 4 of 5 window slots used: the next spawn must not explode.
        for _ in 0..4 {
            s.spawn_times.push_back(now);
        }

        step_until_spawn(&mut s, (400.0, 300.0), now);
        assert!(s.explosion().is_none());
        assert_eq!(s.particle_count(), 31);
        assert_eq!(s.spawn_times.len(), 5);
    }

    #[test]
    fn threshold_zero_never_explodes() {
        let mut s = sim(&["--min-particles", "2", "--explosion-threshold", "0"]);
        let now = Instant::now();
        // A spawn window far beyond any positive threshold.
        for _ in 0..5000 {
            s.spawn_times.push_back(now);
        }

        step_until_spawn(&mut s, (400.0, 300.0), now);
        assert!(s.explosion().is_none());
        assert_eq!(s.particle_count(), 3, "spawning continues instead");
    }

    #[test]
    fn explosion_centers_on_collision_hotspot_in_collision_mode() {
        let site = (200.0, 150.0);
        let mut s = sim(&[
            "--min-particles",
            "30",
            "--explosion-threshold",
            "5",
            "--spawn-at-collision",
        ]);
        arm_collision(&mut s, site.0, site.1);
        let now = Instant::now();
        for _ in 0..5 {
            s.spawn_times.push_back(now);
        }

        s.step(0.01, now, None);
        let exp = s.explosion().expect("explosion must trigger");
        assert!(
            (exp.x - site.0).abs() < 10.0 && (exp.y - site.1).abs() < 10.0,
            "explosion at ({}, {}) should be near the collision site {site:?}",
            exp.x,
            exp.y
        );
    }

    #[test]
    fn spawns_are_blocked_in_crowded_space() {
        let mut s = sim(&["--min-particles", "2"]);
        arm_collision(&mut s, 400.0, 300.0);
        // Pack the neighborhood of the collision site solid: any candidate
        // spawn position within the jitter radius overlaps something.
        for gx in 0..13 {
            for gy in 0..13 {
                let mut p = Particle::new_at_position(
                    &mut s.rng,
                    388.0 + f64::from(gx) * 2.0,
                    288.0 + f64::from(gy) * 2.0,
                );
                p.vx = 0.0;
                p.vy = 0.0;
                s.particles.push(p);
            }
        }
        let before = s.particle_count();

        let events = s.step(0.005, Instant::now(), None);
        assert!(events.max_collision_energy > 0.0, "collisions still happen");
        assert_eq!(
            s.spawn_times.len(),
            0,
            "no spawn may be recorded when every position is occupied"
        );
        assert_eq!(s.particle_count(), before);
    }

    #[test]
    fn population_respects_density_cap() {
        let mut s = sim(&["--min-particles", "2"]);
        s.max_particles = 15;
        for _ in 0..5 {
            s.spawn_burst(400.0, 300.0);
        }
        assert_eq!(s.particle_count(), 15, "burst spawning stops at the cap");
    }

    #[test]
    fn manual_explosion_wipes_to_the_floor_and_completes() {
        let mut s = sim(&["--min-particles", "50"]);
        assert!(s.trigger_manual_explosion(400.0, 300.0));
        assert!(
            !s.trigger_manual_explosion(400.0, 300.0),
            "no explosion while one is active"
        );

        let now = Instant::now();
        let mut steps = 0;
        while s.explosion().is_some() {
            s.step(0.05, now, None);
            steps += 1;
            assert!(steps < 100, "explosion must complete");
        }
        assert_eq!(s.particle_count(), 2, "manual blast leaves 2 survivors");
    }

    #[test]
    fn motion_detection_declares_stopped_and_wake_recovers() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        let now = Instant::now();
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, None);
        }
        assert!(s.stopped());

        // A stopped simulation ignores steps...
        let events = s.step(0.01, now, None);
        assert!(events.max_collision_energy == 0.0);

        // ...until something wakes it.
        s.spawn_burst(100.0, 100.0);
        assert!(!s.stopped());
    }

    #[test]
    fn active_well_suppresses_motion_detection_and_moves_particles() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[0].x = 100.0;
        s.particles[0].y = 300.0;
        s.particles[1].x = 700.0;
        s.particles[1].y = 300.0;

        let well = Well {
            x: 400.0,
            y: 300.0,
            direction: 1,
        };
        let now = Instant::now();
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, Some(well));
        }
        assert!(!s.stopped(), "held well must suppress the stopped state");
        assert!(
            s.particles[0].vx > 0.0,
            "left particle accelerates toward the well"
        );
        assert!(
            s.particles[1].vx < 0.0,
            "right particle accelerates toward the well"
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut s = sim(&["--min-particles", "10"]);
        s.spawn_burst(400.0, 300.0);
        s.trigger_manual_explosion(400.0, 300.0);
        s.stopped = true;
        s.spawn_times.push_back(Instant::now());

        s.reset();
        assert_eq!(s.particle_count(), 10);
        assert!(s.explosion().is_none());
        assert!(!s.stopped());
        assert!(s.spawn_times.is_empty());
    }

    #[test]
    fn seeded_simulations_are_reproducible() {
        let mut a = sim(&["--min-particles", "20"]);
        let mut b = sim(&["--min-particles", "20"]);
        let now = Instant::now();
        for _ in 0..100 {
            a.step(1.0 / 120.0, now, None);
            b.step(1.0 / 120.0, now, None);
        }
        assert_eq!(a.particle_count(), b.particle_count());
        for (pa, pb) in a.particles().iter().zip(b.particles()) {
            assert!((pa.x - pb.x).abs() < 1e-12);
            assert!((pa.y - pb.y).abs() < 1e-12);
        }
    }
}
