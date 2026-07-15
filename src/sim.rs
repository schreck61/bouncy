// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! The headless simulation core: particles, spawning, explosions, and their
//! orchestration. No windowing, rendering, or audio — the `App` layer owns
//! those and drives this struct, which keeps every gameplay rule testable.

use crate::config::{Config, SpawnMode};
use crate::explosion::{EXPLOSION_KILL_RATIO, Explosion, SPAWN_RATE_WINDOW, max_radius_from};
use crate::physics::{
    COLLISION_ENERGY_NORMALIZER, CollisionRecorder, MOTION_STOPPED_FRAMES, Particle, ParticleId,
    Segment, SpatialGrid, SpawnSite, WELL_STRENGTH, apply_attractor, apply_flow,
    apply_self_gravity, collide_with_segments, handle_collisions, has_motion, max_radius,
    motion_crosses_segment, pair_mut, substep_count, update_physics,
};
use crate::presets::Scene;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
// std::time::Instant on native; performance.now() on wasm.
use web_time::Instant;

/// Screen area (in logical pixels) per initial particle.
const PIXELS_PER_PARTICLE: u64 = 375_000;
/// Particles spawned by a left click.
const CLICK_BURST_SIZE: usize = 10;
/// Absolute population ceiling, the flat part of the non-linear cap: the
/// coverage-based limit is the binding constraint for large particles
/// (geometry — a jammed, solid-packed window is neither interesting nor
/// fast), but for small ones it allows populations that are pure noise to
/// look at and heavy to simulate long before coverage binds. 12k keeps
/// the benchmarked worst case — a fully packed self-gravitating clump
/// with matter off, where nothing consumes sustained contacts — above
/// 20 FPS (16k measured ~14 FPS there). On a fullscreen window the
/// crossover between the two regimes lands near particle size 3.
const MAX_PARTICLES: usize = 12_000;
/// Floor for the density-based cap so small windows still allow bursts.
const MIN_PARTICLE_CAP: usize = 1000;
/// Spawn throttle per frame. In a dense cluster the collision count is
/// quadratic in cluster size, so unthrottled spawning can multiply the
/// population by orders of magnitude in a single frame.
const MAX_SPAWNS_PER_FRAME: usize = 200;
/// Ceiling on spawn *attempts* per frame (clearance probes, successful or
/// not). A packed clump yields tens of thousands of collision sites whose
/// clearance all fails; probing each one made grid queries the hottest
/// path at ~24k particles. Twice the spawn cap leaves generous headroom
/// for scenes where spawning actually succeeds.
const MAX_SPAWN_ATTEMPTS_PER_FRAME: usize = 2 * MAX_SPAWNS_PER_FRAME;
/// Seconds a population must sit pegged at the density cap before the
/// saturation valve fires an explosion. Long enough that brushing the cap
/// during normal growth doesn't blast; short enough that a genuinely
/// gridlocked arena visibly relieves itself.
const SATURATION_EXPLOSION_SECS: f64 = 3.0;
/// Random offset applied to collision-point spawns so new particles never
/// stack at an identical position (identical positions defeat both collision
/// separation and the spatial grid).
const SPAWN_JITTER: f64 = 4.0;
/// Half-angle of the ejection cone around the outward spawn direction.
const SPAWN_CONE_HALF_ANGLE: f64 = std::f64::consts::FRAC_PI_4;
/// Ejection speed at zero collision energy, as a fraction of
/// the initial speed; full energy ejects at the full initial speed.
const SPAWN_SPEED_MIN_FRACTION: f64 = 0.25;

// Matter mechanics (--matter / X key): collision energy decides the outcome.
/// At or below this closing speed, touching particles fuse into one.
const FUSION_MAX_ENERGY: f64 = 60.0;
/// At or above this closing speed, the colliding particles shatter.
const FISSION_MIN_ENERGY: f64 = 700.0;
/// Smallest fragment radius, as a fraction of the base particle radius;
/// particles too small to yield fragments this size bounce instead.
const MIN_RADIUS_FACTOR: f64 = 0.5;
/// Largest fused radius, as a multiple of the base particle radius. A blob
/// that would exceed it absorbs only up to the cap (partial fusion) and the
/// smaller particle's remainder survives.
const MAX_RADIUS_FACTOR: f64 = 6.0;
/// Fragment separation speed as a fraction of the collision energy.
const FISSION_KICK_FRACTION: f64 = 0.3;

/// Maximum number of pinned gravity wells (the W key and --wells stop here;
/// keep the clap range on --wells in sync).
pub const MAX_PINNED_WELLS: usize = 16;
/// Maximum number of drawn wall segments (held V). Enough for elaborate
/// marble runs while keeping the per-substep segment sweep cheap.
pub const MAX_WALL_SEGMENTS: usize = 200;
/// Comet size as a multiple of the base particle radius; mass scales with
/// the square, so a comet outweighs a default particle 9:1.
const COMET_RADIUS_FACTOR: f64 = 3.0;
/// Comet speed as a multiple of the configured initial speed.
const COMET_SPEED_FACTOR: f64 = 2.0;

/// Peak acceleration of each pinned well: half the held well's strength, so
/// a constellation of pinned wells shapes orbits without overpowering the
/// cursor well or flinging everything into the walls.
const PINNED_WELL_STRENGTH: f64 = WELL_STRENGTH / 2.0;

/// Calculate the initial/minimum particle count based on screen size.
pub fn calculate_particle_count(width: u32, height: u32) -> usize {
    let total_pixels = u64::from(width) * u64::from(height);
    let count = (total_pixels + PIXELS_PER_PARTICLE / 2) / PIXELS_PER_PARTICLE;
    // Safe: count is always small (screen pixels / 375000)
    usize::try_from(count.max(2)).unwrap_or(2)
}

/// What happened during one simulation step, for the caller to react to
/// (sound effects, logging). The simulation itself has no audio or
/// terminal dependency — everything user-facing is reported here.
#[derive(Default)]
pub struct StepEvents {
    /// Highest collision energy this step (0.0 if no collisions).
    pub max_collision_energy: f64,
    /// Stereo pan (0.0 = left, 1.0 = right) of this step's collisions.
    pub collision_pan: f64,
    /// An automatic explosion was triggered this step.
    pub explosion_started: bool,
    /// An explosion finished this step; carries its total kill count.
    pub explosion_completed: Option<usize>,
    /// The simulation declared all motion stopped this step (fires once
    /// per transition into the stopped state).
    pub motion_stopped: bool,
}

pub use crate::physics::Polarity;

/// A gravity well: the transient cursor well (held G) passed into each
/// step, or a persistent pinned well (W key) stored in the simulation.
#[derive(Copy, Clone)]
pub struct Well {
    pub x: f64,
    pub y: f64,
    pub polarity: Polarity,
}

/// Headless particle simulation.
pub struct Simulation {
    // Geometry (fixed at creation)
    width: u32,
    height: u32,
    /// Radius of newly spawned particles and the initial population; matter
    /// mechanics diversify sizes between `MIN`/`MAX_RADIUS_FACTOR` of this.
    base_radius: f64,
    /// Top speed of newly created particles (they start at 50-100% of it).
    initial_speed: f64,

    // Runtime-tunable parameters
    pub spawn_mode: SpawnMode,
    pub gravity_percent: i32,
    pub wall_elasticity: f64,
    pub particle_elasticity: f64,
    /// Spawns/sec that trigger an automatic explosion; 0 = never.
    pub explosion_threshold: usize,
    /// Fusion/fission mechanics enabled.
    pub matter: bool,
    /// Ambient flow field enabled.
    pub flow: bool,
    /// Mutual particle gravity enabled (mass attracts mass).
    pub self_gravity: bool,

    // State
    particles: Vec<Particle>,
    /// Persistent gravity wells pinned with the W key (or --wells).
    pinned_wells: Vec<Well>,
    /// Static wall segments drawn with the V key.
    segments: Vec<Segment>,
    /// Next stable particle id to stamp (monotonic, never reused).
    next_particle_id: ParticleId,
    /// Attractors pinned at startup (--wells); reset restores this layout.
    initial_wells: usize,
    /// Preset scene geometry (window fractions); placed at creation and
    /// restored on reset.
    scene: Scene,
    explosion: Option<Explosion>,
    spawn_times: VecDeque<Instant>,
    /// When the population last became pegged at `max_particles`, for the
    /// saturation valve; `None` whenever it is below the cap.
    saturated_since: Option<Instant>,
    collisions: CollisionRecorder,
    grid: SpatialGrid,
    /// Scratch: particle centers captured at the start of each substep,
    /// feeding the swept segment test. Reused to avoid reallocation; only
    /// filled while wall segments exist.
    prev_positions: Vec<(f64, f64)>,
    rng: StdRng,
    stopped: bool,
    frames_without_motion: u32,
    /// Accumulated simulated time; drives the flow field's drift.
    sim_time: f64,

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

        let max_particles =
            Self::compute_max_particles(width, height, config.particle_size, base_particle_count);

        let mut sim = Simulation {
            width,
            height,
            base_radius: config.particle_size,
            initial_speed: config.initial_speed,
            spawn_mode: config.effective_spawn_mode(),
            gravity_percent: config.gravity,
            wall_elasticity: config.wall_elasticity,
            particle_elasticity: config.particle_elasticity,
            explosion_threshold: config.explosion_threshold,
            matter: config.matter,
            flow: config.flow,
            self_gravity: config.self_gravity,
            particles: Vec::new(),
            pinned_wells: Vec::new(),
            segments: Vec::new(),
            next_particle_id: 1,
            initial_wells: config.wells as usize,
            scene: config.scene.clone(),
            explosion: None,
            spawn_times: VecDeque::new(),
            saturated_since: None,
            collisions: CollisionRecorder::new(),
            grid: SpatialGrid::new(),
            prev_positions: Vec::new(),
            rng,
            stopped: false,
            frames_without_motion: 0,
            sim_time: 0.0,
            base_particle_count,
            max_particles,
            center_x: f64::from(width) / 2.0,
            center_y: f64::from(height) / 2.0,
        };
        sim.populate();
        sim.place_initial_wells();
        sim.place_scene();
        sim
    }

    /// Pin the startup attractors (--wells): a single well sits at the
    /// screen center, several spread evenly on a circle around it.
    fn place_initial_wells(&mut self) {
        let count = self.initial_wells.min(MAX_PINNED_WELLS);
        if count == 1 {
            self.pinned_wells.push(Well {
                x: self.center_x,
                y: self.center_y,
                polarity: Polarity::Attract,
            });
            return;
        }
        let ring = f64::from(self.width.min(self.height)) / 4.0;
        for i in 0..count {
            #[allow(clippy::cast_precision_loss)]
            let angle =
                std::f64::consts::TAU * i as f64 / count as f64 - std::f64::consts::FRAC_PI_2;
            self.pinned_wells.push(Well {
                x: self.center_x + ring * angle.cos(),
                y: self.center_y + ring * angle.sin(),
                polarity: Polarity::Attract,
            });
        }
    }

    /// Place the preset scene's geometry, scaling window-fraction
    /// coordinates to this window. Runs at creation and on reset, after
    /// the --wells ring; both share the same population caps.
    fn place_scene(&mut self) {
        let (w, h) = (f64::from(self.width), f64::from(self.height));
        for well in &self.scene.wells {
            if self.pinned_wells.len() >= MAX_PINNED_WELLS {
                break;
            }
            self.pinned_wells.push(Well {
                x: well.x * w,
                y: well.y * h,
                polarity: well.polarity,
            });
        }
        for wall in &self.scene.walls {
            if self.segments.len() >= MAX_WALL_SEGMENTS {
                break;
            }
            self.segments.push(Segment {
                x1: wall.x1 * w,
                y1: wall.y1 * h,
                x2: wall.x2 * w,
                y2: wall.y2 * h,
            });
        }
    }

    fn populate(&mut self) {
        self.particles.clear();
        for _ in 0..self.base_particle_count {
            let p = Particle::new_random(
                &mut self.rng,
                self.width,
                self.height,
                self.base_radius,
                self.initial_speed,
            );
            let p = self.stamp(p);
            self.particles.push(p);
        }
    }

    /// Assign the next stable id to a freshly constructed particle. Every
    /// particle entering the population passes through here, so ids are
    /// unique for the lifetime of the simulation and features may keep
    /// cross-frame references by id where Vec indices would dangle.
    fn stamp(&mut self, mut particle: Particle) -> Particle {
        particle.id = self.next_particle_id;
        self.next_particle_id += 1;
        particle
    }

    /// Find the current index of the particle with `id`, if it is still
    /// alive. A linear scan: intended for occasional lookups (user
    /// interactions, future spring endpoints), not per-substep work.
    /// No in-app caller yet — this is the id-resolution API that
    /// cross-frame features build on; tests exercise it.
    #[allow(dead_code)]
    pub fn find_particle(&self, id: ParticleId) -> Option<usize> {
        self.particles.iter().position(|p| p.id == id)
    }

    /// Reset to the initial population and clear all transient state,
    /// restoring the startup layout (--wells ring and preset scene
    /// geometry); runtime-pinned wells and hand-drawn walls are erased.
    pub fn reset(&mut self) {
        self.populate();
        self.pinned_wells.clear();
        self.place_initial_wells();
        self.segments.clear();
        self.place_scene();
        self.explosion = None;
        self.spawn_times.clear();
        self.saturated_since = None;
        self.collisions.clear();
        self.stopped = false;
        self.frames_without_motion = 0;
    }

    /// The population cap for a window: ~20% area coverage (one particle
    /// per four diameter-squared tiles) or the flat `MAX_PARTICLES`
    /// ceiling, whichever is lower — coverage binds for large particles,
    /// the ceiling for small ones. Shared by construction and `resize`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn compute_max_particles(
        width: u32,
        height: u32,
        particle_size: f64,
        base_particle_count: usize,
    ) -> usize {
        let diameter = particle_size * 2.0;
        let density_cap =
            (f64::from(width) * f64::from(height) / (4.0 * diameter * diameter)) as usize;
        density_cap
            .clamp(MIN_PARTICLE_CAP, MAX_PARTICLES)
            .max(base_particle_count)
    }

    /// Resize the arena. Positions, pinned wells, and drawn walls rescale
    /// proportionally — for scene geometry (stored in window fractions)
    /// that is exactly where re-placement would put it, and hand-built
    /// constructions keep their composition. Velocities are untouched.
    /// The population cap is recomputed for the new area; a shrink can
    /// leave the population above the new cap, in which case spawning
    /// halts and, with explosions enabled, the saturation valve relieves
    /// the overcrowding. An in-flight explosion keeps its progress with
    /// its reach recomputed from the new far corner.
    pub fn resize(&mut self, width: u32, height: u32) {
        // The CLI's minimum window is 100 px; hold resize to the same
        // floor so degenerate layouts cannot collapse the arena.
        let width = width.max(100);
        let height = height.max(100);
        if width == self.width && height == self.height {
            return;
        }
        let sx = f64::from(width) / f64::from(self.width);
        let sy = f64::from(height) / f64::from(self.height);
        self.width = width;
        self.height = height;
        self.center_x = f64::from(width) / 2.0;
        self.center_y = f64::from(height) / 2.0;
        for p in &mut self.particles {
            p.x *= sx;
            p.y *= sy;
        }
        for w in &mut self.pinned_wells {
            w.x *= sx;
            w.y *= sy;
        }
        for s in &mut self.segments {
            s.x1 *= sx;
            s.y1 *= sy;
            s.x2 *= sx;
            s.y2 *= sy;
        }
        if let Some(ref mut exp) = self.explosion {
            exp.x *= sx;
            exp.y *= sy;
            exp.max_radius = max_radius_from(exp.x, exp.y, width, height);
        }
        self.max_particles =
            Self::compute_max_particles(width, height, self.base_radius, self.base_particle_count);
        // A resize is motion: rescaled positions must integrate and
        // re-collide even if the world had declared STOPPED.
        self.wake();
    }

    // --- Accessors -------------------------------------------------------

    /// The live particles, in a stable order the renderer can iterate.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Current population size.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Births in the sliding window — collision spawns plus fission
    /// fragments over the last second. This is the rate the explosion
    /// threshold is compared against.
    pub fn birth_rate(&self) -> usize {
        self.spawn_times.len()
    }

    /// The population cap for this window size and particle radius.
    pub fn max_particles(&self) -> usize {
        self.max_particles
    }

    /// The explosion ring in progress, if one is expanding.
    pub fn explosion(&self) -> Option<&Explosion> {
        self.explosion.as_ref()
    }

    /// The persistent gravity wells (W pins them; distinct from the
    /// transient held-G well, which is per-frame input).
    pub fn pinned_wells(&self) -> &[Well] {
        &self.pinned_wells
    }

    /// The drawn wall segments particles bounce off (V draws them).
    pub fn wall_segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The initial and minimum particle count (screen-derived, or the
    /// --min-particles override).
    pub fn base_particle_count(&self) -> usize {
        self.base_particle_count
    }

    /// Top speed newly created particles are given (they start at 50-100%
    /// of it); also normalizes the velocity color mode.
    pub fn initial_speed(&self) -> f64 {
        self.initial_speed
    }

    /// Arena size in simulation (logical) pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether all motion has ceased (velocities below threshold for
    /// ~1 second); cleared by [`Simulation::wake`].
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    // --- Interaction ------------------------------------------------------

    /// Leave the stopped state (something is about to inject motion).
    pub fn wake(&mut self) {
        self.stopped = false;
        self.frames_without_motion = 0;
    }

    /// Spawn a burst of particles around `(x, y)` (left click).
    ///
    /// The burst must not collide with itself: a self-collision registers
    /// like any other and, in center spawn mode, teleports a phantom spawn
    /// to the screen center the instant the user clicks elsewhere. Two
    /// rules guarantee that. Placements are clearance-checked against the
    /// local neighborhood and each other, so nothing materializes
    /// overlapping; and every particle is ejected radially away from the
    /// click with speed increasing with distance, a pure expansion field
    /// in which pairwise distances never shrink — siblings cannot converge
    /// no matter which frame boundaries fall where. (Random headings, the
    /// previous behavior, closed the placement margin within one frame.)
    ///
    /// The placement disc widens when the first tries are occupied, and a
    /// particle whose every candidate is blocked is skipped (a click into
    /// solid-packed space adds fewer than the full burst).
    pub fn spawn_burst(&mut self, x: f64, y: f64) {
        /// Placement tries per particle; the disc grows to ~3x base spread.
        const BURST_PLACEMENT_ATTEMPTS: usize = 16;
        let base_spread = SPAWN_JITTER.max(self.base_radius * 4.0);
        let margin = 0.5;

        // Gather the click's neighborhood once (positions and radii). The
        // spatial grid can be stale between steps, and clicks are rare
        // enough that one linear pass is cheaper than keeping it fresh.
        let max_spread = base_spread * 3.0;
        let reach = max_spread + self.base_radius + max_radius(&self.particles) + margin;
        let mut occupied: Vec<(f64, f64, f64)> = self
            .particles
            .iter()
            .filter(|p| p.distance_squared_from(x, y) < reach * reach)
            .map(|p| (p.x, p.y, p.radius))
            .collect();

        let r = self.base_radius;
        let (width_f, height_f) = (f64::from(self.width), f64::from(self.height));
        for _ in 0..CLICK_BURST_SIZE {
            if self.particles.len() >= self.max_particles {
                break;
            }
            #[allow(clippy::cast_precision_loss)]
            let spot = (0..BURST_PLACEMENT_ATTEMPTS).find_map(|attempt| {
                let spread = base_spread
                    + (max_spread - base_spread) * attempt as f64
                        / (BURST_PLACEMENT_ATTEMPTS - 1) as f64;
                let bx = x + self.rng.random_range(-spread..spread);
                let by = y + self.rng.random_range(-spread..spread);
                let in_bounds = bx >= r && bx <= width_f - r && by >= r && by <= height_f - r;
                let clear = occupied.iter().all(|&(ox, oy, or)| {
                    let (dx, dy) = (bx - ox, by - oy);
                    let clearance = r + or + margin;
                    dx * dx + dy * dy >= clearance * clearance
                });
                // Stay on the click's side of every wall: a burst beside
                // a wall must not scatter particles across it.
                let same_side = !self
                    .segments
                    .iter()
                    .any(|s| motion_crosses_segment(s, x, y, bx, by));
                (in_bounds && clear && same_side).then_some((bx, by))
            });
            if let Some((bx, by)) = spot {
                occupied.push((bx, by, r));
                // Radial ejection: away from the click, faster with
                // distance (50-100% of the initial speed), so the burst
                // expands without internal collisions. A particle exactly
                // on the click point picks a random direction.
                let (dx, dy) = (bx - x, by - y);
                let dist = (dx * dx + dy * dy).sqrt();
                let (ux, uy) = if dist > 1e-9 {
                    (dx / dist, dy / dist)
                } else {
                    let angle = self.rng.random_range(0.0..std::f64::consts::TAU);
                    (angle.cos(), angle.sin())
                };
                let speed = self.initial_speed * (0.5 + 0.5 * (dist / max_spread).min(1.0));
                let p = Particle::new_moving(&mut self.rng, bx, by, ux * speed, uy * speed, r);
                let p = self.stamp(p);
                self.particles.push(p);
            }
        }
        // Fresh particles are moving; leave the stopped state if we were in it.
        self.wake();
    }

    /// Launch a comet (middle click): a fast, heavy particle streaking in
    /// from the farthest arena edge toward `(x, y)` — the long flight
    /// maximizes the streak. Under matter mode its impacts exceed the
    /// fission threshold and shatter what they hit; otherwise its 9x mass
    /// simply plows through the population.
    pub fn launch_comet(&mut self, x: f64, y: f64) {
        if self.particles.len() >= self.max_particles {
            return;
        }
        let radius = self.base_radius * COMET_RADIUS_FACTOR;
        let (w, h) = (f64::from(self.width), f64::from(self.height));
        let (left, right, top, bottom) = (x, w - x, y, h - y);
        let farthest = left.max(right).max(top).max(bottom);
        let (ex, ey) = if farthest == left {
            (radius, y)
        } else if farthest == right {
            (w - radius, y)
        } else if farthest == top {
            (x, radius)
        } else {
            (x, h - radius)
        };
        let (dx, dy) = (x - ex, y - ey);
        let dist = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = if dist > 1e-9 {
            (dx / dist, dy / dist)
        } else {
            (1.0, 0.0)
        };
        let speed = self.initial_speed * COMET_SPEED_FACTOR;
        let comet = Particle::new_moving(&mut self.rng, ex, ey, ux * speed, uy * speed, radius);
        let comet = self.stamp(comet);
        self.particles.push(comet);
        // The comet is moving; leave the stopped state if we were in it.
        self.wake();
    }

    /// Pin a persistent gravity well at `(x, y)` (W key; Shift+W pins a
    /// repeller). Returns false once the well limit is reached.
    pub fn pin_well(&mut self, x: f64, y: f64, polarity: Polarity) -> bool {
        if self.pinned_wells.len() >= MAX_PINNED_WELLS {
            return false;
        }
        self.pinned_wells.push(Well { x, y, polarity });
        // The new well is about to move particles.
        self.wake();
        true
    }

    /// Remove all pinned wells, returning how many were cleared.
    pub fn clear_wells(&mut self) -> usize {
        let cleared = self.pinned_wells.len();
        self.pinned_wells.clear();
        cleared
    }

    /// Add a drawn wall segment (held V). Returns false once the segment
    /// limit is reached.
    pub fn add_wall_segment(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
        if self.segments.len() >= MAX_WALL_SEGMENTS {
            return false;
        }
        self.segments.push(Segment { x1, y1, x2, y2 });
        true
    }

    /// Remove all drawn walls, returning how many segments were cleared.
    pub fn clear_wall_segments(&mut self) -> usize {
        let cleared = self.segments.len();
        self.segments.clear();
        cleared
    }

    /// Trigger a cursor explosion at `(x, y)`: kills everything the ring
    /// reaches, down to the base particle count — the configured minimum
    /// is an invariant no explosion violates. Returns false if an
    /// explosion is already active or there is nothing to explode.
    pub fn trigger_manual_explosion(&mut self, x: f64, y: f64) -> bool {
        if self.explosion.is_some() || self.particles.is_empty() {
            return false;
        }
        self.trigger_explosion(x, y, 1.0, self.base_particle_count);
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
        self.explosion = Some(explosion);
        self.spawn_times.clear();
    }

    // --- Stepping ----------------------------------------------------------

    /// Advance the simulation by `dt` seconds. `now` anchors the sliding
    /// spawn-rate window; `well` is the cursor gravity well, if held.
    pub fn step(&mut self, dt: f64, now: Instant, well: Option<Well>) -> StepEvents {
        let mut events = StepEvents::default();
        if self.stopped {
            // Ambient influences revive a stopped simulation on their own:
            // the held or pinned wells and the flow field all inject
            // motion, so callers need not remember to wake() first.
            if well.is_some() || self.flow || self.self_gravity || !self.pinned_wells.is_empty() {
                self.wake();
            } else {
                return events;
            }
        }

        events.explosion_completed = self.update_explosion(dt);
        self.sim_time += dt;

        // No particle enters physics out of bounds. Out-of-bounds space
        // lies beyond every wall segment's span (walls live inside the
        // arena), so a stale out-of-bounds position can round a wall's
        // endpoint undetected — the crossing happens outside the span the
        // wall pass guards. Every position writer maintains this invariant
        // already; this frame-start clamp is the safety net that keeps a
        // future writer's slip from ever becoming a wall leak.
        let (w, h) = (f64::from(self.width), f64::from(self.height));
        for p in &mut self.particles {
            p.x = p.x.min(w - p.radius).max(p.radius);
            p.y = p.y.min(h - p.radius).max(p.radius);
        }

        let gravity_multiplier = f64::from(self.gravity_percent) / 100.0;
        self.collisions.clear();

        // Self-gravity integrates once per frame, not per substep: the
        // collective field changes on screen-crossing timescales, far
        // slower than the collision-scale resolution substeps exist for,
        // and the all-pairs field is by far the costliest force term.
        // Applied before the substep count so the kick it delivers is
        // included in the tunneling-prevention speed estimate.
        if self.self_gravity {
            apply_self_gravity(&mut self.particles, dt);
        }

        let substeps = substep_count(&self.particles, dt);
        let sub_dt = dt / f64::from(substeps);

        let mut max_energy = 0.0f64;
        for _ in 0..substeps {
            if let Some(w) = well {
                apply_attractor(
                    &mut self.particles,
                    w.x,
                    w.y,
                    w.polarity.signum() * WELL_STRENGTH,
                    sub_dt,
                );
            }
            for w in &self.pinned_wells {
                apply_attractor(
                    &mut self.particles,
                    w.x,
                    w.y,
                    w.polarity.signum() * PINNED_WELL_STRENGTH,
                    sub_dt,
                );
            }
            if self.flow {
                apply_flow(&mut self.particles, self.sim_time, sub_dt);
            }
            // The segment test sweeps each particle's motion over the
            // substep, so it needs the centers from before the position
            // update.
            if !self.segments.is_empty() {
                self.prev_positions.clear();
                self.prev_positions
                    .extend(self.particles.iter().map(|p| (p.x, p.y)));
            }
            update_physics(
                &mut self.particles,
                sub_dt,
                self.width,
                self.height,
                gravity_multiplier,
                self.wall_elasticity,
            );
            // Walls resolve twice per substep, straddling the pair pass.
            // This first pass corrects integration crossers *before*
            // contact detection reads positions: a fast clump striking a
            // wall pushes many particles across in the same integration
            // step, and two simultaneous crossers otherwise look like a
            // legitimate same-side contact — recording a far-side spawn
            // site that outlives every later correction.
            if !self.segments.is_empty() {
                collide_with_segments(
                    &mut self.particles,
                    &self.prev_positions,
                    &self.segments,
                    self.wall_elasticity,
                );
            }
            let energy = handle_collisions(
                &mut self.particles,
                &mut self.grid,
                &mut self.collisions,
                self.width,
                self.height,
                self.particle_elasticity,
                &self.segments,
            );
            max_energy = max_energy.max(energy);
            // ... and this second pass keeps containment the invariant
            // that holds at the end of every substep: the sweep covers
            // any pair-separation pushout that would otherwise shove a
            // particle across a wall. The pair overlap a wall correction
            // may reintroduce is the softer failure — the next substep's
            // pair pass re-detects and resolves it.
            if !self.segments.is_empty() {
                collide_with_segments(
                    &mut self.particles,
                    &self.prev_positions,
                    &self.segments,
                    self.wall_elasticity,
                );
            }
        }

        events.max_collision_energy = max_energy;
        events.collision_pan = self
            .collision_centroid()
            .map_or(0.5, |(x, _)| x / f64::from(self.width));

        if self.matter && self.explosion.is_none() && self.process_matter(now) {
            // Matter ops reorder the particle list; rebuild the grid so the
            // spawn clearance checks below consult correct positions.
            self.grid.build(&self.particles, self.width, self.height);
        }

        events.explosion_started = self.handle_spawning(now);
        events.motion_stopped = self.check_motion(
            well.is_some() || self.flow || self.self_gravity || !self.pinned_wells.is_empty(),
        );
        events
    }

    /// Transfer `mass` worth of matter from `src` into `dst`: the
    /// mass-weighted position/velocity/color blend shared by full and
    /// partial fusion. Area and momentum are conserved exactly; the
    /// donor's own radius is left to the caller (full fusion kills it,
    /// partial fusion shrinks it by the transfer).
    ///
    /// The position blend is skipped when it would carry `dst` across a
    /// wall segment. Cross-wall pairs are already filtered at contact
    /// detection, so for a straight wall the blend (a point on the
    /// center-to-center chord) cannot cross; but contacts are recorded
    /// mid-frame and resolved here after further substeps, so near a
    /// short wall's free end the partner may have rounded the tip by
    /// now — without the guard the merged particle could teleport
    /// through the stub. Skipping the (cosmetic) centroid move keeps
    /// mass and momentum conservation intact.
    fn absorb(dst: &mut Particle, src: &Particle, mass: f64, segments: &[Segment]) {
        let dst_mass = dst.mass();
        let total = dst_mass + mass;
        let bx = (dst.x * dst_mass + src.x * mass) / total;
        let by = (dst.y * dst_mass + src.y * mass) / total;
        if !segments
            .iter()
            .any(|s| motion_crosses_segment(s, dst.x, dst.y, bx, by))
        {
            dst.x = bx;
            dst.y = by;
        }
        dst.vx = (dst.vx * dst_mass + src.vx * mass) / total;
        dst.vy = (dst.vy * dst_mass + src.vy * mass) / total;
        dst.radius = total.sqrt();
        for (c, &sc) in dst.color.iter_mut().zip(src.color.iter()).take(3) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                *c = ((f64::from(*c) * dst_mass + f64::from(sc) * mass) / total) as u8;
            }
        }
    }

    /// Apply fusion/fission outcomes to this step's collisions. Slow
    /// contacts merge (area, momentum, and blended color conserved); hard
    /// impacts shatter both participants into half-area fragments that fly
    /// apart perpendicular to the impact. Each particle takes part in at
    /// most one matter event per step. Returns true if anything changed.
    ///
    /// Fission fragments are recorded in the birth window (`spawn_times`),
    /// so runaway fission chains trip the explosion threshold just like
    /// runaway collision spawning — the threshold measures births per
    /// second, whatever their mechanism.
    fn process_matter(&mut self, now: Instant) -> bool {
        let min_r = self.base_radius * MIN_RADIUS_FACTOR;
        let max_r = self.base_radius * MAX_RADIUS_FACTOR;
        let mut touched: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut dead: Vec<usize> = Vec::new();
        let mut changed = false;

        for k in 0..self.collisions.sites().len() {
            let site = self.collisions.sites()[k];
            let (i, j) = (site.i, site.j);
            if touched.contains(&i) || touched.contains(&j) {
                continue;
            }

            if site.energy <= FUSION_MAX_ENERGY {
                let max_mass = max_r * max_r;
                let (m1, m2) = (self.particles[i].mass(), self.particles[j].mass());

                if m1 + m2 <= max_mass {
                    // Full fusion: i absorbs all of j.
                    if self.particles.len() - dead.len() <= 2 {
                        continue; // never fuse below the minimum population
                    }
                    let (pi, pj) = pair_mut(&mut self.particles, i, j);
                    Self::absorb(pi, pj, m2, &self.segments);
                    touched.insert(i);
                    touched.insert(j);
                    dead.push(j);
                    changed = true;
                } else {
                    // Partial fusion at the cap: the larger particle absorbs
                    // mass from the smaller, up to the cap and never leaving
                    // the smaller below the fragment minimum.
                    let (big, small) = if m1 >= m2 { (i, j) } else { (j, i) };
                    let (mb, ms) = (self.particles[big].mass(), self.particles[small].mass());
                    let min_mass = min_r * min_r;
                    let transfer = (max_mass - mb).min(ms - min_mass);
                    if transfer <= 0.0 {
                        continue; // absorber full or donor at minimum: bounce
                    }
                    let (pb, ps) = pair_mut(&mut self.particles, big, small);
                    Self::absorb(pb, ps, transfer, &self.segments);
                    ps.radius = (ms - transfer).sqrt();
                    touched.insert(i);
                    touched.insert(j);
                    changed = true;
                }
            } else if site.energy >= FISSION_MIN_ENERGY {
                // Fission: shatter each participant that is large enough
                // into two half-area fragments receding perpendicular to
                // the impact.
                let (px, py) = (-site.ny, site.nx);
                let kick = FISSION_KICK_FRACTION * site.energy;
                for idx in [i, j] {
                    let frag_r = self.particles[idx].radius / std::f64::consts::SQRT_2;
                    if frag_r < min_r || self.particles.len() >= self.max_particles {
                        continue;
                    }
                    let offset = frag_r + 0.5;
                    // The perpendicular's sign is arbitrary (it only picks
                    // which fragment goes which way), so choose one that
                    // keeps both fragments on the parent's side of every
                    // wall; against a wall-hugging impact where neither
                    // sign works, bounce instead of shattering across.
                    let (ox, oy) = (self.particles[idx].x, self.particles[idx].y);
                    // Candidate fragment positions clamp into the arena
                    // first — out-of-bounds space lies beyond every wall's
                    // span, so a fragment parked there could round a wall
                    // endpoint and re-enter on the far side (the same hole
                    // the pair-separation clamp closes). The wall-crossing
                    // check then runs on the clamped targets.
                    let (w, h) = (f64::from(self.width), f64::from(self.height));
                    let place = |s: f64| {
                        let clamp = |x: f64, y: f64| {
                            (x.min(w - frag_r).max(frag_r), y.min(h - frag_r).max(frag_r))
                        };
                        (
                            clamp(ox + s * px * offset, oy + s * py * offset),
                            clamp(ox - s * px * offset, oy - s * py * offset),
                        )
                    };
                    let segs = &self.segments;
                    let Some((sign, ((pax, pay), (twx, twy)))) = [1.0f64, -1.0]
                        .into_iter()
                        .map(|s| (s, place(s)))
                        .find(|&(_, (p, t))| {
                            !segs.iter().any(|seg| {
                                motion_crosses_segment(seg, ox, oy, p.0, p.1)
                                    || motion_crosses_segment(seg, ox, oy, t.0, t.1)
                            })
                        })
                    else {
                        continue;
                    };
                    let (px, py) = (sign * px, sign * py);
                    let parent = &mut self.particles[idx];
                    parent.radius = frag_r;
                    parent.x = pax;
                    parent.y = pay;
                    parent.vx += px * kick;
                    parent.vy += py * kick;
                    let twin = Particle {
                        id: 0, // stamped below, once the parent borrow ends
                        x: twx,
                        y: twy,
                        vx: parent.vx - 2.0 * px * kick,
                        vy: parent.vy - 2.0 * py * kick,
                        radius: frag_r,
                        color: parent.color,
                        doomed: false,
                    };
                    let twin = self.stamp(twin);
                    self.particles.push(twin);
                    self.spawn_times.push_back(now);
                    touched.insert(idx);
                    changed = true;
                }
                touched.insert(i);
                touched.insert(j);
            }
        }

        // Remove fused-away particles. Descending order keeps the remaining
        // indices valid under swap_remove, and the swapped-in tail elements
        // (fission fragments or untouched particles) are never in `dead`.
        dead.sort_unstable_by(|a, b| b.cmp(a));
        for idx in dead {
            self.particles.swap_remove(idx);
        }
        changed
    }

    /// Update explosion state, processing kills and checking completion.
    /// Returns the total kill count when the explosion completed this step.
    fn update_explosion(&mut self, dt: f64) -> Option<usize> {
        let exp = self.explosion.as_mut()?;
        exp.update(dt);
        exp.process_kills(&mut self.particles);
        if exp.active {
            return None;
        }
        let killed = exp.killed_count;
        self.explosion = None;
        Some(killed)
    }

    /// Handle particle spawning from collisions, potentially triggering an
    /// automatic explosion. Returns true if one was triggered.
    fn handle_spawning(&mut self, now: Instant) -> bool {
        // Track the birth rate (collision spawns and fission
        // fragments) over a sliding window
        let spawn_window = std::time::Duration::from_secs_f64(SPAWN_RATE_WINDOW);
        if let Some(cutoff) = now.checked_sub(spawn_window) {
            while self.spawn_times.front().is_some_and(|&t| t < cutoff) {
                self.spawn_times.pop_front();
            }
        }

        // Saturation valve: a population pegged at the density cap stops
        // producing births (spawning halts), so the birth-rate trigger is
        // blind to exactly the congestion it exists to relieve. Sustained
        // time at the cap is itself the signal: once the population has
        // been pegged continuously long enough, explode. Gated on the same
        // conditions as the rate trigger — explosions enabled, spawning on,
        // none already running.
        if self.particles.len() < self.max_particles {
            self.saturated_since = None;
        } else if self.explosion_threshold > 0
            && self.spawn_mode != SpawnMode::Off
            && self.explosion.is_none()
        {
            let since = *self.saturated_since.get_or_insert(now);
            if now.duration_since(since).as_secs_f64() >= SATURATION_EXPLOSION_SECS {
                self.saturated_since = None;
                let (ex, ey) = if self.spawn_mode == SpawnMode::Collision {
                    self.collision_centroid()
                        .unwrap_or((self.center_x, self.center_y))
                } else {
                    (self.center_x, self.center_y)
                };
                self.trigger_explosion(ex, ey, EXPLOSION_KILL_RATIO, self.base_particle_count);
                return true;
            }
        }

        if self.collisions.is_empty()
            || self.explosion.is_some()
            || self.spawn_mode == SpawnMode::Off
        {
            return false;
        }

        // Explode instead of spawning once the window fills up.
        if self.explosion_threshold > 0 && self.spawn_times.len() >= self.explosion_threshold {
            // In collision-spawning mode the action is wherever particles
            // are densest; center the blast on the recent collisions.
            let (ex, ey) = if self.spawn_mode == SpawnMode::Collision {
                self.collision_centroid()
                    .unwrap_or((self.center_x, self.center_y))
            } else {
                (self.center_x, self.center_y)
            };
            self.trigger_explosion(ex, ey, EXPLOSION_KILL_RATIO, self.base_particle_count);
            return true;
        }

        // Grid state covers exactly the pre-spawn particles (rebuilt after
        // matter ops); particles spawned this frame are checked separately
        // in `spawn_position_is_free`. Clearance distances are conservative
        // against the largest particle in the population.
        let first_new = self.particles.len();
        let max_r = max_radius(&self.particles);
        let mut spawned = 0;
        let mut attempts = 0;
        for i in 0..self.collisions.sites().len() {
            // The attempt budget bounds the frame's clearance queries, not
            // just its successes: a packed clump produces tens of thousands
            // of collision sites whose clearance checks all fail, and
            // probing every one of them each frame made the grid queries
            // the single hottest path at ~24k particles (profiled at ~600ms
            // per frame). Open scenes hit MAX_SPAWNS_PER_FRAME well within
            // the budget, so spawn behavior is unchanged where spawning
            // actually works.
            if spawned >= MAX_SPAWNS_PER_FRAME
                || attempts >= MAX_SPAWN_ATTEMPTS_PER_FRAME
                || self.particles.len() >= self.max_particles
            {
                break;
            }
            let site = self.collisions.sites()[i];
            // With matter mechanics on, low-energy contacts fused and
            // high-energy impacts shattered; only the middle band spawns.
            if self.matter
                && (site.energy <= FUSION_MAX_ENERGY || site.energy >= FISSION_MIN_ENERGY)
            {
                continue;
            }
            attempts += 1;
            let spawn = if self.spawn_mode == SpawnMode::Collision {
                // Eject the newborn outward, away from the collision.
                self.free_position_beside(site, first_new, max_r)
                    .map(|(pos, out)| {
                        let (vx, vy) = self.ejection_velocity(out, site.energy);
                        (pos, vx, vy)
                    })
            } else {
                // Classic center fountain: random direction and speed.
                self.free_position_near_center(first_new, max_r).map(|pos| {
                    let angle = self.rng.random_range(0.0..std::f64::consts::TAU);
                    let speed = self
                        .rng
                        .random_range(self.initial_speed * 0.5..self.initial_speed);
                    (pos, speed * angle.cos(), speed * angle.sin())
                })
            };

            if let Some(((x, y), vx, vy)) = spawn {
                let p = Particle::new_moving(&mut self.rng, x, y, vx, vy, self.base_radius);
                let p = self.stamp(p);
                self.particles.push(p);
                self.spawn_times.push_back(now);
                spawned += 1;
            }
        }
        false
    }

    /// Velocity for a particle ejected from a collision: aimed within a
    /// cone around the outward direction `out` (away from both parents,
    /// which recede along the collision normal), at a speed scaled by the
    /// collision energy — hard impacts eject fast fragments, grazing ones
    /// release slow debris.
    fn ejection_velocity(&mut self, out: (f64, f64), energy: f64) -> (f64, f64) {
        let angle = out.1.atan2(out.0)
            + self
                .rng
                .random_range(-SPAWN_CONE_HALF_ANGLE..SPAWN_CONE_HALF_ANGLE);
        let t = (energy / COLLISION_ENERGY_NORMALIZER).clamp(0.0, 1.0);
        let speed =
            (SPAWN_SPEED_MIN_FRACTION + (1.0 - SPAWN_SPEED_MIN_FRACTION) * t) * self.initial_speed;
        (speed * angle.cos(), speed * angle.sin())
    }

    /// Pick a spawn position beside a collision site, guaranteed clear of
    /// the colliding parents: they separated along the collision normal, so
    /// a point one diameter out along the *perpendicular* cannot touch them.
    /// Falls back to points along the normal beyond each parent, and gives
    /// up only when third-party particles occupy every candidate — spawning
    /// into occupied space would re-collide instantly and spawn again, a
    /// self-feeding cascade that inflates the spawn rate far beyond what the
    /// moving particles actually produce.
    ///
    /// Returns the position together with its outward unit direction (away
    /// from the collision midpoint), which orients the ejection velocity.
    #[allow(clippy::type_complexity)]
    fn free_position_beside(
        &mut self,
        site: SpawnSite,
        first_new: usize,
        max_r: f64,
    ) -> Option<((f64, f64), (f64, f64))> {
        // Offsets are conservative against the largest particle in play:
        // perpendicular distance to each parent (which sits along the
        // normal) exceeds the largest possible contact distance. The
        // along-normal fallbacks sit beyond the far side of each parent.
        let perp_offset = self.base_radius + max_r + 0.5;
        let normal_offset = 2.0 * max_r + self.base_radius + 1.0;
        let (px, py) = (-site.ny, site.nx);
        // Randomize which side is tried first so spawns don't drift to one
        // side of the collision plane.
        let side = if self.rng.random_bool(0.5) { 1.0 } else { -1.0 };

        let candidates = [
            ((side * px, side * py), perp_offset),
            ((-side * px, -side * py), perp_offset),
            ((site.nx, site.ny), normal_offset),
            ((-site.nx, -site.ny), normal_offset),
        ];
        candidates.into_iter().find_map(|((dx, dy), offset)| {
            let (x, y) = (site.x + dx * offset, site.y + dy * offset);
            self.spawn_position_is_free(site.x, site.y, x, y, first_new, max_r)
                .then_some(((x, y), (dx, dy)))
        })
    }

    /// Pick a free spawn position near the screen center (default mode),
    /// trying a few jittered draws before giving up.
    fn free_position_near_center(&mut self, first_new: usize, max_r: f64) -> Option<(f64, f64)> {
        const CENTER_SPAWN_ATTEMPTS: usize = 4;
        let jitter = SPAWN_JITTER.max(self.base_radius * 2.0);
        for _ in 0..CENTER_SPAWN_ATTEMPTS {
            let x = self.center_x + self.rng.random_range(-jitter..jitter);
            let y = self.center_y + self.rng.random_range(-jitter..jitter);
            if self.spawn_position_is_free(self.center_x, self.center_y, x, y, first_new, max_r) {
                return Some((x, y));
            }
        }
        None
    }

    /// A spawn position is usable when it lies inside the arena, clear
    /// of every existing particle (checked via the grid for pre-existing
    /// particles and linearly for this frame's spawns, which the grid
    /// cannot see yet) and of every drawn wall, and on the same side of
    /// every wall as the spawn's source point `(sx, sy)` — the collision
    /// site, click, or center it emanates from. Without the side check,
    /// a birth beside a collision hugging a wall could materialize on
    /// the far side, crossing a divider no motion ever crossed. The
    /// clearance is conservative: the new particle's radius plus the
    /// largest radius in the population.
    fn spawn_position_is_free(
        &self,
        sx: f64,
        sy: f64,
        x: f64,
        y: f64,
        first_new: usize,
        max_r: f64,
    ) -> bool {
        let r = self.base_radius;
        let clearance = r + max_r;
        if x < r || x > f64::from(self.width) - r || y < r || y > f64::from(self.height) - r {
            return false;
        }
        !self
            .grid
            .any_within(&self.particles[..first_new], x, y, clearance)
            && !self.particles[first_new..]
                .iter()
                .any(|p| p.distance_squared_from(x, y) < clearance * clearance)
            && !self
                .segments
                .iter()
                .any(|s| s.distance_to(x, y) < clearance || motion_crosses_segment(s, sx, sy, x, y))
    }

    /// Average position of this step's collisions.
    fn collision_centroid(&self) -> Option<(f64, f64)> {
        let sites = self.collisions.sites();
        if sites.is_empty() {
            return None;
        }
        let (sx, sy) = sites
            .iter()
            .fold((0.0, 0.0), |(ax, ay), s| (ax + s.x, ay + s.y));
        #[allow(clippy::cast_precision_loss)]
        let n = sites.len() as f64;
        Some((sx / n, sy / n))
    }

    /// Check if all particles have stopped moving. An active gravity well
    /// is about to move them, so it suppresses the check. Returns true on
    /// the transition into the stopped state (once stopped, `step` returns
    /// early, so this fires exactly once per stop).
    fn check_motion(&mut self, well_active: bool) -> bool {
        if self.explosion.is_some() || well_active {
            return false;
        }
        if has_motion(&self.particles) {
            self.frames_without_motion = 0;
        } else {
            self.frames_without_motion += 1;
            if self.frames_without_motion >= MOTION_STOPPED_FRAMES {
                self.stopped = true;
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
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

    /// Full-step timing at the scale of the pathological exported scene:
    /// 5,245 particles, self-gravity, matter, two stacked center wells.
    /// Not a correctness test — run manually:
    /// `cargo test --release bench_pathological_scene -- --ignored --nocapture`
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_pathological_scene_frame_time() {
        // Matter itself is cheap; the scene's cost is self-gravity plus
        // clump collisions at a 5,245-particle population. Matter and
        // spawning churn the population (fusion deflates it, spawns grow
        // it), so pin the count: no matter, spawn cap at the target.
        let mut s = sim(&[
            "--min-particles",
            "100",
            "--self-gravity",
            "--particle-size",
            "1.5",
            "--initial-speed",
            "600",
        ]);
        assert!(s.pin_well(0.4936 * 800.0, 0.5286 * 600.0, Polarity::Attract));
        assert!(s.pin_well(0.5 * 800.0, 0.5286 * 600.0, Polarity::Attract));

        // The real scene reached 5,245 particles through matter spawning;
        // the CLI caps --min-particles at 100, so grow the population by
        // hand the same way handle_spawning does.
        let mut rng = StdRng::seed_from_u64(99);
        while s.particle_count() < 5245 {
            let p = Particle::new_random(&mut rng, 800, 600, 1.5, 600.0);
            let p = s.stamp(p);
            s.particles.push(p);
        }
        s.max_particles = 5245;

        // Let the wells and self-gravity pull the population into the dense
        // clump regime before measuring.
        let mut now = Instant::now();
        for _ in 0..30 {
            now += Duration::from_millis(50);
            s.step(0.05, now, None);
        }

        let frames: u32 = 30;
        let t = std::time::Instant::now();
        for _ in 0..frames {
            now += Duration::from_millis(50);
            s.step(0.05, now, None);
        }
        let per_frame = t.elapsed() / frames;
        let fps = 1.0 / per_frame.as_secs_f64();
        println!(
            "scene-1783523659 regime: {per_frame:?}/frame ({fps:.1} FPS ceiling), \
             {} particles",
            s.particle_count()
        );
    }

    #[test]
    fn super_elastic_energy_pump_saturates_at_terminal_velocity() {
        // Elasticity above 1.0 multiplies speed on every wall bounce and
        // amplifies every collision, and the bounce rate grows with speed
        // — without the terminal-velocity clamp this overflows f64 to
        // infinity within a few thousand bounces, after which ∞−∞ in a
        // collision mints NaN: invisible, frozen particles that no force
        // or setting change can ever recover. Run the pump long enough to
        // overflow many times over and verify every particle stays finite,
        // saturated near the cap rather than runaway.
        let mut s = sim(&[
            "--min-particles",
            "40",
            "--particle-elasticity",
            "1.5",
            "--wall-elasticity",
            "1.5",
            "--gravity",
            "0",
        ]);
        let mut now = Instant::now();
        for _ in 0..1200 {
            now += Duration::from_millis(50);
            s.step(0.05, now, None);
        }
        let mut top_speed = 0.0f64;
        for p in s.particles() {
            assert!(
                p.x.is_finite() && p.y.is_finite() && p.vx.is_finite() && p.vy.is_finite(),
                "no particle may go non-finite: pos=({}, {}) vel=({}, {})",
                p.x,
                p.y,
                p.vx,
                p.vy
            );
            top_speed = top_speed.max(p.speed());
        }
        // The pump must actually have run: something should be saturated
        // near the cap (a collision right after the clamp can push a bit
        // past it; the next substep re-clamps).
        assert!(
            top_speed > crate::physics::MAX_SPEED * 0.5,
            "energy pump should reach saturation, top speed {top_speed}"
        );
        assert!(
            top_speed < crate::physics::MAX_SPEED * 10.0,
            "speeds must stay bounded near the cap, top speed {top_speed}"
        );
    }

    /// Frame timing for the field-report regime: matter run merged pieces,
    /// then matter off and unbounded spawning grew ~24k particles that
    /// self-gravity collapsed into one packed corner clump (elasticity
    /// 0.7/0.4, gravity 0, spawn at collisions, explosions off). Run
    /// manually:
    /// `cargo test --release bench_corner_clump -- --ignored --nocapture`
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_corner_clump_frame_time() {
        let mut s = sim(&[
            "--min-particles",
            "100",
            "--self-gravity",
            "--spawn-mode",
            "collision",
            "--explosion-threshold",
            "0",
            "--gravity",
            "0",
            "--particle-elasticity",
            "0.7",
            "--wall-elasticity",
            "0.4",
        ]);
        s.width = 1728;
        s.height = 1117;
        s.max_particles = 30000;
        let mut rng = StdRng::seed_from_u64(3);
        // Packed corner clump: ~21k small particles at touching distance
        // around (1500, 900), plus a handful of surviving merged giants
        // and scattered dust.
        while s.particle_count() < 21000 {
            let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let r: f64 = 260.0 * rng.random_range(0.0f64..1.0).sqrt();
            let mut p = Particle::new_moving(
                &mut s.rng,
                (1500.0 + r * angle.cos()).clamp(2.0, 1726.0),
                (900.0 + r * angle.sin()).clamp(2.0, 1115.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                1.5,
            );
            p.color = [200, 200, 200, 255];
            let p = s.stamp(p);
            s.particles.push(p);
        }
        for k in 0..30 {
            let mut p = Particle::new_moving(
                &mut s.rng,
                1350.0 + f64::from(k % 6) * 60.0,
                780.0 + f64::from(k / 6) * 60.0,
                0.0,
                0.0,
                rng.random_range(20.0..40.0),
            );
            p.color = [220, 220, 220, 255];
            let p = s.stamp(p);
            s.particles.push(p);
        }
        while s.particle_count() < 23800 {
            let p = Particle::new_random(&mut s.rng, 1728, 1117, 1.5, 200.0);
            let p = s.stamp(p);
            s.particles.push(p);
        }

        let mut now = Instant::now();
        for _ in 0..10 {
            now += Duration::from_millis(50);
            s.step(0.05, now, None);
        }
        let frames: u32 = 10;
        let t = std::time::Instant::now();
        for _ in 0..frames {
            now += Duration::from_millis(50);
            s.step(0.05, now, None);
        }
        let per_frame = t.elapsed() / frames;
        let fps = 1.0 / per_frame.as_secs_f64();
        println!(
            "corner-clump regime: {per_frame:?}/frame ({fps:.1} FPS ceiling), {} particles",
            s.particle_count()
        );
    }

    #[test]
    fn resize_rescales_geometry_and_recomputes_the_cap() {
        let mut s = sim(&["--min-particles", "10", "--particle-size", "5"]);
        assert!(s.pin_well(400.0, 300.0, Polarity::Attract));
        assert!(s.add_wall_segment(100.0, 100.0, 200.0, 100.0));
        let cap_before = s.max_particles();

        s.resize(1600, 1200); // double both axes
        assert_eq!(s.dimensions(), (1600, 1200));
        // Coverage cap scales with area: 800*600/(4*10^2) = 1200 becomes
        // 1600*1200/400 = 4800.
        assert_eq!(cap_before, 1200);
        assert_eq!(s.max_particles(), 4800);
        let well = s.pinned_wells()[0];
        assert!((well.x - 800.0).abs() < 1e-9 && (well.y - 600.0).abs() < 1e-9);
        let seg = s.wall_segments()[0];
        assert!((seg.x1 - 200.0).abs() < 1e-9 && (seg.y1 - 200.0).abs() < 1e-9);
        assert!(
            s.particles().iter().all(|p| p.x <= 1600.0 && p.y <= 1200.0),
            "rescaled particles stay in bounds"
        );

        // Shrinking brings everything back and every particle stays
        // inside the new arena after a step settles the boundaries.
        s.resize(400, 300);
        let mut now = Instant::now();
        now += Duration::from_millis(10);
        s.step(0.01, now, None);
        assert!(
            s.particles()
                .iter()
                .all(|p| p.x <= 400.0 && p.y <= 300.0 && p.x >= 0.0 && p.y >= 0.0),
            "particles inside the shrunk arena"
        );
        assert_eq!(s.max_particles(), MIN_PARTICLE_CAP.max(10)); // 400*300/400 = 300 -> floor 1000
    }

    #[test]
    fn resize_updates_an_in_flight_explosion() {
        let mut s = sim(&["--min-particles", "30"]);
        assert!(s.trigger_manual_explosion(400.0, 300.0));
        let reach_before = s.explosion().expect("active").max_radius;

        s.resize(1600, 1200);
        let exp = s.explosion().expect("still active");
        assert!((exp.x - 800.0).abs() < 1e-9 && (exp.y - 600.0).abs() < 1e-9);
        assert!(
            exp.max_radius > reach_before,
            "reach recomputed from the larger arena"
        );

        // The ring must still complete against the new far corner.
        let mut now = Instant::now();
        let mut steps = 0;
        while s.explosion().is_some() {
            now += Duration::from_millis(50);
            s.step(0.05, now, None);
            steps += 1;
            assert!(steps < 200, "explosion completes after resize");
        }
    }

    #[test]
    fn resize_is_deterministic_across_runs() {
        let run = || {
            let mut s = sim(&["--min-particles", "40", "--self-gravity"]);
            let mut now = Instant::now();
            for frame in 0..60 {
                if frame == 30 {
                    s.resize(1200, 500);
                }
                now += Duration::from_millis(10);
                s.step(0.01, now, None);
            }
            s.particles()
                .iter()
                .map(|p| (p.x.to_bits(), p.y.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed + same resize = same world");
    }

    #[test]
    fn new_populates_base_count_and_density_cap() {
        let s = sim(&[]);
        assert_eq!(s.particle_count(), calculate_particle_count(800, 600));
        // Small particles hit the flat ceiling of the non-linear cap, not
        // the coverage bound: at the default radius 1.5 the coverage bound
        // would be 800*600/(4*3^2) = 13333, and 120,000 at radius 0.5.
        assert_eq!(s.max_particles, MAX_PARTICLES);
        let s = sim(&["--particle-size", "0.5"]);
        assert_eq!(s.max_particles, MAX_PARTICLES);

        // Large particles keep the pure coverage bound: 800*600/(4*4^2)
        // and 800*600/(4*10^2).
        let s = sim(&["--particle-size", "2"]);
        assert_eq!(s.max_particles, 7500);
        let s = sim(&["--particle-size", "5"]);
        assert_eq!(s.max_particles, 1200);

        let s = sim(&["--min-particles", "30"]);
        assert_eq!(s.particle_count(), 30);
    }

    #[test]
    fn particle_count_scales_with_screen_size() {
        assert_eq!(calculate_particle_count(100, 100), 2); // minimum enforced
        assert_eq!(calculate_particle_count(1920, 1080), 6);
        assert_eq!(calculate_particle_count(3840, 2160), 22);
    }

    #[test]
    fn collision_spawns_exactly_one_particle() {
        let mut s = sim(&["--min-particles", "2"]);
        arm_collision(&mut s, 200.0, 300.0);
        let events = s.step(0.01, Instant::now(), None);
        assert!(events.max_collision_energy > 0.0);
        assert_eq!(s.particle_count(), 3, "one collision spawns one particle");
        assert_eq!(s.spawn_times.len(), 1);
        // Pan reflects the collision site (x=200 of 800).
        assert!((events.collision_pan - 0.25).abs() < 0.05);
        // Default mode spawns near the screen center.
        let spawn = &s.particles[2];
        assert!((spawn.x - 400.0).abs() < 10.0 && (spawn.y - 300.0).abs() < 10.0);
    }

    #[test]
    fn spawn_beside_collision_never_touches_the_parents() {
        for seed in ["1", "2", "3", "4", "5"] {
            let mut s = Simulation::new(
                &Config::try_parse_from([
                    "bouncy",
                    "--seed",
                    seed,
                    "--min-particles",
                    "2",
                    "--spawn-at-collision",
                ])
                .unwrap(),
                800,
                600,
            );
            arm_collision(&mut s, 400.0, 300.0);
            s.step(0.01, Instant::now(), None);
            assert_eq!(s.particle_count(), 3, "every collision must spawn");

            let spawn = &s.particles[2];
            let diameter = s.base_radius * 2.0;
            for parent in &s.particles[..2] {
                let dist = parent.distance_squared_from(spawn.x, spawn.y).sqrt();
                assert!(
                    dist > diameter,
                    "spawn at ({:.1}, {:.1}) overlaps a parent (dist {dist:.2})",
                    spawn.x,
                    spawn.y
                );
            }
            // And it appears beside the collision, not somewhere random.
            let ox = spawn.x - 400.0;
            let oy = spawn.y - 300.0;
            let site_dist = (ox * ox + oy * oy).sqrt();
            assert!(site_dist < diameter * 3.0, "spawn stays near the site");

            // Ejected outward: its velocity lies within the 45-degree cone
            // around the direction away from the collision midpoint.
            let speed = spawn.speed();
            let outward_dot = (spawn.vx * ox + spawn.vy * oy) / (speed * site_dist);
            assert!(
                outward_dot >= (std::f64::consts::FRAC_PI_4.cos() - 1e-9),
                "spawn must move away from the collision (cos angle = {outward_dot:.3})"
            );
        }
    }

    #[test]
    fn ejection_speed_scales_with_collision_energy() {
        // Same setup, different closing speeds: the newborn from a hard
        // collision must leave faster than one from a grazing collision.
        let spawn_speed = |closing: f64| -> f64 {
            let mut s = Simulation::new(
                &Config::try_parse_from([
                    "bouncy",
                    "--seed",
                    "9",
                    "--min-particles",
                    "2",
                    "--spawn-at-collision",
                ])
                .unwrap(),
                800,
                600,
            );
            arm_collision(&mut s, 400.0, 300.0);
            s.particles[0].vx = closing;
            s.particles[1].vx = -closing;
            // Slow closings take a few steps to reach contact.
            let now = Instant::now();
            for _ in 0..20 {
                s.step(0.01, now, None);
                if s.particle_count() == 3 {
                    break;
                }
            }
            assert_eq!(s.particle_count(), 3);
            s.particles[2].speed()
        };

        let slow = spawn_speed(30.0);
        let fast = spawn_speed(600.0);
        assert!(
            fast > slow * 1.5,
            "hard collisions must eject faster fragments: slow={slow:.0}, fast={fast:.0}"
        );
        // Both stay within the ejection speed envelope.
        assert!(slow >= SPAWN_SPEED_MIN_FRACTION * 600.0 - 1e-9);
        assert!(fast <= 600.0 + 1e-9);
    }

    #[test]
    fn spawn_window_prunes_entries_older_than_one_second() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        let now = Instant::now();
        let old = now.checked_sub(Duration::from_millis(1500)).unwrap();
        let recent = now.checked_sub(Duration::from_millis(500)).unwrap();
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
        arm_collision(&mut s, 200.0, 300.0);
        let now = Instant::now();
        // 4 of 5 window slots used: the next spawn must not explode.
        for _ in 0..4 {
            s.spawn_times.push_back(now);
        }

        let events = s.step(0.01, now, None);
        assert!(!events.explosion_started);
        assert!(s.explosion().is_none());
        assert_eq!(s.particle_count(), 31);
        assert_eq!(s.spawn_times.len(), 5);
    }

    #[test]
    fn threshold_zero_never_explodes() {
        let mut s = sim(&["--min-particles", "2", "--explosion-threshold", "0"]);
        arm_collision(&mut s, 200.0, 300.0);
        let now = Instant::now();
        // A spawn window far beyond any positive threshold.
        for _ in 0..5000 {
            s.spawn_times.push_back(now);
        }

        let events = s.step(0.01, now, None);
        assert!(!events.explosion_started);
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
        let mut s = sim(&["--min-particles", "2", "--spawn-at-collision"]);
        arm_collision(&mut s, 400.0, 300.0);
        // Pack the neighborhood of the collision site solid: every candidate
        // spawn position (perpendicular and along-normal) overlaps something.
        for gx in 0..13 {
            for gy in 0..13 {
                let mut p = Particle::new_at_position(
                    &mut s.rng,
                    388.0 + f64::from(gx) * 2.0,
                    288.0 + f64::from(gy) * 2.0,
                    1.5,
                    600.0,
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
            s.particle_count(),
            before,
            "no particle may materialize when every position is occupied"
        );
        assert!(
            s.spawn_times.is_empty(),
            "blocked births are not births: the window counts only real spawns"
        );
    }

    #[test]
    fn sustained_saturation_at_the_cap_triggers_an_explosion() {
        // Regression for the cap catch-22: a population pegged at
        // max_particles produces no births, so the birth-rate trigger is
        // blind — the saturation valve must fire after the population has
        // been pegged continuously for SATURATION_EXPLOSION_SECS.
        let mut s = sim(&["--min-particles", "10", "--explosion-threshold", "5"]);
        s.max_particles = s.particle_count(); // no birth can ever fit
        let before = s.particle_count();
        let mut now = Instant::now();
        let mut exploded_after = None;
        for frame in 0..80 {
            arm_collision(&mut s, 400.0, 300.0);
            now += Duration::from_millis(50);
            if s.step(0.05, now, None).explosion_started {
                exploded_after = Some(frame);
                break;
            }
        }
        let frame = exploded_after.expect("saturation valve must fire");
        // 3 seconds at 50ms per frame is 60 frames; allow the off-by-one.
        assert!(
            (59..=61).contains(&frame),
            "valve fires after the saturation delay, not immediately: frame {frame}"
        );
        assert_eq!(s.particle_count(), before, "no births at the cap");
    }

    #[test]
    fn dipping_below_the_cap_resets_the_saturation_clock() {
        let mut s = sim(&["--min-particles", "10", "--explosion-threshold", "5"]);
        s.max_particles = s.particle_count();
        let mut now = Instant::now();
        for frame in 0..100 {
            now += Duration::from_millis(50);
            if frame % 40 == 20 {
                // Fall below the cap briefly (no collision armed, so
                // nothing spawns into the gap): the clock must restart.
                let p = s.particles.pop().expect("population is non-empty");
                assert!(!s.step(0.05, now, None).explosion_started);
                s.particles.push(p);
            } else {
                arm_collision(&mut s, 400.0, 300.0);
                assert!(
                    !s.step(0.05, now, None).explosion_started,
                    "interrupted saturation must not fire (frame {frame})"
                );
            }
        }
    }

    #[test]
    fn click_burst_particles_never_materialize_overlapping() {
        // Regression: burst particles used to be placed with no clearance
        // check, so several materialized interpenetrating; the resulting
        // instant collisions teleported phantom spawns to the screen
        // center (in the default spawn mode) the moment the user clicked
        // anywhere else.
        for seed in ["1", "2", "3", "4", "5"] {
            let mut s = Simulation::new(
                &Config::try_parse_from(["bouncy", "--seed", seed, "--min-particles", "2"])
                    .unwrap(),
                800,
                600,
            );
            freeze(&mut s);
            // Park the base pair far from both the click and the center.
            s.particles[0].x = 700.0;
            s.particles[0].y = 500.0;
            s.particles[1].x = 750.0;
            s.particles[1].y = 550.0;

            s.spawn_burst(100.0, 100.0);
            assert_eq!(
                s.particle_count(),
                12,
                "open space fits the full burst (seed {seed})"
            );
            for i in 0..s.particle_count() {
                for j in i + 1..s.particle_count() {
                    let (a, b) = (&s.particles[i], &s.particles[j]);
                    let dist = a.distance_squared_from(b.x, b.y).sqrt();
                    assert!(
                        dist >= a.radius + b.radius,
                        "{i} and {j} materialized overlapping (dist {dist:.2}, seed {seed})"
                    );
                }
            }

            // Every burst particle recedes from the click point, so the
            // burst expands and siblings can never converge.
            for p in &s.particles[2..] {
                let outward = p.vx * (p.x - 100.0) + p.vy * (p.y - 100.0);
                assert!(outward > 0.0, "ejected outward (seed {seed})");
            }

            // Ten frames at a real 120 FPS frame time: the burst disperses
            // in open space without a single collision, so the center
            // fountain stays quiet. (Random headings — the old behavior —
            // produced center spawns on the very first frame.)
            let now = Instant::now();
            for frame in 1..=10 {
                let events = s.step(1.0 / 120.0, now, None);
                assert!(
                    events.max_collision_energy == 0.0,
                    "burst must not self-collide (frame {frame}, seed {seed})"
                );
            }
            assert_eq!(s.particle_count(), 12, "no phantom spawns (seed {seed})");
            assert!(s.spawn_times.is_empty());
        }
    }

    #[test]
    fn click_burst_near_the_corner_stays_in_bounds() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.spawn_burst(1.0, 1.0);
        assert!(
            s.particle_count() > 2,
            "corner click still places particles"
        );
        for p in s.particles() {
            assert!(p.x >= p.radius && p.x <= 800.0 - p.radius, "x={}", p.x);
            assert!(p.y >= p.radius && p.y <= 600.0 - p.radius, "y={}", p.y);
        }
    }

    #[test]
    fn click_burst_into_packed_space_skips_rather_than_overlaps() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[0].x = 700.0;
        s.particles[0].y = 500.0;
        s.particles[1].x = 750.0;
        s.particles[1].y = 550.0;
        // Pack a solid grid over the whole placement disc around the
        // click: every candidate lies within clearance of some particle.
        for gx in 0..23 {
            for gy in 0..23 {
                let mut p = Particle::new_at_position(
                    &mut s.rng,
                    78.0 + f64::from(gx) * 2.0,
                    78.0 + f64::from(gy) * 2.0,
                    1.5,
                    600.0,
                );
                p.vx = 0.0;
                p.vy = 0.0;
                s.particles.push(p);
            }
        }
        let before = s.particle_count();

        s.spawn_burst(100.0, 100.0);
        assert_eq!(
            s.particle_count(),
            before,
            "solid-packed space places nothing rather than overlapping"
        );
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
    fn manual_explosion_respects_the_base_count_and_completes() {
        // The configured minimum is an invariant: even a full manual wipe
        // must leave the base population alive. The population is grown
        // above the base count first so the blast has something to kill.
        let mut s = sim(&["--min-particles", "50"]);
        let mut rng = StdRng::seed_from_u64(31);
        while s.particle_count() < 80 {
            let p = Particle::new_random(&mut rng, 800, 600, 1.5, 600.0);
            let p = s.stamp(p);
            s.particles.push(p);
        }
        assert!(s.trigger_manual_explosion(400.0, 300.0));
        assert!(
            !s.trigger_manual_explosion(400.0, 300.0),
            "no explosion while one is active"
        );

        let now = Instant::now();
        let mut steps = 0;
        let mut completed = None;
        while s.explosion().is_some() {
            let events = s.step(0.05, now, None);
            if events.explosion_completed.is_some() {
                completed = events.explosion_completed;
            }
            steps += 1;
            assert!(steps < 100, "explosion must complete");
        }
        assert!(
            s.particle_count() >= s.base_particle_count(),
            "manual blast keeps the configured minimum alive: {} of {}",
            s.particle_count(),
            s.base_particle_count()
        );
        assert_eq!(completed, Some(30), "completion event reports the kills");
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
        assert_eq!(events.max_collision_energy, 0.0);

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
            polarity: Polarity::Attract,
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
    fn pinned_well_accelerates_particles_and_suppresses_stopped() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[0].x = 100.0;
        s.particles[0].y = 300.0;
        s.particles[1].x = 700.0;
        s.particles[1].y = 300.0;
        assert!(s.pin_well(400.0, 300.0, Polarity::Attract));

        let now = Instant::now();
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, None);
        }
        assert!(
            !s.stopped(),
            "a pinned well must suppress the stopped state"
        );
        assert!(
            s.particles[0].vx > 0.0,
            "left particle accelerates toward the pinned well"
        );
        assert!(
            s.particles[1].vx < 0.0,
            "right particle accelerates toward the pinned well"
        );
    }

    #[test]
    fn pinned_repeller_pushes_particles_away() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[0].x = 300.0;
        s.particles[0].y = 300.0;
        s.particles[1].x = 500.0;
        s.particles[1].y = 300.0;
        assert!(s.pin_well(400.0, 300.0, Polarity::Repel));

        s.step(0.01, Instant::now(), None);
        assert!(s.particles[0].vx < 0.0, "left particle pushed left");
        assert!(s.particles[1].vx > 0.0, "right particle pushed right");
    }

    #[test]
    fn clear_wells_removes_all_and_motion_detection_resumes() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.pin_well(100.0, 100.0, Polarity::Attract);
        s.pin_well(200.0, 200.0, Polarity::Repel);
        assert_eq!(s.pinned_wells().len(), 2);

        assert_eq!(s.clear_wells(), 2);
        assert!(s.pinned_wells().is_empty());

        let now = Instant::now();
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, None);
        }
        assert!(s.stopped(), "with no wells left, motion detection resumes");
    }

    #[test]
    fn pinned_wells_are_capped() {
        let mut s = sim(&["--min-particles", "2"]);
        for _ in 0..MAX_PINNED_WELLS {
            assert!(s.pin_well(400.0, 300.0, Polarity::Attract));
        }
        assert!(
            !s.pin_well(400.0, 300.0, Polarity::Attract),
            "cap must refuse the next pin"
        );
        assert_eq!(s.pinned_wells().len(), MAX_PINNED_WELLS);
    }

    #[test]
    fn wells_flag_prepins_attractors_and_reset_restores_them() {
        let mut s = sim(&["--wells", "3", "--min-particles", "2"]);
        assert_eq!(s.pinned_wells().len(), 3);
        for w in s.pinned_wells() {
            assert_eq!(w.polarity, Polarity::Attract, "startup wells attract");
            assert!(w.x > 0.0 && w.x < 800.0 && w.y > 0.0 && w.y < 600.0);
        }

        s.pin_well(50.0, 50.0, Polarity::Repel);
        assert_eq!(s.pinned_wells().len(), 4);
        s.reset();
        assert_eq!(s.pinned_wells().len(), 3, "reset restores startup wells");
        assert!(
            s.pinned_wells()
                .iter()
                .all(|w| w.polarity == Polarity::Attract)
        );
    }

    #[test]
    fn particle_ids_are_unique_stable_and_survive_population_churn() {
        use std::collections::HashSet;
        let mut s = sim(&["--min-particles", "30", "--matter"]);

        // Every sim-owned particle is stamped with a distinct nonzero id.
        let initial: HashSet<u64> = s.particles().iter().map(|p| p.id).collect();
        assert_eq!(initial.len(), 30);
        assert!(!initial.contains(&0), "sim particles are always stamped");

        // find_particle resolves an id to its current index.
        let tracked = s.particles()[5].id;
        assert_eq!(
            s.find_particle(tracked).map(|i| s.particles()[i].id),
            Some(tracked)
        );
        assert_eq!(s.find_particle(u64::MAX), None);

        // Churn the population hard: bursts (spawns), matter fission
        // (swap_remove + fresh fragments), and an explosion (retain).
        s.spawn_burst(400.0, 300.0);
        arm_collision(&mut s, 400.0, 300.0);
        s.particles[0].vx = 400.0;
        s.particles[1].vx = -400.0;
        let now = Instant::now();
        for _ in 0..20 {
            s.step(0.01, now, None);
        }
        s.trigger_manual_explosion(400.0, 300.0);
        while s.explosion().is_some() {
            s.step(0.05, now, None);
        }

        // Ids stay unique across all of it, and any survivor of the churn
        // still resolves by id even though its index moved.
        let survivors: HashSet<u64> = s.particles().iter().map(|p| p.id).collect();
        assert_eq!(survivors.len(), s.particle_count(), "ids never collide");
        assert!(
            !survivors.contains(&0),
            "churn never produces unstamped ids"
        );
        if let Some(index) = s.find_particle(tracked) {
            assert_eq!(s.particles()[index].id, tracked);
        }
    }

    #[test]
    fn self_gravity_draws_particles_together_and_suppresses_stopped() {
        let mut s = sim(&["--min-particles", "2", "--self-gravity"]);
        freeze(&mut s);
        s.particles[0].x = 300.0;
        s.particles[0].y = 300.0;
        s.particles[1].x = 500.0;
        s.particles[1].y = 300.0;
        let gap_before = s.particles[1].x - s.particles[0].x;

        let now = Instant::now();
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.01, now, None);
        }
        assert!(!s.stopped(), "self-gravity suppresses the stopped state");
        assert!(s.particles[0].vx > 0.0, "left particle falls right");
        assert!(s.particles[1].vx < 0.0, "right particle falls left");
        assert!(
            s.particles[1].x - s.particles[0].x < gap_before,
            "the pair drifts together"
        );

        // Toggling it off lets motion detection resume.
        s.self_gravity = false;
        freeze(&mut s);
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, None);
        }
        assert!(s.stopped(), "without ambient forces the sim can stop again");
    }

    #[test]
    fn comet_launches_from_the_far_edge_toward_the_cursor() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        let before = s.particle_count();

        // Cursor right of center: the farthest edge is the left one.
        s.launch_comet(600.0, 300.0);
        assert_eq!(s.particle_count(), before + 1);
        let comet = s.particles().last().unwrap();
        assert!(
            (comet.radius - s.base_radius * COMET_RADIUS_FACTOR).abs() < 1e-9,
            "comets are heavy: radius {}",
            comet.radius
        );
        assert!(comet.x < 10.0, "entered from the left edge: x={}", comet.x);
        assert!(
            comet.vx > 0.0 && comet.vy.abs() < 1e-9,
            "streaks horizontally toward the cursor"
        );
        assert!(
            (comet.speed() - s.initial_speed() * COMET_SPEED_FACTOR).abs() < 1e-9,
            "flies at the comet speed multiple"
        );
        assert!(!s.stopped(), "a comet wakes a stopped simulation");
    }

    #[test]
    fn preset_scene_geometry_is_placed_scaled_and_survives_reset() {
        let user = crate::presets::parse(
            "[board]\n\
             wells = [{ x = 0.5, y = 0.5, polarity = \"repel\" }]\n\
             walls = [[0.25, 0.5, 0.75, 0.5]]\n",
            std::path::Path::new("/test/presets.toml"),
        )
        .unwrap();
        let args: Vec<std::ffi::OsString> = [
            "bouncy",
            "--seed",
            "1",
            "--min-particles",
            "2",
            "--preset",
            "board",
        ]
        .iter()
        .map(Into::into)
        .collect();
        let config = Config::try_resolve_with(&args, Some(&user)).unwrap();
        let mut s = Simulation::new(&config, 800, 600);

        assert_eq!(s.pinned_wells().len(), 1);
        let well = s.pinned_wells()[0];
        assert!((well.x - 400.0).abs() < 1e-9 && (well.y - 300.0).abs() < 1e-9);
        assert_eq!(well.polarity, Polarity::Repel);
        assert_eq!(s.wall_segments().len(), 1);
        let wall = s.wall_segments()[0];
        assert!((wall.x1 - 200.0).abs() < 1e-9 && (wall.x2 - 600.0).abs() < 1e-9);

        // Runtime additions are transient; the scene survives reset.
        s.pin_well(100.0, 100.0, Polarity::Attract);
        s.add_wall_segment(10.0, 10.0, 20.0, 20.0);
        s.reset();
        assert_eq!(s.pinned_wells().len(), 1, "scene well restored");
        assert_eq!(s.wall_segments().len(), 1, "scene wall restored");
        assert_eq!(s.pinned_wells()[0].polarity, Polarity::Repel);
    }

    #[test]
    fn drawn_wall_deflects_particles() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[1].x = 790.0;
        s.particles[1].y = 10.0;
        // Particle 0 flies right toward a vertical wall at x=400.
        s.particles[0].x = 300.0;
        s.particles[0].y = 300.0;
        s.particles[0].vx = 200.0;
        assert!(s.add_wall_segment(400.0, 250.0, 400.0, 350.0));

        let now = Instant::now();
        for _ in 0..100 {
            s.step(0.01, now, None);
            if s.particles[0].vx < 0.0 {
                break;
            }
        }
        assert!(s.particles[0].vx < 0.0, "particle must bounce back");
        assert!(
            s.particles[0].x < 400.0,
            "and stay on its side of the wall: x={}",
            s.particles[0].x
        );
    }

    #[test]
    fn fast_particle_cannot_tunnel_a_wall_at_low_frame_rates() {
        // Regression for tunneling at low frame rates: a 50 ms frame (the
        // frame-dt cap) saturates MAX_SUBSTEPS, so a single substep moves
        // a fast particle many radii, and the old end-of-substep overlap
        // test let it step clear over the zero-thickness wall. The swept
        // test must keep it contained for as long as the sim runs.
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[1].x = 790.0;
        s.particles[1].y = 10.0;
        // Particle 0 falls fast onto a horizontal wall spanning the arena.
        s.particles[0].x = 400.0;
        s.particles[0].y = 100.0;
        s.particles[0].vy = 2000.0;
        assert!(s.add_wall_segment(100.0, 300.0, 700.0, 300.0));

        let now = Instant::now();
        for frame in 0..40 {
            s.step(0.05, now, None);
            assert!(
                s.particles[0].y < 300.0,
                "particle stays above the wall: y={} after frame {}",
                s.particles[0].y,
                frame
            );
        }
    }

    #[test]
    fn particles_slide_down_a_near_vertical_wall() {
        // Regression for particles sticking to drawn walls: resolving
        // contact at the motion's closest approach reset a sliding
        // particle to its start-of-substep position every substep,
        // freezing it in place. Against a near-vertical wall under
        // gravity, a resting particle must keep descending.
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.gravity_percent = 100;
        s.particles[1].x = 790.0;
        s.particles[1].y = 10.0;
        // Wall leaning 10 px over 500: essentially the screenshot stroke.
        assert!(s.add_wall_segment(400.0, 50.0, 410.0, 550.0));
        let r = s.particles[0].radius;
        // Start pressed against the wall's upper reach, at rest.
        s.particles[0].x = 401.0 + r;
        s.particles[0].y = 100.0;

        let now = Instant::now();
        for _ in 0..120 {
            s.step(1.0 / 60.0, now, None);
        }
        assert!(
            s.particles[0].y > 300.0,
            "particle must slide down the wall, not stick: y={}",
            s.particles[0].y
        );
    }

    #[test]
    fn pair_pushout_cannot_leave_a_particle_inside_a_wall() {
        // A particle resting on a wall takes a hit from above: the pair
        // solver's separation pushout shoves the rester into the wall
        // within the same substep. The segment pass runs after pair
        // resolution precisely so the substep still ends with the rester
        // on its side of the wall (the old order left it embedded until
        // the next frame's sweep).
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        assert!(s.add_wall_segment(100.0, 300.0, 700.0, 300.0));
        let (r0, r1) = (s.particles[0].radius, s.particles[1].radius);
        s.particles[0].x = 400.0;
        s.particles[0].y = 300.0 - r0;
        s.particles[1].x = 400.0;
        // Half a pixel of clearance; one step's travel closes it and
        // overlaps the pair, triggering impulse plus separation pushout.
        s.particles[1].y = s.particles[0].y - (r0 + r1) - 0.5;
        s.particles[1].vy = 100.0;

        s.step(0.01, Instant::now(), None);
        assert!(
            s.particles[0].y <= 300.0 - r0 + 1e-6,
            "rester stays on its side of the wall: y={}, r={}",
            s.particles[0].y,
            r0
        );
    }

    #[test]
    fn matter_cannot_fuse_across_a_wall() {
        // Two particles pressed against opposite faces of a wall, closing
        // slowly (fusion range): the wall must keep them apart — same
        // count, same sides — instead of letting them pair through it
        // and fuse onto whichever side the blend lands.
        let mut s = sim(&["--min-particles", "2", "--matter"]);
        freeze(&mut s);
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        s.particles[0].x = 399.0;
        s.particles[0].y = 300.0;
        s.particles[0].vx = 10.0;
        s.particles[1].x = 401.0;
        s.particles[1].y = 300.0;
        s.particles[1].vx = -10.0;

        let now = Instant::now();
        for _ in 0..60 {
            s.step(1.0 / 60.0, now, None);
        }
        assert_eq!(s.particle_count(), 2, "no fusion through the wall");
        assert!(
            s.particles[0].x < 400.0 && s.particles[1].x > 400.0,
            "both particles keep their sides: x0={}, x1={}",
            s.particles[0].x,
            s.particles[1].x
        );
    }

    #[test]
    fn collision_spawns_stay_on_their_side_of_a_wall() {
        // Collisions armed just beside a full-height wall: every birth
        // they trigger must land on the collision's side, never across.
        let mut s = sim(&["--min-particles", "2", "--spawn-mode", "collision"]);
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        let now = Instant::now();
        for round in 0..40 {
            arm_collision(&mut s, 403.0, 60.0 + f64::from(round) * 12.0);
            for _ in 0..6 {
                s.step(0.01, now, None);
            }
            assert!(
                s.particles().iter().all(|p| p.x > 400.0),
                "everything stays right of the wall (round {round})"
            );
        }
        assert!(
            s.particle_count() > 2,
            "collisions actually spawned: {}",
            s.particle_count()
        );
    }

    #[test]
    fn bursts_stay_on_their_side_of_a_wall() {
        // A click burst right beside the wall scatters particles around
        // the click; none may land across the wall.
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[0].x = 700.0;
        s.particles[1].x = 720.0;
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        for _ in 0..4 {
            s.spawn_burst(402.0, 300.0);
        }
        assert!(s.particle_count() > 2, "burst added particles");
        assert!(
            s.particles().iter().all(|p| p.x > 400.0),
            "no burst particle crosses the wall"
        );
    }

    #[test]
    fn fission_fragments_stay_on_their_side_of_a_wall() {
        // A shattering impact beside a wall recedes its fragments
        // perpendicular to the impact — here straight at the wall. The
        // perpendicular's sign must be flipped (or the shatter skipped)
        // so no fragment materializes across.
        let mut s = sim(&["--min-particles", "2", "--matter"]);
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        let now = Instant::now();

        // Phase 1: impact pressed against the wall — neither sign keeps
        // both fragments on the right side, so the shatter is skipped.
        freeze(&mut s);
        s.particles[0].x = 401.0;
        s.particles[0].y = 290.0;
        s.particles[0].vy = 400.0;
        s.particles[1].x = 401.0;
        s.particles[1].y = 310.0;
        s.particles[1].vy = -400.0;
        for _ in 0..30 {
            s.step(1.0 / 120.0, now, None);
            assert!(
                s.particles().iter().all(|p| p.x > 400.0),
                "wall-hugging impact leaks no fragment"
            );
        }
        assert_eq!(s.particle_count(), 2, "shatter skipped at the wall");

        // Phase 2: same impact with room on the right — the shatter goes
        // ahead and every fragment stays on the right side.
        freeze(&mut s);
        s.particles[0].x = 405.0;
        s.particles[0].y = 290.0;
        s.particles[0].vy = 400.0;
        s.particles[1].x = 405.0;
        s.particles[1].y = 310.0;
        s.particles[1].vy = -400.0;
        for _ in 0..30 {
            s.step(1.0 / 120.0, now, None);
            assert!(
                s.particles().iter().all(|p| p.x > 400.0),
                "fragments stay on the impact's side"
            );
        }
        assert_eq!(s.particle_count(), 4, "both participants shattered");
    }

    #[test]
    fn fission_fragments_stay_inside_the_arena() {
        // A horizontal shattering impact hugging the top edge recedes
        // its fragments vertically — one target lies above the arena.
        // Out-of-bounds space is beyond every wall's span, so a fragment
        // parked there could round a wall endpoint later; placements
        // must clamp into the arena.
        let mut s = sim(&["--min-particles", "2", "--matter"]);
        freeze(&mut s);
        s.particles[0].x = 300.0;
        s.particles[0].y = 1.6;
        s.particles[0].vx = 400.0;
        s.particles[1].x = 320.0;
        s.particles[1].y = 1.6;
        s.particles[1].vx = -400.0;

        let now = Instant::now();
        for _ in 0..30 {
            s.step(1.0 / 120.0, now, None);
            for p in s.particles() {
                assert!(
                    p.y >= p.radius - 1e-9
                        && p.y <= 600.0 - p.radius + 1e-9
                        && p.x >= p.radius - 1e-9
                        && p.x <= 800.0 - p.radius + 1e-9,
                    "every fragment stays in bounds: ({}, {}) r {}",
                    p.x,
                    p.y,
                    p.radius
                );
            }
        }
        assert_eq!(s.particle_count(), 4, "the impact did shatter");
    }

    #[test]
    fn out_of_bounds_particles_cannot_round_a_wall_endpoint() {
        // Defense in depth: hand-place a particle in the (normally
        // unreachable) out-of-bounds sliver beyond the divider's top
        // endpoint, moving fast toward the far side — the exact state a
        // buggy position writer would have to produce for a leak. The
        // frame-start clamp must pull it in bounds before physics runs,
        // so the wall still blocks the crossing.
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        s.particles[1].x = 700.0;
        s.particles[1].y = 300.0;
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        s.particles[0].x = 400.4;
        s.particles[0].y = -0.8;
        s.particles[0].vx = -269.0;

        let now = Instant::now();
        for _ in 0..30 {
            s.step(1.0 / 60.0, now, None);
        }
        assert!(
            s.particles[0].x > 400.0,
            "clamped back in bounds and kept on its side: x={}",
            s.particles[0].x
        );
    }

    #[test]
    fn clump_strike_cannot_push_spawns_through_a_wall() {
        // Joe's reproduction: a fused-blob clump full of fission-size
        // smalls rams the divider with a pinned well holding it there.
        // Mass-weighted separation shoves smalls across the wall
        // mid-pass; the sites recorded from those transient positions
        // must not seed births on the far side.
        let mut s = sim(&[
            "--explosion-threshold",
            "0",
            "--particle-elasticity",
            "0.7",
            "--wall-elasticity",
            "0.95",
            "--gravity",
            "0",
            "--self-gravity",
            "--matter",
            "--spawn-mode",
            "collision",
            "--particle-size",
            "0.5",
            "--min-particles",
            "100",
        ]);
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        assert!(s.pin_well(402.0, 300.0, Polarity::Attract));
        strike_clump(&mut s, 2400.0, 300.0);

        let now = Instant::now();
        for frame in 0..150 {
            s.step(1.0 / 60.0, now, None);
            assert!(
                s.particles().iter().all(|p| p.x > 400.0),
                "nothing crosses the divider (frame {frame}, population {})",
                s.particle_count()
            );
        }
    }

    #[test]
    fn a_divided_arena_stays_divided() {
        // The user-facing invariant behind all the wall work: an arena
        // split by a full-height wall, with every particle on the right,
        // must keep its left half empty through a spawning-and-matter
        // runaway pressed against the divider — motion, pair pushout,
        // births, bursts, fusion, and fission all respect the wall.
        let mut s = sim(&[
            "--explosion-threshold",
            "0",
            "--particle-elasticity",
            "0.7",
            "--wall-elasticity",
            "0.9",
            "--gravity",
            "0",
            "--self-gravity",
            "--matter",
            "--spawn-mode",
            "collision",
            "--particle-size",
            "0.5",
            "--min-particles",
            "30",
        ]);
        assert!(s.add_wall_segment(400.0, 0.0, 400.0, 600.0));
        for (k, p) in s.particles.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                p.x = 430.0 + (k % 10) as f64 * 35.0;
                p.y = 60.0 + (k / 10) as f64 * 120.0;
            }
        }
        s.spawn_burst(420.0, 200.0);
        s.spawn_burst(420.0, 400.0);

        let now = Instant::now();
        for frame in 0..300 {
            s.step(1.0 / 60.0, now, None);
            assert!(
                s.particles().iter().all(|p| p.x > 400.0),
                "left half stays empty (frame {frame}, population {})",
                s.particle_count()
            );
        }
        assert!(
            s.particle_count() > 55,
            "the runaway actually ran: {}",
            s.particle_count()
        );
    }

    #[test]
    fn wall_segment_count_is_capped() {
        let mut s = sim(&["--min-particles", "2"]);
        for i in 0..MAX_WALL_SEGMENTS {
            #[allow(clippy::cast_precision_loss)]
            let y = 10.0 + i as f64;
            assert!(s.add_wall_segment(10.0, y, 20.0, y));
        }
        assert!(
            !s.add_wall_segment(10.0, 500.0, 20.0, 500.0),
            "cap must refuse the next segment"
        );
        assert_eq!(s.wall_segments().len(), MAX_WALL_SEGMENTS);

        assert_eq!(s.clear_wall_segments(), MAX_WALL_SEGMENTS);
        assert!(s.wall_segments().is_empty());
    }

    #[test]
    fn spawns_are_blocked_near_walls() {
        let mut s = sim(&["--min-particles", "2"]);
        arm_collision(&mut s, 200.0, 300.0);
        // Fence off the center spawn area (default mode spawns within a
        // few pixels of (400, 300)): parallel walls 4px apart leave no
        // point further than the spawn clearance from a wall.
        for y in [292.0, 296.0, 300.0, 304.0, 308.0] {
            s.add_wall_segment(380.0, y, 420.0, y);
        }

        let events = s.step(0.01, Instant::now(), None);
        assert!(events.max_collision_energy > 0.0, "collision still happens");
        assert_eq!(s.particle_count(), 2, "no spawn inside the wall fence");
        assert!(s.spawn_times.is_empty());
    }

    #[test]
    fn single_startup_well_sits_at_the_center() {
        let s = sim(&["--wells", "1", "--min-particles", "2"]);
        let wells = s.pinned_wells();
        assert_eq!(wells.len(), 1);
        assert!((wells[0].x - 400.0).abs() < 1e-9);
        assert!((wells[0].y - 300.0).abs() < 1e-9);
    }

    #[test]
    fn stopping_is_reported_once_and_ambient_forces_self_wake() {
        let mut s = sim(&["--min-particles", "2"]);
        freeze(&mut s);
        let now = Instant::now();

        // The transition into the stopped state fires the event exactly once.
        let mut stop_events = 0;
        for _ in 0..=2 * MOTION_STOPPED_FRAMES {
            if s.step(0.001, now, None).motion_stopped {
                stop_events += 1;
            }
        }
        assert!(s.stopped());
        assert_eq!(
            stop_events, 1,
            "motion_stopped fires on the transition only"
        );

        // Enabling the flow revives the stopped simulation without any
        // caller-side wake() bookkeeping.
        s.flow = true;
        s.step(0.005, now, None);
        assert!(!s.stopped(), "flow self-wakes a stopped simulation");

        // Same for the held well.
        s.flow = false;
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, None);
        }
        // (Flow gave particles some velocity; re-freeze to stop again.)
        freeze(&mut s);
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.001, now, None);
        }
        assert!(s.stopped());
        let well = Well {
            x: 400.0,
            y: 300.0,
            polarity: Polarity::Attract,
        };
        s.step(0.001, now, Some(well));
        assert!(!s.stopped(), "a held well self-wakes a stopped simulation");
    }

    #[test]
    fn spawn_mode_off_never_spawns() {
        let mut s = sim(&["--min-particles", "2", "--spawn-mode", "off"]);
        arm_collision(&mut s, 200.0, 300.0);
        let events = s.step(0.01, Instant::now(), None);
        assert!(events.max_collision_energy > 0.0, "collision still happens");
        assert_eq!(s.particle_count(), 2, "but nothing spawns");
        assert!(s.spawn_times.is_empty());
    }

    /// Step until the population size changes (matter ops need a few steps
    /// for the armed pair to reach contact).
    fn step_until_count_changes(s: &mut Simulation, max_steps: usize) -> usize {
        let before = s.particle_count();
        for _ in 0..max_steps {
            s.step(0.01, Instant::now(), None);
            if s.particle_count() != before {
                return s.particle_count();
            }
        }
        panic!("population never changed within {max_steps} steps");
    }

    #[test]
    fn slow_contact_fuses_conserving_area_and_momentum() {
        let mut s = sim(&["--min-particles", "30", "--matter", "--spawn-mode", "off"]);
        arm_collision(&mut s, 400.0, 300.0);
        // Slow approach: closing speed 50 is below the fusion threshold.
        s.particles[0].vx = 25.0;
        s.particles[1].vx = -25.0;

        let count = step_until_count_changes(&mut s, 50);
        assert_eq!(count, 29, "fusion merges two particles into one");

        let merged = s
            .particles
            .iter()
            .find(|p| p.radius > s.base_radius + 0.1)
            .expect("a merged particle must exist");
        // Area conserved: r = sqrt(1.5^2 + 1.5^2).
        let expected_r = (2.0f64 * 1.5 * 1.5).sqrt();
        assert!((merged.radius - expected_r).abs() < 1e-9);
        // Equal and opposite momenta cancel.
        assert!(merged.vx.abs() < 1.0, "momentum conserved: {}", merged.vx);
    }

    #[test]
    fn hard_impact_fissions_both_particles() {
        let mut s = sim(&["--min-particles", "30", "--matter", "--spawn-mode", "off"]);
        arm_collision(&mut s, 400.0, 300.0);
        // Violent approach: closing speed 800 exceeds the fission threshold.
        s.particles[0].vx = 400.0;
        s.particles[1].vx = -400.0;

        let count = step_until_count_changes(&mut s, 50);
        assert_eq!(count, 32, "both participants split into two fragments");

        let fragments = s
            .particles
            .iter()
            .filter(|p| p.radius < s.base_radius - 0.1)
            .count();
        assert_eq!(fragments, 4, "four half-area fragments exist");
        let frag = s
            .particles
            .iter()
            .find(|p| p.radius < s.base_radius - 0.1)
            .unwrap();
        let expected_r = 1.5 / std::f64::consts::SQRT_2;
        assert!((frag.radius - expected_r).abs() < 1e-9);
    }

    #[test]
    fn fusion_at_the_cap_is_partial_and_conserves_area_and_momentum() {
        let mut s = sim(&["--min-particles", "30", "--matter", "--spawn-mode", "off"]);
        arm_collision(&mut s, 400.0, 300.0);
        // Slow contact between a blob near the 6x cap (r=9 for base 1.5)
        // and a base-size donor: combined area exceeds the cap.
        s.particles[0].vx = 25.0;
        s.particles[1].vx = -25.0;
        s.particles[0].radius = 8.9;
        s.particles[1].radius = 1.5;
        let max_r = s.base_radius * MAX_RADIUS_FACTOR;
        let min_r = s.base_radius * MIN_RADIUS_FACTOR;
        let area_before: f64 = s.particles.iter().map(Particle::mass).sum();
        let momentum_before: f64 = s.particles.iter().map(|p| p.mass() * p.vx).sum();

        s.step(0.01, Instant::now(), None);

        assert_eq!(s.particle_count(), 30, "partial fusion kills nobody");
        let area_after: f64 = s.particles.iter().map(Particle::mass).sum();
        let momentum_after: f64 = s.particles.iter().map(|p| p.mass() * p.vx).sum();
        assert!((area_before - area_after).abs() < 1e-9, "area conserved");
        assert!(
            (momentum_before - momentum_after).abs() < 1e-6,
            "momentum conserved: {momentum_before} vs {momentum_after}"
        );

        let big = s.particles.iter().map(|p| p.radius).fold(0.0f64, f64::max);
        let small = s
            .particles
            .iter()
            .map(|p| p.radius)
            .fold(f64::INFINITY, f64::min);
        assert!(big > 8.9, "the absorber grew: {big}");
        assert!(big <= max_r + 1e-9, "but never past the cap: {big}");
        assert!(
            small >= min_r - 1e-9,
            "the donor never shrinks below the fragment minimum: {small}"
        );
    }

    #[test]
    fn fission_births_count_toward_the_explosion_threshold() {
        // Regression: a matter run grew past 18,000 particles without ever
        // exploding, because fission fragments multiplied the population
        // while the explosion threshold only counted collision *spawns* —
        // and fission-band collisions never spawn. Births now share one
        // window regardless of mechanism.
        let mut s = sim(&[
            "--min-particles",
            "30",
            "--matter",
            "--explosion-threshold",
            "5",
        ]);
        arm_collision(&mut s, 400.0, 300.0);
        // Violent approach: closing speed 800 is in the fission band, so
        // the old accounting recorded nothing for this collision.
        s.particles[0].vx = 400.0;
        s.particles[1].vx = -400.0;
        // Preload the window to just below the threshold; the two fission
        // fragments must push it over.
        let now = Instant::now();
        for _ in 0..4 {
            s.spawn_times.push_back(now);
        }

        let mut exploded = false;
        for _ in 0..50 {
            if s.step(0.01, now, None).explosion_started {
                exploded = true;
                break;
            }
        }
        assert!(exploded, "fission births must fill the explosion window");
    }

    #[test]
    fn matter_off_means_no_size_changes() {
        let mut s = sim(&["--min-particles", "30", "--spawn-mode", "off"]);
        arm_collision(&mut s, 400.0, 300.0);
        s.particles[0].vx = 400.0;
        s.particles[1].vx = -400.0;
        for _ in 0..20 {
            s.step(0.01, Instant::now(), None);
        }
        assert_eq!(s.particle_count(), 30);
        assert!(
            s.particles
                .iter()
                .all(|p| (p.radius - s.base_radius).abs() < 1e-12)
        );
    }

    #[test]
    fn flow_field_keeps_particles_moving_and_suppresses_stopped() {
        let mut s = sim(&["--min-particles", "2", "--flow"]);
        freeze(&mut s);
        let now = Instant::now();
        for _ in 0..=MOTION_STOPPED_FRAMES {
            s.step(0.005, now, None);
        }
        assert!(!s.stopped(), "flow suppresses the stopped state");
        assert!(
            s.particles.iter().any(|p| p.speed() > 0.0),
            "flow pushes particles"
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut s = sim(&["--min-particles", "10"]);
        s.spawn_burst(400.0, 300.0);
        s.trigger_manual_explosion(400.0, 300.0);
        s.pin_well(100.0, 100.0, Polarity::Attract);
        s.add_wall_segment(50.0, 50.0, 150.0, 50.0);
        s.stopped = true;
        s.spawn_times.push_back(Instant::now());

        s.reset();
        assert_eq!(s.particle_count(), 10);
        assert!(s.explosion().is_none());
        assert!(!s.stopped());
        assert!(s.spawn_times.is_empty());
        assert!(s.pinned_wells().is_empty(), "no startup wells to restore");
        assert!(s.wall_segments().is_empty(), "drawn walls are erased");
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

    /// Build the clump-strike scenario at (650, 300): five max-size fused
    /// blobs in a plus, wrapped in dense rings of fission-size smalls,
    /// everything flying at `(-speed, vy)` — toward a wall at x=400.
    fn strike_clump(s: &mut Simulation, speed: f64, vy: f64) {
        for (k, p) in s.particles.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let kf = k as f64;
            if k < 5 {
                p.radius = 3.0;
                let (dx, dy) = [(0.0, 0.0), (6.5, 0.0), (-6.5, 0.0), (0.0, 6.5), (0.0, -6.5)][k];
                p.x = 650.0 + dx;
                p.y = 300.0 + dy;
            } else {
                p.radius = 0.3;
                let ring = 10.0 + (kf % 5.0) * 1.4;
                let angle = kf * 0.66;
                p.x = 650.0 + ring * angle.cos();
                p.y = 300.0 + ring * angle.sin();
            }
            p.vx = -speed;
            p.vy = vy;
        }
    }

    /// Manual experiment for the clump-strike leak report: a fused blob
    /// cluster stuffed with fission-size smalls is hurled at a full-height
    /// wall across a sweep of speeds and pinned-well setups. Run with
    /// `cargo test --release clump_strike_experiment -- --ignored --nocapture`.
    /// Any particle left of the wall is a leak and is printed with context.
    #[test]
    #[ignore = "manual leak experiment"]
    fn clump_strike_experiment() {
        let wall_x = 400.0;
        for &(speed, vy, well) in &[
            (300.0, 0.0, None),
            (600.0, 0.0, None),
            (600.0, 200.0, None),
            (1200.0, 0.0, None),
            (1200.0, 0.0, Some((410.0, 300.0))),
            (2400.0, 0.0, None),
            (2400.0, 300.0, Some((402.0, 300.0))),
            (6000.0, 0.0, None),
            (6000.0, 0.0, Some((402.0, 300.0))),
            (12000.0, 500.0, None),
        ] {
            let mut s = sim(&[
                "--explosion-threshold",
                "0",
                "--particle-elasticity",
                "0.7",
                "--wall-elasticity",
                "0.95",
                "--gravity",
                "0",
                "--self-gravity",
                "--matter",
                "--spawn-mode",
                "collision",
                "--particle-size",
                "0.5",
                "--min-particles",
                "100",
            ]);
            assert!(s.add_wall_segment(wall_x, 0.0, wall_x, 600.0));
            if let Some((wx, wy)) = well {
                assert!(s.pin_well(wx, wy, Polarity::Attract));
            }
            strike_clump(&mut s, speed, vy);
            let mut leaked = 0u32;
            let now = Instant::now();
            for frame in 0..600 {
                s.step(1.0 / 60.0, now, None);
                for p in s.particles() {
                    if p.x < wall_x && leaked < 5 {
                        leaked += 1;
                        println!(
                            "LEAK speed={speed} vy={vy} well={well:?} frame={frame}: \
                             id {} x={:.3} y={:.3} r={:.2} v=({:.0},{:.0})",
                            p.id, p.x, p.y, p.radius, p.vx, p.vy
                        );
                    }
                }
                if leaked >= 5 {
                    break;
                }
            }
            println!(
                "trial speed={speed} vy={vy} well={well:?}: leaked={leaked} pop={}",
                s.particle_count()
            );
        }
    }

    /// Manual experiment reproducing the divided-arena leak: run with
    /// `cargo test --release leak_experiment -- --ignored --nocapture`.
    /// Attributes every side change to a cause: a matter reposition
    /// (radius changed the same frame), a birth, or — the one class the
    /// wall pass is supposed to make impossible — a pure physics flip.
    #[test]
    #[ignore = "manual leak experiment"]
    fn leak_experiment() {
        use std::collections::HashMap;
        let mut s = sim(&[
            "--explosion-threshold",
            "0",
            "--particle-elasticity",
            "0.7",
            "--wall-elasticity",
            "0.9",
            "--gravity",
            "0",
            "--self-gravity",
            "--matter",
            "--spawn-mode",
            "collision",
            "--particle-size",
            "0.5",
            "--min-particles",
            "60",
        ]);
        let wall_x = 400.0;
        assert!(s.add_wall_segment(wall_x, 0.0, wall_x, 600.0));

        // id -> (side sign, radius, x) from the previous frame.
        let mut seen: HashMap<u64, (f64, f64, f64)> = HashMap::new();
        let mut flips_matter = 0u32;
        let mut flips_physics = 0u32;
        let mut births = [0u32; 2];
        let now = Instant::now();

        let audit = |s: &Simulation,
                     seen: &mut HashMap<u64, (f64, f64, f64)>,
                     frame: usize,
                     flips_matter: &mut u32,
                     flips_physics: &mut u32,
                     births: &mut [u32; 2]| {
            for p in s.particles() {
                let side = (p.x - wall_x).signum();
                if let Some(&(prev_side, prev_r, prev_x)) = seen.get(&p.id) {
                    if side != prev_side && side != 0.0 && prev_side != 0.0 {
                        #[allow(clippy::float_cmp)]
                        let same_radius = p.radius == prev_r;
                        if same_radius {
                            *flips_physics += 1;
                            println!(
                                "frame {frame}: PHYSICS flip id {} x {prev_x:.2}->{:.2} \
                                 r {:.2} v ({:.0},{:.0})",
                                p.id, p.x, p.radius, p.vx, p.vy
                            );
                        } else {
                            *flips_matter += 1;
                            println!(
                                "frame {frame}: MATTER flip id {} x {prev_x:.2}->{:.2} \
                                 r {prev_r:.2}->{:.2}",
                                p.id, p.x, p.radius
                            );
                        }
                    }
                } else {
                    births[usize::from(side > 0.0)] += 1;
                    if (p.x - wall_x).abs() < 25.0 {
                        println!(
                            "frame {frame}: near-wall birth id {} at x {:.2} (side {})",
                            p.id,
                            p.x,
                            if side < 0.0 { "L" } else { "R" }
                        );
                    }
                }
            }
            seen.clear();
            for p in s.particles() {
                seen.insert(p.id, ((p.x - wall_x).signum(), p.radius, p.x));
            }
        };

        // Settle phase.
        for frame in 0..240 {
            s.step(1.0 / 60.0, now, None);
            audit(
                &s,
                &mut seen,
                frame,
                &mut flips_matter,
                &mut flips_physics,
                &mut births,
            );
        }
        let count_sides = |s: &Simulation| {
            let l = s.particles().iter().filter(|p| p.x < wall_x).count();
            (l, s.particle_count() - l)
        };
        let (l0, r0) = count_sides(&s);
        println!("after settle: L={l0} R={r0}");

        // Bursts on the heavier side to force a birth runaway there.
        let burst_x = if l0 >= r0 { 200.0 } else { 600.0 };
        for burst in 0..6 {
            s.spawn_burst(burst_x, 150.0 + f64::from(burst) * 60.0);
        }
        println!("bursts fired at x={burst_x}");

        let frames: usize = std::env::var("LEAK_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2400);
        let mut oob_logged: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for frame in 240..frames {
            s.step(1.0 / 60.0, now, None);
            audit(
                &s,
                &mut seen,
                frame,
                &mut flips_matter,
                &mut flips_physics,
                &mut births,
            );
            // A particle out of bounds at frame end escaped every
            // in-frame clamp — name the writer's victim so the leak
            // mechanism identifies itself.
            for p in s.particles() {
                let r = p.radius;
                if (p.x < r || p.x > 800.0 - r || p.y < r || p.y > 600.0 - r)
                    && oob_logged.insert(p.id)
                {
                    println!(
                        "frame {frame}: OUT-OF-BOUNDS id {} at ({:.3},{:.3}) r {:.2} v ({:.0},{:.0})",
                        p.id, p.x, p.y, p.radius, p.vx, p.vy
                    );
                }
            }
            if frame % 2000 == 0 {
                let l = s.particles().iter().filter(|p| p.x < wall_x).count();
                println!(
                    "frame {frame}: L={l} R={} flips m={flips_matter} p={flips_physics}",
                    s.particle_count() - l
                );
            }
        }
        let (l1, r1) = count_sides(&s);
        println!(
            "final: L={l1} R={r1} | births L={} R={} | flips: matter={flips_matter} \
             physics={flips_physics}",
            births[0], births[1]
        );
    }
}
