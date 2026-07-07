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
    apply_self_gravity, collide_with_segments, handle_collisions, has_motion, max_radius, pair_mut,
    substep_count, update_physics,
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

/// Whether a gravity well pulls particles inward or pushes them away.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Polarity {
    Attract,
    Repel,
}

impl Polarity {
    /// Sign applied to the well strength: attraction is positive.
    pub fn signum(self) -> f64 {
        match self {
            Polarity::Attract => 1.0,
            Polarity::Repel => -1.0,
        }
    }
}

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
    explosion: Option<Explosion>,
    spawn_times: VecDeque<Instant>,
    collisions: CollisionRecorder,
    grid: SpatialGrid,
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
            explosion: None,
            spawn_times: VecDeque::new(),
            collisions: CollisionRecorder::new(),
            grid: SpatialGrid::new(),
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
    /// restoring the startup well layout and erasing drawn walls.
    pub fn reset(&mut self) {
        self.populate();
        self.pinned_wells.clear();
        self.place_initial_wells();
        self.segments.clear();
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

    pub fn pinned_wells(&self) -> &[Well] {
        &self.pinned_wells
    }

    pub fn wall_segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The initial and minimum particle count (screen-derived, or the
    /// --min-particles override).
    pub fn base_particle_count(&self) -> usize {
        self.base_particle_count
    }

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
                (in_bounds && clear).then_some((bx, by))
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

        let gravity_multiplier = f64::from(self.gravity_percent) / 100.0;
        self.collisions.clear();
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
            if self.self_gravity {
                apply_self_gravity(&mut self.particles, sub_dt);
            }
            if self.flow {
                apply_flow(&mut self.particles, self.sim_time, sub_dt);
            }
            update_physics(
                &mut self.particles,
                sub_dt,
                self.width,
                self.height,
                gravity_multiplier,
                self.wall_elasticity,
            );
            if !self.segments.is_empty() {
                collide_with_segments(&mut self.particles, &self.segments, self.wall_elasticity);
            }
            let energy = handle_collisions(
                &mut self.particles,
                &mut self.grid,
                &mut self.collisions,
                self.width,
                self.height,
                self.particle_elasticity,
            );
            max_energy = max_energy.max(energy);
        }

        events.max_collision_energy = max_energy;
        events.collision_pan = self
            .collision_centroid()
            .map_or(0.5, |(x, _)| x / f64::from(self.width));

        if self.matter && self.explosion.is_none() && self.process_matter() {
            // Matter ops reorder the particle list; rebuild the grid so the
            // spawn clearance checks below consult correct positions.
            self.grid.build(
                &self.particles,
                self.width,
                self.height,
                max_radius(&self.particles),
            );
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
    fn absorb(dst: &mut Particle, src: &Particle, mass: f64) {
        let dst_mass = dst.mass();
        let total = dst_mass + mass;
        dst.x = (dst.x * dst_mass + src.x * mass) / total;
        dst.y = (dst.y * dst_mass + src.y * mass) / total;
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
    fn process_matter(&mut self) -> bool {
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
                    Self::absorb(pi, pj, m2);
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
                    Self::absorb(pb, ps, transfer);
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
                    let parent = &mut self.particles[idx];
                    parent.radius = frag_r;
                    parent.x += px * offset;
                    parent.y += py * offset;
                    parent.vx += px * kick;
                    parent.vy += py * kick;
                    let twin = Particle {
                        id: 0, // stamped below, once the parent borrow ends
                        x: parent.x - 2.0 * px * offset,
                        y: parent.y - 2.0 * py * offset,
                        vx: parent.vx - 2.0 * px * kick,
                        vy: parent.vy - 2.0 * py * kick,
                        radius: frag_r,
                        color: parent.color,
                        doomed: false,
                    };
                    let twin = self.stamp(twin);
                    self.particles.push(twin);
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
        // Track spawn rate over sliding window
        let spawn_window = std::time::Duration::from_secs_f64(SPAWN_RATE_WINDOW);
        if let Some(cutoff) = now.checked_sub(spawn_window) {
            while self.spawn_times.front().is_some_and(|&t| t < cutoff) {
                self.spawn_times.pop_front();
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
        for i in 0..self.collisions.sites().len() {
            if spawned >= MAX_SPAWNS_PER_FRAME || self.particles.len() >= self.max_particles {
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
            self.spawn_position_is_free(x, y, first_new, max_r)
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
            if self.spawn_position_is_free(x, y, first_new, max_r) {
                return Some((x, y));
            }
        }
        None
    }

    /// A spawn position is usable when it lies inside the arena and clear
    /// of every existing particle (checked via the grid for pre-existing
    /// particles and linearly for this frame's spawns, which the grid
    /// cannot see yet) and of every drawn wall. The clearance is
    /// conservative: the new particle's radius plus the largest radius in
    /// the population.
    fn spawn_position_is_free(&self, x: f64, y: f64, first_new: usize, max_r: f64) -> bool {
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
                .any(|s| s.distance_to(x, y) < clearance)
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
            s.spawn_times.len(),
            0,
            "no spawn may be recorded when every position is occupied"
        );
        assert_eq!(s.particle_count(), before);
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
    fn manual_explosion_wipes_to_the_floor_and_completes() {
        let mut s = sim(&["--min-particles", "50"]);
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
        assert_eq!(s.particle_count(), 2, "manual blast leaves 2 survivors");
        assert_eq!(completed, Some(48), "completion event reports the kills");
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
}
