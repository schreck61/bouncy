// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Particle physics: motion, gravity, wall bounces, and particle-particle
//! collisions accelerated by a uniform spatial grid.

use rand::Rng;

// Physics constants
pub const GRAVITY: f64 = 100.0;
pub const DEFAULT_PARTICLE_RADIUS: f64 = 1.5;
pub const INITIAL_VELOCITY: f64 = 600.0;
pub const PARTICLE_MARGIN: f64 = 10.0;

// Collision constants
pub const COLLISION_ENERGY_NORMALIZER: f64 = 800.0;
const SEPARATION_PADDING: f64 = 0.5;

// Substepping: cap particle travel per physics step to roughly one radius so
// fast particles cannot tunnel through each other between steps.
const MAX_SUBSTEPS: u32 = 8;

// Motion detection constants
pub const MOTION_VELOCITY_THRESHOLD: f64 = 1.0; // Minimum velocity to be considered "moving"
pub const MOTION_STOPPED_FRAMES: u32 = 60; // Frames of no motion before declaring stopped

// Float comparison constants
const DISTANCE_SQ_EPSILON: f64 = 1e-10; // Minimum distance squared for collision normal

/// Convert f64 color component (0.0-255.0) to u8, clamped to valid range.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn color_component(v: f64) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Convert an HSV hue angle to its sector (0-5) for color calculation.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hue_sector(hue: f64) -> u32 {
    (hue / 60.0) as u32
}

/// Convert an HSV color to an RGBA pixel.
pub fn hsv_to_rgba(hue: f64, saturation: f64, value: f64) -> [u8; 4] {
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

/// Generate a random bright color using HSV color space.
pub fn random_bright_color(rng: &mut impl Rng) -> [u8; 4] {
    hsv_to_rgba(rng.random_range(0.0..360.0), 0.4, 1.0)
}

/// Result of a collision between two particles.
pub struct CollisionResult {
    /// Energy of the collision (relative velocity along collision normal).
    pub energy: f64,
    /// Midpoint x-coordinate where the collision occurred.
    pub mid_x: f64,
    /// Midpoint y-coordinate where the collision occurred.
    pub mid_y: f64,
    /// Unit collision normal (the direction from p1 to p2); the separated
    /// parents lie along this line, so space perpendicular to it is clear.
    pub nx: f64,
    pub ny: f64,
}

/// Where and how a collision happened, kept for spawning: the midpoint plus
/// the collision normal along which the parent particles separated.
#[derive(Copy, Clone)]
pub struct SpawnSite {
    pub x: f64,
    pub y: f64,
    pub nx: f64,
    pub ny: f64,
}

/// A particle in the simulation with position, velocity, and color.
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub color: [u8; 4],
    /// Marked for elimination by an active explosion.
    pub doomed: bool,
}

impl Particle {
    /// Generate a random velocity vector with speed between 50-100% of `INITIAL_VELOCITY`.
    fn random_velocity(rng: &mut impl Rng) -> (f64, f64) {
        let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        let speed: f64 = rng.random_range(INITIAL_VELOCITY * 0.5..INITIAL_VELOCITY);
        (speed * angle.cos(), speed * angle.sin())
    }

    /// Create a particle at a random position within the screen bounds.
    pub fn new_random(rng: &mut impl Rng, width: u32, height: u32) -> Self {
        let x = rng.random_range(PARTICLE_MARGIN..(f64::from(width) - PARTICLE_MARGIN));
        let y = rng.random_range(PARTICLE_MARGIN..(f64::from(height) - PARTICLE_MARGIN));
        Self::new_at_position(rng, x, y)
    }

    /// Create a particle at a specific position.
    pub fn new_at_position(rng: &mut impl Rng, x: f64, y: f64) -> Self {
        let (vx, vy) = Self::random_velocity(rng);
        Particle {
            x,
            y,
            vx,
            vy,
            color: random_bright_color(rng),
            doomed: false,
        }
    }

    /// Current speed in pixels per second.
    pub fn speed(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }

    /// Update particle position with gravity and wall bouncing.
    /// `gravity_multiplier`: 1.0 = 100% gravity, negative = upward gravity
    /// `wall_elasticity`: 1.0 = fully elastic, 0.0 = completely inelastic (sticks)
    pub fn update(
        &mut self,
        dt: f64,
        width: u32,
        height: u32,
        radius: f64,
        gravity_multiplier: f64,
        wall_elasticity: f64,
    ) {
        self.vy += GRAVITY * gravity_multiplier * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        let width_f = f64::from(width);
        let height_f = f64::from(height);

        if self.x <= radius {
            self.x = radius;
            self.vx = -self.vx * wall_elasticity;
        } else if self.x >= width_f - radius {
            self.x = width_f - radius;
            self.vx = -self.vx * wall_elasticity;
        }

        if self.y <= radius {
            self.y = radius;
            self.vy = -self.vy * wall_elasticity;
        } else if self.y >= height_f - radius {
            self.y = height_f - radius;
            self.vy = -self.vy * wall_elasticity;
        }
    }

    /// Calculate squared distance from particle to a point.
    pub fn distance_squared_from(&self, x: f64, y: f64) -> f64 {
        let dx = self.x - x;
        let dy = self.y - y;
        dx * dx + dy * dy
    }
}

/// Attempt a collision between two particles of equal mass.
/// Returns collision result if particles are touching and approaching.
///
/// `particle_elasticity` is the coefficient of restitution: 1.0 = fully
/// elastic (velocities along the normal are exchanged), 0.0 = perfectly
/// inelastic (both particles leave with the common normal velocity). The
/// per-particle impulse for equal masses is `dvn * (1 + e) / 2`.
pub fn try_elastic_collision(
    p1: &mut Particle,
    p2: &mut Particle,
    radius: f64,
    particle_elasticity: f64,
) -> Option<CollisionResult> {
    let diameter = radius * 2.0;
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let dist_sq = dx * dx + dy * dy;

    if !(DISTANCE_SQ_EPSILON..=diameter * diameter).contains(&dist_sq) {
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

    let impulse = dvn * (1.0 + particle_elasticity) / 2.0;
    p1.vx -= impulse * nx;
    p1.vy -= impulse * ny;
    p2.vx += impulse * nx;
    p2.vy += impulse * ny;

    let overlap = diameter - dist;
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
        nx,
        ny,
    })
}

/// Uniform spatial grid for broad-phase collision detection.
///
/// Particles are binned into cells at least one particle diameter wide, so
/// any colliding pair is either in the same cell or in directly adjacent
/// cells. Storage uses an intrusive linked list (`heads` per cell, `next` per
/// particle) so rebuilding each frame allocates nothing in the steady state.
pub struct SpatialGrid {
    cell_size: f64,
    cols: usize,
    rows: usize,
    heads: Vec<i32>,
    next: Vec<i32>,
}

/// Forward half of the 3x3 neighborhood (plus the cell itself handled
/// separately). Visiting only these from each cell yields every adjacent
/// cell pair exactly once.
const FORWARD_NEIGHBORS: [(i64, i64); 4] = [(1, 0), (-1, 1), (0, 1), (1, 1)];

impl SpatialGrid {
    pub fn new() -> Self {
        SpatialGrid {
            cell_size: 1.0,
            cols: 0,
            rows: 0,
            heads: Vec::new(),
            next: Vec::new(),
        }
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn cell_of(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = ((x / self.cell_size) as usize).min(self.cols - 1);
        let cy = ((y / self.cell_size) as usize).min(self.rows - 1);
        (cx, cy)
    }

    /// Rebuild the grid from the current particle positions.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn build(&mut self, particles: &[Particle], width: u32, height: u32, radius: f64) {
        // Cells must be at least one diameter wide; larger cells trade a few
        // extra narrow-phase tests for far fewer cells to clear per frame.
        let cell_size = (radius * 4.0).max(12.0);
        let cols = ((f64::from(width) / cell_size).ceil() as usize).max(1);
        let rows = ((f64::from(height) / cell_size).ceil() as usize).max(1);

        self.cell_size = cell_size;
        self.cols = cols;
        self.rows = rows;
        self.heads.clear();
        self.heads.resize(cols * rows, -1);
        self.next.clear();
        self.next.resize(particles.len(), -1);

        for (i, p) in particles.iter().enumerate() {
            let (cx, cy) = self.cell_of(p.x, p.y);
            let cell = cy * cols + cx;
            self.next[i] = self.heads[cell];
            // Particle indices fit in i32: particle counts are far below i32::MAX.
            self.heads[cell] = i as i32;
        }
    }

    /// Check whether any particle center lies within `dist` of `(x, y)`.
    /// Uses the cell lists from the most recent `build`; `particles` must be
    /// the slice the grid was built from. `dist` must not exceed the cell
    /// size (spawn-clearance queries use one particle diameter, which is
    /// always at most half a cell).
    pub fn any_within(&self, particles: &[Particle], x: f64, y: f64, dist: f64) -> bool {
        if self.heads.is_empty() {
            return false;
        }
        debug_assert!(dist <= self.cell_size);
        let dist_sq = dist * dist;
        let (cx, cy) = self.cell_of(x, y);
        for ny in cy.saturating_sub(1)..=(cy + 1).min(self.rows - 1) {
            for nx in cx.saturating_sub(1)..=(cx + 1).min(self.cols - 1) {
                let mut i = self.heads[ny * self.cols + nx];
                while i >= 0 {
                    #[allow(clippy::cast_sign_loss)]
                    let idx = i as usize;
                    if idx < particles.len()
                        && particles[idx].distance_squared_from(x, y) <= dist_sq
                    {
                        return true;
                    }
                    i = self.next[idx];
                }
            }
        }
        false
    }

    /// Visit every pair of particle indices that could be colliding.
    fn for_each_candidate_pair(&self, mut visit: impl FnMut(usize, usize)) {
        #[allow(clippy::cast_sign_loss)]
        for cy in 0..self.rows {
            for cx in 0..self.cols {
                let cell = cy * self.cols + cx;

                // Pairs within this cell.
                let mut a = self.heads[cell];
                while a >= 0 {
                    let mut b = self.next[a as usize];
                    while b >= 0 {
                        visit(a as usize, b as usize);
                        b = self.next[b as usize];
                    }
                    a = self.next[a as usize];
                }

                // Pairs with forward neighbor cells.
                for (dx, dy) in FORWARD_NEIGHBORS {
                    let nx = cx as i64 + dx;
                    let ny = cy as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= self.cols as i64 || ny >= self.rows as i64 {
                        continue;
                    }
                    let neighbor = (ny as usize) * self.cols + (nx as usize);

                    let mut a = self.heads[cell];
                    while a >= 0 {
                        let mut b = self.heads[neighbor];
                        while b >= 0 {
                            visit(a as usize, b as usize);
                            b = self.next[b as usize];
                        }
                        a = self.next[a as usize];
                    }
                }
            }
        }
    }
}

impl Default for SpatialGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrow two distinct particles mutably by index.
fn pair_mut(particles: &mut [Particle], i: usize, j: usize) -> (&mut Particle, &mut Particle) {
    debug_assert!(i != j);
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = particles.split_at_mut(hi);
    if i < j {
        (&mut left[lo], &mut right[0])
    } else {
        (&mut right[0], &mut left[lo])
    }
}

/// Records collision spawn positions across the substeps of one frame,
/// counting each particle pair at most once.
///
/// The physics impulse must run on every contact so sustained contacts
/// (stacks under gravity, wall traps, click-burst clusters) stay resolved,
/// but re-detecting the same pair each substep must not multiply spawns —
/// spawn accounting stays one-per-pair-per-frame, as it was before
/// substepping existed. Clear at the start of each frame.
pub struct CollisionRecorder {
    sites: Vec<SpawnSite>,
    seen_pairs: std::collections::HashSet<(usize, usize)>,
}

impl CollisionRecorder {
    pub fn new() -> Self {
        CollisionRecorder {
            sites: Vec::with_capacity(100),
            seen_pairs: std::collections::HashSet::new(),
        }
    }

    /// Forget the current frame's collisions. Call once per frame, not per
    /// substep — particle indices are stable within a frame, so pairs remain
    /// identifiable across its substeps.
    pub fn clear(&mut self) {
        self.sites.clear();
        self.seen_pairs.clear();
    }

    /// Record a contact between particles `i` and `j`; the site is kept
    /// only the first time the pair collides this frame.
    fn record(&mut self, i: usize, j: usize, site: SpawnSite) {
        if self.seen_pairs.insert((i.min(j), i.max(j))) {
            self.sites.push(site);
        }
    }

    /// Spawn sites recorded this frame (one per colliding pair).
    pub fn sites(&self) -> &[SpawnSite] {
        &self.sites
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }
}

impl Default for CollisionRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Process all particle-particle collisions using the spatial grid.
/// Returns the maximum collision energy and records spawn positions.
pub fn handle_collisions(
    particles: &mut [Particle],
    grid: &mut SpatialGrid,
    recorder: &mut CollisionRecorder,
    width: u32,
    height: u32,
    radius: f64,
    particle_elasticity: f64,
) -> f64 {
    let mut max_energy = 0.0f64;
    if particles.len() < 2 {
        return max_energy;
    }

    grid.build(particles, width, height, radius);
    grid.for_each_candidate_pair(|i, j| {
        let (p1, p2) = pair_mut(particles, i, j);
        if let Some(result) = try_elastic_collision(p1, p2, radius, particle_elasticity) {
            max_energy = max_energy.max(result.energy);
            recorder.record(
                i,
                j,
                SpawnSite {
                    x: result.mid_x,
                    y: result.mid_y,
                    nx: result.nx,
                    ny: result.ny,
                },
            );
        }
    });

    max_energy
}

/// Acceleration scale of the cursor gravity well (pixels/s^2, asymptotic).
/// Kept below typical particle speeds so the well shepherds particles into
/// orbits rather than instantly dominating their motion.
pub const WELL_STRENGTH: f64 = 1000.0;

/// Pull (positive `strength`) or push (negative) every particle toward/away
/// from `(x, y)`. The softened falloff keeps acceleration gentle near the
/// well center — magnitude approaches `|strength|` at distance and fades to
/// zero at the center, so particles orbit instead of jittering.
pub fn apply_attractor(particles: &mut [Particle], x: f64, y: f64, strength: f64, dt: f64) {
    const SOFTENING: f64 = 50.0;
    for p in particles {
        let dx = x - p.x;
        let dy = y - p.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-6 {
            continue;
        }
        let scale = strength / (dist + SOFTENING);
        p.vx += dx * scale * dt;
        p.vy += dy * scale * dt;
    }
}

/// Update all particle positions with physics simulation.
pub fn update_physics(
    particles: &mut [Particle],
    dt: f64,
    width: u32,
    height: u32,
    radius: f64,
    gravity_multiplier: f64,
    wall_elasticity: f64,
) {
    for particle in particles {
        particle.update(
            dt,
            width,
            height,
            radius,
            gravity_multiplier,
            wall_elasticity,
        );
    }
}

/// Number of physics substeps needed so the fastest particle travels no more
/// than one radius per step, capped at `MAX_SUBSTEPS`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn substep_count(particles: &[Particle], dt: f64, radius: f64) -> u32 {
    let max_speed_sq = particles
        .iter()
        .map(|p| p.vx * p.vx + p.vy * p.vy)
        .fold(0.0f64, f64::max);
    let max_travel = max_speed_sq.sqrt() * dt;
    ((max_travel / radius).ceil() as u32).clamp(1, MAX_SUBSTEPS)
}

/// Check if any particles have significant motion.
pub fn has_motion(particles: &[Particle]) -> bool {
    const THRESHOLD_SQ: f64 = MOTION_VELOCITY_THRESHOLD * MOTION_VELOCITY_THRESHOLD;
    particles
        .iter()
        .any(|p| p.vx * p.vx + p.vy * p.vy > THRESHOLD_SQ)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn particle(x: f64, y: f64, vx: f64, vy: f64) -> Particle {
        Particle {
            x,
            y,
            vx,
            vy,
            color: [255, 255, 255, 255],
            doomed: false,
        }
    }

    const RADIUS: f64 = DEFAULT_PARTICLE_RADIUS;

    #[test]
    fn elastic_collision_exchanges_normal_velocities() {
        // Head-on collision of equal masses at e=1.0 swaps velocities.
        let mut p1 = particle(0.0, 0.0, 100.0, 0.0);
        let mut p2 = particle(2.0, 0.0, -100.0, 0.0);
        let result = try_elastic_collision(&mut p1, &mut p2, RADIUS, 1.0).unwrap();
        assert!((p1.vx - (-100.0)).abs() < 1e-9);
        assert!((p2.vx - 100.0).abs() < 1e-9);
        assert!((result.energy - 200.0).abs() < 1e-9);
        assert!((result.mid_x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn inelastic_collision_sticks() {
        // At e=0.0 both particles leave with the common normal velocity.
        let mut p1 = particle(0.0, 0.0, 100.0, 0.0);
        let mut p2 = particle(2.0, 0.0, -100.0, 0.0);
        try_elastic_collision(&mut p1, &mut p2, RADIUS, 0.0).unwrap();
        assert!(
            p1.vx.abs() < 1e-9,
            "expected common velocity, got {}",
            p1.vx
        );
        assert!(
            p2.vx.abs() < 1e-9,
            "expected common velocity, got {}",
            p2.vx
        );
    }

    #[test]
    fn collision_conserves_momentum() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100 {
            let mut p1 = particle(
                0.0,
                0.0,
                rng.random_range(-500.0..500.0),
                rng.random_range(-500.0..500.0),
            );
            let mut p2 = particle(
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(-500.0..500.0),
                rng.random_range(-500.0..500.0),
            );
            let before = (p1.vx + p2.vx, p1.vy + p2.vy);
            for e in [0.0, 0.5, 1.0, 1.5] {
                try_elastic_collision(&mut p1, &mut p2, RADIUS, e);
                let after = (p1.vx + p2.vx, p1.vy + p2.vy);
                assert!((before.0 - after.0).abs() < 1e-9);
                assert!((before.1 - after.1).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn elastic_collision_conserves_energy() {
        let mut p1 = particle(0.0, 0.0, 300.0, 40.0);
        let mut p2 = particle(1.5, 1.5, -200.0, -80.0);
        let energy_before = p1.vx * p1.vx + p1.vy * p1.vy + p2.vx * p2.vx + p2.vy * p2.vy;
        try_elastic_collision(&mut p1, &mut p2, RADIUS, 1.0).unwrap();
        let energy_after = p1.vx * p1.vx + p1.vy * p1.vy + p2.vx * p2.vx + p2.vy * p2.vy;
        assert!((energy_before - energy_after).abs() < 1e-6);
    }

    #[test]
    fn separating_particles_do_not_collide() {
        let mut p1 = particle(0.0, 0.0, -100.0, 0.0);
        let mut p2 = particle(2.0, 0.0, 100.0, 0.0);
        assert!(try_elastic_collision(&mut p1, &mut p2, RADIUS, 1.0).is_none());
    }

    #[test]
    fn distant_particles_do_not_collide() {
        let mut p1 = particle(0.0, 0.0, 100.0, 0.0);
        let mut p2 = particle(100.0, 0.0, -100.0, 0.0);
        assert!(try_elastic_collision(&mut p1, &mut p2, RADIUS, 1.0).is_none());
    }

    #[test]
    fn grid_finds_same_collisions_as_brute_force() {
        let mut rng = StdRng::seed_from_u64(42);
        let (width, height) = (800u32, 600u32);

        for _ in 0..10 {
            let make = |rng: &mut StdRng| -> Vec<Particle> {
                (0..200)
                    .map(|_| {
                        particle(
                            rng.random_range(RADIUS..f64::from(width) - RADIUS),
                            rng.random_range(RADIUS..f64::from(height) - RADIUS),
                            rng.random_range(-500.0..500.0),
                            rng.random_range(-500.0..500.0),
                        )
                    })
                    .collect()
            };
            let particles = make(&mut rng);

            // Brute force: count touching, approaching pairs.
            let mut brute_pairs = 0;
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let (p1, p2) = (&particles[i], &particles[j]);
                    let (dx, dy) = (p2.x - p1.x, p2.y - p1.y);
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq > 1e-10 && dist_sq <= (RADIUS * 2.0) * (RADIUS * 2.0) {
                        let dist = dist_sq.sqrt();
                        let dvn = (p1.vx - p2.vx) * (dx / dist) + (p1.vy - p2.vy) * (dy / dist);
                        if dvn > 0.0 {
                            brute_pairs += 1;
                        }
                    }
                }
            }

            // Grid: process the identical starting state. Resolved collisions
            // can cascade, but the first pass must find at least the brute
            // force pairs; with sparse random particles counts match exactly.
            let mut grid_particles: Vec<Particle> = particles
                .iter()
                .map(|p| particle(p.x, p.y, p.vx, p.vy))
                .collect();
            let mut grid = SpatialGrid::new();
            let mut recorder = CollisionRecorder::new();
            handle_collisions(
                &mut grid_particles,
                &mut grid,
                &mut recorder,
                width,
                height,
                RADIUS,
                1.0,
            );
            assert_eq!(
                recorder.sites().len(),
                brute_pairs,
                "grid and brute force disagree"
            );
        }
    }

    #[test]
    fn grid_finds_pairs_across_cell_boundaries() {
        let (width, height) = (800u32, 600u32);
        // Straddle a cell boundary: cell size is at least 12, so 11.9 / 12.1
        // are in different cells but within collision distance.
        let mut particles = vec![
            particle(11.9, 50.0, 100.0, 0.0),
            particle(12.1 + RADIUS, 50.0, -100.0, 0.0),
        ];
        let mut grid = SpatialGrid::new();
        let mut recorder = CollisionRecorder::new();
        handle_collisions(
            &mut particles,
            &mut grid,
            &mut recorder,
            width,
            height,
            RADIUS,
            1.0,
        );
        assert_eq!(recorder.sites().len(), 1);
    }

    #[test]
    fn sustained_contact_spawns_once_per_frame() {
        // Regression: a pair in resting contact (gravity presses the upper
        // particle into the lower one, which sits on the floor) re-collides
        // on every substep. The impulse must run each time, but only ONE
        // spawn position may be recorded per pair per frame.
        let (width, height) = (200u32, 200u32);
        let floor_y = f64::from(height) - RADIUS;
        let mut particles = vec![
            particle(100.0, floor_y, 0.0, 0.0), // resting on the floor
            particle(100.0, floor_y - 2.0 * RADIUS, 0.0, 0.0), // stacked on top
        ];
        let mut grid = SpatialGrid::new();
        let mut recorder = CollisionRecorder::new();

        // Strong gravity presses the stack back into contact within a few
        // substeps of each separation, but never fast enough to tunnel.
        let gravity_multiplier = 20.0;
        let sub_dt = 0.01;
        let mut run_frame = |particles: &mut Vec<Particle>,
                             recorder: &mut CollisionRecorder|
         -> u32 {
            recorder.clear();
            let mut contact_substeps = 0;
            for _ in 0..8 {
                update_physics(
                    particles,
                    sub_dt,
                    width,
                    height,
                    RADIUS,
                    gravity_multiplier,
                    0.0,
                );
                let energy =
                    handle_collisions(particles, &mut grid, recorder, width, height, RADIUS, 0.0);
                if energy > 0.0 {
                    contact_substeps += 1;
                }
            }
            contact_substeps
        };

        let contact_substeps = run_frame(&mut particles, &mut recorder);
        assert!(
            contact_substeps > 1,
            "test setup must produce repeated contact, got {contact_substeps}"
        );
        assert_eq!(
            recorder.sites().len(),
            1,
            "a pair may spawn only once per frame"
        );

        // Physics still resolved every substep: the stack has not interpenetrated.
        let dist_sq = particles[0].distance_squared_from(particles[1].x, particles[1].y);
        assert!(
            dist_sq.sqrt() >= 2.0 * RADIUS - 0.1,
            "particles sank into each other: dist={}",
            dist_sq.sqrt()
        );

        // A new frame (recorder cleared) records the pair again — the dedup
        // is per frame, not forever.
        let contact_substeps = run_frame(&mut particles, &mut recorder);
        assert!(contact_substeps >= 1);
        assert_eq!(recorder.sites().len(), 1);
    }

    #[test]
    fn grid_any_within_finds_nearby_particles() {
        let (width, height) = (800u32, 600u32);
        let particles = vec![particle(100.0, 100.0, 0.0, 0.0)];
        let mut grid = SpatialGrid::new();

        // Unbuilt grid: nothing is nearby.
        assert!(!grid.any_within(&particles, 100.0, 100.0, 10.0));

        grid.build(&particles, width, height, RADIUS);
        assert!(grid.any_within(&particles, 100.0, 100.0, 10.0), "same spot");
        assert!(
            grid.any_within(&particles, 105.0, 100.0, 10.0),
            "within distance, possibly a neighboring cell"
        );
        assert!(
            !grid.any_within(&particles, 120.0, 100.0, 10.0),
            "beyond distance"
        );
        assert!(!grid.any_within(&particles, 700.0, 500.0, 10.0), "far away");
        // Out-of-bounds query coordinates must clamp, not panic.
        assert!(!grid.any_within(&particles, -50.0, -50.0, 10.0));
        assert!(!grid.any_within(&particles, 5000.0, 5000.0, 10.0));
    }

    #[test]
    fn wall_bounce_reflects_velocity() {
        let mut p = particle(1.0, 50.0, -100.0, 0.0);
        p.update(0.01, 200, 200, RADIUS, 0.0, 1.0);
        assert!(p.vx > 0.0);
        assert!((p.x - RADIUS).abs() < 1e-9);
    }

    #[test]
    fn gravity_accelerates_downward() {
        let mut p = particle(100.0, 100.0, 0.0, 0.0);
        p.update(0.1, 200, 200, RADIUS, 1.0, 1.0);
        assert!((p.vy - GRAVITY * 0.1).abs() < 1e-9);
    }

    #[test]
    fn negative_gravity_accelerates_upward() {
        let mut p = particle(100.0, 100.0, 0.0, 0.0);
        p.update(0.1, 200, 200, RADIUS, -1.0, 1.0);
        assert!((p.vy + GRAVITY * 0.1).abs() < 1e-9);
    }

    #[test]
    fn substep_count_scales_with_speed() {
        let slow = vec![particle(50.0, 50.0, 10.0, 0.0)];
        let fast = vec![particle(50.0, 50.0, 600.0, 0.0)];
        assert_eq!(substep_count(&slow, 0.008, RADIUS), 1);
        assert!(substep_count(&fast, 0.008, RADIUS) > 1);
        assert!(substep_count(&fast, 10.0, RADIUS) <= MAX_SUBSTEPS);
        assert_eq!(substep_count(&[], 0.008, RADIUS), 1);
    }

    #[test]
    fn attractor_pulls_toward_and_repels_from_point() {
        let mut particles = vec![particle(100.0, 100.0, 0.0, 0.0)];
        apply_attractor(&mut particles, 200.0, 100.0, WELL_STRENGTH, 0.01);
        assert!(particles[0].vx > 0.0, "attract must pull toward the well");
        assert!(particles[0].vy.abs() < 1e-9);

        let mut particles = vec![particle(100.0, 100.0, 0.0, 0.0)];
        apply_attractor(&mut particles, 200.0, 100.0, -WELL_STRENGTH, 0.01);
        assert!(particles[0].vx < 0.0, "repel must push away from the well");

        // A particle exactly at the well center must not produce NaN.
        let mut particles = vec![particle(200.0, 100.0, 0.0, 0.0)];
        apply_attractor(&mut particles, 200.0, 100.0, WELL_STRENGTH, 0.01);
        assert!(particles[0].vx.is_finite() && particles[0].vy.is_finite());
        assert_eq!(particles[0].vx, 0.0);
    }

    #[test]
    fn attractor_acceleration_is_softened_near_center() {
        let mut near = vec![particle(195.0, 100.0, 0.0, 0.0)];
        let mut far = vec![particle(0.0, 100.0, 0.0, 0.0)];
        apply_attractor(&mut near, 200.0, 100.0, WELL_STRENGTH, 0.01);
        apply_attractor(&mut far, 200.0, 100.0, WELL_STRENGTH, 0.01);
        assert!(
            near[0].vx.abs() < far[0].vx.abs(),
            "acceleration must fade near the well center"
        );
    }

    #[test]
    fn has_motion_detects_stopped_particles() {
        let moving = vec![particle(0.0, 0.0, 10.0, 0.0)];
        let stopped = vec![particle(0.0, 0.0, 0.1, 0.1)];
        assert!(has_motion(&moving));
        assert!(!has_motion(&stopped));
    }

    #[test]
    fn hsv_produces_valid_bright_colors() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            let c = random_bright_color(&mut rng);
            assert_eq!(c[3], 255);
            // With value=1.0 and saturation=0.4, max channel is always 255.
            assert_eq!(*c.iter().take(3).max().unwrap(), 255);
        }
    }
}
