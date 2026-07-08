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

/// Where and how a collision happened, kept for spawning and for matter
/// (fusion/fission) processing: the participants, the midpoint, the
/// collision normal along which the parent particles separated, and the
/// collision energy.
#[derive(Copy, Clone)]
pub struct SpawnSite {
    /// Indices of the colliding pair, valid until the particle list is next
    /// mutated (spawning/matter processing happens before any mutation).
    pub i: usize,
    pub j: usize,
    pub x: f64,
    pub y: f64,
    pub nx: f64,
    pub ny: f64,
    pub energy: f64,
}

/// Stable particle identity, unique within one `Simulation` run and never
/// reused.
pub type ParticleId = u64;

/// A particle in the simulation with position, velocity, size, and color.
pub struct Particle {
    /// Stable identity. Positions in the particle Vec are invalidated by
    /// `swap_remove` (matter events) and `retain` (explosion kills), so any
    /// feature that stores cross-frame references to a particle must hold
    /// its id, not its index. Ids are stamped by the Simulation; particles
    /// constructed outside one (tests) carry 0.
    pub id: ParticleId,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    /// Radius in pixels; mass is proportional to area (radius squared).
    pub radius: f64,
    pub color: [u8; 4],
    /// Marked for elimination by an active explosion.
    pub doomed: bool,
}

impl Particle {
    /// Generate a random velocity vector with speed between 50-100% of `max_speed`.
    fn random_velocity(rng: &mut impl Rng, max_speed: f64) -> (f64, f64) {
        let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        let speed: f64 = rng.random_range(max_speed * 0.5..max_speed.max(1e-9));
        (speed * angle.cos(), speed * angle.sin())
    }

    /// Create a particle at a random position within the screen bounds.
    pub fn new_random(
        rng: &mut impl Rng,
        width: u32,
        height: u32,
        radius: f64,
        max_speed: f64,
    ) -> Self {
        let x = rng.random_range(PARTICLE_MARGIN..(f64::from(width) - PARTICLE_MARGIN));
        let y = rng.random_range(PARTICLE_MARGIN..(f64::from(height) - PARTICLE_MARGIN));
        Self::new_at_position(rng, x, y, radius, max_speed)
    }

    /// Create a particle at a specific position with a random velocity.
    pub fn new_at_position(
        rng: &mut impl Rng,
        x: f64,
        y: f64,
        radius: f64,
        max_speed: f64,
    ) -> Self {
        let (vx, vy) = Self::random_velocity(rng, max_speed);
        Self::new_moving(rng, x, y, vx, vy, radius)
    }

    /// Create a particle with a specific position and velocity. The id is
    /// 0 (unstamped) until the owning Simulation assigns one.
    pub fn new_moving(rng: &mut impl Rng, x: f64, y: f64, vx: f64, vy: f64, radius: f64) -> Self {
        Particle {
            id: 0,
            x,
            y,
            vx,
            vy,
            radius,
            color: crate::color::random_bright_color(rng),
            doomed: false,
        }
    }

    /// Mass, proportional to area. The proportionality constant cancels in
    /// every use, so area itself serves as the mass.
    pub fn mass(&self) -> f64 {
        self.radius * self.radius
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
        gravity_multiplier: f64,
        wall_elasticity: f64,
    ) {
        self.vy += GRAVITY * gravity_multiplier * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        let width_f = f64::from(width);
        let height_f = f64::from(height);
        let radius = self.radius;

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

/// Attempt a collision between two particles, which may have unequal
/// masses. Returns collision result if particles are touching (center
/// distance within the sum of radii) and approaching.
///
/// `particle_elasticity` is the coefficient of restitution: 1.0 = fully
/// elastic, 0.0 = perfectly inelastic (both leave with the common normal
/// velocity). The impulse uses the standard reduced-mass form
/// `j = (1 + e) * dvn * m1*m2/(m1 + m2)`, which for equal masses reduces to
/// the familiar per-particle velocity change of `dvn * (1 + e) / 2`.
pub fn try_elastic_collision(
    p1: &mut Particle,
    p2: &mut Particle,
    particle_elasticity: f64,
) -> Option<CollisionResult> {
    let contact = p1.radius + p2.radius;
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let dist_sq = dx * dx + dy * dy;

    if !(DISTANCE_SQ_EPSILON..=contact * contact).contains(&dist_sq) {
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

    let (m1, m2) = (p1.mass(), p2.mass());
    let impulse = (1.0 + particle_elasticity) * dvn * (m1 * m2) / (m1 + m2);
    p1.vx -= (impulse / m1) * nx;
    p1.vy -= (impulse / m1) * ny;
    p2.vx += (impulse / m2) * nx;
    p2.vy += (impulse / m2) * ny;

    // Separate overlapping particles, the lighter one moving further.
    let overlap = contact - dist;
    if overlap > 0.0 {
        let total = overlap + 2.0 * SEPARATION_PADDING;
        let share1 = m2 / (m1 + m2);
        p1.x -= total * share1 * nx;
        p1.y -= total * share1 * ny;
        p2.x += total * (1.0 - share1) * nx;
        p2.y += total * (1.0 - share1) * ny;
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

    /// Rebuild the grid from the current particle positions. `max_radius`
    /// is the largest particle radius present; the cell size must cover the
    /// largest possible contact distance so that colliding pairs are always
    /// in the same or adjacent cells.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn build(
        &mut self,
        particles: &[Particle],
        width: u32,
        height: u32,
        max_radius: f64,
    ) {
        // Cells must be at least one max-diameter wide; larger cells trade a
        // few extra narrow-phase tests for fewer cells to clear per frame.
        let cell_size = (max_radius * 4.0).max(12.0);
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
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                self.heads[cell] = i as i32;
            }
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
    /// Grid dimensions are far below any integer limit, so the index casts
    /// in the neighbor arithmetic are lossless.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation
    )]
    fn for_each_candidate_pair(&self, mut visit: impl FnMut(usize, usize)) {
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
pub(crate) fn pair_mut(
    particles: &mut [Particle],
    i: usize,
    j: usize,
) -> (&mut Particle, &mut Particle) {
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

/// The largest particle radius in the population (0.0 when empty).
pub fn max_radius(particles: &[Particle]) -> f64 {
    particles.iter().map(|p| p.radius).fold(0.0f64, f64::max)
}

/// Process all particle-particle collisions using the spatial grid.
/// Returns the maximum collision energy and records spawn positions.
pub fn handle_collisions(
    particles: &mut [Particle],
    grid: &mut SpatialGrid,
    recorder: &mut CollisionRecorder,
    width: u32,
    height: u32,
    particle_elasticity: f64,
) -> f64 {
    let mut max_energy = 0.0f64;
    if particles.len() < 2 {
        return max_energy;
    }

    grid.build(particles, width, height, max_radius(particles));
    grid.for_each_candidate_pair(|i, j| {
        let (p1, p2) = pair_mut(particles, i, j);
        if let Some(result) = try_elastic_collision(p1, p2, particle_elasticity) {
            max_energy = max_energy.max(result.energy);
            recorder.record(
                i,
                j,
                SpawnSite {
                    i,
                    j,
                    x: result.mid_x,
                    y: result.mid_y,
                    nx: result.nx,
                    ny: result.ny,
                    energy: result.energy,
                },
            );
        }
    });

    max_energy
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

/// Peak acceleration of the cursor gravity well (pixels/s^2), reached at
/// roughly `WELL_SOFTENING / sqrt(2)` pixels from the cursor.
pub const WELL_STRENGTH: f64 = 800.0;
/// Softening length of the well: inside it the pull fades linearly to zero
/// (no jitter at the center); outside it the pull falls off as 1/d^2.
const WELL_SOFTENING: f64 = 120.0;
/// Peak of d/(d^2+s^2)^(3/2), times s^2: at d = s/sqrt(2) the curve reaches
/// 1/(sqrt(2) * 1.5^1.5 * s^2). Used to normalize the peak to `WELL_STRENGTH`.
const WELL_PEAK_FACTOR: f64 = 0.384_900_179_459_750_5;

/// Pull (positive `strength`) or push (negative) every particle toward/away
/// from `(x, y)` like a softened point mass (Plummer): acceleration rises
/// linearly from zero at the center, peaks at `|strength|` near the
/// softening radius, and falls off as 1/d^2 at range — local control, not
/// screen-wide suction.
pub fn apply_attractor(particles: &mut [Particle], x: f64, y: f64, strength: f64, dt: f64) {
    let s2 = WELL_SOFTENING * WELL_SOFTENING;
    // a(d) = k * d / (d^2 + s^2)^(3/2), normalized so max |a| = |strength|.
    let k = strength * s2 / WELL_PEAK_FACTOR;
    for p in particles {
        let dx = x - p.x;
        let dy = y - p.y;
        let d2 = dx * dx + dy * dy;
        let scale = k / (d2 + s2).powf(1.5);
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
    gravity_multiplier: f64,
    wall_elasticity: f64,
) {
    for particle in particles {
        particle.update(dt, width, height, gravity_multiplier, wall_elasticity);
    }
}

/// A static wall segment drawn by the user (held V). Particles bounce off
/// it under the same elasticity rule as the arena walls.
#[derive(Copy, Clone)]
pub struct Segment {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl Segment {
    /// Distance from `(x, y)` to the nearest point of this segment.
    pub fn distance_to(&self, x: f64, y: f64) -> f64 {
        let (cx, cy) = self.closest_point(x, y);
        let (dx, dy) = (x - cx, y - cy);
        (dx * dx + dy * dy).sqrt()
    }

    /// The point on the segment closest to `(x, y)`.
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        let (abx, aby) = (self.x2 - self.x1, self.y2 - self.y1);
        let len2 = abx * abx + aby * aby;
        let t = if len2 > 0.0 {
            (((x - self.x1) * abx + (y - self.y1) * aby) / len2).clamp(0.0, 1.0)
        } else {
            0.0 // degenerate segment: a point
        };
        (self.x1 + t * abx, self.y1 + t * aby)
    }
}

/// Bounce particles off the drawn wall segments. A particle overlapping a
/// segment is pushed out along the contact normal (closest point to
/// center), and its approaching velocity component reflects scaled by
/// `wall_elasticity` — the arena-wall rule. A particle already receding
/// only gets the position correction, so a bounce is never re-reflected
/// back into the wall it just left.
pub fn collide_with_segments(
    particles: &mut [Particle],
    segments: &[Segment],
    wall_elasticity: f64,
) {
    for p in particles.iter_mut() {
        for seg in segments {
            // Cheap AABB reject before the exact closest-point test.
            let r = p.radius;
            if p.x < seg.x1.min(seg.x2) - r
                || p.x > seg.x1.max(seg.x2) + r
                || p.y < seg.y1.min(seg.y2) - r
                || p.y > seg.y1.max(seg.y2) + r
            {
                continue;
            }
            let (cx, cy) = seg.closest_point(p.x, p.y);
            let (dx, dy) = (p.x - cx, p.y - cy);
            let d2 = dx * dx + dy * dy;
            if d2 >= r * r {
                continue;
            }
            // Contact normal from the wall toward the particle center. A
            // dead-center hit has no direction; push out perpendicular to
            // the segment (or straight up for a degenerate point segment).
            let d = d2.sqrt();
            let (nx, ny) = if d > 1e-9 {
                (dx / d, dy / d)
            } else {
                let (abx, aby) = (seg.x2 - seg.x1, seg.y2 - seg.y1);
                let len = (abx * abx + aby * aby).sqrt();
                if len > 1e-9 {
                    (-aby / len, abx / len)
                } else {
                    (0.0, -1.0)
                }
            };
            p.x = cx + nx * r;
            p.y = cy + ny * r;
            let vn = p.vx * nx + p.vy * ny;
            if vn < 0.0 {
                p.vx -= (1.0 + wall_elasticity) * vn * nx;
                p.vy -= (1.0 + wall_elasticity) * vn * ny;
            }
        }
    }
}

/// Gravitational constant for particle self-gravity, in pixel/mass units
/// (masses are areas: ~2.25 for a default particle). Tuned for
/// screen-scale dynamics: a single pair barely drifts together, but a
/// clustered population collapses in seconds — collective attraction is
/// the point.
pub const SELF_GRAVITY_G: f64 = 2500.0;
/// Plummer softening length for self-gravity (pixels): caps the 1/d²
/// singularity so close encounters swing by instead of slingshotting to
/// infinity. Pairs this close are in collision range anyway.
const SELF_GRAVITY_SOFTENING: f64 = 10.0;

/// Apply mutual Newtonian gravity between every particle pair, softened
/// like the cursor well and applied symmetrically (Newton's third law),
/// so momentum is conserved exactly and heavier particles both pull
/// harder and yield less. O(n²) per substep by design: correct and fast
/// for the preset-scale populations self-gravity runs with (~5k pair
/// evaluations at 100 particles); the documented next tier for thousands
/// of particles is a far-field approximation over the spatial grid
/// (per-cell mass and center of mass).
pub fn apply_self_gravity(particles: &mut [Particle], dt: f64) {
    let s2 = SELF_GRAVITY_SOFTENING * SELF_GRAVITY_SOFTENING;
    for i in 0..particles.len() {
        let (left, right) = particles.split_at_mut(i + 1);
        let pi = &mut left[i];
        for pj in &mut *right {
            let dx = pj.x - pi.x;
            let dy = pj.y - pi.y;
            let softened = dx * dx + dy * dy + s2;
            // a_i = G * m_j * d / (d² + s²)^(3/2), and symmetrically for j.
            let scale = SELF_GRAVITY_G * dt / (softened * softened.sqrt());
            let (fx, fy) = (dx * scale, dy * scale);
            pi.vx += fx * pj.mass();
            pi.vy += fy * pj.mass();
            pj.vx -= fx * pi.mass();
            pj.vy -= fy * pi.mass();
        }
    }
}

/// Speed of the ambient flow field's currents (pixels/s). Gentle by design:
/// the flow should read as drift, not wind tunnel.
pub const FLOW_SPEED: f64 = 60.0;
/// How quickly particles entrain into the flow (1/s). The flow is a medium,
/// not a force: particles are dragged *toward* the local current velocity,
/// so speeds stay bounded instead of accumulating until a wall eats them.
const FLOW_COUPLING: f64 = 1.2;

/// Entrain every particle toward a slowly-drifting swirl of currents. The
/// field direction is a few layered sinusoids of position and time — cheap,
/// smooth, deterministic — with gentle gusts in magnitude.
pub fn apply_flow(particles: &mut [Particle], t: f64, dt: f64) {
    let k = (FLOW_COUPLING * dt).min(1.0);
    for p in particles {
        let angle = (p.x * 0.008 + t * 0.35).sin() * 2.2
            + (p.y * 0.011 - t * 0.23).cos() * 1.9
            + (p.x * 0.002 - p.y * 0.003 + t * 0.11).sin() * 1.3;
        let gust = 0.7 + 0.3 * (p.x * 0.004 + p.y * 0.005 + t * 0.17).sin();
        let fx = angle.cos() * FLOW_SPEED * gust;
        let fy = angle.sin() * FLOW_SPEED * gust;
        p.vx += (fx - p.vx) * k;
        p.vy += (fy - p.vy) * k;
    }
}

/// Number of physics substeps needed so the fastest particle travels no
/// more than one (smallest) radius per step, capped at `MAX_SUBSTEPS`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn substep_count(particles: &[Particle], dt: f64) -> u32 {
    let max_speed_sq = particles
        .iter()
        .map(|p| p.vx * p.vx + p.vy * p.vy)
        .fold(0.0f64, f64::max);
    let min_radius = particles
        .iter()
        .map(|p| p.radius)
        .fold(f64::INFINITY, f64::min);
    if !min_radius.is_finite() {
        return 1;
    }
    let max_travel = max_speed_sq.sqrt() * dt;
    ((max_travel / min_radius).ceil() as u32).clamp(1, MAX_SUBSTEPS)
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
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const RADIUS: f64 = DEFAULT_PARTICLE_RADIUS;

    fn particle(x: f64, y: f64, vx: f64, vy: f64) -> Particle {
        particle_r(x, y, vx, vy, RADIUS)
    }

    fn particle_r(x: f64, y: f64, vx: f64, vy: f64, radius: f64) -> Particle {
        Particle {
            id: 0,
            x,
            y,
            vx,
            vy,
            radius,
            color: [255, 255, 255, 255],
            doomed: false,
        }
    }

    #[test]
    fn self_gravity_attracts_pairs_and_conserves_momentum() {
        let mut particles = vec![
            particle(100.0, 300.0, 0.0, 0.0),
            particle(300.0, 300.0, 0.0, 0.0),
        ];
        apply_self_gravity(&mut particles, 0.1);
        assert!(particles[0].vx > 0.0, "left particle pulled right");
        assert!(particles[1].vx < 0.0, "right particle pulled left");
        assert!(
            particles[0].vy.abs() < 1e-12 && particles[1].vy.abs() < 1e-12,
            "no lateral force for a horizontal pair"
        );
        // Equal masses: velocities are exactly opposite (momentum zero).
        assert!((particles[0].vx + particles[1].vx).abs() < 1e-12);
    }

    #[test]
    fn heavier_particles_pull_harder_and_accelerate_less() {
        // m = 9 vs m = 2.25: the light particle must gain more speed, and
        // total momentum must stay zero.
        let mut particles = vec![
            particle_r(100.0, 300.0, 0.0, 0.0, 3.0),
            particle(300.0, 300.0, 0.0, 0.0),
        ];
        apply_self_gravity(&mut particles, 0.1);
        let (heavy, light) = (&particles[0], &particles[1]);
        assert!(
            light.vx.abs() > heavy.vx.abs() * 3.0,
            "light particle accelerates ~4x more: heavy={}, light={}",
            heavy.vx,
            light.vx
        );
        let momentum = heavy.mass() * heavy.vx + light.mass() * light.vx;
        assert!(momentum.abs() < 1e-9, "momentum conserved: {momentum}");
    }

    #[test]
    fn self_gravity_softening_caps_close_encounters() {
        // A near-overlapping pair must receive a finite, modest kick, not
        // a slingshot: the softened force peaks near the softening length.
        let mut close = vec![
            particle(400.0, 300.0, 0.0, 0.0),
            particle(400.5, 300.0, 0.0, 0.0),
        ];
        apply_self_gravity(&mut close, 0.01);
        assert!(close[0].vx.is_finite());
        assert!(
            close[0].vx.abs() < 10.0,
            "softening keeps the kick modest: {}",
            close[0].vx
        );
    }

    #[test]
    fn particle_bounces_off_a_wall_segment() {
        // Horizontal wall at y=100; particle overlaps it from above while
        // falling. It must be pushed out to rest on the wall and reflect.
        let wall = Segment {
            x1: 50.0,
            y1: 100.0,
            x2: 150.0,
            y2: 100.0,
        };
        let mut particles = vec![particle(100.0, 99.0, 30.0, 200.0)];
        collide_with_segments(&mut particles, &[wall], 1.0);

        let p = &particles[0];
        assert!((p.y - (100.0 - RADIUS)).abs() < 1e-9, "pushed out: {}", p.y);
        assert!((p.vy - (-200.0)).abs() < 1e-9, "reflected: {}", p.vy);
        assert!((p.vx - 30.0).abs() < 1e-9, "tangential velocity untouched");
    }

    #[test]
    fn segment_bounce_respects_wall_elasticity() {
        let wall = Segment {
            x1: 50.0,
            y1: 100.0,
            x2: 150.0,
            y2: 100.0,
        };
        let mut particles = vec![particle(100.0, 99.0, 0.0, 200.0)];
        collide_with_segments(&mut particles, &[wall], 0.5);
        assert!(
            (particles[0].vy - (-100.0)).abs() < 1e-9,
            "half-elastic wall halves the rebound: {}",
            particles[0].vy
        );
    }

    #[test]
    fn receding_particle_is_pushed_out_but_not_reflected() {
        let wall = Segment {
            x1: 50.0,
            y1: 100.0,
            x2: 150.0,
            y2: 100.0,
        };
        // Overlapping but already moving away from the wall.
        let mut particles = vec![particle(100.0, 99.0, 0.0, -50.0)];
        collide_with_segments(&mut particles, &[wall], 1.0);
        let p = &particles[0];
        assert!((p.y - (100.0 - RADIUS)).abs() < 1e-9, "still pushed out");
        assert!((p.vy - (-50.0)).abs() < 1e-9, "velocity left alone");
    }

    #[test]
    fn segment_endpoints_and_degenerate_segments_are_safe() {
        // Hit past the endpoint: the corner acts like a point obstacle.
        let wall = Segment {
            x1: 50.0,
            y1: 100.0,
            x2: 150.0,
            y2: 100.0,
        };
        let mut particles = vec![particle(151.0, 100.0, -100.0, 0.0)];
        collide_with_segments(&mut particles, &[wall], 1.0);
        let p = &particles[0];
        assert!(p.x >= 150.0 + RADIUS - 1e-9, "pushed off the endpoint");
        assert!(p.vx > 0.0, "reflected off the corner");

        // A zero-length segment exactly at the particle center must not
        // produce NaN; the particle is pushed out along the fallback normal.
        let point = Segment {
            x1: 200.0,
            y1: 200.0,
            x2: 200.0,
            y2: 200.0,
        };
        let mut particles = vec![particle(200.0, 200.0, 0.0, 0.0)];
        collide_with_segments(&mut particles, &[point], 1.0);
        let p = &particles[0];
        assert!(p.x.is_finite() && p.y.is_finite() && p.vx.is_finite() && p.vy.is_finite());
        assert!(
            point.distance_to(p.x, p.y) >= RADIUS - 1e-9,
            "no longer overlapping"
        );
    }

    #[test]
    fn segment_distance_measures_to_closest_point() {
        let seg = Segment {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 0.0,
        };
        assert!((seg.distance_to(5.0, 3.0) - 3.0).abs() < 1e-9, "broadside");
        assert!((seg.distance_to(13.0, 4.0) - 5.0).abs() < 1e-9, "past end");
        assert!((seg.distance_to(5.0, 0.0)).abs() < 1e-9, "on the segment");
    }

    #[test]
    fn elastic_collision_exchanges_normal_velocities() {
        // Head-on collision of equal masses at e=1.0 swaps velocities.
        let mut p1 = particle(0.0, 0.0, 100.0, 0.0);
        let mut p2 = particle(2.0, 0.0, -100.0, 0.0);
        let result = try_elastic_collision(&mut p1, &mut p2, 1.0).unwrap();
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
        try_elastic_collision(&mut p1, &mut p2, 0.0).unwrap();
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
                try_elastic_collision(&mut p1, &mut p2, e);
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
        try_elastic_collision(&mut p1, &mut p2, 1.0).unwrap();
        let energy_after = p1.vx * p1.vx + p1.vy * p1.vy + p2.vx * p2.vx + p2.vy * p2.vy;
        assert!((energy_before - energy_after).abs() < 1e-6);
    }

    #[test]
    fn separating_particles_do_not_collide() {
        let mut p1 = particle(0.0, 0.0, -100.0, 0.0);
        let mut p2 = particle(2.0, 0.0, 100.0, 0.0);
        assert!(try_elastic_collision(&mut p1, &mut p2, 1.0).is_none());
    }

    #[test]
    fn distant_particles_do_not_collide() {
        let mut p1 = particle(0.0, 0.0, 100.0, 0.0);
        let mut p2 = particle(100.0, 0.0, -100.0, 0.0);
        assert!(try_elastic_collision(&mut p1, &mut p2, 1.0).is_none());
    }

    #[test]
    fn unequal_mass_collision_conserves_momentum_and_energy() {
        // Heavy (r=3, m=9) meets light (r=1, m=1) head-on, fully elastic.
        let mut heavy = particle_r(0.0, 0.0, 100.0, 0.0, 3.0);
        let mut light = particle_r(3.5, 0.0, -100.0, 0.0, 1.0);
        let (m1, m2) = (heavy.mass(), light.mass());
        let momentum_before = m1 * heavy.vx + m2 * light.vx;
        let energy_before = m1 * heavy.vx * heavy.vx + m2 * light.vx * light.vx;

        try_elastic_collision(&mut heavy, &mut light, 1.0).unwrap();

        let momentum_after = m1 * heavy.vx + m2 * light.vx;
        let energy_after = m1 * heavy.vx * heavy.vx + m2 * light.vx * light.vx;
        assert!((momentum_before - momentum_after).abs() < 1e-9);
        assert!((energy_before - energy_after).abs() < 1e-6);

        // The heavy particle barely deflects; the light one rebounds hard.
        assert!(
            heavy.vx > 50.0,
            "heavy keeps most of its speed: {}",
            heavy.vx
        );
        assert!(light.vx > 200.0, "light rebounds fast: {}", light.vx);
    }

    #[test]
    fn contact_distance_is_the_sum_of_radii() {
        // r=2 and r=1: contact at 3.0.
        let mut a = particle_r(0.0, 0.0, 100.0, 0.0, 2.0);
        let mut b = particle_r(3.5, 0.0, -100.0, 0.0, 1.0);
        assert!(try_elastic_collision(&mut a, &mut b, 1.0).is_none());

        let mut a = particle_r(0.0, 0.0, 100.0, 0.0, 2.0);
        let mut b = particle_r(2.9, 0.0, -100.0, 0.0, 1.0);
        assert!(try_elastic_collision(&mut a, &mut b, 1.0).is_some());
    }

    #[test]
    fn flow_field_entrains_toward_bounded_speeds() {
        let mut particles = vec![
            particle(100.0, 100.0, 0.0, 0.0),
            particle(500.0, 300.0, 0.0, 0.0),
        ];
        // From rest, the flow pushes every particle.
        apply_flow(&mut particles, 1.0, 0.01);
        for p in &particles {
            assert!(p.vx.is_finite() && p.vy.is_finite());
            assert!(p.speed() > 0.0, "flow must push every particle");
        }

        // The flow is a medium, not a force: speeds converge to the local
        // current instead of growing without bound.
        let mut t = 1.0;
        for _ in 0..10_000 {
            apply_flow(&mut particles, t, 0.01);
            t += 0.01;
        }
        for p in &particles {
            assert!(
                p.speed() <= FLOW_SPEED * 1.01,
                "flow speeds must stay bounded: {}",
                p.speed()
            );
        }

        // A particle far faster than the current is slowed by drag.
        let mut fast = vec![particle(200.0, 200.0, 900.0, 0.0)];
        for _ in 0..300 {
            apply_flow(&mut fast, 5.0, 0.01);
        }
        assert!(fast[0].speed() < 300.0, "drag must bleed off excess speed");
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
        handle_collisions(&mut particles, &mut grid, &mut recorder, width, height, 1.0);
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
                update_physics(particles, sub_dt, width, height, gravity_multiplier, 0.0);
                let energy = handle_collisions(particles, &mut grid, recorder, width, height, 0.0);
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
        p.update(0.01, 200, 200, 0.0, 1.0);
        assert!(p.vx > 0.0);
        assert!((p.x - RADIUS).abs() < 1e-9);
    }

    #[test]
    fn gravity_accelerates_downward() {
        let mut p = particle(100.0, 100.0, 0.0, 0.0);
        p.update(0.1, 200, 200, 1.0, 1.0);
        assert!((p.vy - GRAVITY * 0.1).abs() < 1e-9);
    }

    #[test]
    fn negative_gravity_accelerates_upward() {
        let mut p = particle(100.0, 100.0, 0.0, 0.0);
        p.update(0.1, 200, 200, -1.0, 1.0);
        assert!((p.vy + GRAVITY * 0.1).abs() < 1e-9);
    }

    #[test]
    fn substep_count_scales_with_speed() {
        let slow = vec![particle(50.0, 50.0, 10.0, 0.0)];
        let fast = vec![particle(50.0, 50.0, 600.0, 0.0)];
        assert_eq!(substep_count(&slow, 0.008), 1);
        assert!(substep_count(&fast, 0.008) > 1);
        assert!(substep_count(&fast, 10.0) <= MAX_SUBSTEPS);
        assert_eq!(substep_count(&[], 0.008), 1);
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
    fn attractor_peaks_at_strength_and_falls_off_as_inverse_square() {
        let accel_at = |d: f64| -> f64 {
            let mut p = vec![particle(0.0, 0.0, 0.0, 0.0)];
            apply_attractor(&mut p, d, 0.0, WELL_STRENGTH, 1.0);
            p[0].vx
        };

        // The peak sits at s/sqrt(2) and equals WELL_STRENGTH.
        let peak_d = 120.0 / std::f64::consts::SQRT_2;
        let peak = accel_at(peak_d);
        assert!(
            (peak - WELL_STRENGTH).abs() < WELL_STRENGTH * 0.01,
            "peak acceleration must equal WELL_STRENGTH: {peak}"
        );

        // Beyond the softening radius the pull decays monotonically...
        let (a200, a400, a800) = (accel_at(200.0), accel_at(400.0), accel_at(800.0));
        assert!(peak > a200 && a200 > a400 && a400 > a800);

        // ...approaching a true inverse-square law: doubling the distance
        // divides the pull by ~4.
        let ratio = a400 / a800;
        assert!(
            (3.2..=4.2).contains(&ratio),
            "far field must be ~1/d^2 (got ratio {ratio:.2})"
        );

        // Distant particles feel a small fraction of the peak, not the
        // near-full strength the old asymptotic model applied.
        assert!(a800 < WELL_STRENGTH * 0.1);
    }

    #[test]
    fn has_motion_detects_stopped_particles() {
        let moving = vec![particle(0.0, 0.0, 10.0, 0.0)];
        let stopped = vec![particle(0.0, 0.0, 0.1, 0.1)];
        assert!(has_motion(&moving));
        assert!(!has_motion(&stopped));
    }
}
