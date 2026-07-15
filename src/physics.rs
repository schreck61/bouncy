// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Particle physics: motion, gravity, wall bounces, and particle-particle
//! collisions accelerated by a uniform spatial grid.

use rand::Rng;
use rayon::prelude::*;

/// Whether the parallel passes may fan out. On wasm without the
/// `web-threads` feature there is no thread pool (`std::thread` cannot
/// spawn), so both fan-out branches are compile-time disabled and every
/// population takes the serial paths; the dead parallel code is
/// eliminated. With `web-threads` (wasm threads + wasm-bindgen-rayon)
/// and on every native target, the fan-out gates on population size as
/// usual.
const THREADS_AVAILABLE: bool = cfg!(any(not(target_arch = "wasm32"), feature = "web-threads"));

// Physics constants
/// Downward acceleration (pixels/s²) at --gravity 100.
pub const GRAVITY: f64 = 100.0;
/// Particle radius (pixels) at the default --particle-size.
pub const DEFAULT_PARTICLE_RADIUS: f64 = 1.5;
/// Top speed (pixels/s) of newly created particles at the default
/// --initial-speed; spawns start at 50-100% of it.
pub const INITIAL_VELOCITY: f64 = 600.0;
/// Clearance (pixels) kept from the walls when placing new particles.
pub const PARTICLE_MARGIN: f64 = 10.0;
/// Terminal speed (pixels/s): the ceiling the per-substep clamp in
/// `Particle::update` enforces. ~10 screen-crossings per second on a large
/// display — an order of magnitude above what comets, explosions, or wells
/// produce, low enough that super-elastic energy pumps saturate long
/// before f64 overflow.
pub const MAX_SPEED: f64 = 20_000.0;

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

        // Terminal velocity: elasticity above 1.0 pumps energy into every
        // bounce and collision, and the bounce rate grows with speed, so
        // without a ceiling velocities overflow f64 to infinity in finite
        // time — and one ∞−∞ in a collision then mints NaN, an absorbing
        // state no force or setting change can ever recover (that is why
        // this clamp lives here, at the one integration choke point every
        // substep passes through, rather than at the energy sources). The
        // cap is far above anything a legitimate mechanic produces, so
        // saturated particles still read as ludicrously fast.
        let speed_sq = self.vx * self.vx + self.vy * self.vy;
        if speed_sq > MAX_SPEED * MAX_SPEED {
            let scale = MAX_SPEED / speed_sq.sqrt();
            self.vx *= scale;
            self.vy *= scale;
        }

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

/// Uniform spatial grid for broad-phase collision detection, with an
/// overflow tier for rare oversized particles.
///
/// Cells are sized from the *median* radius, so the grid stays fine-grained
/// when matter fusion produces huge merged particles among thousands of
/// small ones — sizing from the maximum radius would inflate every cell
/// and collapse the clump into a handful of quadratic-cost cells. Particles
/// small enough that any touching pair fits the cell-adjacency invariant
/// (radius ≤ half a cell) are binned normally; the rest go on the
/// `oversized` list and are paired against every particle directly. That
/// sweep is only cheap while the list is short, so when the population
/// holds more than `MAX_OVERSIZED` large particles the cells grow to the
/// smallest size that brings the list back under the cap — never all the
/// way to max-radius sizing, which is exactly the degradation this tier
/// exists to avoid. Storage is a CSR layout (`entries[starts[c]..starts[c+1]]`
/// holds cell `c`'s particle indices, ascending) built by counting sort:
/// the contact sweep walks contiguous memory instead of chasing an
/// intrusive linked list, which halved the detection cost in packed-clump
/// populations. Rebuilding each frame allocates nothing in the steady
/// state.
pub struct SpatialGrid {
    cell_size: f64,
    cols: usize,
    rows: usize,
    /// CSR row offsets: cell `c` owns `entries[starts[c]..starts[c+1]]`.
    starts: Vec<u32>,
    /// Binned particle indices, grouped by cell, ascending within a cell.
    entries: Vec<u32>,
    /// Scratch: each particle's cell id (`u32::MAX` for oversized).
    cell_ids: Vec<u32>,
    /// Scratch: per-cell write cursors for the counting-sort scatter.
    cursors: Vec<u32>,
    /// Particles too large for the cell-adjacency invariant, paired against
    /// everything directly.
    oversized: Vec<u32>,
    /// Per-particle flag mirroring membership in `oversized`.
    is_oversized: Vec<bool>,
    /// Scratch buffer for the median-radius selection.
    radii: Vec<f64>,
}

/// Cap on the direct-pairing tier: beyond this many oversized particles the
/// O(oversized × n) sweep stops being cheap, so cell sizing grows cells
/// until at most this many particles exceed the binning threshold.
const MAX_OVERSIZED: usize = 64;

/// Population size at which contact detection fans out across threads.
/// Below this the whole sweep runs in microseconds and rayon's fork-join
/// overhead would eat the gain.
const COLLISION_PARALLEL_THRESHOLD: usize = 1024;

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
            starts: Vec::new(),
            entries: Vec::new(),
            cell_ids: Vec::new(),
            cursors: Vec::new(),
            oversized: Vec::new(),
            is_oversized: Vec::new(),
            radii: Vec::new(),
        }
    }

    /// The particle indices binned in cell `c`, ascending.
    #[inline]
    fn cell_entries(&self, c: usize) -> &[u32] {
        &self.entries[self.starts[c] as usize..self.starts[c + 1] as usize]
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn cell_of(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = ((x / self.cell_size) as usize).min(self.cols - 1);
        let cy = ((y / self.cell_size) as usize).min(self.rows - 1);
        (cx, cy)
    }

    /// Size cells from the median radius: at least one typical diameter
    /// wide (larger cells trade a few extra narrow-phase tests for fewer
    /// cells to clear per frame). Particles with radius at most half a
    /// cell keep the adjacency invariant (any touching pair spans at most
    /// one cell); larger ones become oversized. The direct tier only stays
    /// cheap while short, so the floor of twice the (`MAX_OVERSIZED`+1)-th
    /// largest radius guarantees at most `MAX_OVERSIZED` particles exceed
    /// the binning threshold — cells grow just enough to absorb a glut of
    /// mid-size merged particles instead of ballooning to the maximum
    /// radius and collapsing a dense clump into a few quadratic cells.
    fn choose_cell_size(particles: &[Particle], radii: &mut Vec<f64>) -> f64 {
        radii.clear();
        radii.extend(particles.iter().map(|p| p.radius));
        if radii.is_empty() {
            return 12.0;
        }
        let mid = radii.len() / 2;
        let median = *radii.select_nth_unstable_by(mid, f64::total_cmp).1;
        let mut cell_size = (median * 4.0).max(12.0);
        if radii.len() > MAX_OVERSIZED {
            let k = radii.len() - 1 - MAX_OVERSIZED;
            let r_k = *radii.select_nth_unstable_by(k, f64::total_cmp).1;
            cell_size = cell_size.max(2.0 * r_k);
        }
        cell_size
    }

    /// Rebuild the grid from the current particle positions. Cell size
    /// must cover the largest possible contact distance among *binned*
    /// particles so that any colliding binned pair is in the same or
    /// adjacent cells; oversized particles satisfy no such invariant and
    /// are swept directly instead.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn build(&mut self, particles: &[Particle], width: u32, height: u32) {
        let cell_size = Self::choose_cell_size(particles, &mut self.radii);
        let cols = ((f64::from(width) / cell_size).ceil() as usize).max(1);
        let rows = ((f64::from(height) / cell_size).ceil() as usize).max(1);

        self.cell_size = cell_size;
        self.cols = cols;
        self.rows = rows;
        self.oversized.clear();
        self.is_oversized.clear();
        self.is_oversized.resize(particles.len(), false);

        // Counting sort into the CSR arrays. Pass 1: classify each
        // particle and count per-cell occupancy in starts[cell + 1].
        self.starts.clear();
        self.starts.resize(cols * rows + 1, 0);
        self.cell_ids.clear();
        for (i, p) in particles.iter().enumerate() {
            if p.radius > cell_size / 2.0 {
                self.oversized.push(i as u32);
                self.is_oversized[i] = true;
                self.cell_ids.push(u32::MAX);
                continue;
            }
            let (cx, cy) = self.cell_of(p.x, p.y);
            let cell = cy * cols + cx;
            self.cell_ids.push(cell as u32);
            self.starts[cell + 1] += 1;
        }

        // Prefix-sum the counts into row offsets, then scatter in
        // ascending particle order so each cell's slice is sorted — the
        // contact sweep's visitation order stays deterministic.
        for c in 1..self.starts.len() {
            self.starts[c] += self.starts[c - 1];
        }
        self.cursors.clear();
        self.cursors.extend_from_slice(&self.starts[..cols * rows]);
        self.entries.clear();
        self.entries
            .resize(particles.len() - self.oversized.len(), 0);
        for (i, &cell) in self.cell_ids.iter().enumerate() {
            if cell == u32::MAX {
                continue;
            }
            let cursor = &mut self.cursors[cell as usize];
            self.entries[*cursor as usize] = i as u32;
            *cursor += 1;
        }
    }

    /// Check whether any particle center lies within `dist` of `(x, y)`.
    /// Uses the cell lists from the most recent `build`; `particles` must be
    /// the slice the grid was built from. `dist` may exceed the cell size
    /// (spawn-clearance queries are conservative against the largest radius
    /// in play, which can dwarf the median-sized cells): the scan covers
    /// exactly the cells the query circle touches, plus the oversized list.
    pub fn any_within(&self, particles: &[Particle], x: f64, y: f64, dist: f64) -> bool {
        if self.starts.len() < 2 {
            return false;
        }
        let dist_sq = dist * dist;
        let (cx0, cy0) = self.cell_of(x - dist, y - dist);
        let (cx1, cy1) = self.cell_of(x + dist, y + dist);
        for ny in cy0..=cy1 {
            for nx in cx0..=cx1 {
                for &i in self.cell_entries(ny * self.cols + nx) {
                    let idx = i as usize;
                    if idx < particles.len()
                        && particles[idx].distance_squared_from(x, y) <= dist_sq
                    {
                        return true;
                    }
                }
            }
        }
        self.oversized.iter().any(|&i| {
            (i as usize) < particles.len()
                && particles[i as usize].distance_squared_from(x, y) <= dist_sq
        })
    }

    /// Collect the contacts of one cell row — every touching-and-approaching
    /// pair whose first particle is binned in row `cy`, via cell adjacency —
    /// in a fixed order (cells left to right, ascending indices within each
    /// cell). Grid dimensions are far below any integer limit, so the index
    /// casts in the neighbor arithmetic are lossless (bounds-checked
    /// non-negative before the sign-losing casts).
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation
    )]
    fn collect_row_contacts(&self, cy: usize, particles: &[Particle], out: &mut Vec<(u32, u32)>) {
        for cx in 0..self.cols {
            let cell = cy * self.cols + cx;
            let a_slice = self.cell_entries(cell);

            // Pairs within this cell.
            for (k, &a) in a_slice.iter().enumerate() {
                for &b in &a_slice[k + 1..] {
                    if in_contact_and_approaching(&particles[a as usize], &particles[b as usize]) {
                        out.push((a, b));
                    }
                }
            }

            // Pairs with forward neighbor cells.
            for (dx, dy) in FORWARD_NEIGHBORS {
                let nx = cx as i64 + dx;
                let ny = cy as i64 + dy;
                if nx < 0 || ny < 0 || nx >= self.cols as i64 || ny >= self.rows as i64 {
                    continue;
                }
                let b_slice = self.cell_entries((ny as usize) * self.cols + (nx as usize));
                for &a in a_slice {
                    for &b in b_slice {
                        if in_contact_and_approaching(
                            &particles[a as usize],
                            &particles[b as usize],
                        ) {
                            out.push((a, b));
                        }
                    }
                }
            }
        }
    }

    /// Find every touching-and-approaching pair against the current
    /// positions: binned pairs via cell adjacency, then each oversized
    /// particle against every binned particle and every later oversized
    /// one. Detection is read-only, so large populations fan the row sweep
    /// out across threads; rows are reassembled in row order, making the
    /// contact list identical for any thread count. The narrow-phase
    /// distance tests dominate the collision pass, so this is the part
    /// worth parallelizing — resolution mutates shared particles and stays
    /// serial.
    fn detect_contacts(&self, particles: &[Particle]) -> Vec<(u32, u32)> {
        let mut contacts: Vec<(u32, u32)> =
            if THREADS_AVAILABLE && particles.len() >= COLLISION_PARALLEL_THRESHOLD {
                (0..self.rows)
                    .into_par_iter()
                    .map(|cy| {
                        let mut row = Vec::new();
                        self.collect_row_contacts(cy, particles, &mut row);
                        row
                    })
                    .collect::<Vec<_>>()
                    .concat()
            } else {
                let mut all = Vec::new();
                for cy in 0..self.rows {
                    self.collect_row_contacts(cy, particles, &mut all);
                }
                all
            };

        // Oversized tier: no adjacency invariant, so sweep directly. Binned
        // partners come from the flag scan; oversized-oversized pairs from
        // the forward half of the list, each pair visited exactly once. At
        // most MAX_OVERSIZED entries by construction — not worth threads.
        #[allow(clippy::cast_possible_truncation)]
        for (k, &a) in self.oversized.iter().enumerate() {
            for (b, &over) in self.is_oversized.iter().enumerate() {
                if !over && in_contact_and_approaching(&particles[a as usize], &particles[b]) {
                    contacts.push((a, b as u32));
                }
            }
            for &b in &self.oversized[k + 1..] {
                if in_contact_and_approaching(&particles[a as usize], &particles[b as usize]) {
                    contacts.push((a, b));
                }
            }
        }
        contacts
    }
}

/// Whether two particles are in collision range (center distance within
/// the sum of radii, beyond the degenerate epsilon) and approaching each
/// other — the same acceptance conditions as `try_elastic_collision`,
/// evaluated without mutation so detection can run concurrently. The sign
/// test uses the unnormalized center offset: `dv·d > 0` exactly when the
/// normal velocity `dvn > 0`.
fn in_contact_and_approaching(p1: &Particle, p2: &Particle) -> bool {
    let contact = p1.radius + p2.radius;
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let dist_sq = dx * dx + dy * dy;
    if !(DISTANCE_SQ_EPSILON..=contact * contact).contains(&dist_sq) {
        return false;
    }
    (p1.vx - p2.vx) * dx + (p1.vy - p2.vy) * dy > 0.0
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
    debug_assert_ne!(i, j);
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
///
/// Runs in two phases: detect every touching-and-approaching pair against
/// a snapshot of the current positions (read-only, parallel for large
/// populations), then resolve the short contact list serially — impulses
/// mutate both partners, so pairs sharing a particle must not resolve
/// concurrently. `try_elastic_collision` re-validates each contact at
/// resolve time, so a pair separated by an earlier resolution in the same
/// pass is skipped, exactly as the old fused sweep skipped it; the one
/// semantic difference is that a pair pushed *into* contact mid-pass
/// resolves on the next substep instead of this one.
pub fn handle_collisions(
    particles: &mut [Particle],
    grid: &mut SpatialGrid,
    recorder: &mut CollisionRecorder,
    width: u32,
    height: u32,
    particle_elasticity: f64,
    segments: &[Segment],
) -> f64 {
    let mut max_energy = 0.0f64;
    if particles.len() < 2 {
        return max_energy;
    }

    grid.build(particles, width, height);
    for (i, j) in grid.detect_contacts(particles) {
        let (i, j) = (i as usize, j as usize);
        let (p1, p2) = pair_mut(particles, i, j);
        // A wall between two centers means they are not touching: without
        // this, particles hugging opposite faces of a zero-thickness wall
        // exchange impulses — and, with matter on, fuse — straight through
        // it. Filtering here also starves the recorder, so matter and
        // spawning never see cross-wall pairs either.
        if wall_between(p1, p2, segments) {
            continue;
        }
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
    }

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

/// Signed side of `(px, py)` relative to the wall's directed infinite
/// line: positive on the `(-aby, abx)` perpendicular's side, negative
/// opposite, zero on the line.
fn line_side(seg: &Segment, px: f64, py: f64) -> f64 {
    (seg.x2 - seg.x1) * (py - seg.y1) - (seg.y2 - seg.y1) * (px - seg.x1)
}

/// Whether the motion segment `(ox, oy)→(nx, ny)` properly crosses the
/// wall segment: strict side changes on both segments, so touches,
/// collinear overlaps, and crossings of the line beyond the wall's span
/// all count as not crossing (the end-position overlap test owns those).
fn motion_crosses_segment(seg: &Segment, ox: f64, oy: f64, nx: f64, ny: f64) -> bool {
    if line_side(seg, ox, oy) * line_side(seg, nx, ny) >= 0.0 {
        return false;
    }
    let (mdx, mdy) = (nx - ox, ny - oy);
    let end1 = mdx * (seg.y1 - oy) - mdy * (seg.x1 - ox);
    let end2 = mdx * (seg.y2 - oy) - mdy * (seg.x2 - ox);
    end1 * end2 < 0.0
}

/// Whether any drawn wall segment lies between the two particle centers
/// (their center-to-center segment properly crosses it). Same strictness
/// as `motion_crosses_segment`: a center exactly on a wall's line does
/// not count as separated.
pub fn wall_between(p1: &Particle, p2: &Particle, segments: &[Segment]) -> bool {
    for seg in segments {
        // Cheap AABB reject before the exact crossing test.
        if p1.x.max(p2.x) < seg.x1.min(seg.x2)
            || p1.x.min(p2.x) > seg.x1.max(seg.x2)
            || p1.y.max(p2.y) < seg.y1.min(seg.y2)
            || p1.y.min(p2.y) > seg.y1.max(seg.y2)
        {
            continue;
        }
        if motion_crosses_segment(seg, p1.x, p1.y, p2.x, p2.y) {
            return true;
        }
    }
    false
}

/// Bounce particles off the drawn wall segments. Two contact conditions,
/// both resolved from the particle's *end* position so that sustained
/// contact never erases tangential motion (a particle sliding along a
/// wall must keep sliding, not stick): the end position overlaps the
/// wall, or the motion over the substep — `prev` holds each particle's
/// center before this substep's position update, same indexing as
/// `particles` — crossed the wall outright. The crossing test is what
/// prevents tunneling when the substep cap saturates at low frame rates:
/// however far past the zero-thickness wall a single step lands, the
/// particle is placed back on the side it came from, at the wall's
/// closest point to where its step ended. On contact the particle is
/// pushed out along the contact normal and its approaching velocity
/// component reflects scaled by `wall_elasticity` — the arena-wall rule.
/// A particle already receding only gets the position correction, so a
/// bounce is never re-reflected back into the wall it just left.
pub fn collide_with_segments(
    particles: &mut [Particle],
    prev: &[(f64, f64)],
    segments: &[Segment],
    wall_elasticity: f64,
) {
    debug_assert_eq!(particles.len(), prev.len());
    for (p, &(ox, oy)) in particles.iter_mut().zip(prev) {
        for seg in segments {
            // Cheap AABB reject (over the whole motion) before the exact
            // tests.
            let r = p.radius;
            if p.x.max(ox) < seg.x1.min(seg.x2) - r
                || p.x.min(ox) > seg.x1.max(seg.x2) + r
                || p.y.max(oy) < seg.y1.min(seg.y2) - r
                || p.y.min(oy) > seg.y1.max(seg.y2) + r
            {
                continue;
            }
            let (cx, cy) = seg.closest_point(p.x, p.y);
            let (dx, dy) = (p.x - cx, p.y - cy);
            let d2 = dx * dx + dy * dy;
            let crossed = motion_crosses_segment(seg, ox, oy, p.x, p.y);
            if !crossed && d2 >= r * r {
                continue;
            }
            // Contact normal. A crossing (or a dead-center overlap) has
            // no radial direction from the end position; push out
            // perpendicular to the segment, toward the side the motion
            // started on (or straight up for a degenerate point segment).
            let d = d2.sqrt();
            let (nx, ny) = if !crossed && d > 1e-9 {
                (dx / d, dy / d)
            } else {
                let (abx, aby) = (seg.x2 - seg.x1, seg.y2 - seg.y1);
                let len = (abx * abx + aby * aby).sqrt();
                if len > 1e-9 {
                    let (px, py) = (-aby / len, abx / len);
                    if line_side(seg, ox, oy) < 0.0 {
                        (-px, -py)
                    } else {
                        (px, py)
                    }
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

/// Population size at which self-gravity switches from the exact pairwise
/// pass to the Barnes-Hut tree. Below this the O(n²) pass is both faster
/// (no tree build) and exactly momentum-conserving; above it the tree's
/// O(n log n) wins and the approximation error is invisible at screen
/// scale.
const BARNES_HUT_THRESHOLD: usize = 256;

/// Apply mutual Newtonian gravity between all particles, softened like the
/// cursor well. Small populations use the exact pairwise pass (symmetric,
/// so momentum is conserved to floating-point exactness); populations of
/// `BARNES_HUT_THRESHOLD` or more use an adaptive Barnes-Hut quadtree,
/// which approximates far-away groups by their center of mass and keeps
/// the cost near O(n log n) even when the population collapses into one
/// dense clump.
pub fn apply_self_gravity(particles: &mut [Particle], dt: f64) {
    if particles.len() < BARNES_HUT_THRESHOLD {
        apply_self_gravity_exact(particles, dt);
    } else {
        apply_self_gravity_barnes_hut(particles, dt);
    }
}

/// Exact pairwise self-gravity, applied symmetrically (Newton's third
/// law), so momentum is conserved exactly and heavier particles both pull
/// harder and yield less. O(n²) per substep: correct and fast for
/// preset-scale populations (~5k pair evaluations at 100 particles), ruinous
/// for the thousands-of-particles populations matter mode can grow — those
/// route to the Barnes-Hut pass instead.
fn apply_self_gravity_exact(particles: &mut [Particle], dt: f64) {
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

/// Barnes-Hut opening angle θ: a node is treated as a single point mass
/// when its width divided by the distance to its center of mass is below
/// θ. 0.6 is the conventional accuracy/speed middle ground; force errors
/// stay well under 1% of the exact pass.
const BARNES_HUT_THETA: f64 = 0.6;
/// Leaves hold up to this many particles and interact exactly. Larger
/// leaves mean a shallower tree (cheaper build) at the cost of more exact
/// pair work; 16 balances the two for the populations matter mode reaches.
const BARNES_HUT_LEAF_CAP: usize = 16;
/// Hard depth cap. Each level halves the node width, so 32 levels shrink a
/// screen-sized root below any physical separation; a leaf that still
/// exceeds the cap here holds effectively coincident particles, which the
/// softening handles.
const BARNES_HUT_MAX_DEPTH: u32 = 32;
/// Population size at which the force pass fans out across threads. Below
/// this the whole pass runs in well under a millisecond and rayon's
/// fork-join overhead would eat the gain.
const BARNES_HUT_PARALLEL_THRESHOLD: usize = 1024;

/// One quadtree node: a square region with the total mass and center of
/// mass of the particles inside it. Internal nodes store the index of
/// their first child (all four children are contiguous); leaves store a
/// range into the build's permuted particle-index buffer.
struct BhNode {
    center_x: f64,
    center_y: f64,
    half: f64,
    mass: f64,
    com_x: f64,
    com_y: f64,
    /// Index of the first of four contiguous children, or -1 for a leaf.
    first_child: i32,
    /// Leaf particle range in `order` (unused for internal nodes).
    start: usize,
    len: usize,
}

/// Adaptive Barnes-Hut self-gravity: build a quadtree over the current
/// positions, then accumulate each particle's acceleration by walking the
/// tree — exact softened pairs inside leaves and near nodes, center-of-mass
/// approximations for nodes that pass the opening test. The tree adapts to
/// the distribution, so a population collapsed into one dense clump (the
/// regime uniform-grid far-field schemes degrade in) still resolves into
/// small leaves. Near-field pair forces cancel exactly; far-field
/// approximation breaks momentum conservation only at the force-error
/// level (<1%), invisible at screen scale.
fn apply_self_gravity_barnes_hut(particles: &mut [Particle], dt: f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in particles.iter() {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let half = ((max_x - min_x).max(max_y - min_y) / 2.0).max(1e-6);
    let root = BhNode {
        center_x: f64::midpoint(min_x, max_x),
        center_y: f64::midpoint(min_y, max_y),
        half,
        mass: 0.0,
        com_x: 0.0,
        com_y: 0.0,
        first_child: -1,
        start: 0,
        len: particles.len(),
    };

    let mut nodes = vec![root];
    #[allow(clippy::cast_possible_truncation)]
    let mut order: Vec<u32> = (0..particles.len() as u32).collect();

    // Iterative build: split each pending node's index range into four
    // quadrant sub-ranges in place, then aggregate mass and center of mass
    // bottom-up (children are always created after their parent, so a
    // reverse sweep sees every child before its parent).
    let mut pending = vec![(0usize, 0u32)]; // (node index, depth)
    while let Some((ni, depth)) = pending.pop() {
        let (start, len, cx, cy, half) = {
            let n = &nodes[ni];
            (n.start, n.len, n.center_x, n.center_y, n.half)
        };
        if len <= BARNES_HUT_LEAF_CAP || depth >= BARNES_HUT_MAX_DEPTH {
            let (mut m, mut mx, mut my) = (0.0, 0.0, 0.0);
            for &i in &order[start..start + len] {
                let p = &particles[i as usize];
                let pm = p.mass();
                m += pm;
                mx += pm * p.x;
                my += pm * p.y;
            }
            let n = &mut nodes[ni];
            n.mass = m;
            n.com_x = mx / m.max(f64::MIN_POSITIVE);
            n.com_y = my / m.max(f64::MIN_POSITIVE);
            continue;
        }

        // Partition the range into quadrants: top/bottom by y, then each
        // half left/right by x. `partition_range` is stable-order-free but
        // in-place and O(len).
        let range = &mut order[start..start + len];
        let split_y = partition_range(range, |i| particles[i as usize].y < cy);
        let (top, bottom) = range.split_at_mut(split_y);
        let split_x_top = partition_range(top, |i| particles[i as usize].x < cx);
        let split_x_bottom = partition_range(bottom, |i| particles[i as usize].x < cx);

        let quarter = half / 2.0;
        let child_ranges = [
            (start, split_x_top),                         // top-left
            (start + split_x_top, split_y - split_x_top), // top-right
            (start + split_y, split_x_bottom),            // bottom-left
            (
                start + split_y + split_x_bottom,
                len - split_y - split_x_bottom,
            ), // bottom-right
        ];
        let child_centers = [
            (cx - quarter, cy - quarter),
            (cx + quarter, cy - quarter),
            (cx - quarter, cy + quarter),
            (cx + quarter, cy + quarter),
        ];

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let first_child = nodes.len() as i32;
        nodes[ni].first_child = first_child;
        for k in 0..4 {
            nodes.push(BhNode {
                center_x: child_centers[k].0,
                center_y: child_centers[k].1,
                half: quarter,
                mass: 0.0,
                com_x: 0.0,
                com_y: 0.0,
                first_child: -1,
                start: child_ranges[k].0,
                len: child_ranges[k].1,
            });
            if child_ranges[k].1 > 0 {
                pending.push((nodes.len() - 1, depth + 1));
            }
        }
    }

    // Bottom-up aggregation: children live at higher indices than their
    // parent, so one reverse pass folds each populated child into its
    // parent's mass and center of mass.
    for ni in (0..nodes.len()).rev() {
        let fc = nodes[ni].first_child;
        if fc < 0 {
            continue;
        }
        let (mut m, mut mx, mut my) = (0.0, 0.0, 0.0);
        #[allow(clippy::cast_sign_loss)]
        for k in 0..4 {
            let c = &nodes[fc as usize + k];
            m += c.mass;
            mx += c.mass * c.com_x;
            my += c.mass * c.com_y;
        }
        let n = &mut nodes[ni];
        n.mass = m;
        n.com_x = mx / m.max(f64::MIN_POSITIVE);
        n.com_y = my / m.max(f64::MIN_POSITIVE);
    }

    // Force pass: walk the tree once per particle. Forces depend only on
    // positions, which the pass never touches, so per-particle results are
    // independent and the pass parallelizes without synchronization. Large
    // populations gather accelerations in parallel; small ones stay on one
    // thread, where the fork-join overhead would exceed the work. Both
    // paths produce bit-identical results: each particle's accumulation
    // order is its own fixed tree traversal, regardless of scheduling.
    if THREADS_AVAILABLE && particles.len() >= BARNES_HUT_PARALLEL_THRESHOLD {
        let accels: Vec<(f64, f64)> = (0..particles.len())
            .into_par_iter()
            .map_init(
                || Vec::with_capacity(64),
                |stack, i| particle_acceleration(&nodes, &order, particles, i, stack),
            )
            .collect();
        for (p, (ax, ay)) in particles.iter_mut().zip(accels) {
            p.vx += SELF_GRAVITY_G * dt * ax;
            p.vy += SELF_GRAVITY_G * dt * ay;
        }
    } else {
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        for i in 0..particles.len() {
            let (ax, ay) = particle_acceleration(&nodes, &order, particles, i, &mut stack);
            let p = &mut particles[i];
            p.vx += SELF_GRAVITY_G * dt * ax;
            p.vy += SELF_GRAVITY_G * dt * ay;
        }
    }
}

/// Accumulate the softened gravitational acceleration on particle `i` (in
/// units of G — the caller applies `SELF_GRAVITY_G * dt`) by walking the
/// quadtree with an explicit `stack`, reused across calls to avoid
/// per-particle allocation. Reads only positions, masses, and the tree, so
/// concurrent calls for different particles are race-free.
fn particle_acceleration(
    nodes: &[BhNode],
    order: &[u32],
    particles: &[Particle],
    i: usize,
    stack: &mut Vec<u32>,
) -> (f64, f64) {
    let s2 = SELF_GRAVITY_SOFTENING * SELF_GRAVITY_SOFTENING;
    let theta_sq = BARNES_HUT_THETA * BARNES_HUT_THETA;
    let (px, py) = (particles[i].x, particles[i].y);
    let (mut ax, mut ay) = (0.0, 0.0);
    stack.push(0);
    while let Some(ni) = stack.pop() {
        let n = &nodes[ni as usize];
        if n.mass <= 0.0 {
            continue;
        }
        let dx = n.com_x - px;
        let dy = n.com_y - py;
        let d2 = dx * dx + dy * dy;
        let width = n.half * 2.0;
        if width * width < theta_sq * d2 {
            // Far field: the whole node as one softened point mass.
            // Leaves qualify too — passing the opening test puts the
            // particle at distance > width/θ, beyond the node's √2·width
            // interior diagonal, so it cannot be among the node's own
            // particles and no self-force is possible.
            let softened = d2 + s2;
            let scale = n.mass / (softened * softened.sqrt());
            ax += dx * scale;
            ay += dy * scale;
            continue;
        }
        if n.first_child >= 0 {
            #[allow(clippy::cast_sign_loss)]
            let fc = n.first_child as u32;
            stack.extend_from_slice(&[fc, fc + 1, fc + 2, fc + 3]);
            continue;
        }
        for &j in &order[n.start..n.start + n.len] {
            if j as usize == i {
                continue;
            }
            let pj = &particles[j as usize];
            let dx = pj.x - px;
            let dy = pj.y - py;
            let softened = dx * dx + dy * dy + s2;
            let scale = pj.mass() / (softened * softened.sqrt());
            ax += dx * scale;
            ay += dy * scale;
        }
    }
    (ax, ay)
}

/// Move every index satisfying `pred` to the front of `range`, returning
/// the count moved. Order within the two groups is not preserved; the
/// quadtree build only needs the split point.
fn partition_range(range: &mut [u32], pred: impl Fn(u32) -> bool) -> usize {
    let mut split = 0;
    for j in 0..range.len() {
        if pred(range[j]) {
            range.swap(split, j);
            split += 1;
        }
    }
    split
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

    /// Clone a population, run the exact and Barnes-Hut passes on the two
    /// copies, and return the worst per-particle relative error of the
    /// Barnes-Hut velocity kick against the exact one.
    fn barnes_hut_worst_relative_error(particles: &[Particle], dt: f64) -> f64 {
        let clone = |ps: &[Particle]| -> Vec<Particle> {
            ps.iter()
                .map(|p| particle_r(p.x, p.y, p.vx, p.vy, p.radius))
                .collect()
        };
        let mut exact = clone(particles);
        let mut approx = clone(particles);
        apply_self_gravity_exact(&mut exact, dt);
        apply_self_gravity_barnes_hut(&mut approx, dt);

        let scale = exact
            .iter()
            .map(|p| (p.vx * p.vx + p.vy * p.vy).sqrt())
            .fold(0.0f64, f64::max)
            .max(1e-12);
        exact
            .iter()
            .zip(&approx)
            .map(|(e, a)| {
                let (dx, dy) = (e.vx - a.vx, e.vy - a.vy);
                (dx * dx + dy * dy).sqrt() / scale
            })
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn barnes_hut_matches_exact_forces_when_spread_out() {
        let mut rng = StdRng::seed_from_u64(7);
        let particles: Vec<Particle> = (0..600)
            .map(|_| Particle::new_random(&mut rng, 800, 600, RADIUS, 0.0))
            .collect();
        let err = barnes_hut_worst_relative_error(&particles, 0.01);
        assert!(err < 0.02, "worst relative force error {err} exceeds 2%");
    }

    #[test]
    fn barnes_hut_matches_exact_forces_in_a_dense_clump() {
        // The regime that motivated the tree: thousands of particles
        // collapsed by stacked wells into a blob a few dozen pixels wide.
        let mut rng = StdRng::seed_from_u64(11);
        let particles: Vec<Particle> = (0..2000)
            .map(|_| {
                let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                let r: f64 = rng.random_range(0.0..30.0);
                particle(400.0 + r * angle.cos(), 300.0 + r * angle.sin(), 0.0, 0.0)
            })
            .collect();
        let err = barnes_hut_worst_relative_error(&particles, 0.01);
        assert!(err < 0.02, "worst relative force error {err} exceeds 2%");
    }

    #[test]
    fn barnes_hut_survives_coincident_particles() {
        // All particles at one point: the build cannot partition them, so
        // the depth cap must terminate it and softening must keep forces
        // finite (and zero, by symmetry of the softened kernel).
        let mut particles: Vec<Particle> =
            (0..300).map(|_| particle(400.0, 300.0, 0.0, 0.0)).collect();
        apply_self_gravity_barnes_hut(&mut particles, 0.01);
        for p in &particles {
            assert!(p.vx.is_finite() && p.vy.is_finite());
        }
    }

    #[test]
    fn barnes_hut_parallel_force_pass_is_deterministic() {
        // Seeded simulations must replay identically, so the parallel
        // force pass may not depend on thread scheduling. Run a population
        // above the parallel threshold through a single-threaded pool and
        // the default multi-threaded pool: every velocity must match to
        // the bit. This holds because each particle's accumulation order
        // is its own tree traversal, never a cross-thread reduction.
        let mut rng = StdRng::seed_from_u64(23);
        let particles: Vec<Particle> = (0..(BARNES_HUT_PARALLEL_THRESHOLD + 500))
            .map(|_| Particle::new_random(&mut rng, 800, 600, RADIUS, 200.0))
            .collect();
        let clone = |ps: &[Particle]| -> Vec<Particle> {
            ps.iter()
                .map(|p| particle_r(p.x, p.y, p.vx, p.vy, p.radius))
                .collect()
        };
        let mut single = clone(&particles);
        let mut multi = clone(&particles);

        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
            .install(|| apply_self_gravity_barnes_hut(&mut single, 0.01));
        apply_self_gravity_barnes_hut(&mut multi, 0.01);

        for (s, m) in single.iter().zip(&multi) {
            assert_eq!(s.vx.to_bits(), m.vx.to_bits(), "vx must match exactly");
            assert_eq!(s.vy.to_bits(), m.vy.to_bits(), "vy must match exactly");
        }
    }

    /// Timing comparison at the scale of the pathological exported scene
    /// (5,245 particles in a dense clump). Not a correctness test — run
    /// manually: `cargo test --release bench_self_gravity -- --ignored --nocapture`
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_self_gravity_exact_vs_barnes_hut() {
        let mut rng = StdRng::seed_from_u64(42);
        let make = |rng: &mut StdRng| -> Vec<Particle> {
            (0..5245)
                .map(|_| {
                    let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                    let r: f64 = rng.random_range(0.0..60.0);
                    particle(400.0 + r * angle.cos(), 300.0 + r * angle.sin(), 0.0, 0.0)
                })
                .collect()
        };
        let mut exact = make(&mut rng);
        let mut approx = make(&mut rng);

        let t = std::time::Instant::now();
        for _ in 0..8 {
            apply_self_gravity_exact(&mut exact, 0.001);
        }
        let exact_time = t.elapsed();

        let t = std::time::Instant::now();
        for _ in 0..8 {
            apply_self_gravity_barnes_hut(&mut approx, 0.001);
        }
        let bh_time = t.elapsed();

        println!("8 substeps at n=5245 (dense clump):");
        println!("  exact:      {exact_time:?}");
        println!("  barnes-hut: {bh_time:?}");
    }

    #[test]
    fn terminal_velocity_clamps_runaway_speed_and_keeps_direction() {
        // A particle at absurd speed (as the super-elastic energy pump
        // produces) must saturate at MAX_SPEED with its heading intact,
        // long before f64 overflow can mint infinities.
        let mut p = particle(400.0, 300.0, 3.0e9, -4.0e9);
        p.update(0.0001, 800, 600, 0.0, 1.0);
        assert!(
            (p.speed() - MAX_SPEED).abs() < 1e-6,
            "speed saturates at the cap: {}",
            p.speed()
        );
        assert!(
            (p.vx / p.vy - (3.0 / -4.0)).abs() < 1e-9,
            "direction preserved through the clamp"
        );
    }

    #[test]
    fn parallel_contact_detection_is_deterministic() {
        // Seeded simulations must replay identically, so the parallel
        // detection sweep may not depend on thread scheduling: rows are
        // reassembled in row order, making the contact list — and thus
        // every impulse and separation the serial resolve phase applies —
        // identical for any thread count. Verify positions and velocities
        // to the bit on a dense population above the parallel threshold.
        let mut rng = StdRng::seed_from_u64(29);
        let particles: Vec<Particle> = (0..(COLLISION_PARALLEL_THRESHOLD + 500))
            .map(|_| {
                let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                let r: f64 = 60.0 * rng.random_range(0.0f64..1.0).sqrt();
                let mut p = particle(400.0 + r * angle.cos(), 300.0 + r * angle.sin(), 0.0, 0.0);
                p.vx = rng.random_range(-100.0..100.0);
                p.vy = rng.random_range(-100.0..100.0);
                p
            })
            .collect();
        let clone = |ps: &[Particle]| -> Vec<Particle> {
            ps.iter()
                .map(|p| particle_r(p.x, p.y, p.vx, p.vy, p.radius))
                .collect()
        };
        let mut single = clone(&particles);
        let mut multi = clone(&particles);

        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
            .install(|| {
                let mut grid = SpatialGrid::new();
                let mut recorder = CollisionRecorder::new();
                handle_collisions(&mut single, &mut grid, &mut recorder, 800, 600, 0.7, &[]);
            });
        let mut grid = SpatialGrid::new();
        let mut recorder = CollisionRecorder::new();
        handle_collisions(&mut multi, &mut grid, &mut recorder, 800, 600, 0.7, &[]);

        for (s, m) in single.iter().zip(&multi) {
            assert_eq!(s.x.to_bits(), m.x.to_bits(), "x must match exactly");
            assert_eq!(s.y.to_bits(), m.y.to_bits(), "y must match exactly");
            assert_eq!(s.vx.to_bits(), m.vx.to_bits(), "vx must match exactly");
            assert_eq!(s.vy.to_bits(), m.vy.to_bits(), "vy must match exactly");
        }
    }

    /// Run one collision pass over `particles` and return the recorded
    /// spawn-site count (i.e. how many distinct pairs actually collided).
    fn collide_all(particles: &mut [Particle]) -> usize {
        let mut grid = SpatialGrid::new();
        let mut recorder = CollisionRecorder::new();
        handle_collisions(particles, &mut grid, &mut recorder, 800, 600, 1.0, &[]);
        recorder.sites().len()
    }

    #[test]
    fn oversized_particles_still_collide_with_small_ones() {
        // A merged blob (r=50) among small particles is too big for the
        // median-sized cells; the oversized tier must still find its
        // contacts. Blob at center, one small particle overlapping its rim
        // and approaching, plus spectators keeping the median radius small.
        let mut particles = vec![particle_r(400.0, 300.0, 0.0, 0.0, 50.0)];
        particles.push(particle(451.0, 300.0, -50.0, 0.0)); // touching, approaching
        for i in 0..20 {
            #[allow(clippy::cast_precision_loss)]
            particles.push(particle(20.0 + 10.0 * f64::from(i), 30.0, 0.0, 0.0));
        }
        assert_eq!(collide_all(&mut particles), 1, "blob-small contact found");
        assert!(
            particles[1].vx > 0.0,
            "small particle bounced off the blob: vx = {}",
            particles[1].vx
        );
    }

    #[test]
    fn two_oversized_particles_collide_with_each_other() {
        let mut particles = vec![
            particle_r(300.0, 300.0, 50.0, 0.0, 40.0),
            particle_r(379.0, 300.0, -50.0, 0.0, 40.0), // touching, approaching
        ];
        for i in 0..20 {
            #[allow(clippy::cast_precision_loss)]
            particles.push(particle(20.0 + 10.0 * f64::from(i), 30.0, 0.0, 0.0));
        }
        assert_eq!(collide_all(&mut particles), 1, "blob-blob contact found");
        assert!(particles[0].vx < 0.0 && particles[1].vx > 0.0);
    }

    #[test]
    fn any_within_sees_beyond_one_cell_and_finds_oversized_particles() {
        // Median radius 1.5 → 12px cells. A clearance query conservative
        // against a 50px blob (dist ≈ 53) spans several cells and must find
        // both a small particle ~40px away and the blob itself, which lives
        // on the oversized list rather than in any cell.
        let mut particles = vec![particle(440.0, 300.0, 0.0, 0.0)];
        for i in 0..20 {
            #[allow(clippy::cast_precision_loss)]
            particles.push(particle(20.0 + 10.0 * f64::from(i), 500.0, 0.0, 0.0));
        }
        let mut grid = SpatialGrid::new();
        grid.build(&particles, 800, 600);
        assert!(
            grid.any_within(&particles, 400.0, 300.0, 53.0),
            "small, 3+ cells away"
        );
        assert!(
            !grid.any_within(&particles, 400.0, 300.0, 30.0),
            "nothing within 30px"
        );

        particles.push(particle_r(400.0, 250.0, 0.0, 0.0, 50.0));
        grid.build(&particles, 800, 600);
        assert!(
            grid.any_within(&particles, 400.0, 300.0, 53.0),
            "oversized blob found via the overflow list"
        );
    }

    #[test]
    fn oversized_flood_grows_cells_just_enough_to_absorb_it() {
        // More than MAX_OVERSIZED mid-size giants above a small-particle
        // median — the "matter off after a long merge run" population. The
        // direct tier would go quadratic, so cells must grow to twice the
        // (MAX_OVERSIZED+1)-th largest radius: 2×20 = 40px here, absorbing
        // every giant into the grid. Crucially the sizing must ignore the
        // single much larger blob — one r=100 outlier stays on the direct
        // tier instead of quadrupling every cell. Two giants overlap on a
        // collision course and must still collide.
        let mut particles: Vec<Particle> = (0..200)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = 4.0 + 3.9 * f64::from(i);
                particle(x, 580.0, 0.0, 0.0)
            })
            .collect();
        for i in 0..70 {
            #[allow(clippy::cast_precision_loss)]
            let (col, row) = (f64::from(i % 14), f64::from(i / 14));
            particles.push(particle_r(
                50.0 + col * 50.0,
                50.0 + row * 50.0,
                0.0,
                0.0,
                20.0,
            ));
        }
        particles.push(particle_r(300.0, 400.0, 50.0, 0.0, 20.0));
        particles.push(particle_r(339.0, 400.0, -50.0, 0.0, 20.0));
        particles.push(particle_r(700.0, 400.0, 0.0, 0.0, 100.0));
        let mut grid = SpatialGrid::new();
        grid.build(&particles, 800, 600);
        assert!(
            (grid.cell_size - 40.0).abs() < 1e-9,
            "cells sized to the giants, not the outlier: {}",
            grid.cell_size
        );
        assert_eq!(
            grid.oversized.len(),
            1,
            "only the r=100 outlier stays on the direct tier"
        );
        assert_eq!(collide_all(&mut particles), 1, "approaching giants collide");
    }

    /// Isolate the cost of each physics term in the "matter off after a big
    /// merge" regime: ~13k particles clumped, with and without a large
    /// merged blob among them. Diagnoses whether the slowdown is gravity or
    /// the collision grid degrading when `max_radius` inflates cell size.
    /// Run manually:
    /// `cargo test --release bench_clump_cost_breakdown -- --ignored --nocapture`
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_clump_cost_breakdown() {
        const N: usize = 13000;
        let mut rng = StdRng::seed_from_u64(5);
        let make = |rng: &mut StdRng| -> Vec<Particle> {
            (0..N)
                .map(|_| {
                    let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                    let r: f64 = 200.0 * rng.random_range(0.0f64..1.0).sqrt();
                    let mut p =
                        particle(400.0 + r * angle.cos(), 300.0 + r * angle.sin(), 0.0, 0.0);
                    p.vx = rng.random_range(-100.0..100.0);
                    p.vy = rng.random_range(-100.0..100.0);
                    p
                })
                .collect()
        };
        let base = make(&mut rng);
        let clone = |ps: &[Particle]| -> Vec<Particle> {
            ps.iter()
                .map(|p| {
                    let mut c = particle_r(p.x, p.y, p.vx, p.vy, p.radius);
                    c.color = p.color;
                    c
                })
                .collect()
        };

        // Gravity term: Barnes-Hut over one frame's 8 substeps.
        let mut gravity = clone(&base);
        let t = std::time::Instant::now();
        for _ in 0..8 {
            apply_self_gravity_barnes_hut(&mut gravity, 0.001);
        }
        let gravity_time = t.elapsed();

        // Steady-state collision cost: the first calls resolve the random
        // clump's initial overlaps (a one-off transient) and run before the
        // CPU has migrated the load to a boosted core, so warm up well past
        // both effects before timing. Timing whole frames (8 substeps) at
        // sustained clock is what the FPS question actually depends on.
        let time_collisions = |particles: &mut Vec<Particle>| {
            let mut grid = SpatialGrid::new();
            let mut recorder = CollisionRecorder::new();
            for _ in 0..200 {
                handle_collisions(particles, &mut grid, &mut recorder, 800, 600, 1.0, &[]);
            }
            let t = std::time::Instant::now();
            for _ in 0..8 {
                handle_collisions(particles, &mut grid, &mut recorder, 800, 600, 1.0, &[]);
            }
            t.elapsed()
        };

        // Uniform small particles: median-sized cells, no oversized tier.
        let mut small = clone(&base);
        let collisions_small = time_collisions(&mut small);

        // One merged blob (r=50) among them: with max-radius cell sizing
        // this inflated every cell to 200px and went quadratic; the
        // oversized tier must keep it near the uniform cost.
        let mut with_blob = clone(&base);
        with_blob[0].radius = 50.0;
        let collisions_blob = time_collisions(&mut with_blob);

        println!("8 substeps at n={N}, clump radius 200px:");
        println!("  barnes-hut gravity:          {gravity_time:?}");
        println!("  collisions (all r=1.5):      {collisions_small:?}");
        println!("  collisions (one r=50 blob):  {collisions_blob:?}");

        // The "matter off after a long merge run" population, as seen in
        // the field (1920x1080): ~120 merged blobs r=20..40 in a packed
        // core with ~3700 small spawns swarming them. Far more than
        // MAX_OVERSIZED mid-size particles, so cell sizing must adapt
        // rather than degrade.
        let mut rng = StdRng::seed_from_u64(17);
        let mut mixed: Vec<Particle> = (0..120)
            .map(|_| {
                let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                let r: f64 = 220.0 * rng.random_range(0.0f64..1.0).sqrt();
                particle_r(
                    960.0 + r * angle.cos(),
                    540.0 + r * angle.sin(),
                    rng.random_range(-50.0..50.0),
                    rng.random_range(-50.0..50.0),
                    rng.random_range(20.0..40.0),
                )
            })
            .collect();
        for _ in 0..3700 {
            let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let r: f64 = 350.0 * rng.random_range(0.0f64..1.0).sqrt();
            let mut p = particle(960.0 + r * angle.cos(), 540.0 + r * angle.sin(), 0.0, 0.0);
            p.vx = rng.random_range(-100.0..100.0);
            p.vy = rng.random_range(-100.0..100.0);
            mixed.push(p);
        }
        let time_mixed = {
            let mut grid = SpatialGrid::new();
            let mut recorder = CollisionRecorder::new();
            for _ in 0..200 {
                handle_collisions(&mut mixed, &mut grid, &mut recorder, 1920, 1080, 0.7, &[]);
            }
            let t = std::time::Instant::now();
            for _ in 0..8 {
                handle_collisions(&mut mixed, &mut grid, &mut recorder, 1920, 1080, 0.7, &[]);
            }
            (t.elapsed(), grid.cell_size, grid.oversized.len())
        };
        println!(
            "  collisions (120 blobs r20-40 + 3700 small, 1080p): {:?} (cell={}, oversized={})",
            time_mixed.0, time_mixed.1, time_mixed.2
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
        collide_with_segments(&mut particles, &[(100.0, 99.0)], &[wall], 1.0);

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
        collide_with_segments(&mut particles, &[(100.0, 99.0)], &[wall], 0.5);
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
        collide_with_segments(&mut particles, &[(100.0, 99.0)], &[wall], 1.0);
        let p = &particles[0];
        assert!((p.y - (100.0 - RADIUS)).abs() < 1e-9, "still pushed out");
        assert!((p.vy - (-50.0)).abs() < 1e-9, "velocity left alone");
    }

    #[test]
    fn fast_particle_cannot_tunnel_through_a_segment() {
        // One substep carried the particle from well above the wall to
        // well below it, never overlapping at either end — the case a
        // discrete end-position test misses when the substep cap
        // saturates at low frame rates. The sweep must still bounce it.
        let wall = Segment {
            x1: 50.0,
            y1: 100.0,
            x2: 150.0,
            y2: 100.0,
        };
        let mut particles = vec![particle(100.0, 140.0, 0.0, 800.0)];
        collide_with_segments(&mut particles, &[(100.0, 60.0)], &[wall], 1.0);
        let p = &particles[0];
        assert!(
            (p.y - (100.0 - RADIUS)).abs() < 1e-9,
            "back on the side it came from: {}",
            p.y
        );
        assert!((p.vy - (-800.0)).abs() < 1e-9, "reflected: {}", p.vy);
    }

    #[test]
    fn sliding_contact_preserves_tangential_motion() {
        // A particle in sustained contact sliding parallel to a vertical
        // wall: the contact must push it out along the normal only —
        // never reset its tangential progress, which reads as the
        // particle sticking to the wall.
        let wall = Segment {
            x1: 100.0,
            y1: 0.0,
            x2: 100.0,
            y2: 200.0,
        };
        // Overlapping by 0.1 px, moving straight down at 100 px/s; this
        // substep carried it from y=50 to y=51.
        let mut particles = vec![particle(101.4, 51.0, 0.0, 100.0)];
        collide_with_segments(&mut particles, &[(101.4, 50.0)], &[wall], 1.0);
        let p = &particles[0];
        assert!(
            (p.y - 51.0).abs() < 1e-9,
            "tangential progress kept: y={}",
            p.y
        );
        assert!(
            (p.x - (100.0 + RADIUS)).abs() < 1e-9,
            "pushed out along the normal: x={}",
            p.x
        );
        assert!((p.vy - 100.0).abs() < 1e-9, "slide velocity kept: {}", p.vy);
    }

    #[test]
    fn sweep_past_the_endpoint_does_not_collide() {
        // Same fast crossing of the wall's infinite line, but beyond its
        // endpoint: no contact, the particle passes untouched.
        let wall = Segment {
            x1: 50.0,
            y1: 100.0,
            x2: 150.0,
            y2: 100.0,
        };
        let mut particles = vec![particle(200.0, 140.0, 0.0, 800.0)];
        collide_with_segments(&mut particles, &[(200.0, 60.0)], &[wall], 1.0);
        let p = &particles[0];
        assert!((p.y - 140.0).abs() < 1e-9, "position untouched: {}", p.y);
        assert!((p.vy - 800.0).abs() < 1e-9, "velocity untouched: {}", p.vy);
    }

    #[test]
    fn stacked_walls_resolve_to_the_near_side_regardless_of_list_order() {
        // One substep's motion crosses two parallel walls; the far one is
        // listed first, so it resolves first and briefly leaves the
        // particle between the walls. The near wall's check sweeps from
        // the original start to the corrected position, so it must still
        // see the crossing and pull the particle back to the approach
        // side, without re-reflecting the now-receding velocity.
        let near = Segment {
            x1: 50.0,
            y1: 300.0,
            x2: 150.0,
            y2: 300.0,
        };
        let far = Segment {
            x1: 50.0,
            y1: 320.0,
            x2: 150.0,
            y2: 320.0,
        };
        let mut particles = vec![particle(100.0, 400.0, 0.0, 900.0)];
        collide_with_segments(&mut particles, &[(100.0, 100.0)], &[far, near], 1.0);
        let p = &particles[0];
        assert!(
            (p.y - (300.0 - RADIUS)).abs() < 1e-9,
            "on the approach side of the near wall: {}",
            p.y
        );
        assert!((p.vy - (-900.0)).abs() < 1e-9, "moving back away: {}", p.vy);
    }

    #[test]
    fn contacts_across_a_wall_are_ignored() {
        // Two particles overlapping across a zero-thickness wall between
        // them: with the wall present they must not exchange impulses or
        // record a contact; without it, the same pair collides.
        let wall = Segment {
            x1: 400.0,
            y1: 0.0,
            x2: 400.0,
            y2: 600.0,
        };
        let make = || {
            vec![
                particle(399.0, 300.0, 50.0, 0.0),
                particle(401.0, 300.0, -50.0, 0.0),
            ]
        };
        let mut grid = SpatialGrid::new();
        let mut recorder = CollisionRecorder::new();

        let mut blocked = make();
        handle_collisions(
            &mut blocked,
            &mut grid,
            &mut recorder,
            800,
            600,
            1.0,
            &[wall],
        );
        assert!(recorder.sites().is_empty(), "no contact through the wall");
        assert!(
            (blocked[0].vx - 50.0).abs() < 1e-9 && (blocked[1].vx + 50.0).abs() < 1e-9,
            "velocities untouched through the wall"
        );

        recorder.clear();
        let mut open = make();
        handle_collisions(&mut open, &mut grid, &mut recorder, 800, 600, 1.0, &[]);
        assert_eq!(
            recorder.sites().len(),
            1,
            "the same pair collides once the wall is gone"
        );
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
        collide_with_segments(&mut particles, &[(151.0, 100.0)], &[wall], 1.0);
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
        collide_with_segments(&mut particles, &[(200.0, 200.0)], &[point], 1.0);
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
                &[],
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
            1.0,
            &[],
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
        let mut run_frame =
            |particles: &mut Vec<Particle>, recorder: &mut CollisionRecorder| -> u32 {
                recorder.clear();
                let mut contact_substeps = 0;
                for _ in 0..8 {
                    update_physics(particles, sub_dt, width, height, gravity_multiplier, 0.0);
                    let energy =
                        handle_collisions(particles, &mut grid, recorder, width, height, 0.0, &[]);
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

        grid.build(&particles, width, height);
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
