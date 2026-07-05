// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Expanding ring explosions that cull the particle population.

use crate::physics::Particle;
use rand::seq::SliceRandom;
use rand::Rng;

pub const SPAWN_RATE_WINDOW: f64 = 1.0;
pub const SPAWN_RATE_THRESHOLD: usize = 30;
pub const EXPLOSION_KILL_RATIO: f64 = 0.99;
pub const EXPLOSION_SPEED: f64 = 800.0;
pub const EXPLOSION_RING_WIDTH: f64 = 20.0;

/// An expanding ring that kills doomed particles as it reaches them.
pub struct Explosion {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    radius_sq: f64,
    max_radius: f64,
    pub active: bool,
    pub doomed_count: usize,
    pub killed_count: usize,
}

/// Distance from `(x, y)` to the farthest corner of the screen — the radius
/// an explosion needs to sweep every particle.
pub fn max_radius_from(x: f64, y: f64, width: u32, height: u32) -> f64 {
    let far_x = x.max(f64::from(width) - x);
    let far_y = y.max(f64::from(height) - y);
    (far_x * far_x + far_y * far_y).sqrt()
}

impl Explosion {
    /// Create a new explosion centered at `(x, y)`, marking a random
    /// `kill_ratio` share of `particles` as doomed while leaving at least
    /// `min_survivors` alive.
    pub fn new(
        rng: &mut impl Rng,
        x: f64,
        y: f64,
        max_radius: f64,
        particles: &mut [Particle],
        kill_ratio: f64,
        min_survivors: usize,
    ) -> Self {
        let particle_count = particles.len();
        // Truncation is intentional (floor); counts are far below f64 mantissa.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let kill_count = ((particle_count as f64 * kill_ratio) as usize)
            .min(particle_count.saturating_sub(min_survivors));

        let mut indices: Vec<usize> = (0..particle_count).collect();
        indices.shuffle(rng);
        for &i in indices.iter().take(kill_count) {
            particles[i].doomed = true;
        }

        Explosion {
            x,
            y,
            radius: 0.0,
            radius_sq: 0.0,
            max_radius,
            active: true,
            doomed_count: kill_count,
            killed_count: 0,
        }
    }

    /// Expand the explosion ring.
    pub fn update(&mut self, dt: f64) {
        if self.active {
            self.radius += EXPLOSION_SPEED * dt;
            self.radius_sq = self.radius * self.radius;
            if self.radius >= self.max_radius {
                self.active = false;
            }
        }
    }

    /// Remove doomed particles that the ring has reached. When the explosion
    /// finishes, any stragglers are pardoned.
    pub fn process_kills(&mut self, particles: &mut Vec<Particle>) {
        let before = particles.len();
        let (x, y, radius_sq) = (self.x, self.y, self.radius_sq);
        particles.retain(|p| !p.doomed || p.distance_squared_from(x, y) > radius_sq);
        self.killed_count += before - particles.len();

        if !self.active {
            for p in particles.iter_mut() {
                p.doomed = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_particles(n: usize) -> Vec<Particle> {
        let mut rng = StdRng::seed_from_u64(3);
        (0..n)
            .map(|_| Particle::new_random(&mut rng, 800, 600, 1.5, 600.0))
            .collect()
    }

    #[test]
    fn respects_min_survivors() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut particles = make_particles(100);
        let mut explosion = Explosion::new(
            &mut rng,
            400.0,
            300.0,
            1000.0,
            &mut particles,
            EXPLOSION_KILL_RATIO,
            10,
        );

        // Sweep the ring past every particle.
        while explosion.active {
            explosion.update(0.016);
            explosion.process_kills(&mut particles);
        }
        assert_eq!(particles.len(), 10);
        assert_eq!(explosion.killed_count, 90);
        assert!(particles.iter().all(|p| !p.doomed));
    }

    #[test]
    fn kills_only_particles_inside_ring() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut particles = make_particles(50);
        // Place one doomed particle far away and one at the center.
        let mut explosion = Explosion::new(
            &mut rng,
            400.0,
            300.0,
            10_000.0,
            &mut particles,
            EXPLOSION_KILL_RATIO,
            0,
        );
        for p in particles.iter_mut() {
            p.doomed = true;
        }
        particles[0].x = 400.0;
        particles[0].y = 300.0;
        particles[1].x = 400.0 + 500.0;
        particles[1].y = 300.0;

        explosion.update(0.05); // radius = 40
        explosion.process_kills(&mut particles);
        assert!(explosion.killed_count >= 1);
        assert!(
            particles.iter().any(|p| (p.x - 900.0).abs() < 1e-9),
            "far particle must survive a small ring"
        );
    }

    #[test]
    fn kill_ratio_applied() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut particles = make_particles(1000);
        let explosion = Explosion::new(
            &mut rng,
            0.0,
            0.0,
            100.0,
            &mut particles,
            EXPLOSION_KILL_RATIO,
            5,
        );
        let doomed = particles.iter().filter(|p| p.doomed).count();
        assert_eq!(doomed, 990);
        assert_eq!(explosion.doomed_count, 990);
    }

    #[test]
    fn full_kill_ratio_dooms_everything_above_the_floor() {
        // Regression: a manual (cursor) explosion at the base population must
        // still doom particles. With ratio 1.0 and a floor of 2, a population
        // of 4 dooms 2 — not 0, as the old base-count floor produced.
        let mut rng = StdRng::seed_from_u64(5);
        let mut particles = make_particles(4);
        let explosion = Explosion::new(&mut rng, 400.0, 300.0, 1000.0, &mut particles, 1.0, 2);
        assert_eq!(explosion.doomed_count, 2);
        assert_eq!(particles.iter().filter(|p| p.doomed).count(), 2);
    }

    #[test]
    fn max_radius_reaches_farthest_corner() {
        // From the center, the farthest corner is half the diagonal.
        let r = max_radius_from(400.0, 300.0, 800, 600);
        assert!((r - 500.0).abs() < 1e-9);
        // From a corner, it is the full diagonal.
        let r = max_radius_from(0.0, 0.0, 800, 600);
        assert!((r - 1000.0).abs() < 1e-9);
    }
}
