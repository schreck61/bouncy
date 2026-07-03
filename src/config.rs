// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Command line configuration.

use clap::{Parser, ValueEnum};

/// How particles are colored.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ColorMode {
    /// Each particle keeps a random bright color.
    #[default]
    Solid,
    /// Particle hue is mapped from its current speed (blue = slow, red = fast).
    Velocity,
}

/// GPU-accelerated particle simulation with elastic collisions, gravity,
/// and explosive chain reactions.
#[derive(Parser, Clone, Debug)]
#[command(name = "bouncy", version, about, after_help = CONTROLS_HELP)]
pub struct Config {
    /// Spawn new particles at collision points instead of screen center
    #[arg(long)]
    pub spawn_at_collision: bool,

    /// Set starting/minimum particle count
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=100))]
    pub min_particles: Option<u32>,

    /// Gravity as a percentage of standard; negative values pull upward
    #[arg(long, default_value_t = 100, allow_negative_numbers = true,
          value_parser = clap::value_parser!(i32).range(-1000..=1000))]
    pub gravity: i32,

    /// Wall bounce elasticity: 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy
    #[arg(long, default_value_t = 1.0, value_parser = parse_elasticity)]
    pub wall_elasticity: f64,

    /// Particle collision elasticity: 0.0 = sticks, 1.0 = elastic, >1.0 = adds energy
    #[arg(long, default_value_t = 1.0, value_parser = parse_elasticity)]
    pub particle_elasticity: f64,

    /// Window width in pixels (omit for fullscreen)
    #[arg(long, requires = "height", value_parser = clap::value_parser!(u32).range(100..=7680))]
    pub width: Option<u32>,

    /// Window height in pixels (omit for fullscreen)
    #[arg(long, requires = "width", value_parser = clap::value_parser!(u32).range(100..=4320))]
    pub height: Option<u32>,

    /// Force CPU rendering (softbuffer) instead of GPU
    #[arg(long)]
    pub cpu: bool,

    /// Start with audio muted (toggle at runtime with M)
    #[arg(long)]
    pub mute: bool,

    /// Leave motion trails behind particles instead of clearing each frame
    #[arg(long)]
    pub trails: bool,

    /// Particle radius in pixels
    #[arg(long, default_value_t = crate::physics::DEFAULT_PARTICLE_RADIUS, value_parser = parse_particle_size)]
    pub particle_size: f64,

    /// How particles are colored
    #[arg(long, value_enum, default_value_t = ColorMode::Solid)]
    pub color_mode: ColorMode,

    /// Spawns per second that trigger an automatic explosion; 0 disables
    /// automatic explosions entirely (right-click still works)
    #[arg(long, default_value_t = crate::explosion::SPAWN_RATE_THRESHOLD as u64,
          value_parser = clap::value_parser!(u64).range(0..=1000))]
    pub explosion_threshold: u64,

    /// Seed the random number generator (reproducible starting conditions)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Print per-second FPS statistics to stdout
    #[arg(long)]
    pub verbose: bool,
}

const CONTROLS_HELP: &str = "Controls:
  Space, Escape, Q   Exit
  P                  Pause / resume
  N                  Advance one frame (while paused)
  R                  Reset the simulation
  M                  Mute / unmute audio
  H                  Cycle the HUD (off / stats / stats+keys)
  Up / Down          Adjust gravity by 10%
  Left / Right       Adjust particle elasticity by 0.05
  [ / ]              Adjust wall elasticity by 0.05
  , / .              Slow down / speed up time (0.1x-4x)
  - / =              Adjust explosion threshold by 5 (0 = off)
  T                  Toggle motion trails
  C                  Cycle color mode
  B                  Toggle spawn location (center / collision point)
  G (hold)           Gravity well at the cursor; Shift+G repels
  Left click         Spawn a burst of particles at the cursor
  Right click        Trigger an explosion at the cursor

Rendering:
  Uses GPU rendering (wgpu) by default. Falls back to CPU rendering
  (softbuffer) if GPU is unavailable. Use --cpu to force CPU rendering.";

fn parse_range_f64(s: &str, name: &str, min: f64, max: f64) -> Result<f64, String> {
    let value: f64 = s.parse().map_err(|_| format!("{name} must be a number"))?;
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be between {min} and {max}"));
    }
    Ok(value)
}

fn parse_elasticity(s: &str) -> Result<f64, String> {
    parse_range_f64(s, "elasticity", 0.0, 1.5)
}

fn parse_particle_size(s: &str) -> Result<f64, String> {
    parse_range_f64(s, "particle size", 0.5, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Config, clap::Error> {
        Config::try_parse_from(std::iter::once("bouncy").chain(args.iter().copied()))
    }

    #[test]
    fn defaults() {
        let config = parse(&[]).unwrap();
        assert!(!config.spawn_at_collision);
        assert_eq!(config.min_particles, None);
        assert_eq!(config.gravity, 100);
        assert_eq!(config.wall_elasticity, 1.0);
        assert_eq!(config.particle_elasticity, 1.0);
        assert_eq!(config.width, None);
        assert_eq!(config.height, None);
        assert!(!config.cpu);
        assert!(!config.mute);
        assert!(!config.trails);
        assert_eq!(config.particle_size, 1.5);
        assert_eq!(config.color_mode, ColorMode::Solid);
        assert_eq!(config.seed, None);
    }

    #[test]
    fn parses_all_options() {
        let config = parse(&[
            "--spawn-at-collision",
            "--min-particles",
            "50",
            "--gravity",
            "-200",
            "--wall-elasticity",
            "0.5",
            "--particle-elasticity",
            "1.2",
            "--width",
            "800",
            "--height",
            "600",
            "--cpu",
            "--mute",
            "--trails",
            "--particle-size",
            "3.0",
            "--color-mode",
            "velocity",
            "--seed",
            "42",
            "--verbose",
        ])
        .unwrap();
        assert!(config.spawn_at_collision);
        assert_eq!(config.min_particles, Some(50));
        assert_eq!(config.gravity, -200);
        assert_eq!(config.wall_elasticity, 0.5);
        assert_eq!(config.particle_elasticity, 1.2);
        assert_eq!(config.width, Some(800));
        assert_eq!(config.height, Some(600));
        assert!(config.cpu);
        assert!(config.mute);
        assert!(config.trails);
        assert_eq!(config.particle_size, 3.0);
        assert_eq!(config.color_mode, ColorMode::Velocity);
        assert_eq!(config.seed, Some(42));
        assert!(config.verbose);
    }

    #[test]
    fn width_requires_height() {
        assert!(parse(&["--width", "800"]).is_err());
        assert!(parse(&["--height", "600"]).is_err());
        assert!(parse(&["--width", "800", "--height", "600"]).is_ok());
    }

    #[test]
    fn explosion_threshold_defaults_and_bounds() {
        assert_eq!(parse(&[]).unwrap().explosion_threshold, 30);
        assert_eq!(
            parse(&["--explosion-threshold", "0"])
                .unwrap()
                .explosion_threshold,
            0
        );
        assert_eq!(
            parse(&["--explosion-threshold", "1000"])
                .unwrap()
                .explosion_threshold,
            1000
        );
        assert!(parse(&["--explosion-threshold", "1001"]).is_err());
        assert!(parse(&["--explosion-threshold", "-1"]).is_err());
    }

    #[test]
    fn rejects_out_of_range_values() {
        assert!(parse(&["--min-particles", "1"]).is_err());
        assert!(parse(&["--min-particles", "101"]).is_err());
        assert!(parse(&["--gravity", "1001"]).is_err());
        assert!(parse(&["--wall-elasticity", "1.6"]).is_err());
        assert!(parse(&["--wall-elasticity", "-0.1"]).is_err());
        assert!(parse(&["--wall-elasticity", "nan"]).is_err());
        assert!(parse(&["--particle-size", "0.4"]).is_err());
        assert!(parse(&["--width", "99", "--height", "600"]).is_err());
    }

    #[test]
    fn rejects_unknown_options() {
        assert!(parse(&["--bogus"]).is_err());
    }
}
