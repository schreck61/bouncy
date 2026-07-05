// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Command line configuration.

use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, ValueEnum};

/// How particles are colored.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ColorMode {
    /// Each particle keeps a random bright color.
    #[default]
    Solid,
    /// Particle hue is mapped from its current speed (blue = slow, red = fast).
    Velocity,
}

/// Where collision-triggered spawns appear.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum SpawnMode {
    /// New particles appear near the screen center (classic fountain).
    #[default]
    Center,
    /// New particles are ejected beside the collision that spawned them.
    Collision,
    /// Collisions do not spawn (fixed population; explosions never trigger).
    Off,
}

impl SpawnMode {
    /// Cycle for the B key: center -> collision -> off.
    pub fn next(self) -> Self {
        match self {
            SpawnMode::Center => SpawnMode::Collision,
            SpawnMode::Collision => SpawnMode::Off,
            SpawnMode::Off => SpawnMode::Center,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SpawnMode::Center => "center",
            SpawnMode::Collision => "at collisions",
            SpawnMode::Off => "off",
        }
    }
}

/// Curated settings bundles. A preset supplies defaults; any option given
/// explicitly on the command line overrides the preset's value for it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Preset {
    /// Low gravity, energetic collision sprays, trails, velocity colors.
    Fireworks,
    /// Slow heavy blobs that merge and drift; fusion/fission enabled.
    #[value(alias = "lava-lamp")]
    Blob,
    /// A fixed rack of large elastic balls; no spawning, no explosions.
    Billiards,
    /// Many tiny particles drifting on the flow field with soft walls.
    #[value(alias = "snow")]
    Peace,
    /// Weightless particles slung around pinned gravity wells; trails
    /// paint the orbits.
    Orbits,
}

impl Preset {
    pub fn label(self) -> &'static str {
        match self {
            Preset::Fireworks => "fireworks",
            Preset::Blob => "blob",
            Preset::Billiards => "billiards",
            Preset::Peace => "peace",
            Preset::Orbits => "orbits",
        }
    }

    /// Apply this preset's bundle to `config`, skipping any field the user
    /// set explicitly on the command line.
    fn apply(self, config: &mut Config, matches: &ArgMatches) {
        fn set<T>(matches: &ArgMatches, id: &str, field: &mut T, value: T) {
            if matches.value_source(id) != Some(ValueSource::CommandLine) {
                *field = value;
            }
        }

        match self {
            Preset::Fireworks => {
                set(matches, "gravity", &mut config.gravity, 40);
                set(
                    matches,
                    "spawn_mode",
                    &mut config.spawn_mode,
                    SpawnMode::Collision,
                );
                set(matches, "trails", &mut config.trails, true);
                set(
                    matches,
                    "color_mode",
                    &mut config.color_mode,
                    ColorMode::Velocity,
                );
                set(
                    matches,
                    "wall_elasticity",
                    &mut config.wall_elasticity,
                    0.85,
                );
                set(
                    matches,
                    "explosion_threshold",
                    &mut config.explosion_threshold,
                    80,
                );
                set(
                    matches,
                    "min_particles",
                    &mut config.min_particles,
                    Some(20),
                );
            }
            Preset::Blob => {
                set(matches, "matter", &mut config.matter, true);
                // Weightless, lossless, and slow: blobs drift below the
                // fusion threshold and merge instead of sinking into a pile.
                set(matches, "gravity", &mut config.gravity, 0);
                set(matches, "initial_speed", &mut config.initial_speed, 60.0);
                set(matches, "particle_size", &mut config.particle_size, 5.0);
                set(
                    matches,
                    "min_particles",
                    &mut config.min_particles,
                    Some(40),
                );
                set(
                    matches,
                    "spawn_mode",
                    &mut config.spawn_mode,
                    SpawnMode::Off,
                );
                set(
                    matches,
                    "explosion_threshold",
                    &mut config.explosion_threshold,
                    0,
                );
            }
            Preset::Billiards => {
                set(matches, "gravity", &mut config.gravity, 0);
                set(matches, "particle_size", &mut config.particle_size, 7.0);
                set(
                    matches,
                    "min_particles",
                    &mut config.min_particles,
                    Some(12),
                );
                set(
                    matches,
                    "spawn_mode",
                    &mut config.spawn_mode,
                    SpawnMode::Off,
                );
                set(
                    matches,
                    "explosion_threshold",
                    &mut config.explosion_threshold,
                    0,
                );
            }
            Preset::Peace => {
                set(matches, "flow", &mut config.flow, true);
                // Gentle flakes: born slow, entrained by the flow, drifting
                // down under light gravity. Silent by default - the constant
                // grazing contacts would otherwise tick continuously (M
                // unmutes at runtime).
                set(matches, "mute", &mut config.mute, true);
                set(matches, "initial_speed", &mut config.initial_speed, 40.0);
                set(matches, "gravity", &mut config.gravity, 25);
                set(
                    matches,
                    "min_particles",
                    &mut config.min_particles,
                    Some(90),
                );
                set(matches, "wall_elasticity", &mut config.wall_elasticity, 0.1);
                set(
                    matches,
                    "particle_elasticity",
                    &mut config.particle_elasticity,
                    0.05,
                );
                set(
                    matches,
                    "spawn_mode",
                    &mut config.spawn_mode,
                    SpawnMode::Off,
                );
                set(
                    matches,
                    "explosion_threshold",
                    &mut config.explosion_threshold,
                    0,
                );
            }
            Preset::Orbits => {
                // A binary system of pinned wells with weightless particles
                // launched slowly enough to stay bound; trails paint the
                // orbit ribbons.
                set(matches, "wells", &mut config.wells, 2);
                set(matches, "gravity", &mut config.gravity, 0);
                set(matches, "trails", &mut config.trails, true);
                set(matches, "initial_speed", &mut config.initial_speed, 220.0);
                set(
                    matches,
                    "min_particles",
                    &mut config.min_particles,
                    Some(40),
                );
                set(
                    matches,
                    "spawn_mode",
                    &mut config.spawn_mode,
                    SpawnMode::Off,
                );
                set(
                    matches,
                    "explosion_threshold",
                    &mut config.explosion_threshold,
                    0,
                );
            }
        }
    }
}

/// GPU-accelerated particle simulation with elastic collisions, gravity,
/// and explosive chain reactions.
#[derive(Parser, Clone, Debug)]
#[command(name = "bouncy", version, about, after_help = CONTROLS_HELP)]
pub struct Config {
    /// Apply a curated settings bundle; explicit options override its values
    #[arg(long, value_enum)]
    pub preset: Option<Preset>,

    /// Where collision-triggered spawns appear
    #[arg(long, value_enum, default_value_t = SpawnMode::Center)]
    pub spawn_mode: SpawnMode,

    /// Alias for --spawn-mode collision (kept for compatibility)
    #[arg(long, conflicts_with = "spawn_mode")]
    pub spawn_at_collision: bool,

    /// Enable matter mechanics: slow contacts fuse particles together,
    /// hard impacts split them into fragments (toggle at runtime with X)
    #[arg(long)]
    pub matter: bool,

    /// Enable the ambient flow field that pushes particles along drifting
    /// currents (toggle at runtime with F)
    #[arg(long)]
    pub flow: bool,

    /// Pin this many attracting gravity wells around the screen center at
    /// startup (pin more at runtime with W; the range mirrors
    /// MAX_PINNED_WELLS)
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=16))]
    pub wells: u32,

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

    /// Top speed of newly created particles in pixels/sec (they start at
    /// 50-100% of this)
    #[arg(long, default_value_t = crate::physics::INITIAL_VELOCITY, value_parser = parse_initial_speed)]
    pub initial_speed: f64,

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

impl Config {
    /// The spawn mode, honoring the deprecated `--spawn-at-collision` alias.
    pub fn effective_spawn_mode(&self) -> SpawnMode {
        if self.spawn_at_collision {
            SpawnMode::Collision
        } else {
            self.spawn_mode
        }
    }

    /// Parse the process command line, then overlay the chosen preset onto
    /// every option the user did not set explicitly.
    pub fn resolve() -> Self {
        match Self::try_resolve_from(std::env::args()) {
            Ok(config) => config,
            Err(e) => e.exit(),
        }
    }

    /// Testable core of [`Config::resolve`].
    pub fn try_resolve_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let matches = Self::command().try_get_matches_from(itr)?;
        let mut config = Self::from_arg_matches(&matches)?;
        if let Some(preset) = config.preset {
            preset.apply(&mut config, &matches);
        }
        Ok(config)
    }
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
  B                  Cycle spawn mode (center / collision / off)
  X                  Toggle matter mechanics (fusion/fission)
  F                  Toggle the flow field
  G (hold)           Gravity well at the cursor; Shift+G repels
  W                  Pin a persistent well at the cursor; Shift+W repels
  Shift+R            Clear all pinned wells
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

fn parse_initial_speed(s: &str) -> Result<f64, String> {
    parse_range_f64(s, "initial speed", 10.0, 2000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Config, clap::Error> {
        Config::try_resolve_from(std::iter::once("bouncy").chain(args.iter().copied()))
    }

    #[test]
    fn defaults() {
        let config = parse(&[]).unwrap();
        assert!(!config.spawn_at_collision);
        assert_eq!(config.min_particles, None);
        assert_eq!(config.wells, 0);
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
        assert!(parse(&["--wells", "17"]).is_err());
        assert!(parse(&["--wells", "-1"]).is_err());
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

    #[test]
    fn spawn_mode_parses_and_alias_works() {
        assert_eq!(
            parse(&[]).unwrap().effective_spawn_mode(),
            SpawnMode::Center
        );
        assert_eq!(
            parse(&["--spawn-mode", "collision"])
                .unwrap()
                .effective_spawn_mode(),
            SpawnMode::Collision
        );
        assert_eq!(
            parse(&["--spawn-mode", "off"])
                .unwrap()
                .effective_spawn_mode(),
            SpawnMode::Off
        );
        // Deprecated alias still selects collision mode.
        assert_eq!(
            parse(&["--spawn-at-collision"])
                .unwrap()
                .effective_spawn_mode(),
            SpawnMode::Collision
        );
        // But it conflicts with the new flag.
        assert!(parse(&["--spawn-at-collision", "--spawn-mode", "off"]).is_err());
    }

    #[test]
    fn preset_applies_its_bundle() {
        let config = parse(&["--preset", "billiards"]).unwrap();
        assert_eq!(config.gravity, 0);
        assert_eq!(config.particle_size, 7.0);
        assert_eq!(config.min_particles, Some(12));
        assert_eq!(config.spawn_mode, SpawnMode::Off);
        assert_eq!(config.explosion_threshold, 0);

        let config = parse(&["--preset", "blob"]).unwrap();
        assert!(config.matter);
        assert_eq!(config.gravity, 0);
        assert_eq!(config.initial_speed, 60.0);

        let config = parse(&["--preset", "peace"]).unwrap();
        assert!(config.flow);
        assert!(config.mute, "peace is silent by default");
        assert_eq!(config.initial_speed, 40.0);

        let config = parse(&["--preset", "fireworks"]).unwrap();
        assert_eq!(config.spawn_mode, SpawnMode::Collision);
        assert!(config.trails);
        assert_eq!(config.color_mode, ColorMode::Velocity);

        let config = parse(&["--preset", "orbits"]).unwrap();
        assert_eq!(config.wells, 2);
        assert_eq!(config.gravity, 0);
        assert!(config.trails);
        assert_eq!(config.initial_speed, 220.0);
        assert_eq!(config.spawn_mode, SpawnMode::Off);
        assert_eq!(config.explosion_threshold, 0);
    }

    #[test]
    fn old_preset_names_still_work_as_aliases() {
        assert_eq!(
            parse(&["--preset", "lava-lamp"]).unwrap().preset,
            Some(Preset::Blob)
        );
        assert_eq!(
            parse(&["--preset", "snow"]).unwrap().preset,
            Some(Preset::Peace)
        );
    }

    #[test]
    fn explicit_options_override_the_preset() {
        let config = parse(&["--preset", "billiards", "--gravity", "50"]).unwrap();
        assert_eq!(config.gravity, 50, "explicit flag wins");
        assert_eq!(config.particle_size, 7.0, "rest of preset still applies");

        let config = parse(&["--preset", "peace", "--min-particles", "10"]).unwrap();
        assert_eq!(config.min_particles, Some(10));
        assert!(config.flow);

        let config = parse(&["--preset", "orbits", "--wells", "5"]).unwrap();
        assert_eq!(config.wells, 5, "explicit well count wins");
        assert!(config.trails, "rest of preset still applies");
    }
}
