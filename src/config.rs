// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Command line configuration.

use crate::presets::{Preset, Scene};
use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum};

/// Limits for runtime-adjustable parameters, shared between the clap
/// value parsers and the hotkey/command clamps in the app layer so the
/// two can never disagree.
pub const GRAVITY_LIMIT: i32 = 1000;
pub const ELASTICITY_MAX: f64 = 1.5;
pub const EXPLOSION_THRESHOLD_MAX: usize = 1000;

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

/// GPU-accelerated particle simulation with elastic collisions, gravity,
/// and explosive chain reactions.
#[derive(Parser, Clone, Debug)]
#[command(name = "bouncy", version, about, after_help = controls_help(),
          args_override_self = true)]
pub struct Config {
    /// Apply a settings bundle: a built-in preset (fireworks, blob,
    /// billiards, peace, orbits, mandala) or one from the user presets
    /// file (see --list-presets); explicit options override its values
    #[arg(long, value_name = "NAME")]
    pub preset: Option<String>,

    /// Load user presets from this TOML file instead of the platform
    /// default location
    #[arg(long, value_name = "PATH")]
    pub presets_file: Option<std::path::PathBuf>,

    /// List built-in and user presets, then exit
    #[arg(long)]
    pub list_presets: bool,

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

    /// Give every particle gravity proportional to its mass, so clumps
    /// attract and accrete (toggle at runtime with A)
    #[arg(long)]
    pub self_gravity: bool,

    /// Pin this many attracting gravity wells around the screen center at
    /// startup (pin more at runtime with W; the range mirrors
    /// `MAX_PINNED_WELLS`)
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=16))]
    pub wells: u32,

    /// Set starting/minimum particle count
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=100))]
    pub min_particles: Option<u32>,

    /// Gravity as a percentage of standard; negative values pull upward
    #[arg(long, default_value_t = 100, allow_negative_numbers = true,
          value_parser = clap::value_parser!(i32).range(i64::from(-GRAVITY_LIMIT)..=i64::from(GRAVITY_LIMIT)))]
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

    /// Quantize collision pings to a pentatonic scale — energy picks the
    /// note (toggle at runtime with S)
    #[arg(long)]
    pub music: bool,

    /// Mirror the frame 4-fold around the screen center (toggle at runtime
    /// with K)
    #[arg(long)]
    pub kaleidoscope: bool,

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

    /// Particle births per second (collision spawns plus fission
    /// fragments) that trigger an automatic explosion; 0 disables
    /// automatic explosions entirely (right-click still works)
    #[arg(long, default_value_t = crate::explosion::SPAWN_RATE_THRESHOLD,
          value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(0..=EXPLOSION_THRESHOLD_MAX as u64))]
    pub explosion_threshold: usize,

    /// Slow time briefly (bullet time) whenever an explosion ring starts
    #[arg(long)]
    pub bullet_time: bool,

    /// Seed the random number generator (reproducible starting conditions)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Print per-second FPS statistics to stdout
    #[arg(long)]
    pub verbose: bool,

    /// Scene geometry from a user preset (walls/wells keys); not a CLI
    /// argument — populated during preset resolution.
    #[arg(skip)]
    pub scene: Scene,
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

    /// Print a startup summary of every setting that differs from the
    /// defaults (plus the preset and seed when given).
    pub fn print_summary(&self) {
        if let Some(ref preset) = self.preset {
            println!("Preset: {preset}");
        }
        println!("Spawn mode: {}", self.effective_spawn_mode().label());
        if self.matter {
            println!("Matter mechanics: fusion/fission enabled");
        }
        if self.flow {
            println!("Flow field: enabled");
        }
        if self.self_gravity {
            println!("Self-gravity: mass attracts mass");
        }
        if self.wells > 0 {
            println!("Pinned wells: {}", self.wells);
        }
        if self.music {
            println!("Musical pings: pentatonic scale");
        }
        if self.kaleidoscope {
            println!("Kaleidoscope: enabled");
        }
        if self.gravity != 100 {
            println!("Gravity: {}%", self.gravity);
        }
        if (self.wall_elasticity - 1.0).abs() > f64::EPSILON {
            println!("Wall elasticity: {}", self.wall_elasticity);
        }
        if (self.particle_elasticity - 1.0).abs() > f64::EPSILON {
            println!("Particle elasticity: {}", self.particle_elasticity);
        }
        if let Some(seed) = self.seed {
            println!("Random seed: {seed}");
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

    /// Testable core of [`Config::resolve`]: parses the arguments and, when
    /// the chosen preset is not a built-in, loads the user presets file.
    pub fn try_resolve_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let args: Vec<std::ffi::OsString> = itr.into_iter().map(Into::into).collect();

        // Peek at the arguments to learn the preset name and presets-file
        // path; built-in presets never touch the filesystem.
        let peek = {
            let matches = Self::command().try_get_matches_from(&args)?;
            Self::from_arg_matches(&matches)?
        };
        let needs_file = peek
            .preset
            .as_ref()
            .is_some_and(|name| Preset::from_str(name, true).is_err());
        let user = if needs_file {
            crate::presets::load(peek.presets_file.as_deref()).map_err(|msg| {
                Self::command().error(clap::error::ErrorKind::ValueValidation, msg)
            })?
        } else {
            None
        };
        Self::try_resolve_with(&args, user.as_ref())
    }

    /// Core resolution with the user presets supplied by the caller, so
    /// tests can inject them without touching the filesystem.
    pub fn try_resolve_with(
        args: &[std::ffi::OsString],
        user: Option<&crate::presets::UserPresets>,
    ) -> Result<Self, clap::Error> {
        let matches = Self::command().try_get_matches_from(args)?;
        let config = Self::from_arg_matches(&matches)?;
        let Some(name) = config.preset.clone() else {
            return Ok(config);
        };

        // Built-ins (including their old-name aliases) win over the file.
        if let Ok(builtin) = Preset::from_str(&name, true) {
            let mut config =
                Self::parse_spliced(args, builtin.args().iter().copied()).map_err(|e| {
                    Self::command().error(
                        e.kind(),
                        format!(
                            "built-in preset '{}' failed to resolve (a bug): {e}",
                            builtin.label()
                        ),
                    )
                })?;
            config.preset = Some(builtin.label().to_string());
            return Ok(config);
        }

        let err = |msg: String| Self::command().error(clap::error::ErrorKind::ValueValidation, msg);
        let Some(user) = user else {
            let looked: Vec<String> = crate::presets::default_paths()
                .iter()
                .map(|p| format!("'{}'", p.display()))
                .collect();
            let looked = if looked.is_empty() {
                String::new()
            } else {
                format!(" (looked for {})", looked.join(", "))
            };
            return Err(err(format!(
                "unknown preset '{name}' and no user presets file was found{looked}; \
                 see --list-presets"
            )));
        };
        let Some(entry) = user.presets.get(&name) else {
            let names: Vec<&str> = user.presets.keys().map(String::as_str).collect();
            return Err(err(format!(
                "unknown preset '{name}'; user presets in '{}': {}",
                user.path.display(),
                if names.is_empty() {
                    "(none)".to_string()
                } else {
                    names.join(", ")
                }
            )));
        };

        // The base built-in's options go first, the user preset's second,
        // so the preset overrides its base wherever both speak.
        let injected = entry
            .base
            .iter()
            .flat_map(|base| base.args().iter().copied())
            .chain(entry.args.iter().map(String::as_str));
        let mut config = Self::parse_spliced(args, injected).map_err(|e| {
            err(format!(
                "in preset '{name}' from '{}':\n{e}",
                user.path.display()
            ))
        })?;
        config.scene = entry.scene.clone();
        config.preset = Some(name);
        Ok(config)
    }

    /// Parse the command line with `injected` preset options spliced in
    /// front of the user's own arguments and re-parse: the same parser
    /// validates preset values, and because explicit arguments come later,
    /// they always win (`args_override_self` keeps the last occurrence).
    fn parse_spliced<'a>(
        args: &[std::ffi::OsString],
        injected: impl Iterator<Item = &'a str>,
    ) -> Result<Self, clap::Error> {
        let mut spliced: Vec<std::ffi::OsString> = Vec::with_capacity(args.len() + 16);
        spliced.push(args.first().cloned().unwrap_or_else(|| "bouncy".into()));
        spliced.extend(injected.map(std::ffi::OsString::from));
        spliced.extend(args.iter().skip(1).cloned());
        let matches = Self::command().try_get_matches_from(spliced)?;
        Self::from_arg_matches(&matches)
    }
}

/// Keyboard and mouse controls as (input, action) pairs: the single
/// source for the --help footer. The HUD key reference (app.rs) uses
/// curated compressed lines for screen space, and the README repeats the
/// table for browsers — a test checks each input here appears in the
/// README so the copies cannot drift silently.
pub const CONTROLS: &[(&str, &str)] = &[
    ("Space, Escape, Q", "Exit"),
    ("P", "Pause / resume"),
    ("N", "Advance one frame (while paused)"),
    ("R", "Reset the simulation"),
    ("M", "Mute / unmute audio"),
    ("H", "Cycle the HUD (off / stats / stats+keys)"),
    ("Up / Down", "Adjust gravity by 10%"),
    ("Left / Right", "Adjust particle elasticity by 0.05"),
    ("[ / ]", "Adjust wall elasticity by 0.05"),
    (", / .", "Slow down / speed up time by 0.05 (0.1x-4x)"),
    ("- / =", "Adjust explosion threshold by 5 (0 = off)"),
    ("T", "Toggle motion trails"),
    ("C", "Cycle color mode"),
    ("B", "Cycle spawn mode (center / collision / off)"),
    ("X", "Toggle matter mechanics (fusion/fission)"),
    ("F", "Toggle the flow field"),
    ("A", "Toggle self-gravity (mass attracts mass)"),
    ("S", "Toggle musical pings (pentatonic scale)"),
    ("K", "Toggle kaleidoscope rendering"),
    ("G (hold)", "Gravity well at the cursor; Shift+G repels"),
    ("W", "Pin a persistent well at the cursor; Shift+W repels"),
    ("Shift+R", "Clear all pinned wells"),
    (
        "V (hold + drag)",
        "Draw wall segments that particles bounce off",
    ),
    ("Shift+V", "Clear all drawn walls"),
    ("O", "Save a screenshot (PNG in the working directory)"),
    (
        "E",
        "Export settings and scene (wells/walls) as a preset file",
    ),
    ("Left click", "Spawn a burst of particles at the cursor"),
    (
        "Middle click",
        "Launch a comet from the far edge toward the cursor",
    ),
    ("Right click", "Trigger an explosion at the cursor"),
];

/// Render the --help footer from the controls table.
fn controls_help() -> String {
    let lines: Vec<String> = CONTROLS
        .iter()
        .map(|(input, action)| format!("  {input:<18} {action}"))
        .collect();
    format!(
        "Controls:\n{}\n\nRendering:\n  Uses GPU rendering (wgpu) by default. \
         Falls back to CPU rendering\n  (softbuffer) if GPU is unavailable. \
         Use --cpu to force CPU rendering.",
        lines.join("\n")
    )
}

fn parse_range_f64(s: &str, name: &str, min: f64, max: f64) -> Result<f64, String> {
    let value: f64 = s.parse().map_err(|_| format!("{name} must be a number"))?;
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be between {min} and {max}"));
    }
    Ok(value)
}

fn parse_elasticity(s: &str) -> Result<f64, String> {
    parse_range_f64(s, "elasticity", 0.0, ELASTICITY_MAX)
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
        assert!(!config.bullet_time, "bullet time is opt-in");
        assert!(!config.music);
        assert!(!config.kaleidoscope);
        assert!(!config.self_gravity);
    }

    #[test]
    fn music_and_kaleidoscope_flags_parse() {
        assert!(parse(&["--music"]).unwrap().music);
        assert!(parse(&["--kaleidoscope"]).unwrap().kaleidoscope);
    }

    #[test]
    fn bullet_time_flag_parses() {
        assert!(parse(&["--bullet-time"]).unwrap().bullet_time);
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
    fn every_control_is_documented_in_the_readme() {
        // Drift insurance for the three hand-maintained control listings:
        // the README table must mention every input from the CONTROLS
        // source of truth (backticks stripped to match its formatting).
        let readme = include_str!("../README.md").replace('`', "");
        for (input, _) in CONTROLS {
            assert!(
                readme.contains(input),
                "control '{input}' is missing from the README controls table"
            );
        }
    }

    #[test]
    fn every_builtin_preset_resolves_cleanly() {
        // Built-in bundles are static CLI-arg lists; a typo in one would
        // only surface at runtime, so resolve every variant here.
        for preset in Preset::value_variants() {
            let name = preset.label();
            let config = parse(&["--preset", name])
                .unwrap_or_else(|e| panic!("built-in '{name}' must resolve: {e}"));
            assert_eq!(config.preset.as_deref(), Some(name));
        }
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
        assert!(config.bullet_time, "fireworks slows down its explosions");

        let config = parse(&["--preset", "orbits"]).unwrap();
        assert_eq!(config.wells, 2);
        assert_eq!(config.gravity, 0);
        assert!(config.trails);
        assert_eq!(config.initial_speed, 220.0);
        assert_eq!(config.spawn_mode, SpawnMode::Off);
        assert_eq!(config.explosion_threshold, 0);

        let config = parse(&["--preset", "accretion"]).unwrap();
        assert!(config.self_gravity);
        assert!(config.matter);
        assert!(config.trails);
        assert_eq!(config.gravity, 0);
        assert_eq!(config.particle_elasticity, 0.6);
        assert_eq!(config.spawn_mode, SpawnMode::Off);
        assert_eq!(config.explosion_threshold, 0);

        let config = parse(&["--preset", "mandala"]).unwrap();
        assert!(config.kaleidoscope);
        assert!(config.bullet_time);
        assert!(config.trails);
        assert_eq!(config.gravity, 0);
        assert_eq!(config.spawn_mode, SpawnMode::Collision);
        assert_eq!(config.color_mode, ColorMode::Velocity);
        assert_eq!(config.explosion_threshold, 80);
    }

    #[test]
    fn old_preset_names_still_work_as_aliases() {
        let config = parse(&["--preset", "lava-lamp"]).unwrap();
        assert_eq!(config.preset.as_deref(), Some("blob"), "canonical label");
        assert!(config.matter, "blob bundle applied");
        let config = parse(&["--preset", "snow"]).unwrap();
        assert_eq!(config.preset.as_deref(), Some("peace"));
        assert!(config.flow, "peace bundle applied");
    }

    /// Resolve `args` against user presets parsed from `toml` (no
    /// filesystem involved).
    fn parse_with(args: &[&str], toml: &str) -> Result<Config, clap::Error> {
        let user = crate::presets::parse(toml, std::path::Path::new("/test/presets.toml"))
            .expect("test presets must parse");
        let args: Vec<std::ffi::OsString> = std::iter::once("bouncy")
            .chain(args.iter().copied())
            .map(Into::into)
            .collect();
        Config::try_resolve_with(&args, Some(&user))
    }

    const PACHINKO: &str = "[pachinko]\n\
        base = \"billiards\"\n\
        gravity = 80\n\
        particle-size = 4.0\n";

    #[test]
    fn user_preset_applies_its_values_and_its_base() {
        let config = parse_with(&["--preset", "pachinko"], PACHINKO).unwrap();
        assert_eq!(config.preset.as_deref(), Some("pachinko"));
        assert_eq!(config.gravity, 80, "preset value wins over base");
        assert_eq!(config.particle_size, 4.0, "preset value wins over default");
        assert_eq!(
            config.spawn_mode,
            SpawnMode::Off,
            "unset options fall through to the base (billiards)"
        );
        assert_eq!(config.explosion_threshold, 0, "base value applied");
        assert_eq!(config.wall_elasticity, 1.0, "everything else stays default");
    }

    #[test]
    fn explicit_flags_override_a_user_preset() {
        let config = parse_with(&["--preset", "pachinko", "--gravity", "10"], PACHINKO).unwrap();
        assert_eq!(config.gravity, 10, "the command line always wins");
        assert_eq!(config.particle_size, 4.0, "rest of the preset holds");
        assert_eq!(config.spawn_mode, SpawnMode::Off, "and so does its base");
    }

    #[test]
    fn user_preset_without_base_leaves_defaults_alone() {
        let config = parse_with(&["--preset", "slow"], "[slow]\ninitial-speed = 60.0\n").unwrap();
        assert_eq!(config.initial_speed, 60.0);
        assert_eq!(config.gravity, 100, "defaults untouched");
        assert_eq!(config.spawn_mode, SpawnMode::Center);
    }

    #[test]
    fn user_preset_values_are_validated_like_the_command_line() {
        let err = parse_with(&["--preset", "broken"], "[broken]\ngravity = 5000\n").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("broken"), "error names the preset: {msg}");

        // Cross-option rules apply too: width requires height.
        let err = parse_with(&["--preset", "sized"], "[sized]\nwidth = 800\n").unwrap_err();
        assert!(err.to_string().contains("sized"), "{err}");
    }

    #[test]
    fn unknown_preset_lists_the_available_user_presets() {
        let err = parse_with(&["--preset", "nope"], PACHINKO).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown preset 'nope'"), "{msg}");
        assert!(msg.contains("pachinko"), "lists what exists: {msg}");

        let args: Vec<std::ffi::OsString> = ["bouncy", "--preset", "nope"]
            .iter()
            .map(Into::into)
            .collect();
        let err = Config::try_resolve_with(&args, None).unwrap_err();
        assert!(
            err.to_string().contains("no user presets file"),
            "explains the missing file: {err}"
        );
    }

    #[test]
    fn builtin_presets_win_over_the_user_file() {
        // A built-in name resolves without consulting user presets even
        // when a file is present.
        let config = parse_with(&["--preset", "billiards"], PACHINKO).unwrap();
        assert_eq!(config.preset.as_deref(), Some("billiards"));
        assert_eq!(config.particle_size, 7.0);
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
