// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! User-defined presets, loaded from a TOML file.
//!
//! Each top-level table is a preset; its keys are the CLI option names from
//! `--help` (kebab-case; underscores are accepted and normalized). Values
//! are converted to their command-line form and spliced in front of the
//! user's real arguments, so the same clap parser validates them and the
//! usual precedence holds: explicit flag > user preset > `base` built-in >
//! default. A `base` key names a built-in preset to inherit from (one
//! level; user presets cannot base on each other).
//!
//! ```toml
//! [pinball]
//! description = "Big slow balls under heavy gravity"
//! base = "billiards"
//! gravity = 80
//! particle-size = 4.0
//! ```
//!
//! The reserved keys `base` and `description` are preset metadata, not
//! options; `description` is shown by --list-presets. The file lives at
//! the first of the [`default_paths`] candidates that exists, or wherever
//! `--presets-file` points. A missing default file is fine; a malformed
//! one is a loud error — silently ignoring it would cost users an hour of
//! wondering why their preset didn't load.

use crate::physics::Polarity;
use clap::ValueEnum;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    /// Kaleidoscope over weightless collision sprays: symmetric blooms
    /// of trails.
    Mandala,
    /// Self-gravitating dust with dissipative collisions: clumps form,
    /// fuse, and sweep their orbits clean.
    Accretion,
    /// Four silent channels with drumming end caps: weightless particles
    /// rattle the box and thump a stereo four-note tom pattern.
    Percussion,
    /// A stair of eleven pitched bars, high to low: falling particles
    /// play descending pentatonic runs and floor-bounced ascents.
    Marimba,
    /// A staggered peg field under gravity: cascades plink down through
    /// descending rows and funnel back up for another pass.
    Pachinko,
    /// A fan of six pitched strings grazed by particles orbiting a
    /// binary of wells: slow rolled arpeggios, painted with trails.
    Harp,
}

impl Preset {
    pub fn label(self) -> &'static str {
        match self {
            Preset::Fireworks => "fireworks",
            Preset::Blob => "blob",
            Preset::Billiards => "billiards",
            Preset::Peace => "peace",
            Preset::Orbits => "orbits",
            Preset::Mandala => "mandala",
            Preset::Accretion => "accretion",
            Preset::Percussion => "percussion",
            Preset::Marimba => "marimba",
            Preset::Pachinko => "pachinko",
            Preset::Harp => "harp",
        }
    }

    /// The preset's bundle as command-line arguments. Presets apply by
    /// splicing these ahead of the user's real arguments and re-parsing,
    /// exactly like user presets from the TOML file: built-in values pass
    /// the same validation as the command line, and the same precedence
    /// holds — explicit flag > preset > default. A test resolves every
    /// variant to keep these lists valid.
    pub(crate) fn args(self) -> &'static [&'static str] {
        match self {
            Preset::Fireworks => &[
                "--bullet-time",
                "--gravity",
                "40",
                "--spawn-mode",
                "collision",
                "--trails",
                "--color-mode",
                "velocity",
                "--wall-elasticity",
                "0.85",
                "--explosion-threshold",
                "80",
                "--min-particles",
                "20",
            ],
            // Weightless, lossless, and slow: blobs drift below the fusion
            // threshold and merge instead of sinking into a pile.
            Preset::Blob => &[
                "--matter",
                "--gravity",
                "0",
                "--initial-speed",
                "60",
                "--particle-size",
                "5",
                "--min-particles",
                "40",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            Preset::Billiards => &[
                "--gravity",
                "0",
                "--particle-size",
                "7",
                "--min-particles",
                "12",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            // Gentle flakes: born slow, entrained by the flow, drifting down
            // under light gravity. Silent by default — the constant grazing
            // contacts would otherwise tick continuously (M unmutes).
            Preset::Peace => &[
                "--flow",
                "--mute",
                "--initial-speed",
                "40",
                "--gravity",
                "25",
                "--min-particles",
                "90",
                "--wall-elasticity",
                "0.1",
                "--particle-elasticity",
                "0.05",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            // A binary system of pinned wells with weightless particles
            // launched slowly enough to stay bound; trails paint the orbits.
            Preset::Orbits => &[
                "--wells",
                "2",
                "--gravity",
                "0",
                "--trails",
                "--initial-speed",
                "220",
                "--min-particles",
                "40",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            // The fireworks recipe under a kaleidoscope, minus gravity:
            // weightless sprays keep the bloom radially symmetric.
            Preset::Mandala => &[
                "--bullet-time",
                "--kaleidoscope",
                "--trails",
                "--gravity",
                "0",
                "--spawn-mode",
                "collision",
                "--color-mode",
                "velocity",
                "--wall-elasticity",
                "0.85",
                "--explosion-threshold",
                "80",
                "--min-particles",
                "20",
            ],
            // Dissipation is what makes self-gravity clump: sub-elastic
            // collisions bleed off the energy that would otherwise keep
            // dust swinging through the cluster forever, and matter mode
            // fuses whatever slows into contact. No global gravity, no
            // spawning - the population evolves purely by accretion.
            Preset::Accretion => &[
                "--self-gravity",
                "--matter",
                "--trails",
                "--gravity",
                "0",
                "--particle-elasticity",
                "0.6",
                "--wall-elasticity",
                "0.75",
                "--particle-size",
                "2.5",
                "--initial-speed",
                "80",
                "--min-particles",
                "100",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            // The four instrument scenes share one recipe: chimes on,
            // musical pings so incidental collisions stay in key, a fixed
            // population (spawn off, explosions off), and the default
            // 1.0 elasticities — energy is conserved, so the instrument
            // plays itself indefinitely. Only gravity, speed, and sizing
            // differ per instrument.
            Preset::Percussion => &[
                "--wall-chimes",
                "--music",
                "--gravity",
                "0",
                "--initial-speed",
                "240",
                "--particle-size",
                "3",
                "--min-particles",
                "16",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            Preset::Marimba => &[
                "--wall-chimes",
                "--music",
                "--gravity",
                "110",
                "--initial-speed",
                "140",
                "--particle-size",
                "3",
                "--min-particles",
                "12",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            Preset::Pachinko => &[
                "--wall-chimes",
                "--music",
                "--gravity",
                "130",
                "--initial-speed",
                "100",
                "--particle-size",
                "3.5",
                "--min-particles",
                "8",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
            Preset::Harp => &[
                "--wall-chimes",
                "--music",
                "--trails",
                "--gravity",
                "0",
                "--initial-speed",
                "220",
                "--particle-size",
                "2",
                "--min-particles",
                "30",
                "--spawn-mode",
                "off",
                "--explosion-threshold",
                "0",
            ],
        }
    }

    /// The preset's scene geometry, placed exactly like a user preset's
    /// `walls`/`wells` keys. Most bundles are settings-only; the
    /// instrument presets carry the walls that make them instruments.
    /// Exhaustive so a new variant forces a conscious scene decision.
    pub(crate) fn scene(self) -> Scene {
        match self {
            Preset::Fireworks
            | Preset::Blob
            | Preset::Billiards
            | Preset::Peace
            | Preset::Orbits
            | Preset::Mandala
            | Preset::Accretion => Scene::default(),
            Preset::Percussion => percussion_scene(),
            Preset::Marimba => marimba_scene(),
            Preset::Pachinko => pachinko_scene(),
            Preset::Harp => harp_scene(),
        }
    }
}

// ---- Instrument scenes -------------------------------------------------
//
// Shared geometry language: window fractions, sounding walls always pin
// their pentatonic degree (auto pitch keys off the window diagonal and
// would drift a degree across aspect ratios), silent walls are pure
// geometry. All four rely on the default 1.0 elasticities: lossless
// bounces are the "motor" that keeps an unattended scene playing.

/// Four horizontal channels bounded by silent rails; each channel is
/// sealed by a cap at both ends carrying the same low note, so every
/// channel is one drum and stereo pan ping-pongs as particles shuttle.
fn percussion_scene() -> Scene {
    const RAIL_YS: [f64; 5] = [0.110, 0.305, 0.500, 0.695, 0.890];
    // Channel caps top-to-bottom: G4, E4, D4, C4 — lowest channel,
    // lowest drum.
    const CAP_NOTES: [u8; 4] = [3, 2, 1, 0];
    let mut walls: Vec<SceneWall> = RAIL_YS
        .iter()
        .map(|&y| silent(0.150, y, 0.850, y))
        .collect();
    for (k, &note) in CAP_NOTES.iter().enumerate() {
        let (top, bottom) = (RAIL_YS[k], RAIL_YS[k + 1]);
        walls.push(noted(0.150, top, 0.150, bottom, note));
        walls.push(noted(0.850, top, 0.850, bottom, note));
    }
    Scene {
        wells: Vec::new(),
        walls,
    }
}

/// Eleven tilted bars, one per pentatonic degree, top-left high to
/// bottom-right low; each bar tips down-right (rise = 30% of half-length)
/// so bounces shed particles toward the next lower bar, and the visual
/// length gradient matches the pitch gradient.
fn marimba_scene() -> Scene {
    let walls = (0u8..11)
        .map(|k| {
            let kf = f64::from(k);
            let (cx, cy) = (0.115 + 0.077 * kf, 0.140 + 0.068 * kf);
            let half = 0.030 + 0.0045 * kf;
            let rise = half * 0.3;
            noted(cx - half, cy - rise, cx + half, cy + rise, 10 - k)
        })
        .collect();
    Scene {
        wells: Vec::new(),
        walls,
    }
}

/// Seven staggered rows of short pegs, pitch descending with depth, and
/// two silent corner funnels that steer floor-bounced balls back into
/// the field (the always-elastic arena floor is the return spring).
fn pachinko_scene() -> Scene {
    const PEG_HALF: f64 = 0.012;
    let mut walls = Vec::new();
    for row in 0u8..7 {
        let y = 0.20 + 0.08 * f64::from(row);
        let note = 9 - row;
        let (count, x0) = if row % 2 == 0 { (8, 0.15) } else { (7, 0.20) };
        for j in 0..count {
            let cx = x0 + 0.10 * f64::from(j);
            walls.push(noted(cx - PEG_HALF, y, cx + PEG_HALF, y, note));
        }
    }
    // Funnels leave escape gaps at both ends so nothing can be trapped.
    walls.push(silent(0.020, 0.780, 0.300, 0.940));
    walls.push(silent(0.980, 0.780, 0.700, 0.940));
    Scene {
        wells: Vec::new(),
        walls,
    }
}

/// Six strings tangent to circles around the screen center, grazed by
/// particles orbiting a vertical binary of wells: orbital motion crosses
/// tangent chords at shallow angles, so passes pluck gently instead of
/// slamming. Notes are the open set C4 E4 A4 C5 E5 A5.
fn harp_scene() -> Scene {
    const STRING_NOTES: [u8; 6] = [0, 2, 4, 5, 7, 9];
    const RADII: [f64; 6] = [0.400, 0.362, 0.324, 0.286, 0.248, 0.210];
    const HALF_LENS: [f64; 6] = [0.200, 0.175, 0.152, 0.130, 0.110, 0.092];
    let walls = (0..6)
        .map(|k| {
            let kf = f64::from(u8::try_from(k).expect("six strings"));
            let (r, half) = (RADII[k], HALF_LENS[k]);
            let phi = (kf - 2.5) * 7.0_f64.to_radians();
            let (tx, ty) = (0.5 - r * phi.cos(), 0.5 + r * phi.sin());
            let (dx, dy) = (phi.sin(), phi.cos());
            noted(
                tx - half * dx,
                ty - half * dy,
                tx + half * dx,
                ty + half * dy,
                STRING_NOTES[k],
            )
        })
        .collect();
    Scene {
        wells: vec![well(0.500, 0.300), well(0.500, 0.700)],
        walls,
    }
}

/// A sounding wall with a pinned scale degree (scene walls pin their
/// notes so pitch survives aspect-ratio changes).
const fn noted(x1: f64, y1: f64, x2: f64, y2: f64, note: u8) -> SceneWall {
    SceneWall {
        x1,
        y1,
        x2,
        y2,
        note: WallNote::Note(note),
    }
}

/// Pure geometry: bounces, never chimes or flashes.
const fn silent(x1: f64, y1: f64, x2: f64, y2: f64) -> SceneWall {
    SceneWall {
        x1,
        y1,
        x2,
        y2,
        note: WallNote::Silent,
    }
}

/// An attracting pinned well.
const fn well(x: f64, y: f64) -> SceneWell {
    SceneWell {
        x,
        y,
        polarity: Polarity::Attract,
    }
}

/// Scene geometry a preset can carry: pinned wells and wall segments in
/// window-fraction coordinates (0.0-1.0 of the window width/height), so a
/// scene lays out identically on any screen size.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub wells: Vec<SceneWell>,
    pub walls: Vec<SceneWall>,
}

/// A pinned gravity well in window-fraction coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SceneWell {
    pub x: f64,
    pub y: f64,
    pub polarity: Polarity,
}

/// How a wall sounds when chimes are on.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WallNote {
    /// Pitch derived from the stroke's length (the drawn-wall default).
    #[default]
    Auto,
    /// Pinned pentatonic scale degree, 0 (lowest) up to `NOTE_COUNT`.
    Note(u8),
    /// Never sounds (and never flashes): pure geometry, like the
    /// percussion box's guide rails.
    Silent,
}

/// A wall segment in window-fraction coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SceneWall {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    /// How this wall sounds when chimes are on.
    pub note: WallNote,
}

/// One user preset: an optional built-in to inherit from, an optional
/// description for --list-presets, the remaining options in command-line
/// form (`["--gravity", "80", ...]`), and any scene geometry.
#[derive(Debug)]
pub struct UserPreset {
    pub base: Option<Preset>,
    pub description: Option<String>,
    pub args: Vec<String>,
    pub scene: Scene,
}

/// All user presets from one file, keyed by name.
#[derive(Debug)]
pub struct UserPresets {
    pub path: PathBuf,
    pub presets: BTreeMap<String, UserPreset>,
}

/// Candidate locations of the user presets file, in lookup order. The
/// XDG-style `~/.config/bouncy/presets.toml` comes first on every platform
/// — command-line users expect it, macOS included — followed by the
/// platform-blessed config directory (`~/Library/Application Support` on
/// macOS, `%APPDATA%` on Windows; on Linux the two coincide and collapse
/// to one entry). `$XDG_CONFIG_HOME` overrides `~/.config` when set.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_paths() -> Vec<PathBuf> {
    let file = |dir: PathBuf| dir.join("bouncy").join("presets.toml");
    let mut paths = Vec::new();
    match std::env::var_os("XDG_CONFIG_HOME") {
        Some(xdg) if !xdg.is_empty() => paths.push(file(PathBuf::from(xdg))),
        _ => {
            if let Some(home) = dirs::home_dir() {
                paths.push(file(home.join(".config")));
            }
        }
    }
    if let Some(config) = dirs::config_dir() {
        paths.push(file(config));
    }
    paths.dedup();
    paths
}

/// Load the user presets file. An explicitly given path must exist; the
/// default locations are searched in order and are all optional
/// (`Ok(None)` when none exists).
#[cfg(not(target_arch = "wasm32"))]
pub fn load(explicit: Option<&Path>) -> Result<Option<UserPresets>, String> {
    if let Some(path) = explicit {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read presets file '{}': {e}", path.display()))?;
        return parse(&text, path).map(Some);
    }
    for path in default_paths() {
        match std::fs::read_to_string(&path) {
            Ok(text) => return parse(&text, &path).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(format!(
                    "cannot read presets file '{}': {e}",
                    path.display()
                ));
            }
        }
    }
    Ok(None)
}

/// Parse presets from TOML text. `path` is only used in error messages and
/// listings, which keeps this testable without a filesystem.
pub fn parse(text: &str, path: &Path) -> Result<UserPresets, String> {
    let table: toml::Table = text
        .parse()
        .map_err(|e| format!("invalid TOML in '{}': {e}", path.display()))?;

    let mut presets = BTreeMap::new();
    for (name, value) in table {
        let toml::Value::Table(entries) = value else {
            return Err(format!(
                "'{name}' in '{}' must be a preset table ([{name}])",
                path.display()
            ));
        };
        if Preset::from_str(&name, true).is_ok() {
            return Err(format!(
                "preset '{name}' shadows a built-in; name it something else \
                 and use base = \"{name}\" to inherit from the built-in"
            ));
        }

        let mut base = None;
        let mut description = None;
        let mut args = Vec::new();
        let mut scene = Scene::default();
        for (key, value) in entries {
            let key = key.replace('_', "-");
            match key.as_str() {
                "walls" => scene.walls = parse_scene_walls(&name, &value)?,
                // `wells` is double-duty: an integer is the --wells ring
                // count (a CLI option), an array is scene geometry.
                "wells" if value.is_array() => {
                    scene.wells = parse_scene_wells(&name, &value)?;
                }
                "description" => {
                    let toml::Value::String(text) = value else {
                        return Err(format!("preset '{name}': description must be a string"));
                    };
                    description = Some(text);
                }
                "base" => {
                    let toml::Value::String(base_name) = value else {
                        return Err(format!(
                            "preset '{name}': base must be a string naming a built-in preset"
                        ));
                    };
                    base = Some(Preset::from_str(&base_name, true).map_err(|_| {
                        format!(
                            "preset '{name}': unknown base '{base_name}' \
                             (base must be a built-in preset)"
                        )
                    })?);
                }
                "preset" | "presets-file" | "list-presets" | "help" | "version" => {
                    return Err(format!(
                        "preset '{name}': '{key}' cannot be set from a preset"
                    ));
                }
                _ => match value {
                    toml::Value::Boolean(true) => args.push(format!("--{key}")),
                    toml::Value::Boolean(false) => {
                        return Err(format!(
                            "preset '{name}': '{key} = false' is not supported — like \
                             the command line, presets can only enable boolean options"
                        ));
                    }
                    toml::Value::Integer(i) => {
                        args.push(format!("--{key}"));
                        args.push(i.to_string());
                    }
                    toml::Value::Float(f) => {
                        args.push(format!("--{key}"));
                        args.push(f.to_string());
                    }
                    toml::Value::String(s) => {
                        args.push(format!("--{key}"));
                        args.push(s);
                    }
                    other => {
                        return Err(format!(
                            "preset '{name}': unsupported value type '{}' for '{key}'",
                            other.type_str()
                        ));
                    }
                },
            }
        }
        presets.insert(
            name,
            UserPreset {
                base,
                description,
                args,
                scene,
            },
        );
    }

    Ok(UserPresets {
        path: path.to_path_buf(),
        presets,
    })
}

/// A window-fraction coordinate: a number in 0.0..=1.0.
fn parse_fraction(preset: &str, key: &str, value: &toml::Value) -> Result<f64, String> {
    let f = match value {
        toml::Value::Integer(i) => {
            #[allow(clippy::cast_precision_loss)]
            {
                *i as f64
            }
        }
        toml::Value::Float(f) => *f,
        _ => {
            return Err(format!(
                "preset '{preset}': '{key}' coordinates must be numbers"
            ));
        }
    };
    if !(0.0..=1.0).contains(&f) {
        return Err(format!(
            "preset '{preset}': '{key}' coordinates are window fractions and \
             must be between 0.0 and 1.0 (got {f})"
        ));
    }
    Ok(f)
}

/// Parse scene walls (window fractions). Two forms per entry: the legacy
/// bare array `[x1, y1, x2, y2]`, or a table
/// `{ x1 = .., y1 = .., x2 = .., y2 = .., note = 4 }` whose optional
/// `note` pins a chime scale degree instead of deriving pitch from length.
fn parse_scene_walls(preset: &str, value: &toml::Value) -> Result<Vec<SceneWall>, String> {
    let toml::Value::Array(entries) = value else {
        return Err(format!(
            "preset '{preset}': walls must be an array of [x1, y1, x2, y2] \
             arrays or {{ x1, y1, x2, y2, note }} tables"
        ));
    };
    let mut walls = Vec::with_capacity(entries.len());
    for entry in entries {
        match entry {
            toml::Value::Array(coords) => {
                if coords.len() != 4 {
                    return Err(format!(
                        "preset '{preset}': each wall needs exactly 4 coordinates, got {}",
                        coords.len()
                    ));
                }
                let mut c = [0.0; 4];
                for (slot, coord) in c.iter_mut().zip(coords) {
                    *slot = parse_fraction(preset, "walls", coord)?;
                }
                walls.push(SceneWall {
                    x1: c[0],
                    y1: c[1],
                    x2: c[2],
                    y2: c[3],
                    note: WallNote::Auto,
                });
            }
            toml::Value::Table(table) => {
                let coord = |key: &str| -> Result<f64, String> {
                    table.get(key).map_or_else(
                        || {
                            Err(format!(
                                "preset '{preset}': a wall table is missing '{key}'"
                            ))
                        },
                        |v| parse_fraction(preset, "walls", v),
                    )
                };
                let silent = match table.get("silent") {
                    None => false,
                    Some(toml::Value::Boolean(true)) => true,
                    Some(toml::Value::Boolean(false)) => {
                        return Err(format!(
                            "preset '{preset}': like the command line, a wall \
                             can only be marked silent, not un-silent — drop \
                             the 'silent = false' key"
                        ));
                    }
                    Some(other) => {
                        return Err(format!(
                            "preset '{preset}': wall 'silent' must be the \
                             boolean true (got {other})"
                        ));
                    }
                };
                if silent && table.contains_key("note") {
                    return Err(format!(
                        "preset '{preset}': a wall cannot be both silent and \
                         note-pinned — drop one of the keys"
                    ));
                }
                let note = match table.get("note") {
                    _ if silent => WallNote::Silent,
                    None => WallNote::Auto,
                    Some(v) => {
                        let max = crate::audio::NOTE_COUNT - 1;
                        let range = 0..=i64::try_from(max).unwrap_or(i64::MAX);
                        let degree = v.as_integer().filter(|d| range.contains(d));
                        let Some(degree) = degree else {
                            return Err(format!(
                                "preset '{preset}': wall note must be an integer \
                                 scale degree 0-{max} (got {v})"
                            ));
                        };
                        // Bounds-checked against NOTE_COUNT - 1 (10) above.
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        WallNote::Note(degree as u8)
                    }
                };
                walls.push(SceneWall {
                    x1: coord("x1")?,
                    y1: coord("y1")?,
                    x2: coord("x2")?,
                    y2: coord("y2")?,
                    note,
                });
            }
            _ => {
                return Err(format!(
                    "preset '{preset}': each wall must be an [x1, y1, x2, y2] \
                     array or a {{ x1, y1, x2, y2, note/silent }} table"
                ));
            }
        }
    }
    Ok(walls)
}

/// Parse `wells = [{ x = 0.5, y = 0.25, polarity = "attract" }, ...]`
/// (window fractions; polarity defaults to attract).
fn parse_scene_wells(preset: &str, value: &toml::Value) -> Result<Vec<SceneWell>, String> {
    let toml::Value::Array(entries) = value else {
        return Err(format!(
            "preset '{preset}': wells geometry must be an array of tables"
        ));
    };
    let mut wells = Vec::with_capacity(entries.len());
    for entry in entries {
        let toml::Value::Table(table) = entry else {
            return Err(format!(
                "preset '{preset}': each well must be a table like \
                 {{ x = 0.5, y = 0.25, polarity = \"attract\" }}"
            ));
        };
        let coord = |key: &str| -> Result<f64, String> {
            table.get(key).map_or_else(
                || Err(format!("preset '{preset}': a well is missing '{key}'")),
                |v| parse_fraction(preset, "wells", v),
            )
        };
        let polarity = match table.get("polarity").and_then(toml::Value::as_str) {
            None | Some("attract") => Polarity::Attract,
            Some("repel") => Polarity::Repel,
            Some(other) => {
                return Err(format!(
                    "preset '{preset}': well polarity must be \"attract\" or \
                     \"repel\" (got '{other}')"
                ));
            }
        };
        wells.push(SceneWell {
            x: coord("x")?,
            y: coord("y")?,
            polarity,
        });
    }
    Ok(wells)
}

/// Serialize a captured scene as a complete `[name]` preset table that
/// [`parse`] round-trips. Pure (no filesystem), so tests can verify the
/// round trip.
pub fn scene_to_toml(
    name: &str,
    settings: &[(&str, toml::Value)],
    wells: &[SceneWell],
    walls: &[SceneWall],
) -> String {
    let mut table = toml::Table::new();
    for (key, value) in settings {
        table.insert((*key).to_string(), value.clone());
    }
    if !wells.is_empty() {
        let wells = wells
            .iter()
            .map(|w| {
                let mut t = toml::Table::new();
                t.insert("x".into(), w.x.into());
                t.insert("y".into(), w.y.into());
                let polarity = match w.polarity {
                    Polarity::Attract => "attract",
                    Polarity::Repel => "repel",
                };
                t.insert("polarity".into(), polarity.into());
                toml::Value::Table(t)
            })
            .collect();
        table.insert("wells".into(), toml::Value::Array(wells));
    }
    if !walls.is_empty() {
        let walls = walls
            .iter()
            .map(|w| {
                let table_with = |key: &str, value: toml::Value| {
                    let mut t = toml::Table::new();
                    t.insert("x1".into(), w.x1.into());
                    t.insert("y1".into(), w.y1.into());
                    t.insert("x2".into(), w.x2.into());
                    t.insert("y2".into(), w.y2.into());
                    t.insert(key.into(), value);
                    toml::Value::Table(t)
                };
                match w.note {
                    // Auto walls keep the legacy array form so exports
                    // stay readable by older binaries and diff-identical.
                    WallNote::Auto => {
                        toml::Value::Array(vec![w.x1.into(), w.y1.into(), w.x2.into(), w.y2.into()])
                    }
                    WallNote::Note(note) => table_with("note", i64::from(note).into()),
                    WallNote::Silent => table_with("silent", true.into()),
                }
            })
            .collect();
        table.insert("walls".into(), toml::Value::Array(walls));
    }
    let mut document = toml::Table::new();
    document.insert(name.to_string(), toml::Value::Table(table));
    document.to_string()
}

/// Write a captured scene to a uniquely named preset file in the working
/// directory, returning its path. The user copies the table into their
/// presets file (or points --presets-file at it) and runs --preset with
/// the generated name.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_scene(
    settings: &[(&str, toml::Value)],
    wells: &[SceneWell],
    walls: &[SceneWall],
) -> Result<std::path::PathBuf, String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    for n in 0..100 {
        let suffix = if n == 0 {
            String::new()
        } else {
            format!("-{n}")
        };
        let path = std::path::PathBuf::from(format!("bouncy-scene-{secs}{suffix}.toml"));
        if path.exists() {
            continue;
        }
        let text = scene_to_toml(&format!("scene-{secs}{suffix}"), settings, wells, walls);
        std::fs::write(&path, text)
            .map_err(|e| format!("cannot write '{}': {e}", path.display()))?;
        return Ok(path);
    }
    Err("too many scene exports this second".to_string())
}

/// Print built-in and user presets (--list-presets), including where the
/// user presets file was (or would be) loaded from.
#[cfg(not(target_arch = "wasm32"))]
pub fn print_list(explicit: Option<&Path>) {
    println!("Built-in presets:");
    for preset in Preset::value_variants() {
        let pv = preset
            .to_possible_value()
            .expect("built-in presets are never skipped");
        let help = pv.get_help().map(ToString::to_string).unwrap_or_default();
        println!("  {:<12} {help}", pv.get_name());
    }

    match load(explicit) {
        Ok(Some(user)) => {
            println!("\nUser presets (from '{}'):", user.path.display());
            for (name, preset) in &user.presets {
                // A description reads like the built-in listings; without
                // one, fall back to the preset's settings themselves.
                let info = preset.description.clone().unwrap_or_else(|| {
                    let base = preset
                        .base
                        .map_or_else(String::new, |b| format!("base: {}; ", b.label()));
                    format!("{base}{}", preset.args.join(" "))
                });
                println!("  {name:<12} {info}");
            }
        }
        Ok(None) => {
            // Ok(None) only happens for the default lookup: an explicit
            // --presets-file that cannot be read is an error instead.
            let looked = default_paths();
            if looked.is_empty() {
                println!("\nNo platform config directory available for user presets.");
            } else {
                println!("\nNo user presets file found. Locations checked, in order:");
                for path in looked {
                    println!("  {}", path.display());
                }
            }
            println!("Create one or pass --presets-file; see the README for the format.");
        }
        Err(msg) => eprintln!("\n{msg}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(text: &str) -> Result<UserPresets, String> {
        parse(text, Path::new("/test/presets.toml"))
    }

    #[test]
    fn labels_agree_with_clap_names() {
        // label() is a hand-written copy of the kebab-case names clap
        // derives; this pins the two together so a renamed variant cannot
        // drift silently.
        for preset in Preset::value_variants() {
            assert_eq!(
                preset.label(),
                preset.to_possible_value().unwrap().get_name(),
                "label() and the clap-derived name must agree"
            );
        }
    }

    #[test]
    fn parses_presets_into_cli_args_and_base() {
        let user = parse_str(
            "[pinball]\n\
             base = \"billiards\"\n\
             gravity = 80\n\
             particle-size = 4.5\n\
             spawn-mode = \"off\"\n\
             trails = true\n",
        )
        .unwrap();
        let preset = &user.presets["pinball"];
        assert_eq!(preset.base, Some(Preset::Billiards));
        assert_eq!(
            preset.args,
            [
                "--gravity",
                "80",
                "--particle-size",
                "4.5",
                "--spawn-mode",
                "off",
                "--trails"
            ]
        );
    }

    #[test]
    fn underscore_keys_are_normalized_to_kebab_case() {
        // Keys are written in alphabetical order so the assertion holds
        // whether the TOML map preserves or sorts key order.
        let user = parse_str("[a]\nmin_particles = 5\nspawn_mode = \"off\"\n").unwrap();
        assert_eq!(
            user.presets["a"].args,
            ["--min-particles", "5", "--spawn-mode", "off"]
        );
    }

    #[test]
    fn negative_numbers_pass_through() {
        let user = parse_str("[antigrav]\ngravity = -150\n").unwrap();
        assert_eq!(user.presets["antigrav"].args, ["--gravity", "-150"]);
    }

    #[test]
    fn description_is_kept_out_of_the_args() {
        let user = parse_str(
            "[calm]\n\
             description = \"Slow and peaceful drift\"\n\
             initial-speed = 40.0\n",
        )
        .unwrap();
        let preset = &user.presets["calm"];
        assert_eq!(
            preset.description.as_deref(),
            Some("Slow and peaceful drift")
        );
        assert_eq!(preset.args, ["--initial-speed", "40"]);

        let err = parse_str("[a]\ndescription = 5\n").unwrap_err();
        assert!(err.contains("description must be a string"), "{err}");
    }

    #[test]
    fn scene_geometry_parses_and_wells_stays_double_duty() {
        let user = parse_str(
            "[board]\n\
             wells = [{ x = 0.5, y = 0.25 }, { x = 0.5, y = 0.75, polarity = \"repel\" }]\n\
             walls = [[0.1, 0.5, 0.4, 0.5], [0.6, 0.5, 0.9, 0.5]]\n",
        )
        .unwrap();
        let scene = &user.presets["board"].scene;
        assert_eq!(scene.wells.len(), 2);
        assert_eq!(scene.wells[0].polarity, Polarity::Attract, "default");
        assert_eq!(scene.wells[1].polarity, Polarity::Repel);
        assert_eq!(
            scene.walls[1],
            SceneWall {
                x1: 0.6,
                y1: 0.5,
                x2: 0.9,
                y2: 0.5,
                note: WallNote::Auto
            }
        );

        // An integer `wells` is still the --wells ring count, not geometry.
        let user = parse_str("[ring]\nwells = 3\n").unwrap();
        let ring = &user.presets["ring"];
        assert_eq!(ring.args, ["--wells", "3"]);
        assert!(ring.scene.wells.is_empty());
    }

    #[test]
    fn scene_geometry_is_validated_loudly() {
        let err = parse_str("[a]\nwalls = [[0.1, 0.2, 0.3]]\n").unwrap_err();
        assert!(err.contains("exactly 4"), "{err}");
        let err = parse_str("[a]\nwalls = [[0.1, 0.2, 0.3, 1.5]]\n").unwrap_err();
        assert!(err.contains("between 0.0 and 1.0"), "{err}");
        let err = parse_str("[a]\nwells = [{ x = 0.5 }]\n").unwrap_err();
        assert!(err.contains("missing 'y'"), "{err}");
        let err = parse_str("[a]\nwells = [{ x = 0.5, y = 0.5, polarity = \"sideways\" }]\n")
            .unwrap_err();
        assert!(err.contains("attract"), "{err}");
    }

    #[test]
    fn exported_scenes_round_trip_through_the_parser() {
        let wells = [SceneWell {
            x: 0.5,
            y: 0.25,
            polarity: Polarity::Repel,
        }];
        let walls = [
            SceneWall {
                x1: 0.1,
                y1: 0.2,
                x2: 0.3,
                y2: 0.4,
                note: WallNote::Auto,
            },
            SceneWall {
                x1: 0.5,
                y1: 0.6,
                x2: 0.7,
                y2: 0.6,
                note: WallNote::Note(4),
            },
            SceneWall {
                x1: 0.15,
                y1: 0.8,
                x2: 0.85,
                y2: 0.8,
                note: WallNote::Silent,
            },
        ];
        let text = scene_to_toml(
            "saved",
            &[
                ("description", "Exported scene".into()),
                ("gravity", 40i64.into()),
                ("trails", true.into()),
                ("spawn-mode", "off".into()),
            ],
            &wells,
            &walls,
        );

        let user = parse(&text, Path::new("/test/export.toml")).unwrap();
        let preset = &user.presets["saved"];
        assert_eq!(preset.description.as_deref(), Some("Exported scene"));
        assert!(preset.args.contains(&"--gravity".to_string()));
        assert!(preset.args.contains(&"--trails".to_string()));
        assert_eq!(preset.scene.wells, wells);
        assert_eq!(preset.scene.walls, walls);
        // Note-free walls keep the legacy bare-array form on disk.
        assert!(text.contains("[0.1, 0.2, 0.3, 0.4]"), "{text}");
        // Silent walls carry the marker explicitly.
        assert!(text.contains("silent = true"), "{text}");
    }

    #[test]
    fn wall_tables_with_notes_parse_alongside_legacy_arrays() {
        let user = parse_str(
            "[chimes]\ngravity = 10\nwalls = [\n  [0.1, 0.2, 0.3, 0.2],\n  { x1 = 0.4, y1 = 0.5, x2 = 0.6, y2 = 0.5, note = 7 },\n  { x1 = 0.7, y1 = 0.8, x2 = 0.9, y2 = 0.8 },\n]\n",
        )
        .unwrap();
        let walls = &user.presets["chimes"].scene.walls;
        assert_eq!(walls.len(), 3);
        assert_eq!(walls[0].note, WallNote::Auto, "legacy array form");
        assert_eq!(walls[1].note, WallNote::Note(7), "table form with note");
        assert_eq!(walls[2].note, WallNote::Auto, "table form without note");
        assert!((walls[1].x1 - 0.4).abs() < 1e-12);
    }

    #[test]
    fn silent_walls_parse_and_are_validated_loudly() {
        let user = parse_str(
            "[box]\nwalls = [\n  [0.1, 0.2, 0.3, 0.2],\n  { x1 = 0.4, y1 = 0.5, x2 = 0.6, y2 = 0.5, silent = true },\n]\n",
        )
        .unwrap();
        let walls = &user.presets["box"].scene.walls;
        assert_eq!(walls[0].note, WallNote::Auto);
        assert_eq!(walls[1].note, WallNote::Silent);

        let err = parse_str(
            "[bad]\nwalls = [{ x1 = 0.1, y1 = 0.2, x2 = 0.3, y2 = 0.2, silent = false }]\n",
        )
        .unwrap_err();
        assert!(err.contains("only be marked silent"), "{err}");

        let err =
            parse_str("[bad]\nwalls = [{ x1 = 0.1, y1 = 0.2, x2 = 0.3, y2 = 0.2, silent = 1 }]\n")
                .unwrap_err();
        assert!(err.contains("must be the boolean true"), "{err}");

        let err = parse_str(
            "[bad]\nwalls = [{ x1 = 0.1, y1 = 0.2, x2 = 0.3, y2 = 0.2, silent = true, note = 3 }]\n",
        )
        .unwrap_err();
        assert!(err.contains("both silent and note-pinned"), "{err}");
    }

    #[test]
    fn wall_note_out_of_range_is_rejected_loudly() {
        let err =
            parse_str("[bad]\nwalls = [{ x1 = 0.1, y1 = 0.2, x2 = 0.3, y2 = 0.2, note = 11 }]\n")
                .unwrap_err();
        assert!(err.contains("scale degree 0-10"), "{err}");
        let err =
            parse_str("[bad]\nwalls = [{ x1 = 0.1, y1 = 0.2, x2 = 0.3, y2 = 0.2, note = -1 }]\n")
                .unwrap_err();
        assert!(err.contains("scale degree 0-10"), "{err}");
        let err =
            parse_str("[bad]\nwalls = [{ x1 = 0.1, y1 = 0.2, x2 = 0.3, note = 2 }]\n").unwrap_err();
        assert!(err.contains("missing 'y2'"), "{err}");
    }

    #[test]
    fn false_booleans_are_rejected() {
        let err = parse_str("[a]\ntrails = false\n").unwrap_err();
        assert!(err.contains("only enable boolean options"), "{err}");
    }

    #[test]
    fn shadowing_a_builtin_is_rejected() {
        let err = parse_str("[fireworks]\ngravity = 0\n").unwrap_err();
        assert!(err.contains("shadows a built-in"), "{err}");
        // Aliases of built-ins are protected too.
        let err = parse_str("[snow]\ngravity = 0\n").unwrap_err();
        assert!(err.contains("shadows a built-in"), "{err}");
    }

    #[test]
    fn reserved_and_recursive_keys_are_rejected() {
        for key in ["preset", "presets-file", "list-presets"] {
            let err = parse_str(&format!("[a]\n\"{key}\" = \"x\"\n")).unwrap_err();
            assert!(err.contains("cannot be set from a preset"), "{key}: {err}");
        }
    }

    #[test]
    fn unknown_base_and_bad_shapes_are_rejected() {
        let err = parse_str("[a]\nbase = \"nonsense\"\n").unwrap_err();
        assert!(err.contains("unknown base"), "{err}");

        let err = parse_str("top-level = 5\n").unwrap_err();
        assert!(err.contains("must be a preset table"), "{err}");

        let err = parse_str("[a]\ngravity = [1, 2]\n").unwrap_err();
        assert!(err.contains("unsupported value type"), "{err}");

        assert!(
            parse_str("not valid toml [")
                .unwrap_err()
                .contains("invalid TOML")
        );
    }

    #[test]
    fn default_paths_check_dot_config_first() {
        let paths = default_paths();
        assert!(!paths.is_empty());
        assert!(
            paths.iter().all(|p| p.ends_with("bouncy/presets.toml")),
            "{paths:?}"
        );
        // The XDG-style location is always a candidate ($XDG_CONFIG_HOME
        // when set, otherwise ~/.config), and it is checked first.
        let first = paths[0].to_string_lossy().into_owned();
        assert!(
            std::env::var_os("XDG_CONFIG_HOME").is_some_and(|x| !x.is_empty())
                || first.contains(".config"),
            "first candidate must be XDG-style: {first}"
        );
    }

    #[test]
    fn explicit_missing_file_is_an_error_for_load() {
        let err = load(Some(Path::new("/nonexistent/bouncy-presets.toml"))).unwrap_err();
        assert!(err.contains("cannot read presets file"), "{err}");
    }
}
