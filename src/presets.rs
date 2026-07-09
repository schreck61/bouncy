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
//! [pachinko]
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
        }
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

/// A wall segment in window-fraction coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SceneWall {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
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

/// Parse `walls = [[x1, y1, x2, y2], ...]` (window fractions).
fn parse_scene_walls(preset: &str, value: &toml::Value) -> Result<Vec<SceneWall>, String> {
    let toml::Value::Array(entries) = value else {
        return Err(format!(
            "preset '{preset}': walls must be an array of [x1, y1, x2, y2] arrays"
        ));
    };
    let mut walls = Vec::with_capacity(entries.len());
    for entry in entries {
        let toml::Value::Array(coords) = entry else {
            return Err(format!(
                "preset '{preset}': each wall must be an [x1, y1, x2, y2] array"
            ));
        };
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
        });
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
            .map(|w| toml::Value::Array(vec![w.x1.into(), w.y1.into(), w.x2.into(), w.y2.into()]))
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
            "[pachinko]\n\
             base = \"billiards\"\n\
             gravity = 80\n\
             particle-size = 4.5\n\
             spawn-mode = \"off\"\n\
             trails = true\n",
        )
        .unwrap();
        let preset = &user.presets["pachinko"];
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
                y2: 0.5
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
        let walls = [SceneWall {
            x1: 0.1,
            y1: 0.2,
            x2: 0.3,
            y2: 0.4,
        }];
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
