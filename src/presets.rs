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
        }
    }
}

/// One user preset: an optional built-in to inherit from, an optional
/// description for --list-presets, and the remaining options in
/// command-line form (`["--gravity", "80", ...]`).
#[derive(Debug)]
pub struct UserPreset {
    pub base: Option<Preset>,
    pub description: Option<String>,
    pub args: Vec<String>,
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
                ))
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
        for (key, value) in entries {
            let key = key.replace('_', "-");
            match key.as_str() {
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
            },
        );
    }

    Ok(UserPresets {
        path: path.to_path_buf(),
        presets,
    })
}

/// Print built-in and user presets (--list-presets), including where the
/// user presets file was (or would be) loaded from.
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

        assert!(parse_str("not valid toml [")
            .unwrap_err()
            .contains("invalid TOML"));
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
