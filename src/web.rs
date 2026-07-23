// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! The WebAssembly shell: a thin JS boundary around the same `App` the
//! native binary runs. The control panel talks to the simulation through
//! a shared mailbox — commands queue in, a state snapshot publishes out,
//! once per frame — so the panel is just another control surface entering
//! at `App::apply`, inheriting clamping and semantics. Nothing in the
//! core knows the panel exists.

use crate::app::{App, Command};
use crate::config::{Config, query_to_args};
use crate::sim::Polarity;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::web::EventLoopExtWebSys;

pub use crate::app::PanelCommand as WebCommand;

/// The HUD as data: everything the panel's readouts show, refreshed once
/// per rendered frame.
#[derive(Clone, Default, serde::Serialize)]
pub struct Snapshot {
    pub fps: f64,
    pub particles: usize,
    pub max_particles: usize,
    pub birth_rate: usize,
    pub explosion_threshold: usize,
    pub gravity: i32,
    pub particle_elasticity: f64,
    pub wall_elasticity: f64,
    pub time_scale: f64,
    pub paused: bool,
    /// All motion has ceased (the STOPPED banner state).
    pub stopped: bool,
    /// An explosion ring is currently expanding.
    pub exploding: bool,
    pub matter: bool,
    pub flow: bool,
    pub self_gravity: bool,
    pub trails: bool,
    pub kaleidoscope: bool,
    pub wall_chimes: bool,
    pub wells: usize,
    pub walls: usize,
    pub emitters: usize,
    pub width: u32,
    pub height: u32,
    pub muted: bool,
    pub music: bool,
    pub ping_volume: i32,
    /// Quantize tempo (0 = off) and beat-grid ticks per beat.
    pub bpm: f64,
    pub beat_div: u32,
    /// Whether the `WebAudio` engine has been created (needs a user
    /// gesture; see [`WebHandle::enable_audio`]).
    pub audio_ready: bool,
    /// CLI value names (the same strings the parser accepts), so the
    /// share link can carry them.
    pub spawn_mode: String,
    pub color_mode: String,
    /// Current HUD overlay level (hidden / stats / stats+keys), so the
    /// panel's cycle button can show where in the cycle it is.
    pub hud: String,
    /// Inspector: the selected entity, flattened to per-field optionals
    /// (all None when nothing is selected). Flat so a stale cached page
    /// sees absent fields and simply never shows the inspector.
    pub selection_kind: Option<String>,
    pub selection_id: Option<u32>,
    pub selection_rate: Option<f64>,
    pub selection_cap: Option<usize>,
    /// Aim as compass degrees (0 = up), the scene-export convention.
    pub selection_angle: Option<f64>,
    /// Chime-note label: "auto", "degree N", or "silent".
    pub selection_note: Option<String>,
    pub selection_segments: Option<usize>,
    /// Gate label: "off" or "every N" (pre-formatted, like the note).
    pub selection_gate: Option<String>,
    /// Pass-note label: "off" or "degree D".
    pub selection_pass: Option<String>,
    /// Emitter stamped-note label: "none" or "degree D".
    pub selection_emitter_note: Option<String>,
    /// Stroke MIDI-key label: "auto" or "60 (C4)".
    pub selection_midi_key: Option<String>,
    /// Stroke MIDI channel, 1-based like a DAW.
    pub selection_midi_channel: Option<u32>,
    /// `WebMIDI`: a browser output port is connected and ready.
    pub midi_ready: bool,
    /// `WebMIDI`: the last enable attempt failed (no API, no ports, or
    /// permission denied).
    pub midi_failed: bool,
    /// Note sending is gated on (Y / the panel toggle).
    pub midi_enabled: bool,
    /// `WebMIDI`: every output port's name, in the browser's enumeration
    /// order (the dropdown's option list; empty unless ready).
    pub midi_ports: Vec<String>,
    /// `WebMIDI`: the connected port's name, if ready.
    pub midi_port: Option<String>,
}

/// The mailbox shared between the running [`App`] and the [`WebHandle`]
/// the page holds. Single-threaded (wasm), so `Rc<RefCell>` suffices.
#[derive(Default)]
pub struct Shared {
    pub commands: Vec<WebCommand>,
    pub snapshot: Snapshot,
    /// The current scene as preset TOML, refreshed with the snapshot;
    /// the panel turns it into a download.
    pub scene_toml: Option<String>,
}

/// The page's handle to a running simulation. Constructing it parses the
/// URL query string through the CLI parser, attaches to the `#bouncy`
/// canvas, and spawns the winit event loop (which never returns; the
/// browser drives frames via requestAnimationFrame).
#[wasm_bindgen]
pub struct WebHandle {
    shared: Rc<RefCell<Shared>>,
}

#[wasm_bindgen]
impl WebHandle {
    /// Start the simulation. `query` is the page's `location.search`
    /// (with or without the leading `?`), mapped onto CLI options.
    #[wasm_bindgen(constructor)]
    pub fn new(query: &str) -> Result<WebHandle, JsValue> {
        console_error_panic_hook::set_once();

        let args = query_to_args(query);
        let config =
            Config::try_resolve_with(&args, None).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let shared = Rc::new(RefCell::new(Shared::default()));
        let app = App::new_web(config, Rc::clone(&shared));

        let event_loop =
            EventLoop::new().map_err(|e| JsValue::from_str(&format!("event loop: {e}")))?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.spawn_app(app);

        Ok(WebHandle { shared })
    }

    /// The latest per-frame state snapshot, as a plain JS object.
    pub fn state(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.shared.borrow().snapshot).unwrap_or(JsValue::NULL)
    }

    /// The current scene as preset-file TOML (for download/sharing).
    pub fn scene_toml(&self) -> Option<String> {
        self.shared.borrow().scene_toml.clone()
    }

    fn push(&self, cmd: WebCommand) {
        self.shared.borrow_mut().commands.push(cmd);
    }

    /// Absolute pause state (the P key toggles; the panel sends state).
    pub fn set_paused(&self, paused: bool) {
        self.push(WebCommand::SetPaused(paused));
    }

    /// Advance one frame (only meaningful while paused; the N key).
    pub fn step_frame(&self) {
        self.push(WebCommand::Plain(Command::StepFrame));
    }

    /// Restart the simulation with its launch configuration (the R key).
    pub fn reset(&self) {
        self.push(WebCommand::Plain(Command::Reset));
    }

    /// Gravity as a percentage of standard, clamped like --gravity.
    pub fn set_gravity(&self, percent: i32) {
        self.push(WebCommand::SetGravity(percent));
    }

    /// Particle restitution, clamped like --particle-elasticity.
    pub fn set_particle_elasticity(&self, e: f64) {
        self.push(WebCommand::SetParticleElasticity(e));
    }

    /// Wall restitution, clamped like --wall-elasticity.
    pub fn set_wall_elasticity(&self, e: f64) {
        self.push(WebCommand::SetWallElasticity(e));
    }

    /// Simulation speed multiplier, clamped to the hotkeys' 0.1x-4x range.
    pub fn set_time_scale(&self, scale: f64) {
        self.push(WebCommand::SetTimeScale(scale));
    }

    /// Births-per-second explosion trigger (0 disables), clamped like
    /// --explosion-threshold.
    pub fn set_explosion_threshold(&self, threshold: i32) {
        self.push(WebCommand::SetExplosionThreshold(threshold));
    }

    /// Toggle matter mechanics (fusion/fission; the X key).
    pub fn toggle_matter(&self) {
        self.push(WebCommand::Plain(Command::ToggleMatter));
    }

    /// Toggle the ambient flow field (the F key).
    pub fn toggle_flow(&self) {
        self.push(WebCommand::Plain(Command::ToggleFlow));
    }

    /// Toggle self-gravity — mass attracts mass (the A key).
    pub fn toggle_self_gravity(&self) {
        self.push(WebCommand::Plain(Command::ToggleSelfGravity));
    }

    /// Toggle motion trails (the T key).
    pub fn toggle_trails(&self) {
        self.push(WebCommand::Plain(Command::ToggleTrails));
    }

    /// Toggle the 4-fold kaleidoscope mirror (the K key).
    pub fn toggle_kaleidoscope(&self) {
        self.push(WebCommand::Plain(Command::ToggleKaleidoscope));
    }

    /// Toggle wall chimes — walls play notes on impact (the I key).
    pub fn toggle_wall_chimes(&self) {
        self.push(WebCommand::Plain(Command::ToggleWallChimes));
    }

    /// Set the particle-ping volume (0-100 percent).
    pub fn set_ping_volume(&self, percent: i32) {
        self.push(WebCommand::SetPingVolume(percent));
    }

    /// Quantize tempo in BPM (0 = off; the app snaps 1-29 up to 30 and
    /// clamps at 300, like --bpm).
    pub fn set_bpm(&self, bpm: f64) {
        self.push(WebCommand::SetBpm(bpm));
    }

    /// Beat-grid resolution in ticks per beat; only 1, 2, 4, or 8 take
    /// effect, like --beat-div.
    pub fn set_beat_div(&self, div: u32) {
        self.push(WebCommand::SetBeatDiv(div));
    }

    /// Place an emitter at `(x, y)`, aimed at the arena center.
    pub fn place_emitter(&self, x: f64, y: f64) {
        self.push(WebCommand::PlaceEmitter(x, y));
    }

    /// Remove every pinned emitter (Shift+U).
    pub fn clear_emitters(&self) {
        self.push(WebCommand::Plain(Command::ClearEmitters));
    }

    /// Cycle solid/velocity coloring (the C key).
    pub fn cycle_color_mode(&self) {
        self.push(WebCommand::Plain(Command::CycleColorMode));
    }

    /// Cycle spawn mode: center / collision points / off (the B key).
    pub fn cycle_spawn_mode(&self) {
        self.push(WebCommand::Plain(Command::CycleSpawnMode));
    }

    /// Cycle the canvas HUD: off / stats / stats+keys (the H key).
    pub fn cycle_hud(&self) {
        self.push(WebCommand::Plain(Command::CycleHud));
    }

    /// Remove every pinned gravity well (Shift+R).
    pub fn clear_wells(&self) {
        self.push(WebCommand::Plain(Command::ClearWells));
    }

    /// Remove every drawn wall segment (Shift+V).
    pub fn clear_walls(&self) {
        self.push(WebCommand::Plain(Command::ClearWalls));
    }

    /// Spawn a burst of particles at simulation coordinates (left click).
    pub fn spawn_burst(&self, x: f64, y: f64) {
        self.push(WebCommand::SpawnBurst(x, y));
    }

    /// Launch a comet from the far edge toward the point (middle click).
    pub fn launch_comet(&self, x: f64, y: f64) {
        self.push(WebCommand::LaunchComet(x, y));
    }

    /// Pin a persistent gravity well at the point; `repel` inverts its
    /// polarity (the W / Shift+W keys).
    pub fn pin_well(&self, x: f64, y: f64, repel: bool) {
        let polarity = if repel {
            Polarity::Repel
        } else {
            Polarity::Attract
        };
        self.push(WebCommand::PinWell(x, y, polarity));
    }

    /// Trigger an explosion ring centered on the point (right click).
    pub fn trigger_explosion(&self, x: f64, y: f64) {
        self.push(WebCommand::TriggerExplosion(x, y));
    }

    /// Resize the simulation to a new logical size (the page calls this
    /// from a debounced `ResizeObserver`, so the arena tracks the canvas).
    pub fn resize(&self, width: u32, height: u32) {
        self.push(WebCommand::Resize(width, height));
    }

    /// Create (or resume) the `WebAudio` engine and unmute. Browsers only
    /// allow audio to start inside a user gesture, so the page must call
    /// this synchronously from a click handler — the wasm call inherits
    /// the gesture's user activation. Returns whether audio is ready.
    pub fn enable_audio(&self) -> bool {
        let ready = crate::audio::web_enable();
        if ready {
            self.push(WebCommand::SetMuted(false));
        }
        ready
    }

    /// Whether the `WebAudio` engine exists (direct read, same value the
    /// snapshot's `audio_ready` reports).
    pub fn audio_ready(&self) -> bool {
        crate::audio::web_ready()
    }

    /// Absolute mute state (see [`WebHandle::enable_audio`] for the
    /// user-gesture path that unmutes for the first time).
    pub fn set_muted(&self, muted: bool) {
        self.push(WebCommand::SetMuted(muted));
    }

    /// Absolute musical-pings state (pentatonic collision pitches).
    pub fn set_music(&self, music: bool) {
        self.push(WebCommand::SetMusic(music));
    }

    /// Resolve a click to a selection; a miss deselects (the panel's
    /// Select tool). The result arrives via the snapshot's
    /// `selection_*` fields.
    pub fn select_at(&self, x: f64, y: f64) {
        self.push(WebCommand::SelectAt(x, y));
    }

    /// Drop the selection.
    pub fn deselect(&self) {
        self.push(WebCommand::Deselect);
    }

    /// Set the selected emitter's emission rate (particles/second,
    /// clamped like scene TOML).
    pub fn set_emitter_rate(&self, id: u32, rate: f64) {
        self.push(WebCommand::SetEmitterRate(id, rate));
    }

    /// Set the selected emitter's live-particle cap (clamped like scene
    /// TOML).
    pub fn set_emitter_cap(&self, id: u32, cap: i32) {
        self.push(WebCommand::SetEmitterCap(id, cap));
    }

    /// Point the emitter from its position toward `(x, y)` (the panel's
    /// armed Re-aim tool).
    pub fn aim_emitter_at(&self, id: u32, x: f64, y: f64) {
        self.push(WebCommand::AimEmitterAt(id, x, y));
    }

    /// Step the stroke's chime note: Auto → degrees → Silent → Auto.
    pub fn cycle_stroke_note(&self, id: u32) {
        self.push(WebCommand::CycleStrokeNote(id));
    }

    /// Step the stroke's gate: off → every 2/3/4/8 → off (replacing any
    /// pass filter).
    pub fn cycle_stroke_gate(&self, id: u32) {
        self.push(WebCommand::CycleStrokeGate(id));
    }

    /// Step the stroke's pass-note: off → degree 0..10 → off (replacing
    /// any gate).
    pub fn cycle_stroke_pass(&self, id: u32) {
        self.push(WebCommand::CycleStrokePass(id));
    }

    /// Step the emitter's stamped note: none → degree 0..10 → none.
    pub fn cycle_emitter_note(&self, id: u32) {
        self.push(WebCommand::CycleEmitterNote(id));
    }

    /// Pin the stroke's MIDI key (negative = the pentatonic auto
    /// mapping; 0..=127 pins, clamped like the scene TOML).
    pub fn set_stroke_midi_key(&self, id: u32, key: i32) {
        let key = u8::try_from(key).ok().map(|k| k.min(127));
        self.push(WebCommand::SetStrokeMidiKey(id, key));
    }

    /// Set the stroke's MIDI channel (1-based, like a DAW and the
    /// scene TOML; clamped to 1..=16).
    pub fn set_stroke_midi_channel(&self, id: u32, channel: u32) {
        #[allow(clippy::cast_possible_truncation)]
        let ch = (channel.clamp(1, 16) - 1) as u8;
        self.push(WebCommand::SetStrokeMidiChannel(id, ch));
    }

    /// Draw one wall segment from the panel's Draw-wall drag tool: a
    /// fresh stroke for the drag's first segment (`extend` false),
    /// chained onto it for the rest — the same polyline semantics as
    /// the held-V gesture.
    pub fn draw_wall(&self, x1: f64, y1: f64, x2: f64, y2: f64, extend: bool) {
        self.push(WebCommand::DrawWall {
            x1,
            y1,
            x2,
            y2,
            extend,
        });
    }

    /// Kick off the `WebMIDI` permission request (must come from a user
    /// gesture, like `enable_audio`). Readiness is asynchronous: watch
    /// `midi_ready` / `midi_failed` in the polled snapshot.
    pub fn enable_midi(&self) {
        self.push(WebCommand::EnableMidi);
    }

    /// Toggle note sending while the browser port is connected (the
    /// panel twin of the Y key).
    pub fn toggle_midi(&self) {
        self.push(WebCommand::Plain(Command::ToggleMidi));
    }

    /// Switch the live `WebMIDI` connection to the port at `index` in
    /// the snapshot's `midi_ports` (the panel dropdown). Live: the old
    /// port is silenced, the scene never resets.
    pub fn set_midi_port(&self, index: u32) {
        self.push(WebCommand::SetMidiPort(index));
    }

    /// Delete one emitter by id (its particles keep flying).
    pub fn delete_emitter(&self, id: u32) {
        self.push(WebCommand::DeleteEmitter(id));
    }

    /// Delete one wall stroke (all its segments) by id.
    pub fn delete_stroke(&self, id: u32) {
        self.push(WebCommand::DeleteStroke(id));
    }
}

/// The crate version, for the demo page's header stamp — read from
/// Cargo.toml at compile time, so the page can never disagree with the
/// wasm it is running.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Names of the built-in presets, for the panel's launch-options
/// dropdown — sourced from the same enum the CLI parses, so the list
/// cannot drift.
#[wasm_bindgen]
pub fn preset_names() -> Vec<String> {
    use clap::ValueEnum;
    crate::presets::Preset::value_variants()
        .iter()
        .map(|p| p.label().to_string())
        .collect()
}

/// Initialize the rayon thread pool over wasm threads. Exposed only when
/// built with the `web-threads` feature on a cross-origin-isolated page;
/// the loader awaits this before constructing a [`WebHandle`].
#[cfg(feature = "web-threads")]
pub use wasm_bindgen_rayon::init_thread_pool;
