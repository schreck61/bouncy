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

/// A control-panel action. Plain variants wrap the same [`Command`] enum
/// the keyboard uses; the rest carry data the keyboard path would take
/// from held-key or cursor state (absolute values from sliders,
/// coordinates from panel tools).
pub enum WebCommand {
    Plain(Command),
    SetPaused(bool),
    SetGravity(i32),
    SetParticleElasticity(f64),
    SetWallElasticity(f64),
    SetTimeScale(f64),
    SetExplosionThreshold(i32),
    SpawnBurst(f64, f64),
    LaunchComet(f64, f64),
    PinWell(f64, f64, Polarity),
    TriggerExplosion(f64, f64),
    /// Live-resize the arena to a new logical size (`Simulation::resize`
    /// plus a frame-buffer reallocation).
    Resize(u32, u32),
    /// Absolute mute state (the panel sends state, not toggles).
    SetMuted(bool),
    /// Absolute musical-mode state.
    SetMusic(bool),
}

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
    pub matter: bool,
    pub flow: bool,
    pub self_gravity: bool,
    pub trails: bool,
    pub kaleidoscope: bool,
    pub wells: usize,
    pub walls: usize,
    pub width: u32,
    pub height: u32,
    pub muted: bool,
    pub music: bool,
    /// Whether the `WebAudio` engine has been created (needs a user
    /// gesture; see [`WebHandle::enable_audio`]).
    pub audio_ready: bool,
    /// CLI value names (the same strings the parser accepts), so the
    /// share link can carry them.
    pub spawn_mode: String,
    pub color_mode: String,
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

    pub fn set_paused(&self, paused: bool) {
        self.push(WebCommand::SetPaused(paused));
    }

    pub fn step_frame(&self) {
        self.push(WebCommand::Plain(Command::StepFrame));
    }

    pub fn reset(&self) {
        self.push(WebCommand::Plain(Command::Reset));
    }

    pub fn set_gravity(&self, percent: i32) {
        self.push(WebCommand::SetGravity(percent));
    }

    pub fn set_particle_elasticity(&self, e: f64) {
        self.push(WebCommand::SetParticleElasticity(e));
    }

    pub fn set_wall_elasticity(&self, e: f64) {
        self.push(WebCommand::SetWallElasticity(e));
    }

    pub fn set_time_scale(&self, scale: f64) {
        self.push(WebCommand::SetTimeScale(scale));
    }

    pub fn set_explosion_threshold(&self, threshold: i32) {
        self.push(WebCommand::SetExplosionThreshold(threshold));
    }

    pub fn toggle_matter(&self) {
        self.push(WebCommand::Plain(Command::ToggleMatter));
    }

    pub fn toggle_flow(&self) {
        self.push(WebCommand::Plain(Command::ToggleFlow));
    }

    pub fn toggle_self_gravity(&self) {
        self.push(WebCommand::Plain(Command::ToggleSelfGravity));
    }

    pub fn toggle_trails(&self) {
        self.push(WebCommand::Plain(Command::ToggleTrails));
    }

    pub fn toggle_kaleidoscope(&self) {
        self.push(WebCommand::Plain(Command::ToggleKaleidoscope));
    }

    pub fn cycle_color_mode(&self) {
        self.push(WebCommand::Plain(Command::CycleColorMode));
    }

    pub fn cycle_spawn_mode(&self) {
        self.push(WebCommand::Plain(Command::CycleSpawnMode));
    }

    pub fn cycle_hud(&self) {
        self.push(WebCommand::Plain(Command::CycleHud));
    }

    pub fn clear_wells(&self) {
        self.push(WebCommand::Plain(Command::ClearWells));
    }

    pub fn clear_walls(&self) {
        self.push(WebCommand::Plain(Command::ClearWalls));
    }

    pub fn spawn_burst(&self, x: f64, y: f64) {
        self.push(WebCommand::SpawnBurst(x, y));
    }

    pub fn launch_comet(&self, x: f64, y: f64) {
        self.push(WebCommand::LaunchComet(x, y));
    }

    pub fn pin_well(&self, x: f64, y: f64, repel: bool) {
        let polarity = if repel {
            Polarity::Repel
        } else {
            Polarity::Attract
        };
        self.push(WebCommand::PinWell(x, y, polarity));
    }

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

    pub fn set_muted(&self, muted: bool) {
        self.push(WebCommand::SetMuted(muted));
    }

    pub fn set_music(&self, music: bool) {
        self.push(WebCommand::SetMusic(music));
    }
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
