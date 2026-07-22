// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Application shell: windowing, input handling, HUD, audio dispatch, and
//! the winit event loop glue around the headless [`Simulation`] core.

use crate::audio::Audio;
use crate::config::{ColorMode, Config, ELASTICITY_MAX, EXPLOSION_THRESHOLD_MAX, GRAVITY_LIMIT};
#[cfg(not(target_arch = "wasm32"))]
use crate::render::write_png;
use crate::render::{
    RenderContext, create_render_context, dim_rect, fade_frame, kaleidoscope_frame,
    render_emitter_highlight, render_emitters, render_explosion, render_particles, render_segments,
    render_stroke_highlight, render_wells,
};
use crate::sim::{
    MAX_EMITTERS, MAX_PINNED_WELLS, MAX_WALL_SEGMENTS, Polarity, SELECT_RADIUS, Selection,
    Simulation, Well,
};
use crate::text::{draw_text, draw_text_centered, measure_text};
use clap::ValueEnum;
use std::collections::HashMap;
use std::rc::Rc;
// std::time::Instant on native; performance.now() on wasm.
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
#[cfg(not(target_arch = "wasm32"))]
use winit::{dpi::LogicalSize, window::Fullscreen};

/// Frames of physics skipped at startup to let the GPU initialize.
const WARMUP_FRAMES: u32 = 3;
/// Runtime gravity adjustment step (percent) for Up/Down arrows.
const GRAVITY_STEP: i32 = 10;
/// Defaults for interactively placed emitters (scene files set their own).
const EMITTER_DEFAULT_RATE: f64 = 2.0;
const EMITTER_DEFAULT_CAP: usize = 12;
/// Minimum U-drag distance that counts as an aim; a bare tap aims at the
/// screen center instead.
const EMITTER_MIN_AIM_DRAG: f64 = 12.0;
/// Percent step for the ; and ' ping-volume hotkeys.
const PING_VOLUME_STEP: i32 = 10;
/// Runtime elasticity adjustment step for arrow and bracket keys.
const ELASTICITY_STEP: f64 = 0.05;
/// Runtime explosion-threshold adjustment step for the -/= keys.
const THRESHOLD_STEP: i32 = 5;
/// Additive step for time-scale adjustment (comma/period). Linear steps
/// retrace exactly — speeding up and slowing down again always returns to
/// the starting value — because the floor, 1.0, and the ceiling all sit
/// on the same 0.05 grid. (The old multiplicative step could never reach
/// 1.0 again after clamping at either end.)
const TIME_SCALE_STEP: f64 = 0.05;
const TIME_SCALE_MIN: f64 = 0.1;
const TIME_SCALE_MAX: f64 = 4.0;
/// Simulated time for a single frame-step while paused (N key).
const FRAME_STEP_DT: f64 = 1.0 / 120.0;
/// Bullet time: fraction of the chosen time scale while an explosion ring
/// gets under way (opt in with --bullet-time).
const BULLET_TIME_SCALE: f64 = 0.1;
/// Wall-clock seconds the bullet-time dip holds at full slowdown.
const BULLET_TIME_HOLD_SECS: f64 = 1.0;
/// Wall-clock seconds the ramp back to the prior time scale takes.
const BULLET_TIME_RAMP_SECS: f64 = 0.4;
/// Consecutive failed presents before giving up (a few seconds at 60 FPS).
const MAX_RENDER_FAILURES: u32 = 300;
/// Seconds of cursor inactivity before the cursor is hidden.
const CURSOR_HIDE_DELAY: f64 = 2.0;
/// Minimum cursor travel (pixels) before the wall being drawn (held V)
/// gains another segment; short strokes stay smooth without flooding the
/// segment budget.
const WALL_SEGMENT_MIN_LENGTH: f64 = 12.0;

const HUD_FONT_SIZE: f32 = 16.0;
const HUD_LINE_HEIGHT: f32 = 20.0;
const HUD_MARGIN: f32 = 8.0;
const HUD_COLOR: [u8; 3] = [200, 200, 200];
const MESSAGE_COLOR: [u8; 3] = [255, 100, 100];

/// HUD overlay state, cycled with the H key.
#[derive(Copy, Clone, PartialEq, Eq)]
enum HudMode {
    Hidden,
    Stats,
    StatsAndKeys,
}

impl HudMode {
    fn next(self) -> Self {
        match self {
            HudMode::Hidden => HudMode::Stats,
            HudMode::Stats => HudMode::StatsAndKeys,
            HudMode::StatsAndKeys => HudMode::Hidden,
        }
    }

    /// Short name for the panels' HUD cycle button (web and native).
    fn label(self) -> &'static str {
        match self {
            HudMode::Hidden => "hidden",
            HudMode::Stats => "stats",
            HudMode::StatsAndKeys => "stats+keys",
        }
    }
}

/// A runtime control action, decoupled from its input source. The
/// keyboard dispatches these today; any future control surface (a GUI
/// panel, a script, a replay) issues the same commands and gets
/// identical behavior, logging included.
#[derive(Copy, Clone, Debug)]
pub enum Command {
    TogglePause,
    /// Advance one frame (only meaningful while paused).
    StepFrame,
    Reset,
    ClearWells,
    ClearWalls,
    ClearEmitters,
    ToggleMute,
    ToggleMusic,
    CycleHud,
    ToggleTrails,
    CycleColorMode,
    CycleSpawnMode,
    ToggleMatter,
    ToggleFlow,
    ToggleSelfGravity,
    ToggleKaleidoscope,
    ToggleWallChimes,
    /// Step gravity by a signed percentage amount.
    AdjustGravity(i32),
    /// Step the particle-ping volume by a signed percentage amount.
    AdjustPingVolume(i32),
    AdjustParticleElasticity(f64),
    AdjustWallElasticity(f64),
    AdjustTimeScale(f64),
    AdjustExplosionThreshold(i32),
    /// Pin a persistent gravity well at the cursor.
    PinWell(Polarity),
    /// Save the next presented frame as a PNG in the working directory.
    Screenshot,
    /// Export current settings plus wells/walls as a preset file.
    ExportScene,
    /// Launch a comet from the far edge toward the cursor (J or middle
    /// click).
    LaunchComet,
}

/// A control-panel action, shared by every panel surface (the native
/// edge panel and the web demo's HTML panel). Plain variants wrap the
/// same [`Command`] enum the keyboard uses; the rest carry data the
/// keyboard path would take from held-key or cursor state (absolute
/// values from sliders, coordinates from panel tools). Commands enter at
/// `apply_panel_command`, inheriting the same clamping and semantics as
/// every other control surface.
pub enum PanelCommand {
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
    /// Absolute mute state (panels send state, not toggles).
    SetMuted(bool),
    /// Absolute musical-mode state.
    SetMusic(bool),
    /// Absolute particle-ping volume from the panel slider (percent).
    SetPingVolume(i32),
    /// Place an emitter at the click, aimed at the arena center.
    PlaceEmitter(f64, f64),
    /// Resolve the click to a selection (a miss deselects — in select
    /// mode a click never bursts).
    SelectAt(f64, f64),
    /// Drop the selection.
    Deselect,
    /// Set the selected emitter's emission rate (particles/second).
    SetEmitterRate(u32, f64),
    /// Set the selected emitter's live-particle cap.
    SetEmitterCap(u32, i32),
    /// Re-aim the emitter from its position toward the clicked point.
    AimEmitterAt(u32, f64, f64),
    /// Step the stroke's chime note: Auto → degrees → Silent → Auto.
    CycleStrokeNote(u32),
    /// Delete one emitter by id.
    DeleteEmitter(u32),
    /// Delete one wall stroke (all its segments) by id.
    DeleteStroke(u32),
    /// Rebuild the simulation with new construction-time options (the
    /// native panel's launch section; the web demo restarts via URL).
    Relaunch {
        preset: Option<String>,
        /// None = untouched; the preset or default decides.
        particle_size: Option<f64>,
        initial_speed: Option<f64>,
        min_particles: Option<u32>,
    },
}

/// Live settings that differed from the launched config when a panel
/// relaunch was requested. None = the user never touched it, so the new
/// launch's bundle decides; Some = the session's deliberate adjustment,
/// re-asserted over the fresh simulation (the web share-link
/// philosophy).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
#[allow(clippy::struct_excessive_bools)]
struct SessionDeltas {
    gravity: Option<i32>,
    particle_elasticity: Option<f64>,
    wall_elasticity: Option<f64>,
    explosion_threshold: Option<usize>,
    spawn_mode: Option<crate::config::SpawnMode>,
    color_mode: Option<ColorMode>,
    matter: Option<bool>,
    flow: Option<bool>,
    self_gravity: Option<bool>,
    wall_chimes: Option<bool>,
    trails: Option<bool>,
    kaleidoscope: Option<bool>,
    music: Option<bool>,
    muted: Option<bool>,
    ping_volume: Option<i32>,
    time_scale: Option<f64>,
}

/// Canonical CLI value name of a `ValueEnum` variant — the same string
/// the parser accepts, shared by scene export and the web snapshot.
fn value_name(pv: Option<clap::builder::PossibleValue>) -> String {
    pv.expect("no skipped variants").get_name().to_string()
}

/// Convert physical pixels to logical pixels given a scale factor.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[cfg(not(target_arch = "wasm32"))]
fn physical_to_logical(physical: u32, scale_factor: f64) -> u32 {
    (f64::from(physical) / scale_factor) as u32
}

/// Simulation size for a physical window size: the physical size divided
/// by the display scale factor *rounded to an integer*. The GPU renderer
/// (pixels) presents the frame at the largest integer scale that fits, so
/// sizing the buffer this way makes that scale fill the window exactly —
/// at fractional scale factors (Windows 125%/150%) the old
/// physical/scale sizing left the frame letterboxed inside invisible
/// margins. The trade: at 150% the simulation's pixels are 2x instead of
/// 1.5x, slightly chunkier than the OS intends; owning the whole screen
/// wins for a fullscreen toy.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[cfg(not(target_arch = "wasm32"))]
fn simulation_size(physical: u32, scale_factor: f64) -> u32 {
    let k = (scale_factor.round() as u32).max(1);
    (physical / k).max(1)
}

/// Frame-rate accounting for the HUD and the verbose per-second print.
struct FpsCounter {
    frame_count: u64,
    timer: Instant,
    current: f64,
}

impl FpsCounter {
    fn new() -> Self {
        FpsCounter {
            frame_count: 0,
            timer: Instant::now(),
            current: 0.0,
        }
    }

    /// Reset the measurement window (warmup frames must not count).
    fn restart(&mut self, now: Instant) {
        self.frame_count = 0;
        self.timer = now;
    }

    /// Count one frame; returns the refreshed rate once per second.
    fn tick(&mut self) -> Option<f64> {
        self.frame_count += 1;
        let elapsed = self.timer.elapsed().as_secs_f64();
        if elapsed < 1.0 {
            return None;
        }
        // Precision loss acceptable: frame_count is small relative to f64 mantissa
        #[allow(clippy::cast_precision_loss)]
        let fps = self.frame_count as f64 / elapsed;
        self.current = fps;
        self.frame_count = 0;
        self.timer = Instant::now();
        Some(fps)
    }
}

/// Cursor state: position in simulation coordinates plus the inputs to
/// the auto-hide policy.
struct CursorState {
    x: f64,
    y: f64,
    /// What visibility we last set on the window.
    visible: bool,
    last_move: Instant,
    inside: bool,
    window_focused: bool,
}

impl CursorState {
    fn new() -> Self {
        CursorState {
            x: 0.0,
            y: 0.0,
            visible: true,
            last_move: Instant::now(),
            inside: true,
            window_focused: true,
        }
    }

    fn moved(&mut self) {
        self.inside = true;
        self.last_move = Instant::now();
    }
}

/// Bullet-time choreography: purely presentational — the simulation only
/// ever sees a smaller dt.
struct BulletTime {
    /// Enabled via --bullet-time (and the fireworks/mandala presets).
    enabled: bool,
    /// When the current slowdown started, if one is active.
    start: Option<Instant>,
}

impl BulletTime {
    /// Enter bullet time when an explosion ring starts, if enabled. A
    /// ring starting during another ring's ramp restarts the dip.
    fn trigger(&mut self, now: Instant) {
        if self.enabled {
            self.start = Some(now);
        }
    }

    fn active(&self) -> bool {
        self.start.is_some()
    }

    /// This frame's multiplier on the time scale, expiring the effect
    /// once the ramp back to full speed completes.
    fn multiplier(&mut self, now: Instant) -> f64 {
        let Some(start) = self.start else {
            return 1.0;
        };
        let elapsed = now.duration_since(start).as_secs_f64();
        if elapsed >= BULLET_TIME_HOLD_SECS + BULLET_TIME_RAMP_SECS {
            self.start = None;
            return 1.0;
        }
        bullet_time_factor(elapsed)
    }
}

/// Main application state: I/O around the simulation core.
pub struct App {
    // Configuration. `config` is retained only to construct the simulation
    // once window dimensions are known; the fields below are the
    // runtime-mutable copies of its presentation options and are the sole
    // source of truth after startup (reset never re-reads `config`).
    config: Config,
    trails: bool,
    color_mode: ColorMode,
    /// Kaleidoscope post-process enabled (K toggles).
    kaleidoscope: bool,
    verbose: bool,

    // Subsystems
    audio: Audio,
    sim: Option<Simulation>,

    // Window and rendering (initialized on resume)
    render: Option<RenderContext>,
    scale_factor: f64,

    // Timing
    last_time: Instant,
    warmup_frames: u32,
    fps: FpsCounter,

    // Interaction state
    paused: bool,
    step_once: bool,
    hud_mode: HudMode,
    /// Cursor gravity well while G is held (Shift+G repels).
    held_well: Option<Polarity>,
    /// Wall drawing (V held): where the next segment starts, if active.
    wall_anchor: Option<(f64, f64)>,
    /// Emitter placement (U held): the emitter position; releasing aims
    /// it along the drag (a bare tap aims at the screen center).
    emitter_anchor: Option<(f64, f64)>,
    /// Whether the current V-drag has landed its first segment; later
    /// segments extend the same stroke so the polyline chimes as one bar.
    wall_stroke_open: bool,
    /// Chime flash intensity per stroke id, decayed each frame;
    /// presentational state, so it lives here and not in the sim.
    wall_flash: HashMap<u32, f32>,
    /// The inspected emitter or wall stroke, if any; presentational
    /// state like `wall_flash`. A frame sweep drops it when its id dies.
    selection: Option<Selection>,
    /// D is held: the next left click selects instead of bursting.
    d_down: bool,
    shift_down: bool,
    time_scale: f64,
    bullet_time: BulletTime,
    cursor: CursorState,

    /// A screenshot was requested; the next rendered frame is saved.
    screenshot_requested: bool,

    // Render failure tracking (transient surface loss should not crash)
    consecutive_render_failures: u32,

    /// The native control panel (Tab), drawn into the frame buffer.
    #[cfg(not(target_arch = "wasm32"))]
    gui: crate::gui::Gui,

    /// Mailbox shared with the JS control panel: commands flow in, a
    /// state snapshot flows out, once per frame (see `web.rs`).
    #[cfg(target_arch = "wasm32")]
    web_shared: Option<std::rc::Rc<std::cell::RefCell<crate::web::Shared>>>,
}

impl App {
    /// Create a new App with the given configuration.
    pub fn new(config: Config) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let panel_open = config.panel;
        App {
            trails: config.trails,
            color_mode: config.color_mode,
            kaleidoscope: config.kaleidoscope,
            verbose: config.verbose,
            bullet_time: BulletTime {
                enabled: config.bullet_time,
                start: None,
            },
            audio: Audio::new(
                config.mute,
                config.music,
                config.chime_timbre,
                config.ping_volume,
            ),
            sim: None,
            config,
            render: None,
            scale_factor: 1.0,
            last_time: Instant::now(),
            warmup_frames: WARMUP_FRAMES,
            fps: FpsCounter::new(),
            paused: false,
            step_once: false,
            hud_mode: HudMode::Hidden,
            held_well: None,
            wall_anchor: None,
            emitter_anchor: None,
            wall_stroke_open: false,
            wall_flash: HashMap::new(),
            selection: None,
            d_down: false,
            shift_down: false,
            time_scale: 1.0,
            cursor: CursorState::new(),
            screenshot_requested: false,
            consecutive_render_failures: 0,
            #[cfg(not(target_arch = "wasm32"))]
            gui: {
                let mut gui = crate::gui::Gui::new();
                if panel_open {
                    gui.toggle_open();
                }
                gui
            },
            #[cfg(target_arch = "wasm32")]
            web_shared: None,
        }
    }

    /// Create an App wired to the JS control panel's mailbox (web shell).
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn new_web(
        config: Config,
        shared: std::rc::Rc<std::cell::RefCell<crate::web::Shared>>,
    ) -> Self {
        let mut app = Self::new(config);
        app.web_shared = Some(shared);
        app
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.render.as_ref().map(|r| (r.width(), r.height()))
    }

    /// The gravity well input for this frame, if held.
    fn well(&self) -> Option<Well> {
        self.held_well.map(|polarity| Well {
            x: self.cursor.x,
            y: self.cursor.y,
            polarity,
        })
    }

    /// Update FPS counter and (in verbose mode) print statistics.
    fn update_fps_counter(&mut self) {
        if let Some(fps) = self.fps.tick() {
            if self.verbose {
                let count = self.sim.as_ref().map_or(0, Simulation::particle_count);
                println!("FPS: {fps:.1}, Particles: {count}");
            }
        }
    }

    /// Advance the simulation and dispatch resulting audio.
    fn simulate(&mut self, dt: f64, now: Instant) {
        let well = self.well();
        let Some(ref mut sim) = self.sim else {
            return;
        };
        let events = sim.step(dt, now, well);

        // The headless core reports what happened; presentation (logging,
        // audio, choreography) is this layer's job.
        if events.explosion_started {
            println!(
                "EXPLOSION! Birth rate or saturation limit reached; will kill {} of {} particles",
                sim.explosion().map_or(0, |e| e.doomed_count),
                sim.particle_count()
            );
        }
        if let Some(killed) = events.explosion_completed {
            println!(
                "Explosion complete: killed {killed}, {} remaining",
                sim.particle_count()
            );
        }
        if events.motion_stopped {
            println!("All motion has stopped");
        }

        if events.max_collision_energy > 0.0 {
            #[allow(clippy::cast_possible_truncation)]
            self.audio
                .play_ping(events.max_collision_energy, events.collision_pan as f32);
        }
        for chime in &events.wall_hits {
            #[allow(clippy::cast_possible_truncation)]
            self.audio
                .play_note(chime.note, chime.energy, chime.pan as f32);
            self.wall_flash.insert(chime.stroke, 1.0);
        }
        // Decay hit flashes over ~250 ms of simulated time (paused frames
        // take no steps, so flashes hold — the frame is frozen anyway).
        #[allow(clippy::cast_possible_truncation)]
        let decay = (-dt * 12.0).exp() as f32;
        self.wall_flash.retain(|_, v| {
            *v *= decay;
            *v > 0.02
        });
        // Selection sweep: an id that died this frame (clear-all, reset,
        // Shift+V/U) drops the selection with it.
        if let Some(sel) = self.selection {
            let alive = match sel {
                Selection::Emitter(id) => sim.emitter(id).is_some(),
                Selection::WallStroke(id) => sim.stroke_segment_count(id) > 0,
            };
            if !alive {
                self.selection = None;
            }
        }
        if events.explosion_started {
            self.audio.play_explosion();
            self.bullet_time.trigger(now);
        }
    }

    /// Build the HUD overlay text for the current mode.
    fn hud_lines(&self, sim: &Simulation) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(ref preset) = self.config.preset {
            lines.push(format!("Preset: {preset}"));
        }
        lines.extend([
            format!("FPS: {:.1}", self.fps.current),
            format!("Particles: {}", sim.particle_count()),
            format!("Gravity: {}%  (Up/Down)", sim.gravity_percent),
            format!(
                "Elasticity: particle {:.2} (Left/Right) / wall {:.2} ([/])",
                sim.particle_elasticity, sim.wall_elasticity
            ),
            format!("Time scale: {:.2}x  (,/.)", self.time_scale),
            if sim.explosion_threshold == 0 {
                "Explosions: off  (-/=)".to_string()
            } else {
                format!(
                    "Explosions: at {}/s birth rate (now {}/s)  (-/=)",
                    sim.explosion_threshold,
                    sim.birth_rate()
                )
            },
            format!("Spawn: {}  (B)", sim.spawn_mode.label()),
            format!(
                "Matter: {}  (X)   Flow: {}  (F)   Self-grav: {}  (A)",
                if sim.matter { "on" } else { "off" },
                if sim.flow { "on" } else { "off" },
                if sim.self_gravity { "on" } else { "off" },
            ),
            format!(
                "Wells: {}  (W pins, Shift+W repels)",
                sim.pinned_wells().len()
            ),
            format!(
                "Walls: {}  (V draws, Shift+V clears)",
                sim.wall_segments().len()
            ),
            format!(
                "Emitters: {}  (U aims, Shift+U clears)",
                sim.emitters().len()
            ),
            format!(
                "Music: {}  (S)   Pings: {}%  (;/')   Chimes: {}  (I)   Kaleidoscope: {}  (K)",
                if self.audio.is_music() { "on" } else { "off" },
                self.audio.ping_volume_percent(),
                if sim.wall_chimes { "on" } else { "off" },
                if self.kaleidoscope { "on" } else { "off" },
            ),
        ]);

        let mut flags = Vec::new();
        if self.paused {
            flags.push("PAUSED");
        }
        if self.audio.is_muted() {
            flags.push("MUTED");
        }
        if sim.stopped() {
            flags.push("STOPPED");
        }
        match self.held_well {
            Some(Polarity::Attract) => flags.push("WELL: ATTRACT"),
            Some(Polarity::Repel) => flags.push("WELL: REPEL"),
            None => {}
        }
        if self.bullet_time.active() {
            flags.push("BULLET TIME");
        }
        if !flags.is_empty() {
            lines.push(flags.join("  "));
        }

        if self.hud_mode == HudMode::StatsAndKeys {
            lines.push(String::new());
            // Curated compressed lines (screen space); the canonical list
            // is config::CONTROLS, which --help renders and a test checks
            // against the README.
            for key_line in [
                "P pause   N step   R reset   M mute",
                "T trails   C colors   B spawn mode",
                "X matter (fusion/fission)   F flow   A self-gravity",
                "S musical pings   K kaleidoscope",
                "G hold: gravity well (Shift+G repels)",
                "W pin well (Shift+W repel, Shift+R clear)",
                "V hold+drag: draw walls (Shift+V clears)",
                "D hold+click: select emitter/wall (Esc deselects)",
                "J comet   O screenshot   E export scene",
                "Click: burst   Middle: comet   Right-click: explosion",
                "H cycle HUD   Space/Esc/Q quit",
            ] {
                lines.push(key_line.to_string());
            }
        }

        lines
    }

    /// Draw the current frame and present it, tolerating transient failures.
    fn render_frame(&mut self, width: u32, height: u32) {
        let Some(ref sim) = self.sim else {
            return;
        };
        let hud_lines = if self.hud_mode == HudMode::Hidden {
            None
        } else {
            Some(self.hud_lines(sim))
        };

        #[cfg(not(target_arch = "wasm32"))]
        let panel_state = &self.panel_state();

        let Some(ref mut render) = self.render else {
            return;
        };
        #[cfg(not(target_arch = "wasm32"))]
        let gui = &self.gui;

        let trails = self.trails;
        let color_mode = self.color_mode;
        let kaleidoscope = self.kaleidoscope;
        // Per-segment chime-flash intensities: each segment glows with its
        // stroke's current flash level.
        let wall_flash: Vec<f32> = sim
            .wall_meta()
            .iter()
            .map(|m| self.wall_flash.get(&m.stroke).copied().unwrap_or(0.0))
            .collect();
        let stopped = sim.stopped();
        let paused = self.paused;
        let selection = self.selection;
        let capture = self.screenshot_requested;
        self.screenshot_requested = false;

        let mut shot: Option<Vec<u8>> = None;
        render.with_frame(|frame| {
            if trails {
                fade_frame(frame);
            } else {
                frame.fill(0);
            }
            if let Some(exp) = sim.explosion() {
                render_explosion(frame, exp, width, height);
            }
            render_wells(frame, sim.pinned_wells(), width, height);
            render_segments(frame, sim.wall_segments(), &wall_flash, width, height);
            render_emitters(frame, sim.emitters(), width, height);
            // Selection highlight: a dedicated amber pass over the flash
            // pass (a flash-vector sentinel would be ambiguous mid-chime).
            match selection {
                Some(Selection::WallStroke(id)) => {
                    let mask: Vec<bool> = sim.wall_meta().iter().map(|m| m.stroke == id).collect();
                    render_stroke_highlight(frame, sim.wall_segments(), &mask, width, height);
                }
                Some(Selection::Emitter(id)) => {
                    if let Some(e) = sim.emitter(id) {
                        render_emitter_highlight(frame, e, width, height);
                    }
                }
                None => {}
            }
            render_particles(
                frame,
                sim.particles(),
                width,
                height,
                color_mode,
                sim.initial_speed(),
            );
            if kaleidoscope {
                kaleidoscope_frame(frame, width, height);
            }
            if stopped {
                draw_text_centered(frame, width, height, "STOPPED", 72.0, MESSAGE_COLOR);
            } else if paused {
                draw_text_centered(frame, width, height, "PAUSED", 72.0, MESSAGE_COLOR);
            }
            if let Some(ref lines) = hud_lines {
                draw_hud(frame, width, height, lines);
            }
            #[cfg(not(target_arch = "wasm32"))]
            gui.draw(frame, panel_state, width, height);
            if capture {
                shot = Some(frame.to_vec());
            }
        });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(frame) = shot {
            match save_screenshot(&frame, width, height) {
                Ok(path) => println!("Screenshot saved to '{}'", path.display()),
                Err(e) => eprintln!("Screenshot failed: {e}"),
            }
        }
        // On the web the panel captures the canvas directly (toBlob);
        // the O-key capture path has nowhere to write.
        #[cfg(target_arch = "wasm32")]
        let _ = shot;

        match render.present() {
            Ok(()) => self.consecutive_render_failures = 0,
            Err(e) => {
                // Transient surface loss (display sleep, mode change) should
                // not crash; give up only if it never recovers.
                self.consecutive_render_failures += 1;
                eprintln!("Render failed ({e}); skipping frame");
            }
        }
    }

    /// Apply the cursor auto-hide policy, changing OS cursor state only on
    /// transitions.
    fn update_cursor_visibility(&mut self) {
        let idle = self.cursor.last_move.elapsed().as_secs_f64();
        #[cfg(not(target_arch = "wasm32"))]
        let interacting = self.held_well.is_some()
            || self.wall_anchor.is_some()
            || self.emitter_anchor.is_some()
            || self.gui.is_open();
        #[cfg(target_arch = "wasm32")]
        let interacting =
            self.held_well.is_some() || self.wall_anchor.is_some() || self.emitter_anchor.is_some();
        let desired = cursor_should_be_visible(
            self.cursor.window_focused,
            self.cursor.inside,
            interacting,
            idle,
        );
        if desired != self.cursor.visible {
            if let Some(ref render) = self.render {
                render.window().set_cursor_visible(desired);
            }
            self.cursor.visible = desired;
        }
    }

    fn update_and_render(&mut self) {
        // Drain the control panel's queued commands before stepping, and
        // publish a fresh state snapshot afterward (via render_frame's
        // caller returning) — the panel is just another control surface
        // entering at apply().
        #[cfg(target_arch = "wasm32")]
        self.drain_web_commands();

        self.update_cursor_visibility();
        let Some((width, height)) = self.dimensions() else {
            return;
        };
        let now = Instant::now();

        // Warmup frames for GPU initialization
        if self.warmup_frames > 0 {
            self.warmup_frames -= 1;
            self.last_time = now;
            self.fps.restart(now);
            self.render_frame(width, height);
            return;
        }

        let dt = now.duration_since(self.last_time).as_secs_f64().min(0.05);
        self.last_time = now;

        // The panel acts before the step so its commands shape this
        // frame, exactly like the web panel's drained mailbox.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = self.panel_state();
            let idle = self.cursor.last_move.elapsed().as_secs_f64();
            let commands = self
                .gui
                .tick(dt, &state, width, height, self.shift_down, idle);
            for command in commands {
                self.apply_panel_command(command);
            }
        }

        let time_scale = self.time_scale * self.bullet_time.multiplier(now);
        if !self.paused {
            self.simulate(dt * time_scale, now);
        } else if self.step_once {
            // Frame-step (N while paused): advance one fixed-size step.
            self.simulate(FRAME_STEP_DT * time_scale, now);
        }
        self.step_once = false;

        self.render_frame(width, height);
        self.update_fps_counter();

        #[cfg(target_arch = "wasm32")]
        self.publish_web_snapshot();
    }

    /// Apply every command the JS panel queued since the last frame.
    #[cfg(target_arch = "wasm32")]
    fn drain_web_commands(&mut self) {
        let Some(shared) = self.web_shared.clone() else {
            return;
        };
        // Drain into a local Vec first: apply() must not run while the
        // RefCell is borrowed (a command could re-enter the mailbox).
        let commands: Vec<crate::web::WebCommand> =
            shared.borrow_mut().commands.drain(..).collect();
        for cmd in commands {
            self.apply_panel_command(cmd);
        }
    }

    /// Execute one panel command: plain [`Command`]s go through `apply()`;
    /// pointer-shaped ones (which carry coordinates the cursor state
    /// would otherwise supply) act directly, mirroring `handle_mouse`.
    /// By-value deliberately: commands are consumed from queues.
    #[allow(clippy::needless_pass_by_value)]
    fn apply_panel_command(&mut self, cmd: PanelCommand) {
        match cmd {
            PanelCommand::Plain(command) => self.apply(command),
            PanelCommand::SetPaused(paused) => {
                if self.paused != paused {
                    self.apply(Command::TogglePause);
                }
            }
            PanelCommand::SetGravity(pct) => {
                if let Some(ref mut sim) = self.sim {
                    sim.gravity_percent = pct.clamp(-GRAVITY_LIMIT, GRAVITY_LIMIT);
                }
            }
            PanelCommand::SetParticleElasticity(e) => {
                if let Some(ref mut sim) = self.sim {
                    sim.particle_elasticity = e.clamp(0.0, ELASTICITY_MAX);
                }
            }
            PanelCommand::SetWallElasticity(e) => {
                if let Some(ref mut sim) = self.sim {
                    sim.wall_elasticity = e.clamp(0.0, ELASTICITY_MAX);
                }
            }
            PanelCommand::SetTimeScale(scale) => {
                self.time_scale = scale.clamp(TIME_SCALE_MIN, TIME_SCALE_MAX);
            }
            PanelCommand::SetExplosionThreshold(t) => {
                if let Some(ref mut sim) = self.sim {
                    sim.explosion_threshold =
                        usize::try_from(t.max(0)).map_or(0, |v| v.min(EXPLOSION_THRESHOLD_MAX));
                }
            }
            PanelCommand::SpawnBurst(x, y) => {
                if let Some(ref mut sim) = self.sim {
                    sim.spawn_burst(x, y);
                }
            }
            PanelCommand::LaunchComet(x, y) => {
                if let Some(ref mut sim) = self.sim {
                    sim.launch_comet(x, y);
                }
            }
            PanelCommand::PinWell(x, y, polarity) => {
                if let Some(ref mut sim) = self.sim {
                    sim.pin_well(x, y, polarity);
                }
            }
            PanelCommand::TriggerExplosion(x, y) => {
                if let Some(ref mut sim) = self.sim {
                    sim.trigger_manual_explosion(x, y);
                }
            }
            #[cfg(target_arch = "wasm32")]
            PanelCommand::Resize(width, height) => {
                if let Some(ref mut sim) = self.sim {
                    sim.resize(width, height);
                    let (w, h) = sim.dimensions();
                    if let Some(ref mut render) = self.render {
                        render.resize_sim(w, h);
                    }
                }
            }
            // Native windows resize through the surface path; the panel
            // command is a web-only concept (the canvas tracks CSS size).
            #[cfg(not(target_arch = "wasm32"))]
            PanelCommand::Resize(..) => {}
            PanelCommand::SetMuted(muted) => {
                if self.audio.is_muted() != muted {
                    self.audio.toggle_mute();
                }
            }
            PanelCommand::SetMusic(music) => {
                if self.audio.is_music() != music {
                    self.audio.toggle_music();
                }
            }
            PanelCommand::SetPingVolume(percent) => {
                self.audio
                    .set_ping_volume(percent.clamp(0, crate::config::PING_VOLUME_MAX));
            }
            PanelCommand::PlaceEmitter(x, y) => {
                if let Some(ref mut sim) = self.sim {
                    let (w, h) = sim.dimensions();
                    let (dx, dy) = (f64::from(w) / 2.0 - x, f64::from(h) / 2.0 - y);
                    sim.place_emitter(x, y, dx, dy, EMITTER_DEFAULT_RATE, EMITTER_DEFAULT_CAP);
                }
            }
            PanelCommand::SelectAt(x, y) => {
                self.selection = self
                    .sim
                    .as_ref()
                    .and_then(|sim| sim.select_at(x, y, SELECT_RADIUS));
            }
            PanelCommand::Deselect => self.selection = None,
            PanelCommand::SetEmitterRate(id, rate) => {
                if let Some(ref mut sim) = self.sim {
                    sim.set_emitter_rate(id, rate);
                }
            }
            PanelCommand::SetEmitterCap(id, cap) => {
                if let Some(ref mut sim) = self.sim {
                    sim.set_emitter_cap(id, usize::try_from(cap).unwrap_or(1));
                }
            }
            PanelCommand::AimEmitterAt(id, x, y) => {
                if let Some(ref mut sim) = self.sim
                    && let Some(e) = sim.emitter(id)
                {
                    let (dx, dy) = (x - e.x, y - e.y);
                    sim.aim_emitter(id, dx, dy);
                }
            }
            PanelCommand::CycleStrokeNote(id) => {
                if let Some(ref mut sim) = self.sim
                    && let Some(note) = sim.stroke_note_setting(id)
                {
                    sim.set_stroke_note(id, note.cycled());
                }
            }
            PanelCommand::DeleteEmitter(id) => {
                if let Some(ref mut sim) = self.sim
                    && sim.remove_emitter(id)
                {
                    println!("Deleted emitter {id}");
                }
                if self.selection == Some(Selection::Emitter(id)) {
                    self.selection = None;
                }
            }
            PanelCommand::DeleteStroke(id) => {
                if let Some(ref mut sim) = self.sim {
                    let removed = sim.remove_stroke(id);
                    if removed > 0 {
                        println!("Deleted wall stroke {id} ({removed} segments)");
                    }
                }
                self.wall_flash.remove(&id);
                if self.selection == Some(Selection::WallStroke(id)) {
                    self.selection = None;
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            PanelCommand::Relaunch {
                preset,
                particle_size,
                initial_speed,
                min_particles,
            } => {
                if self.relaunch(preset, particle_size, initial_speed, min_particles) {
                    self.gui.reseed_launch();
                }
            }
            // The web demo relaunches by navigating to a new URL; the
            // command never arrives from the HTML panel.
            #[cfg(target_arch = "wasm32")]
            PanelCommand::Relaunch { .. } => {}
        }
    }

    /// Rebuild the simulation with new construction-time options, at the
    /// current arena size. Runs the same preset resolution as the command
    /// line (explicit values override the preset bundle, user presets
    /// load from the configured file), so a panel relaunch and a CLI
    /// launch can never disagree. On a bad configuration the old
    /// simulation keeps running.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::needless_pass_by_value)]
    fn relaunch(
        &mut self,
        preset: Option<String>,
        particle_size: Option<f64>,
        initial_speed: Option<f64>,
        min_particles: Option<u32>,
    ) -> bool {
        let Some((width, height)) = self.sim.as_ref().map(Simulation::dimensions) else {
            return false;
        };
        // Session deltas: live values the user changed since launch
        // override the new bundle — touched settings travel, untouched
        // ones follow the preset. But the most recent deliberate act
        // wins: when the preset *selection* changes, the new bundle
        // takes precedence over tweaks made while exploring the old
        // one (the user cannot know which would conflict), so the
        // deltas drop and the cycle starts fresh from the new baseline.
        let deltas = if preset == self.config.preset {
            self.session_deltas()
        } else {
            SessionDeltas::default()
        };
        let mut args: Vec<std::ffi::OsString> = vec!["bouncy".into()];
        if let Some(ref name) = preset {
            args.push("--preset".into());
            args.push(name.into());
        }
        if let Some(ref path) = self.config.presets_file {
            args.push("--presets-file".into());
            args.push(path.into());
        }
        if let Some(size) = particle_size {
            args.push("--particle-size".into());
            args.push(format!("{size}").into());
        }
        if let Some(speed) = initial_speed {
            args.push("--initial-speed".into());
            args.push(format!("{speed}").into());
        }
        if let Some(min) = min_particles {
            args.push("--min-particles".into());
            args.push(format!("{min}").into());
        }
        // Carry the launch context that describes this process, not the
        // scene: window geometry, backend, audio, verbosity.
        if let (Some(w), Some(h)) = (self.config.width, self.config.height) {
            args.push("--width".into());
            args.push(format!("{w}").into());
            args.push("--height".into());
            args.push(format!("{h}").into());
        }
        if self.config.cpu {
            args.push("--cpu".into());
        }
        if self.config.verbose {
            args.push("--verbose".into());
        }

        match Config::try_resolve_from(args) {
            Ok(config) => {
                self.config = config;
                // Presentation state follows the new launch, exactly as a
                // fresh process would set it...
                self.trails = self.config.trails;
                self.color_mode = self.config.color_mode;
                self.kaleidoscope = self.config.kaleidoscope;
                self.bullet_time.enabled = self.config.bullet_time;
                self.time_scale = 1.0;
                self.paused = false;
                // Ids restart with the new simulation; a stale selection
                // could silently alias a fresh entity.
                self.selection = None;
                self.sim = Some(Simulation::new(&self.config, width, height));
                // ...and then the session's touched settings reassert
                // themselves over the new bundle.
                self.apply_session_deltas(&deltas);
                println!(
                    "Relaunched: preset {}, particle size {}, initial speed {}",
                    self.config.preset.as_deref().unwrap_or("(none)"),
                    self.config.particle_size,
                    self.config.initial_speed
                );
                true
            }
            Err(e) => {
                eprintln!("Relaunch failed; keeping the running simulation: {e}");
                false
            }
        }
    }

    /// Live settings that differ from the launched config: the session's
    /// deliberate adjustments (None = untouched, follows the next launch).
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::struct_excessive_bools)]
    fn session_deltas(&self) -> SessionDeltas {
        let Some(ref sim) = self.sim else {
            return SessionDeltas::default();
        };
        let cfg = &self.config;
        SessionDeltas {
            gravity: (sim.gravity_percent != cfg.gravity).then_some(sim.gravity_percent),
            particle_elasticity: ((sim.particle_elasticity - cfg.particle_elasticity).abs()
                > f64::EPSILON)
                .then_some(sim.particle_elasticity),
            wall_elasticity: ((sim.wall_elasticity - cfg.wall_elasticity).abs() > f64::EPSILON)
                .then_some(sim.wall_elasticity),
            explosion_threshold: (sim.explosion_threshold != cfg.explosion_threshold)
                .then_some(sim.explosion_threshold),
            spawn_mode: (sim.spawn_mode != cfg.effective_spawn_mode()).then_some(sim.spawn_mode),
            color_mode: (self.color_mode != cfg.color_mode).then_some(self.color_mode),
            matter: (sim.matter != cfg.matter).then_some(sim.matter),
            flow: (sim.flow != cfg.flow).then_some(sim.flow),
            self_gravity: (sim.self_gravity != cfg.self_gravity).then_some(sim.self_gravity),
            wall_chimes: (sim.wall_chimes != cfg.wall_chimes).then_some(sim.wall_chimes),
            trails: (self.trails != cfg.trails).then_some(self.trails),
            kaleidoscope: (self.kaleidoscope != cfg.kaleidoscope).then_some(self.kaleidoscope),
            music: (self.audio.is_music() != cfg.music).then_some(self.audio.is_music()),
            muted: (self.audio.is_muted() != cfg.mute).then_some(self.audio.is_muted()),
            ping_volume: (self.audio.ping_volume_percent() != cfg.ping_volume)
                .then_some(self.audio.ping_volume_percent()),
            time_scale: ((self.time_scale - 1.0).abs() > f64::EPSILON).then_some(self.time_scale),
        }
    }

    /// Re-assert the session's touched settings over a fresh launch.
    #[cfg(not(target_arch = "wasm32"))]
    fn apply_session_deltas(&mut self, deltas: &SessionDeltas) {
        if let Some(ref mut sim) = self.sim {
            if let Some(v) = deltas.gravity {
                sim.gravity_percent = v;
            }
            if let Some(v) = deltas.particle_elasticity {
                sim.particle_elasticity = v;
            }
            if let Some(v) = deltas.wall_elasticity {
                sim.wall_elasticity = v;
            }
            if let Some(v) = deltas.explosion_threshold {
                sim.explosion_threshold = v;
            }
            if let Some(v) = deltas.spawn_mode {
                sim.spawn_mode = v;
            }
            if let Some(v) = deltas.matter {
                sim.matter = v;
            }
            if let Some(v) = deltas.flow {
                sim.flow = v;
            }
            if let Some(v) = deltas.self_gravity {
                sim.self_gravity = v;
            }
            if let Some(v) = deltas.wall_chimes {
                sim.wall_chimes = v;
            }
        }
        if let Some(v) = deltas.color_mode {
            self.color_mode = v;
        }
        if let Some(v) = deltas.trails {
            self.trails = v;
        }
        if let Some(v) = deltas.kaleidoscope {
            self.kaleidoscope = v;
        }
        if let Some(v) = deltas.time_scale {
            self.time_scale = v;
        }
        let music = deltas.music.unwrap_or(self.config.music);
        if self.audio.is_music() != music {
            self.audio.toggle_music();
        }
        self.audio
            .set_ping_volume(deltas.ping_volume.unwrap_or(self.config.ping_volume));
        let muted = deltas.muted.unwrap_or(self.config.mute);
        if self.audio.is_muted() != muted {
            self.audio.toggle_mute();
        }
    }

    /// Snapshot the state the native panel shows and edits — the native
    /// analogue of the web snapshot, rebuilt once per frame.
    #[cfg(not(target_arch = "wasm32"))]
    fn panel_state(&self) -> crate::gui::PanelState {
        let Some(ref sim) = self.sim else {
            return crate::gui::PanelState::default();
        };
        crate::gui::PanelState {
            fps: self.fps.current,
            particles: sim.particle_count(),
            max_particles: sim.max_particles(),
            wells: sim.pinned_wells().len(),
            walls: sim.wall_segments().len(),
            emitters: sim.emitters().len(),
            paused: self.paused,
            gravity: sim.gravity_percent,
            particle_elasticity: sim.particle_elasticity,
            wall_elasticity: sim.wall_elasticity,
            time_scale: self.time_scale,
            explosion_threshold: i32::try_from(sim.explosion_threshold).unwrap_or(i32::MAX),
            matter: sim.matter,
            flow: sim.flow,
            self_gravity: sim.self_gravity,
            trails: self.trails,
            kaleidoscope: self.kaleidoscope,
            music: self.audio.is_music(),
            wall_chimes: sim.wall_chimes,
            muted: self.audio.is_muted(),
            ping_volume: self.audio.ping_volume_percent(),
            spawn_mode: value_name(sim.spawn_mode.to_possible_value()),
            color_mode: value_name(self.color_mode.to_possible_value()),
            hud: self.hud_mode.label().to_string(),
            launch_particle_size: self.config.particle_size,
            launch_initial_speed: self.config.initial_speed,
            launch_min_particles: self.config.min_particles,
            launch_preset: self.config.preset.clone().unwrap_or_default(),
            // Pulled fresh by id: a dead id yields None here even before
            // the frame sweep drops the selection.
            selection: self.selection.and_then(|sel| match sel {
                Selection::Emitter(id) => {
                    sim.emitter(id)
                        .map(|e| crate::gui::PanelSelection::Emitter {
                            id,
                            rate: e.rate,
                            cap: e.cap,
                            angle_deg: e.dx.atan2(-e.dy).to_degrees().rem_euclid(360.0),
                        })
                }
                Selection::WallStroke(id) => {
                    sim.stroke_note_setting(id)
                        .map(|note| crate::gui::PanelSelection::Stroke {
                            id,
                            segments: sim.stroke_segment_count(id),
                            note,
                        })
                }
            }),
        }
    }

    /// Publish this frame's state for the JS panel to poll.
    #[cfg(target_arch = "wasm32")]
    fn publish_web_snapshot(&mut self) {
        let Some(ref shared) = self.web_shared else {
            return;
        };
        let Some(ref sim) = self.sim else {
            return;
        };
        let scene_toml = self.scene_toml();
        // Pulled fresh by id, like the native panel: a dead id publishes
        // all-None even before the frame sweep drops the selection.
        let mut selection_kind = None;
        let mut selection_id = None;
        let mut selection_rate = None;
        let mut selection_cap = None;
        let mut selection_angle = None;
        let mut selection_note = None;
        let mut selection_segments = None;
        match self.selection {
            Some(Selection::Emitter(id)) => {
                if let Some(e) = sim.emitter(id) {
                    selection_kind = Some("emitter".to_string());
                    selection_id = Some(id);
                    selection_rate = Some(e.rate);
                    selection_cap = Some(e.cap);
                    selection_angle = Some(e.dx.atan2(-e.dy).to_degrees().rem_euclid(360.0));
                }
            }
            Some(Selection::WallStroke(id)) => {
                if let Some(note) = sim.stroke_note_setting(id) {
                    selection_kind = Some("stroke".to_string());
                    selection_id = Some(id);
                    selection_note = Some(match note {
                        crate::presets::WallNote::Auto => "auto".to_string(),
                        crate::presets::WallNote::Note(n) => format!("degree {n}"),
                        crate::presets::WallNote::Silent => "silent".to_string(),
                    });
                    selection_segments = Some(sim.stroke_segment_count(id));
                }
            }
            None => {}
        }
        let mut shared = shared.borrow_mut();
        shared.snapshot = crate::web::Snapshot {
            fps: self.fps.current,
            particles: sim.particle_count(),
            max_particles: sim.max_particles(),
            birth_rate: sim.birth_rate(),
            explosion_threshold: sim.explosion_threshold,
            gravity: sim.gravity_percent,
            particle_elasticity: sim.particle_elasticity,
            wall_elasticity: sim.wall_elasticity,
            time_scale: self.time_scale,
            paused: self.paused,
            stopped: sim.stopped(),
            exploding: sim.explosion().is_some(),
            matter: sim.matter,
            flow: sim.flow,
            self_gravity: sim.self_gravity,
            trails: self.trails,
            kaleidoscope: self.kaleidoscope,
            wall_chimes: sim.wall_chimes,
            wells: sim.pinned_wells().len(),
            walls: sim.wall_segments().len(),
            emitters: sim.emitters().len(),
            width: sim.dimensions().0,
            height: sim.dimensions().1,
            muted: self.audio.is_muted(),
            music: self.audio.is_music(),
            ping_volume: self.audio.ping_volume_percent(),
            audio_ready: crate::audio::web_ready(),
            spawn_mode: value_name(sim.spawn_mode.to_possible_value()),
            color_mode: value_name(self.color_mode.to_possible_value()),
            hud: self.hud_mode.label().to_string(),
            selection_kind,
            selection_id,
            selection_rate,
            selection_cap,
            selection_angle,
            selection_note,
            selection_segments,
        };
        shared.scene_toml = scene_toml;
    }

    /// Handle a key press or release: pure input mapping. Everything a
    /// key *does* is a [`Command`] dispatched through [`App::apply`], so
    /// any other control surface (a GUI panel, a script) can trigger the
    /// identical behavior. Only inherently input-shaped state stays here:
    /// exiting, the held-well lifetime, and the wall-drawing anchor.
    fn handle_key(
        &mut self,
        key_code: KeyCode,
        state: ElementState,
        repeat: bool,
        event_loop: &ActiveEventLoop,
    ) {
        if state == ElementState::Released {
            if key_code == KeyCode::KeyG {
                self.held_well = None;
            }
            if key_code == KeyCode::KeyV {
                self.wall_anchor = None;
            }
            if key_code == KeyCode::KeyD {
                self.d_down = false;
            }
            if key_code == KeyCode::KeyU
                && let Some((ax, ay)) = self.emitter_anchor.take()
            {
                self.place_aimed_emitter(ax, ay);
            }
            return;
        }
        let shift_polarity = if self.shift_down {
            Polarity::Repel
        } else {
            Polarity::Attract
        };

        match key_code {
            // Esc cancels an armed panel placement tool before it means
            // exit — you are mid-gesture, not asking to leave.
            #[cfg(not(target_arch = "wasm32"))]
            KeyCode::Escape if self.gui.is_armed() => self.gui.disarm(),
            // ...and drops a selection before that: back out of the
            // inspector first, exit on the next press.
            KeyCode::Escape if self.selection.is_some() => self.selection = None,
            KeyCode::Space | KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
            // D arms select-on-click while held (not a winit modifier,
            // so it is tracked by hand on both key edges).
            KeyCode::KeyD => self.d_down = true,
            KeyCode::KeyG => {
                // A stopped simulation self-wakes while the well is held.
                self.held_well = Some(shift_polarity);
            }
            KeyCode::KeyV if !repeat => {
                if self.shift_down {
                    self.apply(Command::ClearWalls);
                } else {
                    // Start drawing: segments are added as the cursor moves.
                    self.wall_anchor = Some((self.cursor.x, self.cursor.y));
                    self.wall_stroke_open = false;
                }
            }
            KeyCode::KeyU if !repeat => {
                if self.shift_down {
                    self.apply(Command::ClearEmitters);
                } else {
                    // Anchor here; the release drag aims the emitter.
                    self.emitter_anchor = Some((self.cursor.x, self.cursor.y));
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            KeyCode::Tab if !repeat => self.gui.toggle_open(),
            KeyCode::KeyP if !repeat => self.apply(Command::TogglePause),
            KeyCode::KeyN => self.apply(Command::StepFrame),
            KeyCode::KeyR if !repeat => self.apply(if self.shift_down {
                Command::ClearWells
            } else {
                Command::Reset
            }),
            KeyCode::KeyM if !repeat => self.apply(Command::ToggleMute),
            KeyCode::KeyH if !repeat => self.apply(Command::CycleHud),
            KeyCode::KeyT if !repeat => self.apply(Command::ToggleTrails),
            KeyCode::KeyC if !repeat => self.apply(Command::CycleColorMode),
            KeyCode::KeyB if !repeat => self.apply(Command::CycleSpawnMode),
            KeyCode::KeyX if !repeat => self.apply(Command::ToggleMatter),
            KeyCode::KeyF if !repeat => self.apply(Command::ToggleFlow),
            KeyCode::KeyA if !repeat => self.apply(Command::ToggleSelfGravity),
            KeyCode::KeyO if !repeat => self.apply(Command::Screenshot),
            KeyCode::KeyE if !repeat => self.apply(Command::ExportScene),
            KeyCode::KeyJ if !repeat => self.apply(Command::LaunchComet),
            KeyCode::KeyS if !repeat => self.apply(Command::ToggleMusic),
            KeyCode::Semicolon => self.apply(Command::AdjustPingVolume(-PING_VOLUME_STEP)),
            KeyCode::Quote => self.apply(Command::AdjustPingVolume(PING_VOLUME_STEP)),
            KeyCode::KeyI if !repeat => self.apply(Command::ToggleWallChimes),
            KeyCode::KeyK if !repeat => self.apply(Command::ToggleKaleidoscope),
            KeyCode::KeyW if !repeat => self.apply(Command::PinWell(shift_polarity)),
            KeyCode::ArrowUp => self.apply(Command::AdjustGravity(GRAVITY_STEP)),
            KeyCode::ArrowDown => self.apply(Command::AdjustGravity(-GRAVITY_STEP)),
            KeyCode::ArrowRight => {
                self.apply(Command::AdjustParticleElasticity(ELASTICITY_STEP));
            }
            KeyCode::ArrowLeft => {
                self.apply(Command::AdjustParticleElasticity(-ELASTICITY_STEP));
            }
            KeyCode::BracketRight => self.apply(Command::AdjustWallElasticity(ELASTICITY_STEP)),
            KeyCode::BracketLeft => self.apply(Command::AdjustWallElasticity(-ELASTICITY_STEP)),
            KeyCode::Period => self.apply(Command::AdjustTimeScale(TIME_SCALE_STEP)),
            KeyCode::Comma => self.apply(Command::AdjustTimeScale(-TIME_SCALE_STEP)),
            KeyCode::Equal => self.apply(Command::AdjustExplosionThreshold(THRESHOLD_STEP)),
            KeyCode::Minus => self.apply(Command::AdjustExplosionThreshold(-THRESHOLD_STEP)),
            _ => {}
        }
    }

    /// Run a closure over the simulation if it exists (it is None only
    /// before the window opens).
    fn with_sim(&mut self, f: impl FnOnce(&mut Simulation)) {
        if let Some(ref mut sim) = self.sim {
            f(sim);
        }
    }

    /// Apply a runtime control action. Every parameter mutation and its
    /// log line lives here, shared by all input paths; clamps use the
    /// same limits as the CLI value parsers, so a hotkey can never reach
    /// a value the command line would reject.
    fn apply(&mut self, command: Command) {
        match command {
            Command::TogglePause => {
                self.paused = !self.paused;
                println!("{}", if self.paused { "Paused" } else { "Resumed" });
            }
            Command::StepFrame => {
                if self.paused {
                    self.step_once = true;
                }
            }
            Command::Reset => {
                if let Some(ref mut sim) = self.sim {
                    println!("Simulation reset");
                    sim.reset();
                    self.paused = false;
                }
            }
            Command::ClearWells => self.with_sim(|sim| {
                println!("Cleared {} pinned wells", sim.clear_wells());
            }),
            Command::ClearWalls => self.with_sim(|sim| {
                println!("Cleared {} wall segments", sim.clear_wall_segments());
            }),
            Command::ClearEmitters => self.with_sim(|sim| {
                let cleared = sim.clear_emitters();
                println!("Cleared {cleared} emitters");
            }),
            Command::ToggleMute => {
                let muted = self.audio.toggle_mute();
                println!("Audio {}", if muted { "muted" } else { "unmuted" });
            }
            Command::ToggleMusic => {
                let music = self.audio.toggle_music();
                println!(
                    "Musical pings {}",
                    if music { "on (pentatonic)" } else { "off" }
                );
            }
            Command::CycleHud => self.hud_mode = self.hud_mode.next(),
            Command::ToggleTrails => {
                self.trails = !self.trails;
                println!("Trails {}", if self.trails { "on" } else { "off" });
            }
            Command::CycleColorMode => {
                self.color_mode = match self.color_mode {
                    ColorMode::Solid => ColorMode::Velocity,
                    ColorMode::Velocity => ColorMode::Solid,
                };
            }
            Command::ToggleKaleidoscope => {
                self.kaleidoscope = !self.kaleidoscope;
                println!(
                    "Kaleidoscope {}",
                    if self.kaleidoscope { "on" } else { "off" }
                );
            }
            Command::CycleSpawnMode => self.with_sim(|sim| {
                sim.spawn_mode = sim.spawn_mode.next();
                println!("Spawn mode: {}", sim.spawn_mode.label());
            }),
            Command::ToggleWallChimes => self.with_sim(|sim| {
                sim.wall_chimes = !sim.wall_chimes;
                println!("Wall chimes {}", if sim.wall_chimes { "on" } else { "off" });
            }),
            Command::AdjustPingVolume(step) => {
                let volume = (self.audio.ping_volume_percent() + step)
                    .clamp(0, crate::config::PING_VOLUME_MAX);
                self.audio.set_ping_volume(volume);
                println!("Ping volume: {volume}%");
            }
            Command::ToggleMatter => self.with_sim(|sim| {
                sim.matter = !sim.matter;
                println!("Matter mechanics {}", if sim.matter { "on" } else { "off" });
            }),
            Command::ToggleFlow => self.with_sim(|sim| {
                // A stopped simulation self-wakes when the flow is on.
                sim.flow = !sim.flow;
                println!("Flow field {}", if sim.flow { "on" } else { "off" });
            }),
            Command::ToggleSelfGravity => self.with_sim(|sim| {
                // Also an ambient force: a stopped simulation self-wakes.
                sim.self_gravity = !sim.self_gravity;
                println!(
                    "Self-gravity {}",
                    if sim.self_gravity { "on" } else { "off" }
                );
            }),
            Command::AdjustGravity(step) => self.with_sim(|sim| {
                sim.gravity_percent =
                    (sim.gravity_percent + step).clamp(-GRAVITY_LIMIT, GRAVITY_LIMIT);
                println!("Gravity: {}%", sim.gravity_percent);
            }),
            Command::AdjustParticleElasticity(delta) => self.with_sim(|sim| {
                sim.particle_elasticity =
                    (sim.particle_elasticity + delta).clamp(0.0, ELASTICITY_MAX);
                println!("Particle elasticity: {:.2}", sim.particle_elasticity);
            }),
            Command::AdjustWallElasticity(delta) => self.with_sim(|sim| {
                sim.wall_elasticity = (sim.wall_elasticity + delta).clamp(0.0, ELASTICITY_MAX);
                println!("Wall elasticity: {:.2}", sim.wall_elasticity);
            }),
            Command::AdjustTimeScale(delta) => self.adjust_time_scale(delta),
            Command::AdjustExplosionThreshold(step) => self.with_sim(|sim| {
                let magnitude = step.unsigned_abs() as usize;
                sim.explosion_threshold = if step >= 0 {
                    (sim.explosion_threshold + magnitude).min(EXPLOSION_THRESHOLD_MAX)
                } else {
                    sim.explosion_threshold.saturating_sub(magnitude)
                };
                if sim.explosion_threshold == 0 {
                    println!("Explosion threshold: off");
                } else {
                    println!("Explosion threshold: {}/s", sim.explosion_threshold);
                }
            }),
            Command::Screenshot => self.screenshot_requested = true,
            #[cfg(not(target_arch = "wasm32"))]
            Command::ExportScene => self.export_scene(),
            // On the web the panel pulls scene_toml() and downloads it;
            // the E key has no file to write.
            #[cfg(target_arch = "wasm32")]
            Command::ExportScene => {}
            Command::LaunchComet => {
                let (x, y) = (self.cursor.x, self.cursor.y);
                self.with_sim(|sim| {
                    sim.launch_comet(x, y);
                    println!("Comet inbound");
                });
            }
            Command::PinWell(polarity) => {
                let (x, y) = (self.cursor.x, self.cursor.y);
                self.with_sim(|sim| {
                    if sim.pin_well(x, y, polarity) {
                        println!(
                            "Pinned {} well at ({x:.0}, {y:.0}); {} total (Shift+R clears)",
                            match polarity {
                                Polarity::Attract => "attracting",
                                Polarity::Repel => "repelling",
                            },
                            sim.pinned_wells().len()
                        );
                    } else {
                        println!("Pinned well limit reached ({MAX_PINNED_WELLS})");
                    }
                });
            }
        }
    }

    /// Step the time scale by `delta` (comma/period), clamped to
    /// [`TIME_SCALE_MIN`], [`TIME_SCALE_MAX`]. The result is snapped onto
    /// the step grid so repeated float additions cannot drift — up and
    /// down retrace each other exactly, always able to land back on 1.0.
    fn adjust_time_scale(&mut self, delta: f64) {
        let stepped = (self.time_scale + delta).clamp(TIME_SCALE_MIN, TIME_SCALE_MAX);
        self.time_scale = (stepped / TIME_SCALE_STEP).round() * TIME_SCALE_STEP;
        println!("Time scale: {:.2}x", self.time_scale);
    }

    /// Extend the wall being drawn (V held): each time the cursor gets far
    /// enough from the anchor, drop a segment and move the anchor forward,
    /// so a curved drag becomes a polyline.
    /// Place an emitter anchored at `(ax, ay)`, aimed along the drag to
    /// the current cursor — or at the screen center on a bare tap.
    fn place_aimed_emitter(&mut self, ax: f64, ay: f64) {
        let Some(ref mut sim) = self.sim else {
            return;
        };
        let (mut dx, mut dy) = (self.cursor.x - ax, self.cursor.y - ay);
        if dx.hypot(dy) < EMITTER_MIN_AIM_DRAG {
            let (w, h) = sim.dimensions();
            (dx, dy) = (f64::from(w) / 2.0 - ax, f64::from(h) / 2.0 - ay);
        }
        if sim.place_emitter(ax, ay, dx, dy, EMITTER_DEFAULT_RATE, EMITTER_DEFAULT_CAP) {
            println!(
                "Emitter placed ({} of {})",
                sim.emitters().len(),
                MAX_EMITTERS
            );
        } else {
            println!("Emitter limit reached ({MAX_EMITTERS})");
        }
    }

    fn extend_wall(&mut self) {
        let Some((ax, ay)) = self.wall_anchor else {
            return;
        };
        let (dx, dy) = (self.cursor.x - ax, self.cursor.y - ay);
        if dx * dx + dy * dy < WALL_SEGMENT_MIN_LENGTH * WALL_SEGMENT_MIN_LENGTH {
            return;
        }
        let Some(ref mut sim) = self.sim else {
            return;
        };
        // The drag's first segment opens a stroke; the rest extend it, so
        // the whole polyline sounds (and flashes) as one instrument bar.
        let added = if self.wall_stroke_open {
            sim.extend_last_wall_stroke(ax, ay, self.cursor.x, self.cursor.y)
        } else {
            sim.add_wall_segment(ax, ay, self.cursor.x, self.cursor.y)
        };
        if added {
            self.wall_anchor = Some((self.cursor.x, self.cursor.y));
            self.wall_stroke_open = true;
        } else {
            // Stop the stroke at the cap instead of retrying every motion.
            println!("Wall segment limit reached ({MAX_WALL_SEGMENTS})");
            self.wall_anchor = None;
        }
    }

    /// Export the current settings and scene geometry (pinned wells and
    /// drawn walls, normalized to window fractions) as a preset table the
    /// user can copy into their presets file. Boolean options are emitted
    /// only when on: like the command line, presets cannot switch one off.
    /// Snapshot the current settings and normalized geometry — the pure
    /// half of scene export, shared by the native file writer and the
    /// web download path.
    #[allow(clippy::type_complexity)]
    fn scene_export_parts(
        &self,
    ) -> Option<(
        Vec<(&'static str, toml::Value)>,
        Vec<crate::presets::SceneWell>,
        Vec<crate::presets::SceneWall>,
        Vec<crate::presets::SceneEmitter>,
    )> {
        use crate::presets::{SceneEmitter, SceneWall, SceneWell};
        let sim = self.sim.as_ref()?;
        let (w, h) = sim.dimensions();
        let (wf, hf) = (f64::from(w), f64::from(h));
        // Four decimals keeps files tidy; a fraction of a pixel at 8K.
        let frac = |v: f64, span: f64| ((v / span) * 10_000.0).round() / 10_000.0;
        let wells: Vec<SceneWell> = sim
            .pinned_wells()
            .iter()
            .map(|well| SceneWell {
                x: frac(well.x, wf),
                y: frac(well.y, hf),
                polarity: well.polarity,
            })
            .collect();
        let walls: Vec<SceneWall> = sim
            .wall_segments()
            .iter()
            .zip(sim.wall_export_notes())
            .map(|(seg, note)| SceneWall {
                x1: frac(seg.x1, wf),
                y1: frac(seg.y1, hf),
                x2: frac(seg.x2, wf),
                y2: frac(seg.y2, hf),
                note,
            })
            .collect();

        // Counts are far below i64::MAX.
        #[allow(clippy::cast_possible_wrap)]
        let mut settings: Vec<(&str, toml::Value)> = vec![
            ("description", "Exported scene".into()),
            ("gravity", i64::from(sim.gravity_percent).into()),
            ("wall-elasticity", sim.wall_elasticity.into()),
            ("particle-elasticity", sim.particle_elasticity.into()),
            (
                "spawn-mode",
                value_name(sim.spawn_mode.to_possible_value()).into(),
            ),
            (
                "explosion-threshold",
                (sim.explosion_threshold as i64).into(),
            ),
            ("particle-size", self.config.particle_size.into()),
            ("initial-speed", self.config.initial_speed.into()),
            ("min-particles", (sim.base_particle_count() as i64).into()),
            (
                "color-mode",
                value_name(self.color_mode.to_possible_value()).into(),
            ),
        ];
        for (flag, on) in [
            ("matter", sim.matter),
            ("flow", sim.flow),
            ("self-gravity", sim.self_gravity),
            ("trails", self.trails),
            ("kaleidoscope", self.kaleidoscope),
            ("music", self.audio.is_music()),
            ("wall-chimes", sim.wall_chimes),
            ("mute", self.audio.is_muted()),
            ("bullet-time", self.bullet_time.enabled),
        ] {
            if on {
                settings.push((flag, true.into()));
            }
        }

        let emitters: Vec<SceneEmitter> = sim
            .emitters()
            .iter()
            .map(|e| SceneEmitter {
                x: frac(e.x, wf),
                y: frac(e.y, hf),
                // The angle convention's inverse: degrees clockwise from
                // straight up. Two decimals round-trips direction well
                // within visual and musical tolerance.
                angle: (e.dx.atan2(-e.dy).to_degrees().rem_euclid(360.0) * 100.0).round() / 100.0,
                rate: e.rate,
                cap: e.cap,
            })
            .collect();

        Some((settings, wells, walls, emitters))
    }

    /// Export the scene to a uniquely named preset file (native).
    #[cfg(not(target_arch = "wasm32"))]
    fn export_scene(&mut self) {
        let Some((settings, wells, walls, emitters)) = self.scene_export_parts() else {
            return;
        };
        match crate::presets::export_scene(&settings, &wells, &walls, &emitters) {
            Ok(path) => println!(
                "Scene exported to '{}' — copy the table into your presets \
                 file (see --list-presets) or run with --presets-file",
                path.display()
            ),
            Err(e) => eprintln!("Scene export failed: {e}"),
        }
    }

    /// The current scene as preset-file TOML text; the web panel turns
    /// this into a download.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn scene_toml(&self) -> Option<String> {
        let (settings, wells, walls, emitters) = self.scene_export_parts()?;
        Some(crate::presets::scene_to_toml(
            "scene-web",
            &settings,
            &wells,
            &walls,
            &emitters,
        ))
    }

    /// Handle a mouse button press at the current cursor position.
    fn handle_mouse(&mut self, button: MouseButton) {
        if button == MouseButton::Middle {
            // Same command the J key issues: one code path for comets.
            self.apply(Command::LaunchComet);
            return;
        }
        let (x, y) = (self.cursor.x, self.cursor.y);
        let Some(ref mut sim) = self.sim else {
            return;
        };
        if button == MouseButton::Left {
            sim.spawn_burst(x, y);
        } else if button == MouseButton::Right && sim.trigger_manual_explosion(x, y) {
            println!(
                "Explosion at cursor; will kill {} of {} particles",
                sim.explosion().map_or(0, |e| e.doomed_count),
                sim.particle_count()
            );
            self.audio.play_explosion();
            self.bullet_time.trigger(Instant::now());
        }
    }
}

/// Write `frame` to a uniquely named PNG in the working directory,
/// returning the path. Named by Unix timestamp, with a numeric suffix if
/// several screenshots land in the same second.
#[cfg(not(target_arch = "wasm32"))]
fn save_screenshot(frame: &[u8], width: u32, height: u32) -> Result<std::path::PathBuf, String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    for n in 0..100 {
        let name = if n == 0 {
            format!("bouncy-{secs}.png")
        } else {
            format!("bouncy-{secs}-{n}.png")
        };
        let path = std::path::PathBuf::from(name);
        if path.exists() {
            continue;
        }
        write_png(&path, frame, width, height)?;
        return Ok(path);
    }
    Err("too many screenshots this second".to_string())
}

/// Bullet-time multiplier at `elapsed` wall-clock seconds since the
/// explosion started: a hard dip to `BULLET_TIME_SCALE`, held for
/// `BULLET_TIME_HOLD_SECS`, then a linear ramp back to full speed over
/// `BULLET_TIME_RAMP_SECS`.
fn bullet_time_factor(elapsed: f64) -> f64 {
    if elapsed < BULLET_TIME_HOLD_SECS {
        BULLET_TIME_SCALE
    } else {
        let t = ((elapsed - BULLET_TIME_HOLD_SECS) / BULLET_TIME_RAMP_SECS).clamp(0.0, 1.0);
        BULLET_TIME_SCALE + (1.0 - BULLET_TIME_SCALE) * t
    }
}

/// Cursor auto-hide policy: visible while the window is unfocused, while
/// the cursor is outside the window, while a cursor interaction (gravity
/// well, wall drawing) is engaged, or until `CURSOR_HIDE_DELAY` seconds
/// have passed since the last movement.
fn cursor_should_be_visible(
    focused: bool,
    inside: bool,
    interacting: bool,
    idle_secs: f64,
) -> bool {
    !focused || !inside || interacting || idle_secs < CURSOR_HIDE_DELAY
}

/// Draw the HUD overlay lines top-left over a dark semi-transparent panel
/// so the text stays readable against a dense particle field. Free function
/// so it can run inside the frame closure while the render context is
/// mutably borrowed.
fn draw_hud(frame: &mut [u8], width: u32, height: u32, lines: &[String]) {
    const PANEL_PADDING: f32 = 6.0;
    const PANEL_KEEP: u16 = 77; // retain ~30% background brightness

    let max_line_width = lines
        .iter()
        .filter(|line| !line.is_empty())
        .map(|line| measure_text(line, HUD_FONT_SIZE).0)
        .fold(0.0f32, f32::max);
    #[allow(clippy::cast_precision_loss)]
    let panel_height = lines.len() as f32 * HUD_LINE_HEIGHT + 2.0 * PANEL_PADDING;
    let panel_width = max_line_width + 2.0 * (HUD_MARGIN + PANEL_PADDING);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let panel = (0, 0, panel_width.ceil() as u32, panel_height.ceil() as u32);
    dim_rect(frame, width, height, panel, PANEL_KEEP);

    for (i, line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let y = HUD_MARGIN + i as f32 * HUD_LINE_HEIGHT;
        draw_text(
            frame,
            width,
            height,
            line,
            HUD_FONT_SIZE,
            (HUD_MARGIN, y),
            HUD_COLOR,
        );
    }
}

impl ApplicationHandler for App {
    #[cfg(target_arch = "wasm32")]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use winit::platform::web::WindowAttributesExtWebSys;

        if self.render.is_some() {
            return; // Already initialized
        }

        // Attach to the page's canvas (the demo page provides #bouncy).
        // prevent_default keeps right-click on the explosion instead of
        // the context menu and stops scrolling keys from moving the page.
        let document = web_sys::window()
            .and_then(|w| w.document())
            .expect("no document");
        let canvas = document
            .get_element_by_id("bouncy")
            .and_then(|e| wasm_bindgen::JsCast::dyn_into::<web_sys::HtmlCanvasElement>(e).ok())
            .expect("no #bouncy canvas in the page");

        // Simulation size: explicit --width/--height (from the URL query)
        // win; otherwise the canvas's CSS size at load, floored to
        // something playable.
        #[allow(clippy::cast_sign_loss)]
        let (width, height) = if let (Some(w), Some(h)) = (self.config.width, self.config.height) {
            (w, h)
        } else {
            (
                (canvas.client_width().max(320)) as u32,
                (canvas.client_height().max(240)) as u32,
            )
        };

        let window_attrs = Window::default_attributes()
            .with_canvas(Some(canvas))
            .with_prevent_default(true)
            .with_focusable(true);
        let window = event_loop
            .create_window(window_attrs)
            .expect("Failed to create window");

        let sim = Simulation::new(&self.config, width, height);
        self.sim = Some(sim);
        self.last_time = Instant::now();
        self.fps.restart(Instant::now());

        let window = Rc::new(window);
        self.render = Some(create_render_context(
            &window, width, height, width, height, false,
        ));
        window.request_redraw();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render.is_some() {
            return; // Already initialized
        }

        // Build window attributes based on requested size or fullscreen
        let window_attrs = Window::default_attributes()
            .with_title("Bouncy Particles - Press SPACE to exit")
            .with_resizable(false);

        let window_attrs = if let (Some(w), Some(h)) = (self.config.width, self.config.height) {
            // Validate against actual display size
            if let Some(monitor) = event_loop.available_monitors().next() {
                let monitor_size = monitor.size();
                let scale = monitor.scale_factor();
                let max_width = physical_to_logical(monitor_size.width, scale);
                let max_height = physical_to_logical(monitor_size.height, scale);

                if w > max_width || h > max_height {
                    eprintln!(
                        "Error: Requested size {w}x{h} exceeds display size {max_width}x{max_height}"
                    );
                    event_loop.exit();
                    return;
                }
            }
            println!("Window mode: {w}x{h} (fixed size)");
            window_attrs.with_inner_size(LogicalSize::new(w, h))
        } else {
            println!("Window mode: fullscreen");
            window_attrs.with_fullscreen(Some(Fullscreen::Borderless(None)))
        };

        let window = event_loop
            .create_window(window_attrs)
            .expect("Failed to create window");

        let physical_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let width = simulation_size(physical_size.width, scale_factor);
        let height = simulation_size(physical_size.height, scale_factor);
        self.scale_factor = scale_factor;

        println!(
            "Window: {}x{} physical, {}x{} logical, scale={}",
            physical_size.width, physical_size.height, width, height, scale_factor
        );

        let sim = Simulation::new(&self.config, width, height);
        println!(
            "Base particle count{}: {}",
            if self.config.min_particles.is_some() {
                " (override)"
            } else {
                " for this screen"
            },
            sim.base_particle_count()
        );
        self.sim = Some(sim);
        self.last_time = Instant::now();
        self.fps.restart(Instant::now());

        let window = Rc::new(window);
        self.render = Some(create_render_context(
            &window,
            width,
            height,
            physical_size.width,
            physical_size.height,
            self.config.cpu,
        ));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                self.handle_key(key_code, state, repeat, event_loop);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.shift_down = modifiers.state().shift_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Ask the renderer for the mapping: dividing by the scale
                // factor is wrong whenever the GPU backend letterboxes
                // (fractional scale factors present the frame smaller than
                // the window, centered).
                if let Some(ref render) = self.render {
                    (self.cursor.x, self.cursor.y) =
                        render.window_pos_to_sim(position.x, position.y);
                }
                self.cursor.moved();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.gui.set_cursor(self.cursor.x, self.cursor.y);
                    // Drawing a wall must stop at the panel's edge: the
                    // pointer belongs to the panel there.
                    if let Some((w, _)) = self.dimensions() {
                        if self.gui.wants_pointer(w) {
                            return;
                        }
                    }
                }
                self.extend_wall();
            }
            WindowEvent::CursorEntered { .. } => {
                self.cursor.moved();
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor.inside = false;
            }
            WindowEvent::Focused(focused) => {
                self.cursor.window_focused = focused;
                if focused {
                    // Fresh focus should not instantly hide the cursor.
                    self.cursor.last_move = Instant::now();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                // The panel owns clicks over it; nothing may fall
                // through and fire a burst or start a wall.
                #[cfg(not(target_arch = "wasm32"))]
                if let Some((w, h)) = self.dimensions() {
                    if button == MouseButton::Left && self.gui.on_press_at(w, h) {
                        return;
                    }
                    // An armed placement tool claims the next arena
                    // click: place at the cursor, exactly like the web
                    // panel's one-shot tools.
                    if button == MouseButton::Left {
                        if let Some(cmd) = self.gui.place_armed(self.cursor.x, self.cursor.y) {
                            self.apply_panel_command(cmd);
                            return;
                        }
                    }
                }
                // Held D turns the click into a pick: it must never fall
                // through and burst (a miss deselects instead).
                if button == MouseButton::Left && self.d_down {
                    self.apply_panel_command(PanelCommand::SelectAt(self.cursor.x, self.cursor.y));
                    return;
                }
                self.handle_mouse(button);
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.gui.on_release();
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => -f64::from(y),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => -pos.y / 24.0,
                };
                if let Some((w, _)) = self.dimensions() {
                    self.gui.on_wheel(lines, w);
                }
            }
            WindowEvent::Resized(new_size) => {
                if let Some(ref mut render) = self.render {
                    render.resize_surface(new_size.width, new_size.height);
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                // The window is not resizable, but DPI can change when it
                // moves between monitors. The simulation keeps its original
                // logical size; log the change deliberately. (winit also
                // fires this at window creation with the initial factor.)
                if (scale_factor - self.scale_factor).abs() > f64::EPSILON {
                    println!(
                        "Scale factor changed: {} -> {scale_factor} (simulation keeps original size)",
                        self.scale_factor
                    );
                }
                self.scale_factor = scale_factor;
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if self.consecutive_render_failures >= MAX_RENDER_FAILURES {
                    eprintln!("Rendering failed repeatedly; exiting");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref render) = self.render {
            render.window().request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Restore the cursor on every exit path (exit keys, window close,
        // repeated render failures) so it is never left hidden.
        if let Some(ref render) = self.render {
            render.window().set_cursor_visible(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_to_logical_conversion() {
        assert_eq!(physical_to_logical(3024, 2.0), 1512);
        assert_eq!(physical_to_logical(1920, 1.0), 1920);
    }

    #[test]
    fn simulation_size_fills_the_window_at_any_scale_factor() {
        // Integer scales are unchanged from the old physical/scale sizing.
        assert_eq!(simulation_size(3024, 2.0), 1512);
        assert_eq!(simulation_size(1920, 1.0), 1920);
        // Fractional scales round the render scale so integer scaling
        // fills: Joe's 4K at 150% gets a 1920-wide buffer shown at 2x
        // (previously 2560 wide shown at 1x, letterboxed).
        assert_eq!(simulation_size(3840, 1.5), 1920);
        assert_eq!(simulation_size(3840, 1.25), 3840);
        assert_eq!(simulation_size(2560, 1.75), 1280);

        // The fill invariant: the renderer's integer scale (floor of
        // physical/buffer) times the buffer covers the window to within
        // one render pixel — no visible letterbox bars.
        for &(physical, scale) in &[
            (3840u32, 1.5f64),
            (3840, 1.25),
            (2560, 1.5),
            (1920, 1.0),
            (3024, 2.0),
            (1001, 2.0),
            (2879, 1.5),
        ] {
            let buffer = simulation_size(physical, scale);
            let render_scale = physical / buffer; // pixels' floor scaling
            let covered = buffer * render_scale;
            assert!(
                physical - covered < render_scale,
                "{physical}@{scale}: buffer {buffer} x{render_scale} covers {covered}"
            );
        }
    }

    #[test]
    fn bullet_time_dips_holds_then_ramps_back() {
        // The dip applies immediately and holds for the full window.
        assert!((bullet_time_factor(0.0) - BULLET_TIME_SCALE).abs() < 1e-9);
        assert!(
            (bullet_time_factor(BULLET_TIME_HOLD_SECS * 0.99) - BULLET_TIME_SCALE).abs() < 1e-9
        );

        // Mid-ramp sits strictly between the dip and full speed.
        let mid = bullet_time_factor(BULLET_TIME_HOLD_SECS + BULLET_TIME_RAMP_SECS / 2.0);
        assert!(mid > BULLET_TIME_SCALE && mid < 1.0, "mid-ramp: {mid}");

        // Fully recovered at the end of the ramp and beyond.
        let done = BULLET_TIME_HOLD_SECS + BULLET_TIME_RAMP_SECS;
        assert!((bullet_time_factor(done) - 1.0).abs() < 1e-9);
        assert!((bullet_time_factor(100.0) - 1.0).abs() < 1e-9);

        // Monotonic: time only speeds back up, never jumps backward.
        let mut prev = 0.0;
        for i in 0..200 {
            let f = bullet_time_factor(f64::from(i) * 0.01);
            assert!(f >= prev, "factor must not decrease (at {i})");
            prev = f;
        }
    }

    /// An App with a live simulation, as if the window had opened.
    fn test_app() -> App {
        let config = Config::try_resolve_from(["bouncy", "--seed", "7", "--mute"]).unwrap();
        let mut app = App::new(config.clone());
        app.sim = Some(Simulation::new(&config, 800, 600));
        app
    }

    #[test]
    fn relaunch_rebuilds_the_simulation_with_new_options() {
        let mut app = test_app();
        let before = app.sim.as_ref().unwrap().dimensions();
        app.apply_panel_command(PanelCommand::Relaunch {
            preset: Some("billiards".to_string()),
            particle_size: Some(3.0),
            initial_speed: Some(200.0),
            min_particles: Some(12),
        });
        assert!((app.config.particle_size - 3.0).abs() < 1e-9);
        assert!((app.config.initial_speed - 200.0).abs() < 1e-9);
        assert_eq!(app.config.preset.as_deref(), Some("billiards"));
        let sim = app.sim.as_ref().unwrap();
        assert_eq!(sim.dimensions(), before, "arena size preserved");
        assert!(
            sim.particles()
                .iter()
                .all(|p| (p.radius - 3.0).abs() < 1e-9),
            "explicit particle size overrides the preset bundle"
        );
        assert!(sim.particle_count() >= 12);

        // Session-delta preservation: touched live settings survive a
        // relaunch, untouched ones follow the new bundle.
        {
            let sim = app.sim.as_mut().unwrap();
            sim.gravity_percent = 250; // touched (config says 0 now)
            sim.matter = true; // touched
        }
        app.time_scale = 2.0; // touched (no config flag; baseline 1.0)
        app.apply_panel_command(PanelCommand::Relaunch {
            preset: Some("billiards".to_string()),
            particle_size: Some(2.0),
            initial_speed: Some(300.0),
            min_particles: None,
        });
        let sim = app.sim.as_ref().unwrap();
        assert_eq!(sim.gravity_percent, 250, "touched gravity survives");
        assert!(sim.matter, "touched matter survives");
        assert!((app.time_scale - 2.0).abs() < 1e-9, "time scale survives");
        assert_eq!(
            sim.spawn_mode,
            crate::config::SpawnMode::Off,
            "untouched spawn mode follows the billiards bundle"
        );
        assert_eq!(
            sim.explosion_threshold, 0,
            "untouched threshold follows the bundle"
        );

        // A bad configuration keeps the running simulation and config.
        app.apply_panel_command(PanelCommand::Relaunch {
            preset: Some("no-such-preset".to_string()),
            particle_size: Some(1.0),
            initial_speed: Some(600.0),
            min_particles: None,
        });
        assert!(
            (app.config.particle_size - 2.0).abs() < 1e-9,
            "failed relaunch leaves the old config in place"
        );
        assert!(app.sim.is_some());
    }

    #[test]
    fn changing_the_preset_drops_session_deltas_and_the_cycle_restarts() {
        let mut app = test_app();

        // Phase A: tweak, then relaunch with the same (no) preset —
        // adjustments travel.
        app.sim.as_mut().unwrap().gravity_percent = 250;
        app.apply_panel_command(PanelCommand::Relaunch {
            preset: None,
            particle_size: Some(2.0),
            initial_speed: None,
            min_particles: None,
        });
        assert_eq!(
            app.sim.as_ref().unwrap().gravity_percent,
            250,
            "same-preset relaunch preserves the touched gravity"
        );

        // Phase B: change the preset — the new bundle wins over stale
        // tweaks; untouched launch fields defer to the bundle too.
        app.sim.as_mut().unwrap().matter = true;
        app.time_scale = 2.0;
        app.apply_panel_command(PanelCommand::Relaunch {
            preset: Some("peace".to_string()),
            particle_size: None,
            initial_speed: None,
            min_particles: None,
        });
        let sim = app.sim.as_ref().unwrap();
        assert_eq!(sim.gravity_percent, 25, "peace's gravity takes precedence");
        assert!(!sim.matter, "stale matter tweak dropped");
        assert!(sim.flow, "peace's flow arrives");
        assert!(
            (app.time_scale - 1.0).abs() < 1e-9,
            "time scale returns to baseline"
        );
        assert!(
            (app.config.initial_speed - 40.0).abs() < 1e-9,
            "untouched initial speed defers to the bundle"
        );

        // Phase C: fresh tweaks against the new baseline travel through
        // a same-preset relaunch.
        app.sim.as_mut().unwrap().gravity_percent = 77;
        app.apply_panel_command(PanelCommand::Relaunch {
            preset: Some("peace".to_string()),
            particle_size: None,
            initial_speed: None,
            min_particles: Some(30),
        });
        let sim = app.sim.as_ref().unwrap();
        assert_eq!(sim.gravity_percent, 77, "fresh tweak survives");
        assert_eq!(app.config.min_particles, Some(30));
        assert!(
            sim.particle_count() < 90,
            "explicit min particles overrides peace's 90: {}",
            sim.particle_count()
        );
    }

    #[test]
    fn command_clamps_agree_with_the_cli_limits() {
        let mut app = test_app();

        // Slam every adjustable parameter into both of its rails; the
        // command clamps share their constants with the clap parsers, and
        // this pins the agreement: the CLI accepts exactly the hotkey
        // maximum and rejects one past it.
        for _ in 0..500 {
            app.apply(Command::AdjustGravity(GRAVITY_STEP));
        }
        assert_eq!(app.sim.as_ref().unwrap().gravity_percent, GRAVITY_LIMIT);
        for _ in 0..500 {
            app.apply(Command::AdjustGravity(-GRAVITY_STEP));
        }
        assert_eq!(app.sim.as_ref().unwrap().gravity_percent, -GRAVITY_LIMIT);
        let max = GRAVITY_LIMIT.to_string();
        assert!(Config::try_resolve_from(["bouncy", "--gravity", &max]).is_ok());
        let over = (GRAVITY_LIMIT + 1).to_string();
        assert!(Config::try_resolve_from(["bouncy", "--gravity", &over]).is_err());

        for _ in 0..100 {
            app.apply(Command::AdjustParticleElasticity(ELASTICITY_STEP));
            app.apply(Command::AdjustWallElasticity(ELASTICITY_STEP));
        }
        let sim = app.sim.as_ref().unwrap();
        assert_eq!(sim.particle_elasticity, ELASTICITY_MAX);
        assert_eq!(sim.wall_elasticity, ELASTICITY_MAX);
        let max = ELASTICITY_MAX.to_string();
        assert!(Config::try_resolve_from(["bouncy", "--wall-elasticity", &max]).is_ok());

        for _ in 0..500 {
            app.apply(Command::AdjustExplosionThreshold(THRESHOLD_STEP));
        }
        assert_eq!(
            app.sim.as_ref().unwrap().explosion_threshold,
            EXPLOSION_THRESHOLD_MAX
        );
        for _ in 0..500 {
            app.apply(Command::AdjustExplosionThreshold(-THRESHOLD_STEP));
        }
        assert_eq!(app.sim.as_ref().unwrap().explosion_threshold, 0);
    }

    #[test]
    fn commands_toggle_and_act_like_their_hotkeys() {
        let mut app = test_app();

        app.apply(Command::TogglePause);
        assert!(app.paused);
        app.apply(Command::StepFrame);
        assert!(app.step_once, "step-frame works while paused");
        app.apply(Command::TogglePause);
        assert!(!app.paused);

        app.apply(Command::ToggleTrails);
        assert!(app.trails);
        app.apply(Command::CycleColorMode);
        assert_eq!(app.color_mode, ColorMode::Velocity);
        app.apply(Command::ToggleKaleidoscope);
        assert!(app.kaleidoscope);

        app.apply(Command::PinWell(Polarity::Repel));
        assert_eq!(app.sim.as_ref().unwrap().pinned_wells().len(), 1);
        app.apply(Command::ClearWells);
        assert!(app.sim.as_ref().unwrap().pinned_wells().is_empty());

        app.apply(Command::ToggleFlow);
        assert!(app.sim.as_ref().unwrap().flow);

        app.apply_panel_command(PanelCommand::PlaceEmitter(100.0, 100.0));
        {
            let sim = app.sim.as_ref().unwrap();
            assert_eq!(sim.emitters().len(), 1, "panel tool places an emitter");
            let e = sim.emitters()[0];
            // Placed at (100, 100) in an 800x600 arena: the default aim
            // points at the center (400, 300), a (300, 200) vector.
            assert!(e.dx > 0.0 && e.dy > 0.0, "aims at the arena center");
            assert!((e.dx / e.dy - 1.5).abs() < 1e-9, "direction ratio 300:200");
        }
        app.apply(Command::ClearEmitters);
        assert!(app.sim.as_ref().unwrap().emitters().is_empty());

        assert_eq!(app.audio.ping_volume_percent(), 100, "default full");
        app.apply(Command::AdjustPingVolume(-30));
        assert_eq!(app.audio.ping_volume_percent(), 70);
        app.apply(Command::AdjustPingVolume(-100));
        assert_eq!(app.audio.ping_volume_percent(), 0, "clamps at silent");
        app.apply(Command::AdjustPingVolume(500));
        assert_eq!(app.audio.ping_volume_percent(), 100, "clamps at full");

        assert!(!app.sim.as_ref().unwrap().wall_chimes, "default off");
        app.apply(Command::ToggleWallChimes);
        assert!(app.sim.as_ref().unwrap().wall_chimes, "I arms the chimes");
        app.apply(Command::ToggleWallChimes);
        assert!(!app.sim.as_ref().unwrap().wall_chimes);

        let before = app.sim.as_ref().unwrap().particle_count();
        app.apply(Command::LaunchComet);
        assert_eq!(
            app.sim.as_ref().unwrap().particle_count(),
            before + 1,
            "J launches a comet"
        );
        app.apply(Command::ToggleSelfGravity);
        assert!(app.sim.as_ref().unwrap().self_gravity);
        app.apply(Command::Reset);
        assert!(!app.paused);
    }

    #[test]
    fn time_scale_steps_retrace_to_the_starting_point() {
        // Regression: the old multiplicative step (x1.25) could never
        // land back on 1.0 after clamping at either end of the range.
        let config = Config::try_resolve_from(["bouncy", "--mute"]).unwrap();
        let mut app = App::new(config);
        assert_eq!(app.time_scale, 1.0);

        for _ in 0..10 {
            app.adjust_time_scale(TIME_SCALE_STEP);
        }
        for _ in 0..10 {
            app.adjust_time_scale(-TIME_SCALE_STEP);
        }
        assert_eq!(app.time_scale, 1.0, "up then down retraces exactly");

        // Slamming into the floor keeps the value on the grid, so 1.0 is
        // still reachable afterward — the reported bug.
        for _ in 0..200 {
            app.adjust_time_scale(-TIME_SCALE_STEP);
        }
        assert_eq!(app.time_scale, TIME_SCALE_MIN, "clamped at the floor");
        for _ in 0..18 {
            app.adjust_time_scale(TIME_SCALE_STEP);
        }
        assert_eq!(app.time_scale, 1.0, "floor back to 1.0 in 18 steps");

        for _ in 0..200 {
            app.adjust_time_scale(TIME_SCALE_STEP);
        }
        assert_eq!(app.time_scale, TIME_SCALE_MAX, "clamped at the ceiling");
    }

    #[test]
    fn cursor_visibility_policy() {
        let recently = CURSOR_HIDE_DELAY - 0.1;
        let idle = CURSOR_HIDE_DELAY + 0.1;

        // Hidden only when focused, inside, well off, and idle.
        assert!(!cursor_should_be_visible(true, true, false, idle));

        // Any of these forces visibility.
        assert!(cursor_should_be_visible(true, true, false, recently)); // moving
        assert!(cursor_should_be_visible(true, true, true, idle)); // well held or wall drawing
        assert!(cursor_should_be_visible(false, true, false, idle)); // unfocused
        assert!(cursor_should_be_visible(true, false, false, idle)); // outside window
    }

    #[test]
    fn select_click_sets_and_empty_click_clears_selection() {
        let mut app = test_app();
        {
            let sim = app.sim.as_mut().unwrap();
            assert!(sim.place_emitter(100.0, 100.0, 1.0, 0.0, 2.0, 12));
        }
        let id = app.sim.as_ref().unwrap().emitters()[0].id;
        app.apply_panel_command(PanelCommand::SelectAt(103.0, 100.0));
        assert_eq!(app.selection, Some(Selection::Emitter(id)));
        // A miss deselects — and select mode never bursts.
        let before = app.sim.as_ref().unwrap().particle_count();
        app.apply_panel_command(PanelCommand::SelectAt(700.0, 500.0));
        assert_eq!(app.selection, None);
        assert_eq!(
            app.sim.as_ref().unwrap().particle_count(),
            before,
            "no burst from a select-mode miss"
        );
        app.apply_panel_command(PanelCommand::Deselect);
        assert_eq!(app.selection, None);
    }

    #[test]
    fn delete_commands_clear_their_selection_and_flash() {
        let mut app = test_app();
        {
            let sim = app.sim.as_mut().unwrap();
            assert!(sim.place_emitter(100.0, 100.0, 1.0, 0.0, 2.0, 12));
            assert!(sim.add_wall_segment(200.0, 300.0, 300.0, 300.0));
        }
        let eid = app.sim.as_ref().unwrap().emitters()[0].id;
        let stroke = app.sim.as_ref().unwrap().wall_meta()[0].stroke;

        app.selection = Some(Selection::Emitter(eid));
        app.apply_panel_command(PanelCommand::DeleteEmitter(eid));
        assert_eq!(
            app.selection, None,
            "deleting the inspected emitter deselects"
        );
        assert!(app.sim.as_ref().unwrap().emitters().is_empty());

        app.selection = Some(Selection::WallStroke(stroke));
        app.wall_flash.insert(stroke, 1.0);
        app.apply_panel_command(PanelCommand::DeleteStroke(stroke));
        assert_eq!(
            app.selection, None,
            "deleting the inspected stroke deselects"
        );
        assert!(app.sim.as_ref().unwrap().wall_segments().is_empty());
        assert!(
            app.wall_flash.is_empty(),
            "flash entry dies with the stroke"
        );
    }

    #[test]
    fn selection_sweep_drops_dead_ids_at_the_next_frame() {
        let mut app = test_app();
        {
            let sim = app.sim.as_mut().unwrap();
            assert!(sim.add_wall_segment(200.0, 300.0, 300.0, 300.0));
        }
        let stroke = app.sim.as_ref().unwrap().wall_meta()[0].stroke;
        app.selection = Some(Selection::WallStroke(stroke));
        // A clear-all bypasses the delete commands' own deselect...
        app.apply(Command::ClearWalls);
        // ...so the per-frame sweep catches the dead id.
        app.simulate(0.016, Instant::now());
        assert_eq!(app.selection, None, "dead id swept at the next frame");
    }

    #[test]
    fn aim_emitter_at_points_the_stream_at_the_click() {
        let mut app = test_app();
        {
            let sim = app.sim.as_mut().unwrap();
            assert!(sim.place_emitter(100.0, 100.0, 0.0, -1.0, 2.0, 12));
        }
        let id = app.sim.as_ref().unwrap().emitters()[0].id;
        app.apply_panel_command(PanelCommand::AimEmitterAt(id, 300.0, 100.0));
        let e = *app.sim.as_ref().unwrap().emitter(id).unwrap();
        assert!(
            (e.dx - 1.0).abs() < 1e-9 && e.dy.abs() < 1e-9,
            "aimed from the emitter toward the click"
        );
        // A dead id is a no-op, never a panic.
        app.apply_panel_command(PanelCommand::AimEmitterAt(id + 1, 0.0, 0.0));
    }

    #[test]
    fn cycle_stroke_note_steps_the_setting() {
        let mut app = test_app();
        {
            let sim = app.sim.as_mut().unwrap();
            assert!(sim.add_wall_segment(200.0, 300.0, 300.0, 300.0));
        }
        let stroke = app.sim.as_ref().unwrap().wall_meta()[0].stroke;
        app.apply_panel_command(PanelCommand::CycleStrokeNote(stroke));
        assert_eq!(
            app.sim.as_ref().unwrap().stroke_note_setting(stroke),
            Some(crate::presets::WallNote::Note(0)),
            "Auto steps to the first pinned degree"
        );
    }
}
