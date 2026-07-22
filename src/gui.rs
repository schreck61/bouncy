// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! The native control panel: an immediate-mode GUI drawn directly into
//! the RGBA frame buffer, the same way the HUD draws. No UI toolkit —
//! the widget set is small and bounded (sliders, toggles, buttons,
//! headers, a scrollbar), and owning it keeps the panel working on both
//! render backends (GPU and CPU), keeps the pixel aesthetic consistent
//! with the bitmap HUD, and makes the roadmap's interaction details
//! (snap-to detents, translucency, the edge-reveal handle) first-class
//! rather than fights with a toolkit's styling.
//!
//! The panel is just another control surface: widgets emit the same
//! [`PanelCommand`]s the web demo's HTML panel sends, entering the app
//! at `apply_panel_command` with identical clamping and semantics.
//!
//! Layout is immediate-mode: rebuilt every frame from [`PanelState`] by
//! `layout`, used once for interaction and once for drawing, so
//! hit-testing and pixels can never disagree.

use crate::app::{Command, PanelCommand};
use crate::presets::WallNote;
use crate::sim::Polarity;
use crate::text::draw_text;

/// Panel width in simulation pixels.
pub const PANEL_WIDTH: f64 = 200.0;
/// Inner padding between the panel edge and its content.
const PAD: f64 = 10.0;
/// Snap radius for slider detents, as a fraction of the track's length
/// (track space, not value space: magnified sliders must snap uniformly).
const DETENT_FRACTION: f64 = 0.025;
/// Scroll wheel speed, pixels per line.
const WHEEL_STEP: f64 = 24.0;
/// Slide animation rate (1/s): the exponential approach constant for the
/// panel's settle — critically-damped feel, no bounce.
const SLIDE_RATE: f64 = 14.0;
/// Width of the strip along the right edge that arms the reveal handle.
const REVEAL_STRIP: f64 = 6.0;
/// How long the cursor must dwell in the strip before the handle shows —
/// quick passes (drawing a wall to the edge) never see it.
const HANDLE_DWELL_SECS: f64 = 0.15;
/// Handle geometry: a thin vertical pill at mid-height.
const HANDLE_W: f64 = 5.0;
const HANDLE_H: f64 = 56.0;
/// Handle fade rate (1/s).
const HANDLE_FADE_RATE: f64 = 10.0;
/// Cursor idle time after which the handle fades with the cursor.
const HANDLE_IDLE_SECS: f64 = 2.0;
/// Press-to-release travel under this is a click (toggle), not a drag.
const CLICK_SLOP: f64 = 3.0;

/// Gravity spans the CLI's full ±1000%, but everyday play lives in
/// -100..100 — so that band gets 60% of the track (every 10% step is
/// several pixels) and the extremes compress into the outer fifths. A
/// linear ±1000 track put 0 and 100 nine pixels apart and the detents
/// swallowed everything between them.
const GRAVITY_POINTS: &[(f64, f64)] = &[(-1000.0, 0.0), (-100.0, 0.2), (100.0, 0.8), (1000.0, 1.0)];
const ELASTICITY_POINTS: &[(f64, f64)] = &[(0.0, 0.0), (1.5, 1.0)];
const TIME_POINTS: &[(f64, f64)] = &[(0.1, 0.0), (4.0, 1.0)];
/// Same treatment as gravity: thresholds people actually set (0-100
/// births/s) get most of the track.
const THRESHOLD_POINTS: &[(f64, f64)] = &[(0.0, 0.0), (100.0, 0.6), (1000.0, 1.0)];
/// Ping volume is a plain linear percent.
const PING_VOLUME_POINTS: &[(f64, f64)] = &[(0.0, 0.0), (100.0, 1.0)];
/// Launch-option sliders (draft values; applied by Apply & relaunch).
const SIZE_POINTS: &[(f64, f64)] = &[(0.5, 0.0), (10.0, 1.0)];
/// Everyday speeds (up to 1000) get most of the track.
const SPEED_POINTS: &[(f64, f64)] = &[(10.0, 0.0), (1000.0, 0.75), (2000.0, 1.0)];
/// 0 means "auto" (sized from the window), drawn as such.
const MIN_PARTICLES_POINTS: &[(f64, f64)] = &[(0.0, 0.0), (100.0, 1.0)];
/// Inspector sliders for the selected emitter. Musical rates (a few
/// notes a second) get most of the track; machine-gun speeds compress
/// into the top fifth. Bounds match the scene TOML validation.
const EMITTER_RATE_POINTS: &[(f64, f64)] = &[(0.1, 0.0), (5.0, 0.7), (20.0, 1.0)];
/// Everyday caps (a hand of live particles) get most of the track.
const EMITTER_CAP_POINTS: &[(f64, f64)] = &[(1.0, 0.0), (30.0, 0.7), (200.0, 1.0)];
/// Built-in presets the launch section cycles through; index 0 is
/// "none". Derived from the preset enum itself — the same source the
/// web panel's dropdown uses — so the list can never drift.
fn preset_names() -> &'static [&'static str] {
    use clap::ValueEnum;
    static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names = vec!["none"];
        names.extend(
            crate::presets::Preset::value_variants()
                .iter()
                .map(|p| p.label()),
        );
        names
    })
}

/// Identity of a value slider.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SliderId {
    Gravity,
    ParticleElasticity,
    WallElasticity,
    TimeScale,
    ExplosionThreshold,
    PingVolume,
    LaunchSize,
    LaunchSpeed,
    LaunchMinParticles,
    /// Inspector: the selected emitter's emission rate.
    EmitterRate,
    /// Inspector: the selected emitter's live-particle cap.
    EmitterCap,
}

/// Identity of an on/off row.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ToggleId {
    Matter,
    Flow,
    SelfGravity,
    Trails,
    Kaleidoscope,
    Music,
    WallChimes,
    Mute,
}

/// Identity of a push button.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ButtonId {
    PauseResume,
    StepFrame,
    Reset,
    CycleSpawn,
    CycleColor,
    CycleHud,
    Burst,
    Comet,
    Explode,
    Screenshot,
    PinWell,
    PinRepeller,
    PlaceEmitter,
    ClearWells,
    ClearWalls,
    ClearEmitters,
    ExportScene,
    CyclePreset,
    Relaunch,
    /// Arms select-on-click, the panel twin of hold-D.
    Select,
    /// Arms re-aim: the next arena click points the selected emitter.
    ReAim,
    /// Steps the selected stroke's note: Auto → degrees → Silent.
    CycleStrokeNote,
    /// Deletes the selected entity (either kind).
    DeleteSelected,
}

/// The selected entity's live values, pulled fresh by id each frame so
/// the inspector can never show a dead or stale entity.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PanelSelection {
    Emitter {
        id: u32,
        rate: f64,
        cap: usize,
        /// Aim as compass degrees (0 = up), the scene-export convention.
        angle_deg: f64,
    },
    Stroke {
        id: u32,
        segments: usize,
        note: WallNote,
    },
}

/// Everything the panel shows or edits, snapshotted once per frame by
/// the app (the native analogue of the web `Snapshot`).
#[derive(Clone, Default)]
pub struct PanelState {
    pub fps: f64,
    pub particles: usize,
    pub max_particles: usize,
    pub wells: usize,
    pub walls: usize,
    pub emitters: usize,
    pub paused: bool,
    pub gravity: i32,
    pub particle_elasticity: f64,
    pub wall_elasticity: f64,
    pub time_scale: f64,
    pub explosion_threshold: i32,
    pub matter: bool,
    pub flow: bool,
    pub self_gravity: bool,
    pub trails: bool,
    pub kaleidoscope: bool,
    pub music: bool,
    pub wall_chimes: bool,
    pub muted: bool,
    pub ping_volume: i32,
    pub spawn_mode: String,
    pub color_mode: String,
    pub hud: String,
    /// Construction-time values from the current config, used once to
    /// seed the launch draft.
    pub launch_particle_size: f64,
    pub launch_initial_speed: f64,
    pub launch_min_particles: Option<u32>,
    pub launch_preset: String,
    /// The inspected entity, if any (None hides the selected section).
    pub selection: Option<PanelSelection>,
}

/// An axis-aligned rectangle in frame coordinates.
#[derive(Copy, Clone, Debug, Default)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl Rect {
    fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }
}

/// One laid-out control: what it is plus where it landed this frame.
enum Item {
    Header(&'static str),
    Readout(String),
    Slider {
        id: SliderId,
        label: &'static str,
        /// Piecewise-linear (value, track-t) control points, ascending in
        /// both coordinates, spanning t = 0..1. A plain linear slider is
        /// two points; a magnified slider gives its everyday band most of
        /// the track and compresses the extremes into the ends.
        points: &'static [(f64, f64)],
        value: f64,
        detents: &'static [f64],
        text: String,
        track: Rect,
    },
    Toggle {
        id: ToggleId,
        label: &'static str,
        on: bool,
        hit: Rect,
    },
    Button {
        id: ButtonId,
        label: String,
        hit: Rect,
    },
}

/// A control plus the full row rectangle it occupies.
struct Laid {
    item: Item,
    row: Rect,
}

/// Pending launch options: edited by the launch sliders, applied only
/// by the Apply & relaunch button (they are construction-time values,
/// not live ones).
#[derive(Clone, Debug)]
pub struct LaunchDraft {
    pub preset_idx: usize,
    pub particle_size: f64,
    pub initial_speed: f64,
    /// 0 = auto (sized from the window).
    pub min_particles: f64,
    /// Which fields the user edited this session: untouched fields are
    /// omitted from the relaunch so the preset (or default) decides,
    /// matching the web's empty placeholder semantics.
    pub size_touched: bool,
    pub speed_touched: bool,
    pub min_touched: bool,
}

impl Default for LaunchDraft {
    fn default() -> Self {
        LaunchDraft {
            preset_idx: 0,
            particle_size: 1.5,
            initial_speed: 600.0,
            min_particles: 0.0,
            size_touched: false,
            speed_touched: false,
            min_touched: false,
        }
    }
}

/// Pointer and wheel input accumulated from window events since the
/// last tick. Edges (pressed/released) are consumed by the tick.
#[derive(Default)]
struct PanelInput {
    pressed: bool,
    released: bool,
    wheel: f64,
}

/// The native control panel's retained state.
pub struct Gui {
    /// Whether the panel is requested open (Tab or the edge handle).
    open: bool,
    /// Slide progress, 0 = fully hidden, 1 = fully out; animated toward
    /// `open` each tick.
    slide: f64,
    scroll: f64,
    content_height: f64,
    dragging: Option<SliderId>,
    pressed_button: Option<ButtonId>,
    hover: Option<Rect>,
    input: PanelInput,
    /// Cursor position in frame coordinates (mirrors the app's cursor).
    cursor: (f64, f64),
    /// Edge-reveal handle visibility (0..1, animated).
    handle_alpha: f64,
    /// How long the cursor has dwelt in the right-edge reveal strip.
    edge_dwell: f64,
    /// The handle is currently grabbed.
    handle_drag: bool,
    /// Cursor x when the handle was grabbed, and whether it has moved
    /// beyond the click slop (drag) or not (click).
    handle_grab_x: f64,
    handle_moved: bool,
    /// A one-shot placement tool armed by an action button: the next
    /// arena click places it (web-panel semantics; a second press or
    /// Esc cancels).
    armed: Option<ButtonId>,
    /// The emitter the armed Re-aim tool will point, captured when the
    /// tool arms — `place_armed` has no `PanelState` to consult, and the
    /// selection could change between arm and click.
    reaim_target: Option<u32>,
    /// Pending launch options; seeded from the config on first tick.
    launch: LaunchDraft,
    launch_seeded: bool,
}

impl Default for Gui {
    fn default() -> Self {
        Gui {
            open: false,
            slide: 0.0,
            scroll: 0.0,
            content_height: 0.0,
            dragging: None,
            pressed_button: None,
            hover: None,
            input: PanelInput::default(),
            cursor: (-1.0, -1.0),
            handle_alpha: 0.0,
            edge_dwell: 0.0,
            handle_drag: false,
            handle_grab_x: 0.0,
            handle_moved: false,
            armed: None,
            reaim_target: None,
            launch: LaunchDraft::default(),
            launch_seeded: false,
        }
    }
}

impl Gui {
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle the panel (Tab).
    pub fn toggle_open(&mut self) {
        self.open = !self.open;
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Record the cursor position (frame coordinates).
    pub fn set_cursor(&mut self, x: f64, y: f64) {
        self.cursor = (x, y);
    }

    /// Whether a placement tool is currently armed.
    pub fn is_armed(&self) -> bool {
        self.armed.is_some()
    }

    /// Cancel the armed placement tool (Esc, or the app's discretion).
    pub fn disarm(&mut self) {
        self.armed = None;
        self.reaim_target = None;
    }

    /// The relaunch command carrying the current draft: only fields the
    /// user touched travel; the rest defer to the preset or default.
    fn relaunch_command(&self) -> PanelCommand {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        PanelCommand::Relaunch {
            preset: (self.launch.preset_idx > 0)
                .then(|| preset_names()[self.launch.preset_idx].to_string()),
            particle_size: self
                .launch
                .size_touched
                .then_some(self.launch.particle_size),
            initial_speed: self
                .launch
                .speed_touched
                .then_some(self.launch.initial_speed),
            min_particles: (self.launch.min_touched && self.launch.min_particles >= 1.5)
                .then(|| (self.launch.min_particles.round() as u32).max(2)),
        }
    }

    /// Forget the draft so the next tick re-seeds it from the (new)
    /// running config — called after a successful relaunch, so the
    /// touched/untouched cycle restarts from the fresh baseline.
    pub fn reseed_launch(&mut self) {
        self.launch = LaunchDraft::default();
        self.launch_seeded = false;
    }

    /// Consume the armed tool with an arena click at `(x, y)`: returns
    /// the placement command and disarms.
    pub fn place_armed(&mut self, x: f64, y: f64) -> Option<PanelCommand> {
        let tool = self.armed.take()?;
        Some(match tool {
            ButtonId::Burst => PanelCommand::SpawnBurst(x, y),
            ButtonId::Comet => PanelCommand::LaunchComet(x, y),
            ButtonId::Explode => PanelCommand::TriggerExplosion(x, y),
            ButtonId::PinWell => PanelCommand::PinWell(x, y, Polarity::Attract),
            ButtonId::PinRepeller => PanelCommand::PinWell(x, y, Polarity::Repel),
            ButtonId::PlaceEmitter => PanelCommand::PlaceEmitter(x, y),
            ButtonId::Select => PanelCommand::SelectAt(x, y),
            ButtonId::ReAim => PanelCommand::AimEmitterAt(self.reaim_target.take()?, x, y),
            // Non-placement ids never arm.
            _ => return None,
        })
    }

    /// The panel's left edge for the current slide progress. Linear in
    /// `slide`: the exponential approach in `tick` supplies the settle,
    /// and a grabbed handle must track the cursor exactly.
    fn panel_x(&self, width: u32) -> f64 {
        f64::from(width) - PANEL_WIDTH * self.slide
    }

    /// The reveal handle's rectangle: a thin pill hugging the panel's
    /// left edge (the window's right edge while the panel is hidden).
    fn handle_rect(&self, width: u32, height: u32) -> Rect {
        Rect {
            x: self.panel_x(width) - HANDLE_W,
            y: f64::from(height) / 2.0 - HANDLE_H / 2.0,
            w: HANDLE_W,
            h: HANDLE_H,
        }
    }

    /// Whether the panel currently owns the pointer: over the visible
    /// panel, or mid-interaction (a drag must not leak to the sim even
    /// if the cursor strays off the panel).
    pub fn wants_pointer(&self, width: u32) -> bool {
        if self.dragging.is_some() || self.pressed_button.is_some() || self.handle_drag {
            return true;
        }
        if self.handle_alpha > 0.3 && self.cursor.0 >= self.panel_x(width) - HANDLE_W {
            return true;
        }
        self.slide > 0.0 && self.cursor.0 >= self.panel_x(width)
    }

    /// A mouse press: returns true when the panel consumes it. A press
    /// on the visible handle grabs it (drag decides open/closed; a
    /// no-travel release toggles); `height` locates the handle.
    pub fn on_press_at(&mut self, width: u32, height: u32) -> bool {
        if self.handle_alpha > 0.3
            && self
                .handle_rect(width, height)
                .contains(self.cursor.0, self.cursor.1)
        {
            self.handle_drag = true;
            self.handle_grab_x = self.cursor.0;
            self.handle_moved = false;
            return true;
        }
        if self.wants_pointer(width) {
            self.input.pressed = true;
            true
        } else {
            false
        }
    }

    /// A mouse release always reaches the panel (it ends drags).
    pub fn on_release(&mut self) {
        self.input.released = true;
    }

    /// A scroll wheel tick (positive = scroll down): returns true when
    /// the panel consumes it.
    pub fn on_wheel(&mut self, lines: f64, width: u32) -> bool {
        if self.wants_pointer(width) {
            self.input.wheel += lines;
            true
        } else {
            false
        }
    }

    /// Advance animation, resolve interactions against this frame's
    /// layout, and emit the resulting panel commands.
    pub fn tick(
        &mut self,
        dt: f64,
        state: &PanelState,
        width: u32,
        height: u32,
        shift: bool,
        cursor_idle: f64,
    ) -> Vec<PanelCommand> {
        let dt = dt.max(0.0);
        let (cx0, cy0) = self.cursor;
        let w_f = f64::from(width);

        // Edge dwell arms the reveal handle; quick passes never see it.
        if cx0 >= w_f - REVEAL_STRIP && cy0 >= 0.0 {
            self.edge_dwell += dt;
        } else {
            self.edge_dwell = 0.0;
        }
        let handle_visible = self.handle_drag
            || self.open
            || self.slide > 0.0
            || (self.edge_dwell >= HANDLE_DWELL_SECS && cursor_idle < HANDLE_IDLE_SECS);
        let handle_target = if handle_visible { 1.0 } else { 0.0 };
        self.handle_alpha +=
            (handle_target - self.handle_alpha) * (1.0 - (-HANDLE_FADE_RATE * dt).exp());
        if (self.handle_alpha - handle_target).abs() < 0.001 {
            self.handle_alpha = handle_target;
        }

        // A grabbed handle drives the slide directly and decides
        // open/closed by where it is let go.
        if self.handle_drag {
            if (cx0 - self.handle_grab_x).abs() > CLICK_SLOP {
                self.handle_moved = true;
            }
            if self.handle_moved {
                self.slide = ((w_f - cx0 - HANDLE_W / 2.0) / PANEL_WIDTH).clamp(0.0, 1.0);
                self.open = self.slide >= 0.5;
            }
            if self.input.released {
                self.input.released = false;
                if !self.handle_moved {
                    // A clean click toggles.
                    self.open = !self.open;
                }
                self.handle_drag = false;
            }
        }

        // Exponential approach: critically-damped settle, no bounce.
        if !self.handle_drag || !self.handle_moved {
            let target = if self.open { 1.0 } else { 0.0 };
            self.slide += (target - self.slide) * (1.0 - (-SLIDE_RATE * dt).exp());
            if (self.slide - target).abs() < 0.001 {
                self.slide = target;
            }
        }

        let mut commands = Vec::new();
        if self.slide <= 0.0 {
            // Hidden: drop stale interaction state and swallow edges.
            // The armed placement tool deliberately survives — hiding
            // the panel to place under where it sat is a legitimate
            // gesture (Esc or a second button press still cancels).
            self.dragging = None;
            self.pressed_button = None;
            self.hover = None;
            self.input = PanelInput::default();
            return commands;
        }

        if !self.launch_seeded {
            self.launch_seeded = true;
            self.launch.particle_size = state.launch_particle_size;
            self.launch.initial_speed = state.launch_initial_speed;
            self.launch.min_particles = state.launch_min_particles.map_or(0.0, f64::from);
            self.launch.preset_idx = preset_names()
                .iter()
                .position(|n| *n == state.launch_preset)
                .unwrap_or(0);
        }

        let (layout, content_height) =
            layout(state, &self.launch, self.panel_x(width), self.scroll);
        self.content_height = content_height;
        let (cx, cy) = self.cursor;

        // Wheel scrolls the content, clamped to what exists.
        if self.input.wheel != 0.0 {
            self.scroll = (self.scroll + self.input.wheel * WHEEL_STEP)
                .clamp(0.0, (self.content_height - f64::from(height)).max(0.0));
            self.input.wheel = 0.0;
        }

        self.hover = None;
        for laid in &layout {
            if laid.row.contains(cx, cy) {
                match &laid.item {
                    Item::Header(_) | Item::Readout(_) => {}
                    _ => self.hover = Some(laid.row),
                }
            }
        }

        if self.input.pressed {
            self.input.pressed = false;
            for laid in &layout {
                if !laid.row.contains(cx, cy) {
                    continue;
                }
                match &laid.item {
                    Item::Slider { id, .. } => self.dragging = Some(*id),
                    Item::Toggle { id, .. } => commands.push(toggle_command(*id)),
                    Item::Button { id, .. } => self.pressed_button = Some(*id),
                    _ => {}
                }
            }
        }

        // An active slider drag tracks the cursor continuously.
        if let Some(drag) = self.dragging {
            for laid in &layout {
                if let Item::Slider {
                    id,
                    points,
                    value,
                    detents,
                    track,
                    ..
                } = &laid.item
                {
                    if *id != drag {
                        continue;
                    }
                    let raw = slider_value_at(cx, track, points);
                    let snapped = snap_to_detents(raw, detents, points, shift);
                    match drag {
                        SliderId::LaunchSize => {
                            self.launch.particle_size = (snapped * 2.0).round() / 2.0;
                            self.launch.size_touched = true;
                        }
                        SliderId::LaunchSpeed => {
                            self.launch.initial_speed = (snapped / 10.0).round() * 10.0;
                            self.launch.speed_touched = true;
                        }
                        SliderId::LaunchMinParticles => {
                            self.launch.min_particles = snapped.round();
                            self.launch.min_touched = true;
                        }
                        // Inspector sliders carry the selected id; a
                        // selection that died mid-drag emits nothing.
                        SliderId::EmitterRate => {
                            if let Some(PanelSelection::Emitter { id, .. }) = state.selection {
                                let rate = (snapped * 10.0).round() / 10.0;
                                if (rate - value).abs() > 1e-9 {
                                    commands.push(PanelCommand::SetEmitterRate(id, rate));
                                }
                            }
                        }
                        SliderId::EmitterCap => {
                            if let Some(PanelSelection::Emitter { id, .. }) = state.selection {
                                #[allow(clippy::cast_possible_truncation)]
                                let cap = snapped.round() as i32;
                                if (f64::from(cap) - value).abs() > 1e-9 {
                                    commands.push(PanelCommand::SetEmitterCap(id, cap));
                                }
                            }
                        }
                        _ => {
                            if (snapped - value).abs() > 1e-9 {
                                commands.push(slider_command(drag, snapped));
                            }
                        }
                    }
                }
            }
        }

        if self.input.released {
            self.input.released = false;
            self.dragging = None;
            if let Some(pressed) = self.pressed_button.take() {
                let over = layout.iter().any(|laid| {
                    matches!(&laid.item, Item::Button { id, .. } if *id == pressed)
                        && laid.row.contains(cx, cy)
                });
                if over {
                    if is_placement(pressed) {
                        // Arm (or, pressed again, cancel) the one-shot
                        // placement tool — the arena click does the work.
                        // Re-aim captures its target now: the selection
                        // could change between arm and click.
                        if self.armed == Some(pressed) {
                            self.disarm();
                        } else if pressed == ButtonId::ReAim {
                            if let Some(PanelSelection::Emitter { id, .. }) = state.selection {
                                self.armed = Some(pressed);
                                self.reaim_target = Some(id);
                            }
                        } else {
                            self.armed = Some(pressed);
                            self.reaim_target = None;
                        }
                    } else if pressed == ButtonId::DeleteSelected {
                        match state.selection {
                            Some(PanelSelection::Emitter { id, .. }) => {
                                commands.push(PanelCommand::DeleteEmitter(id));
                            }
                            Some(PanelSelection::Stroke { id, .. }) => {
                                commands.push(PanelCommand::DeleteStroke(id));
                            }
                            None => {}
                        }
                    } else if pressed == ButtonId::CycleStrokeNote {
                        if let Some(PanelSelection::Stroke { id, .. }) = state.selection {
                            commands.push(PanelCommand::CycleStrokeNote(id));
                        }
                    } else if pressed == ButtonId::CyclePreset {
                        self.launch.preset_idx =
                            (self.launch.preset_idx + 1) % preset_names().len();
                    } else if pressed == ButtonId::Relaunch {
                        commands.push(self.relaunch_command());
                    } else {
                        commands.push(button_command(pressed, state));
                    }
                }
            }
        }

        commands
    }

    /// Draw the panel over the finished frame. Uses the same layout the
    /// tick resolved interactions against.
    pub fn draw(&self, frame: &mut [u8], state: &PanelState, width: u32, height: u32) {
        // The reveal handle draws whenever it has any presence — panel
        // open or closed.
        if self.handle_alpha > 0.01 {
            let hr = self.handle_rect(width, height);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let a = (self.handle_alpha * 225.0) as u8;
            fill_rect(frame, width, height, hr, [205, 210, 220], a);
            // Grip nicks so the pill reads as grabbable.
            for k in 0..3 {
                let ny = hr.y + hr.h / 2.0 - 8.0 + f64::from(k) * 8.0;
                hline(
                    frame,
                    width,
                    height,
                    hr.x + 1.0,
                    hr.x + hr.w - 1.0,
                    ny,
                    [90, 96, 108, 255],
                );
            }
        }
        // An armed placement tool follows the cursor with its hint —
        // panel open or hidden (arming and then hiding the panel to
        // place beneath where it sat is a supported gesture).
        if let Some(tool) = self.armed {
            let (cx, cy) = self.cursor;
            if cx >= 0.0 && cy >= 0.0 {
                draw_text(
                    frame,
                    width,
                    height,
                    placement_label(tool),
                    11.0,
                    to_f32(cx + 12.0, cy + 12.0),
                    [200, 215, 240],
                );
            }
        }
        if self.slide <= 0.0 {
            return;
        }
        let x0 = self.panel_x(width);
        let panel = Rect {
            x: x0,
            y: 0.0,
            w: PANEL_WIDTH,
            h: f64::from(height),
        };
        // Translucent backdrop: the show never stops behind the panel.
        fill_rect(frame, width, height, panel, [12, 14, 18], 215);
        vline(
            frame,
            width,
            height,
            x0,
            0.0,
            f64::from(height),
            [70, 80, 95, 255],
        );

        let (layout, _) = layout(state, &self.launch, x0, self.scroll);
        for laid in &layout {
            let hovered = self
                .hover
                .is_some_and(|h| (h.y - laid.row.y).abs() < 0.5 && laid.row.contains_rect(&h));
            self.draw_item(frame, width, height, laid, hovered);
        }

        // Scrollbar sliver when the content overflows.
        let view_h = f64::from(height);
        if self.content_height > view_h {
            let track_x = f64::from(width) - 3.0;
            let knob_h = (view_h / self.content_height * view_h).max(20.0);
            let knob_y = self.scroll / (self.content_height - view_h) * (view_h - knob_h);
            fill_rect(
                frame,
                width,
                height,
                Rect {
                    x: track_x,
                    y: knob_y,
                    w: 2.0,
                    h: knob_h,
                },
                [120, 130, 145],
                200,
            );
        }
    }

    fn draw_item(&self, frame: &mut [u8], width: u32, height: u32, laid: &Laid, hovered: bool) {
        let row = laid.row;
        match &laid.item {
            Item::Header(title) => {
                draw_text(
                    frame,
                    width,
                    height,
                    title,
                    13.0,
                    to_f32(row.x, row.y + 3.0),
                    [140, 190, 255],
                );
                let rule_y = row.y + row.h - 3.0;
                hline(
                    frame,
                    width,
                    height,
                    row.x,
                    row.x + row.w,
                    rule_y,
                    [60, 70, 85, 255],
                );
            }
            Item::Readout(text) => {
                draw_text(
                    frame,
                    width,
                    height,
                    text,
                    12.0,
                    to_f32(row.x, row.y + 1.0),
                    [185, 190, 198],
                );
            }
            Item::Slider {
                label,
                points,
                value,
                detents,
                text,
                track,
                ..
            } => {
                draw_text(
                    frame,
                    width,
                    height,
                    label,
                    12.0,
                    to_f32(row.x, row.y),
                    [210, 213, 220],
                );
                let (tw, _) = crate::text::measure_text(text, 11.0);
                draw_text(
                    frame,
                    width,
                    height,
                    text,
                    11.0,
                    to_f32(row.x + row.w - f64::from(tw), row.y + 0.5),
                    [150, 160, 172],
                );
                // Track, detent ticks, filled portion, then the knob.
                fill_rect(frame, width, height, *track, [55, 60, 70], 255);
                for d in *detents {
                    let dx = track.x + value_to_t(points, *d) * track.w;
                    vline(
                        frame,
                        width,
                        height,
                        dx,
                        track.y - 2.0,
                        track.y + track.h + 2.0,
                        [130, 140, 155, 255],
                    );
                }
                let t = value_to_t(points, *value).clamp(0.0, 1.0);
                let filled = Rect {
                    w: track.w * t,
                    ..*track
                };
                fill_rect(frame, width, height, filled, [90, 140, 220], 255);
                let knob = Rect {
                    x: track.x + track.w * t - 2.0,
                    y: track.y - 3.0,
                    w: 4.0,
                    h: track.h + 6.0,
                };
                fill_rect(frame, width, height, knob, [235, 238, 245], 255);
            }
            Item::Toggle { label, on, hit, .. } => {
                if hovered {
                    fill_rect(frame, width, height, row, [255, 255, 255], 14);
                }
                draw_text(
                    frame,
                    width,
                    height,
                    label,
                    12.0,
                    to_f32(row.x, row.y + 2.0),
                    [210, 213, 220],
                );
                // A capsule switch: border pill, inset fill pill, and a
                // round knob that slides right when on.
                let (fill, border, knob) = if *on {
                    ([90, 140, 220], [120, 165, 235], [240, 244, 250])
                } else {
                    ([40, 44, 52], [90, 96, 108], [140, 146, 158])
                };
                fill_pill(frame, width, height, *hit, border, 255);
                let inner = Rect {
                    x: hit.x + 1.0,
                    y: hit.y + 1.0,
                    w: hit.w - 2.0,
                    h: hit.h - 2.0,
                };
                fill_pill(frame, width, height, inner, fill, 255);
                let kr = hit.h / 2.0 - 2.5;
                let kx = if *on {
                    hit.x + hit.w - hit.h / 2.0
                } else {
                    hit.x + hit.h / 2.0
                };
                fill_circle(
                    frame,
                    width,
                    height,
                    (kx, hit.y + hit.h / 2.0),
                    kr,
                    knob,
                    255,
                );
            }
            Item::Button { id, label, hit } => {
                let pressed = self.pressed_button == Some(*id);
                let armed = self.armed == Some(*id);
                let bg = if armed {
                    [90, 140, 220]
                } else if pressed {
                    [28, 32, 40]
                } else if hovered {
                    [56, 62, 74]
                } else {
                    [42, 47, 57]
                };
                fill_rect(frame, width, height, *hit, bg, 240);
                stroke_rect(
                    frame,
                    width,
                    height,
                    *hit,
                    if armed {
                        [160, 195, 245]
                    } else {
                        [95, 102, 116]
                    },
                );
                let (tw, th) = crate::text::measure_text(label, 12.0);
                draw_text(
                    frame,
                    width,
                    height,
                    label,
                    12.0,
                    to_f32(
                        hit.x + (hit.w - f64::from(tw)) / 2.0,
                        hit.y + (hit.h - f64::from(th)) / 2.0,
                    ),
                    [222, 226, 233],
                );
            }
        }
    }
}

impl Rect {
    /// Whether `other` is the same row rectangle (used to match the
    /// cached hover rect against this frame's layout).
    fn contains_rect(&self, other: &Rect) -> bool {
        (self.x - other.x).abs() < 0.5 && (self.y - other.y).abs() < 0.5
    }
}

/// Build the control list for one frame: every control with its resolved
/// rectangle, plus the total content height (for scroll clamping). A free
/// function of (state, panel edge, scroll) so interaction and drawing can
/// never disagree about where things are.
fn layout(state: &PanelState, draft: &LaunchDraft, panel_x: f64, scroll: f64) -> (Vec<Laid>, f64) {
    let x = panel_x + PAD;
    let w = PANEL_WIDTH - 2.0 * PAD;
    let mut y = PAD - scroll;
    let mut out = Vec::new();

    push_item(&mut out, x, w, Item::Header("bouncy"), 22.0, &mut y);
    push_item(
        &mut out,
        x,
        w,
        Item::Readout(format!("{:.0} fps", state.fps)),
        15.0,
        &mut y,
    );
    push_item(
        &mut out,
        x,
        w,
        Item::Readout(format!(
            "{} / {} particles",
            state.particles, state.max_particles
        )),
        15.0,
        &mut y,
    );
    push_item(
        &mut out,
        x,
        w,
        Item::Readout(format!(
            "{} wells   {} walls   {} emitters",
            state.wells, state.walls, state.emitters
        )),
        17.0,
        &mut y,
    );

    let pause_label = if state.paused { "Resume" } else { "Pause" };
    button_row(
        &mut out,
        &[
            (ButtonId::PauseResume, pause_label),
            (ButtonId::StepFrame, "Step"),
            (ButtonId::Reset, "Reset"),
        ],
        x,
        w,
        &mut y,
    );

    push_item(&mut out, x, w, Item::Header("physics"), 24.0, &mut y);
    push_slider(
        &mut out,
        x,
        w,
        SliderId::Gravity,
        "Gravity",
        GRAVITY_POINTS,
        f64::from(state.gravity),
        &[0.0, 100.0],
        format!("{}%", state.gravity),
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::ParticleElasticity,
        "Particle elasticity",
        ELASTICITY_POINTS,
        state.particle_elasticity,
        &[1.0],
        format!("{:.2}", state.particle_elasticity),
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::WallElasticity,
        "Wall elasticity",
        ELASTICITY_POINTS,
        state.wall_elasticity,
        &[1.0],
        format!("{:.2}", state.wall_elasticity),
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::TimeScale,
        "Time scale",
        TIME_POINTS,
        state.time_scale,
        &[1.0],
        format!("{:.2}x", state.time_scale),
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::ExplosionThreshold,
        "Explosions at",
        THRESHOLD_POINTS,
        f64::from(state.explosion_threshold),
        &[0.0],
        if state.explosion_threshold == 0 {
            "off".to_string()
        } else {
            format!("{}/s births", state.explosion_threshold)
        },
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::PingVolume,
        "Ping volume",
        PING_VOLUME_POINTS,
        f64::from(state.ping_volume),
        &[0.0, 100.0],
        if state.ping_volume == 0 {
            "silent".to_string()
        } else {
            format!("{}%", state.ping_volume)
        },
        &mut y,
    );

    push_item(&mut out, x, w, Item::Header("mechanics"), 24.0, &mut y);
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::Matter,
        "Matter (fusion/fission)",
        state.matter,
        &mut y,
    );
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::Flow,
        "Flow field",
        state.flow,
        &mut y,
    );
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::SelfGravity,
        "Self-gravity",
        state.self_gravity,
        &mut y,
    );
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::Trails,
        "Trails",
        state.trails,
        &mut y,
    );
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::Kaleidoscope,
        "Kaleidoscope",
        state.kaleidoscope,
        &mut y,
    );
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::Music,
        "Musical pings",
        state.music,
        &mut y,
    );
    push_toggle(
        &mut out,
        x,
        w,
        ToggleId::WallChimes,
        "Wall chimes",
        state.wall_chimes,
        &mut y,
    );
    push_toggle(&mut out, x, w, ToggleId::Mute, "Mute", state.muted, &mut y);
    y += 4.0;

    button_row(
        &mut out,
        &[(
            ButtonId::CycleSpawn,
            &format!("Spawn: {}", state.spawn_mode),
        )],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[(
            ButtonId::CycleColor,
            &format!("Color: {}", state.color_mode),
        )],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[(ButtonId::CycleHud, &format!("Hud: {}", state.hud))],
        x,
        w,
        &mut y,
    );

    push_item(&mut out, x, w, Item::Header("actions"), 24.0, &mut y);
    button_row(&mut out, &[(ButtonId::Select, "Select")], x, w, &mut y);
    button_row(
        &mut out,
        &[(ButtonId::Burst, "Burst"), (ButtonId::Comet, "Comet")],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[
            (ButtonId::Explode, "Explode"),
            (ButtonId::Screenshot, "Screenshot"),
        ],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[
            (ButtonId::PinWell, "Pin well"),
            (ButtonId::PinRepeller, "Pin repeller"),
        ],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[
            (ButtonId::PlaceEmitter, "Place emitter"),
            (ButtonId::ClearEmitters, "Clear emitters"),
        ],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[
            (ButtonId::ClearWells, "Clear wells"),
            (ButtonId::ClearWalls, "Clear walls"),
        ],
        x,
        w,
        &mut y,
    );
    button_row(
        &mut out,
        &[(ButtonId::ExportScene, "Export scene")],
        x,
        w,
        &mut y,
    );

    if let Some(sel) = state.selection {
        push_item(&mut out, x, w, Item::Header("selected"), 24.0, &mut y);
        match sel {
            PanelSelection::Emitter {
                id,
                rate,
                cap,
                angle_deg,
            } => {
                push_item(
                    &mut out,
                    x,
                    w,
                    Item::Readout(format!("Emitter #{id}   aim {angle_deg:.0}°")),
                    17.0,
                    &mut y,
                );
                push_slider(
                    &mut out,
                    x,
                    w,
                    SliderId::EmitterRate,
                    "Rate",
                    EMITTER_RATE_POINTS,
                    rate,
                    &[2.0],
                    format!("{rate:.1}/s"),
                    &mut y,
                );
                #[allow(clippy::cast_precision_loss)]
                push_slider(
                    &mut out,
                    x,
                    w,
                    SliderId::EmitterCap,
                    "Cap",
                    EMITTER_CAP_POINTS,
                    cap as f64,
                    &[12.0],
                    format!("{cap} live"),
                    &mut y,
                );
                button_row(
                    &mut out,
                    &[
                        (ButtonId::ReAim, "Re-aim"),
                        (ButtonId::DeleteSelected, "Delete"),
                    ],
                    x,
                    w,
                    &mut y,
                );
            }
            PanelSelection::Stroke { id, segments, note } => {
                let plural = if segments == 1 { "" } else { "s" };
                push_item(
                    &mut out,
                    x,
                    w,
                    Item::Readout(format!("Wall #{id}   {segments} segment{plural}")),
                    17.0,
                    &mut y,
                );
                button_row(
                    &mut out,
                    &[(
                        ButtonId::CycleStrokeNote,
                        &format!("Note: {}", note_label(note)),
                    )],
                    x,
                    w,
                    &mut y,
                );
                button_row(
                    &mut out,
                    &[(ButtonId::DeleteSelected, "Delete")],
                    x,
                    w,
                    &mut y,
                );
            }
        }
    }

    push_item(&mut out, x, w, Item::Header("launch"), 24.0, &mut y);
    button_row(
        &mut out,
        &[(
            ButtonId::CyclePreset,
            &format!("Preset: {}", preset_names()[draft.preset_idx]),
        )],
        x,
        w,
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::LaunchSize,
        "Particle size",
        SIZE_POINTS,
        draft.particle_size,
        &[1.5],
        format!("{:.1}", draft.particle_size),
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::LaunchSpeed,
        "Initial speed",
        SPEED_POINTS,
        draft.initial_speed,
        &[600.0],
        format!("{:.0}", draft.initial_speed),
        &mut y,
    );
    push_slider(
        &mut out,
        x,
        w,
        SliderId::LaunchMinParticles,
        "Min particles",
        MIN_PARTICLES_POINTS,
        draft.min_particles,
        &[0.0],
        if draft.min_particles < 1.5 {
            "auto".to_string()
        } else {
            format!("{:.0}", draft.min_particles)
        },
        &mut y,
    );
    button_row(
        &mut out,
        &[(ButtonId::Relaunch, "Apply & relaunch")],
        x,
        w,
        &mut y,
    );

    let content_height = y + scroll + PAD;
    (out, content_height)
}

/// Append one full-width control row.
fn push_item(out: &mut Vec<Laid>, x: f64, w: f64, item: Item, row_h: f64, y: &mut f64) {
    out.push(Laid {
        item,
        row: Rect {
            x,
            y: *y,
            w,
            h: row_h,
        },
    });
    *y += row_h;
}

/// Append a labeled slider row with its track rectangle.
#[allow(clippy::too_many_arguments)]
fn push_slider(
    out: &mut Vec<Laid>,
    x: f64,
    w: f64,
    id: SliderId,
    label: &'static str,
    points: &'static [(f64, f64)],
    value: f64,
    detents: &'static [f64],
    text: String,
    y: &mut f64,
) {
    let track = Rect {
        x,
        y: *y + 17.0,
        w,
        h: 5.0,
    };
    out.push(Laid {
        item: Item::Slider {
            id,
            label,
            points,
            value,
            detents,
            text,
            track,
        },
        row: Rect {
            x,
            y: *y,
            w,
            h: 30.0,
        },
    });
    *y += 30.0;
}

/// Append a toggle row with its switch rectangle.
fn push_toggle(
    out: &mut Vec<Laid>,
    x: f64,
    w: f64,
    id: ToggleId,
    label: &'static str,
    on: bool,
    y: &mut f64,
) {
    let hit = Rect {
        x: x + w - 30.0,
        y: *y + 2.0,
        w: 30.0,
        h: 14.0,
    };
    out.push(Laid {
        item: Item::Toggle { id, label, on, hit },
        row: Rect {
            x,
            y: *y,
            w,
            h: 20.0,
        },
    });
    *y += 20.0;
}

/// Append a row of equally sized buttons.
fn button_row(out: &mut Vec<Laid>, buttons: &[(ButtonId, &str)], x: f64, w: f64, y: &mut f64) {
    let gap = 6.0;
    #[allow(clippy::cast_precision_loss)]
    let n = buttons.len() as f64;
    let bw = (w - gap * (n - 1.0)) / n;
    for (i, (id, label)) in buttons.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let bx = x + (bw + gap) * i as f64;
        let hit = Rect {
            x: bx,
            y: *y,
            w: bw,
            h: 20.0,
        };
        out.push(Laid {
            item: Item::Button {
                id: *id,
                label: (*label).to_string(),
                hit,
            },
            row: hit,
        });
    }
    *y += 26.0;
}

/// Track position (0..1) for a value under a piecewise-linear mapping.
fn value_to_t(points: &[(f64, f64)], v: f64) -> f64 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if v <= first.0 {
        return first.1;
    }
    if v >= last.0 {
        return last.1;
    }
    for pair in points.windows(2) {
        let ((v0, t0), (v1, t1)) = (pair[0], pair[1]);
        if v <= v1 {
            return t0 + (v - v0) / (v1 - v0) * (t1 - t0);
        }
    }
    last.1
}

/// Value for a track position (0..1) under a piecewise-linear mapping.
fn t_to_value(points: &[(f64, f64)], t: f64) -> f64 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if t <= first.1 {
        return first.0;
    }
    if t >= last.1 {
        return last.0;
    }
    for pair in points.windows(2) {
        let ((v0, t0), (v1, t1)) = (pair[0], pair[1]);
        if t <= t1 {
            return v0 + (t - t0) / (t1 - t0) * (v1 - v0);
        }
    }
    last.0
}

/// Slider value for a cursor x position over a track.
fn slider_value_at(cx: f64, track: &Rect, points: &[(f64, f64)]) -> f64 {
    let t = ((cx - track.x) / track.w).clamp(0.0, 1.0);
    t_to_value(points, t)
}

/// Snap a value to the nearest detent within reach, measured along the
/// track so magnified sliders snap uniformly; Shift bypasses for fine
/// control.
fn snap_to_detents(value: f64, detents: &[f64], points: &[(f64, f64)], shift: bool) -> f64 {
    if shift {
        return value;
    }
    let t = value_to_t(points, value);
    detents
        .iter()
        .copied()
        .find(|&d| (t - value_to_t(points, d)).abs() <= DETENT_FRACTION)
        .unwrap_or(value)
}

/// The command a toggle press emits.
fn toggle_command(id: ToggleId) -> PanelCommand {
    match id {
        ToggleId::Matter => PanelCommand::Plain(Command::ToggleMatter),
        ToggleId::Flow => PanelCommand::Plain(Command::ToggleFlow),
        ToggleId::SelfGravity => PanelCommand::Plain(Command::ToggleSelfGravity),
        ToggleId::Trails => PanelCommand::Plain(Command::ToggleTrails),
        ToggleId::Kaleidoscope => PanelCommand::Plain(Command::ToggleKaleidoscope),
        ToggleId::Music => PanelCommand::Plain(Command::ToggleMusic),
        ToggleId::WallChimes => PanelCommand::Plain(Command::ToggleWallChimes),
        ToggleId::Mute => PanelCommand::Plain(Command::ToggleMute),
    }
}

/// The command a slider value change emits.
fn slider_command(id: SliderId, value: f64) -> PanelCommand {
    #[allow(clippy::cast_possible_truncation)]
    match id {
        SliderId::Gravity => PanelCommand::SetGravity((value / 10.0).round() as i32 * 10),
        SliderId::ParticleElasticity => {
            PanelCommand::SetParticleElasticity((value * 100.0).round() / 100.0)
        }
        SliderId::WallElasticity => {
            PanelCommand::SetWallElasticity((value * 100.0).round() / 100.0)
        }
        SliderId::TimeScale => PanelCommand::SetTimeScale((value * 100.0).round() / 100.0),
        SliderId::ExplosionThreshold => {
            PanelCommand::SetExplosionThreshold((value / 5.0).round() as i32 * 5)
        }
        SliderId::PingVolume => PanelCommand::SetPingVolume((value / 5.0).round() as i32 * 5),
        SliderId::LaunchSize | SliderId::LaunchSpeed | SliderId::LaunchMinParticles => {
            unreachable!("launch sliders edit the draft; Apply & relaunch emits")
        }
        SliderId::EmitterRate | SliderId::EmitterCap => {
            unreachable!("inspector sliders emit id-carrying commands in the drag arm")
        }
    }
}

/// The chime-note label the stroke inspector shows.
fn note_label(note: WallNote) -> String {
    match note {
        WallNote::Auto => "auto".to_string(),
        WallNote::Note(n) => format!("degree {n}"),
        WallNote::Silent => "silent".to_string(),
    }
}

/// The buttons that arm a one-shot placement tool.
fn is_placement(id: ButtonId) -> bool {
    matches!(
        id,
        ButtonId::Burst
            | ButtonId::Comet
            | ButtonId::Explode
            | ButtonId::PinWell
            | ButtonId::PinRepeller
            | ButtonId::PlaceEmitter
            | ButtonId::Select
            | ButtonId::ReAim
    )
}

/// Tooltip label for an armed placement tool.
fn placement_label(id: ButtonId) -> &'static str {
    match id {
        ButtonId::Burst => "click to place burst",
        ButtonId::Comet => "click to aim comet",
        ButtonId::Explode => "click to place explosion",
        ButtonId::PinWell => "click to pin well",
        ButtonId::PinRepeller => "click to pin repeller",
        ButtonId::PlaceEmitter => "click to place emitter (aims at center)",
        ButtonId::Select => "click an emitter or wall to select",
        ButtonId::ReAim => "click to aim the emitter",
        _ => "",
    }
}

/// The command a non-placement button click emits.
fn button_command(id: ButtonId, state: &PanelState) -> PanelCommand {
    match id {
        ButtonId::PauseResume => PanelCommand::SetPaused(!state.paused),
        ButtonId::StepFrame => PanelCommand::Plain(Command::StepFrame),
        ButtonId::Reset => PanelCommand::Plain(Command::Reset),
        ButtonId::CycleSpawn => PanelCommand::Plain(Command::CycleSpawnMode),
        ButtonId::CycleColor => PanelCommand::Plain(Command::CycleColorMode),
        ButtonId::CycleHud => PanelCommand::Plain(Command::CycleHud),
        ButtonId::Screenshot => PanelCommand::Plain(Command::Screenshot),
        ButtonId::ClearWells => PanelCommand::Plain(Command::ClearWells),
        ButtonId::ClearWalls => PanelCommand::Plain(Command::ClearWalls),
        ButtonId::ClearEmitters => PanelCommand::Plain(Command::ClearEmitters),
        ButtonId::ExportScene => PanelCommand::Plain(Command::ExportScene),
        ButtonId::Burst
        | ButtonId::Comet
        | ButtonId::Explode
        | ButtonId::PinWell
        | ButtonId::PinRepeller
        | ButtonId::PlaceEmitter
        | ButtonId::Select
        | ButtonId::ReAim
        | ButtonId::CycleStrokeNote
        | ButtonId::DeleteSelected
        | ButtonId::CyclePreset
        | ButtonId::Relaunch => {
            unreachable!("handled draft-side before dispatch")
        }
    }
}

fn to_f32(x: f64, y: f64) -> (f32, f32) {
    #[allow(clippy::cast_possible_truncation)]
    (x as f32, y as f32)
}

/// Alpha-blend a solid color over a rectangle of the frame.
fn fill_rect(frame: &mut [u8], width: u32, height: u32, rect: Rect, rgb: [u8; 3], alpha: u8) {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (x0, y0) = (rect.x.max(0.0) as u32, rect.y.max(0.0) as u32);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (x1, y1) = (
        (rect.x + rect.w).clamp(0.0, f64::from(width)) as u32,
        (rect.y + rect.h).clamp(0.0, f64::from(height)) as u32,
    );
    let a = u16::from(alpha);
    for py in y0..y1 {
        let row = (py * width) as usize * 4;
        for px in x0..x1 {
            let idx = row + px as usize * 4;
            for (channel, &c) in rgb.iter().enumerate() {
                let dst = u16::from(frame[idx + channel]);
                #[allow(clippy::cast_possible_truncation)]
                {
                    frame[idx + channel] = ((u16::from(c) * a + dst * (255 - a)) / 255) as u8;
                }
            }
            frame[idx + 3] = 255;
        }
    }
}

/// Alpha-blend a filled capsule (pill): a center rectangle with a
/// semicircle at each end.
fn fill_pill(frame: &mut [u8], width: u32, height: u32, rect: Rect, rgb: [u8; 3], alpha: u8) {
    let r = rect.h / 2.0;
    let (cy, lx, rx) = (rect.y + r, rect.x + r, rect.x + rect.w - r);
    fill_rect(
        frame,
        width,
        height,
        Rect {
            x: lx,
            y: rect.y,
            w: rect.w - 2.0 * r,
            h: rect.h,
        },
        rgb,
        alpha,
    );
    fill_circle(frame, width, height, (lx, cy), r, rgb, alpha);
    fill_circle(frame, width, height, (rx, cy), r, rgb, alpha);
}

/// Alpha-blend a filled circle centered at `center`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn fill_circle(
    frame: &mut [u8],
    width: u32,
    height: u32,
    center: (f64, f64),
    r: f64,
    rgb: [u8; 3],
    alpha: u8,
) {
    let (cx, cy) = center;
    let (x0, x1) = (
        (cx - r).max(0.0) as u32,
        (cx + r + 1.0).min(f64::from(width)) as u32,
    );
    let (y0, y1) = (
        (cy - r).max(0.0) as u32,
        (cy + r + 1.0).min(f64::from(height)) as u32,
    );
    for py in y0..y1 {
        for px in x0..x1 {
            let (dx, dy) = (f64::from(px) + 0.5 - cx, f64::from(py) + 0.5 - cy);
            // Coverage fades over the outermost pixel so the rim is
            // smooth instead of stair-stepped.
            let coverage = (r + 0.5 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let a = (f64::from(alpha) * coverage) as u16;
            let idx = ((py * width + px) * 4) as usize;
            for (channel, &c) in rgb.iter().enumerate() {
                let dst = u16::from(frame[idx + channel]);
                frame[idx + channel] = ((u16::from(c) * a + dst * (255 - a)) / 255) as u8;
            }
            frame[idx + 3] = 255;
        }
    }
}

/// One-pixel border around a rectangle.
fn stroke_rect(frame: &mut [u8], width: u32, height: u32, rect: Rect, rgb: [u8; 3]) {
    let rgba = [rgb[0], rgb[1], rgb[2], 255];
    hline(frame, width, height, rect.x, rect.x + rect.w, rect.y, rgba);
    hline(
        frame,
        width,
        height,
        rect.x,
        rect.x + rect.w,
        rect.y + rect.h - 1.0,
        rgba,
    );
    vline(frame, width, height, rect.x, rect.y, rect.y + rect.h, rgba);
    vline(
        frame,
        width,
        height,
        rect.x + rect.w - 1.0,
        rect.y,
        rect.y + rect.h,
        rgba,
    );
}

fn hline(frame: &mut [u8], width: u32, height: u32, x0: f64, x1: f64, y: f64, rgba: [u8; 4]) {
    fill_rect(
        frame,
        width,
        height,
        Rect {
            x: x0,
            y,
            w: x1 - x0,
            h: 1.0,
        },
        [rgba[0], rgba[1], rgba[2]],
        rgba[3],
    );
}

fn vline(frame: &mut [u8], width: u32, height: u32, x: f64, y0: f64, y1: f64, rgba: [u8; 4]) {
    fill_rect(
        frame,
        width,
        height,
        Rect {
            x,
            y: y0,
            w: 1.0,
            h: y1 - y0,
        },
        [rgba[0], rgba[1], rgba[2]],
        rgba[3],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_gui() -> Gui {
        let mut gui = Gui::new();
        gui.open = true;
        gui.slide = 1.0;
        gui
    }

    fn state() -> PanelState {
        PanelState {
            gravity: 100,
            particle_elasticity: 1.0,
            wall_elasticity: 1.0,
            time_scale: 1.0,
            spawn_mode: "center".into(),
            color_mode: "solid".into(),
            hud: "hidden".into(),
            launch_particle_size: 1.5,
            launch_initial_speed: 600.0,
            ..PanelState::default()
        }
    }

    /// The rect of a control, straight from the layout.
    fn find_slider_track(s: &PanelState, id: SliderId) -> Rect {
        layout(s, &LaunchDraft::default(), 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Slider { id: sid, track, .. } if *sid == id => Some(*track),
                _ => None,
            })
            .expect("slider in layout")
    }

    /// Manual visual check: render the open panel over a synthetic
    /// scene and write a PNG. Run with
    /// `cargo test --release render_panel_preview -- --ignored --nocapture`,
    /// then open target/panel-preview.png.
    #[test]
    #[ignore = "manual visual preview"]
    fn render_panel_preview() {
        let (w, h) = (800u32, 600u32);
        let mut frame = vec![0u8; (w * h * 4) as usize];
        // A dim synthetic scene so translucency is visible: scattered
        // "particles" behind the panel.
        for i in 0..600 {
            let x = (i * 37) % w;
            let y = (i * 53) % h;
            let idx = ((y * w + x) * 4) as usize;
            frame[idx] = 90 + (i % 160) as u8;
            frame[idx + 1] = 200 - (i % 120) as u8;
            frame[idx + 2] = 120 + (i % 100) as u8;
            frame[idx + 3] = 255;
        }

        let mut gui = open_gui();
        let mut s = state();
        s.fps = 119.6;
        s.particles = 4210;
        s.max_particles = 12000;
        s.wells = 2;
        s.walls = 26;
        s.matter = true;
        s.self_gravity = true;
        s.explosion_threshold = 30;
        // Hover the reset button so the hover style shows.
        let (layout, _) = layout(&s, &LaunchDraft::default(), 800.0 - PANEL_WIDTH, 0.0);
        let reset = layout
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Button { id, hit, .. } if *id == ButtonId::Reset => Some(*hit),
                _ => None,
            })
            .expect("reset");
        gui.set_cursor(reset.x + 4.0, reset.y + 4.0);
        gui.handle_alpha = 1.0;
        let _ = gui.tick(0.016, &s, w, h, false, 0.0);
        // Show the bottom of the panel (the launch section) when asked.
        if std::env::var("PANEL_PREVIEW_BOTTOM").is_ok() {
            gui.scroll = (gui.content_height - f64::from(h)).max(0.0);
        }
        gui.draw(&mut frame, &s, w, h);

        let path = std::path::Path::new("target/panel-preview.png");
        crate::render::write_png(path, &frame, w, h).expect("png written");
        println!("panel preview written to {}", path.display());
    }

    #[test]
    fn edge_dwell_reveals_the_handle_and_quick_passes_do_not() {
        let mut gui = Gui::new();
        let s = state();
        // A quick pass through the strip: two short ticks, no reveal.
        gui.set_cursor(797.0, 300.0);
        gui.tick(0.05, &s, 800, 600, false, 0.0);
        gui.set_cursor(400.0, 300.0);
        gui.tick(0.05, &s, 800, 600, false, 0.0);
        assert!(gui.handle_alpha < 0.3, "quick pass never arms the handle");
        // Dwelling in the strip reveals it.
        gui.set_cursor(797.0, 300.0);
        for _ in 0..20 {
            gui.tick(0.05, &s, 800, 600, false, 0.0);
        }
        assert!(
            gui.handle_alpha > 0.9,
            "dwell reveals: {}",
            gui.handle_alpha
        );
        // An idle cursor fades it back out.
        for _ in 0..20 {
            gui.tick(0.05, &s, 800, 600, false, 10.0);
        }
        assert!(gui.handle_alpha < 0.1, "idle fades: {}", gui.handle_alpha);
    }

    #[test]
    fn handle_click_toggles_and_drag_slides() {
        let mut gui = Gui::new();
        let s = state();
        // Arm the handle by dwelling at the edge, at handle height.
        gui.set_cursor(797.0, 300.0);
        for _ in 0..20 {
            gui.tick(0.05, &s, 800, 600, false, 0.0);
        }
        // Click (press + release without travel) opens.
        assert!(gui.on_press_at(800, 600), "handle press consumed");
        gui.on_release();
        for _ in 0..40 {
            gui.tick(0.05, &s, 800, 600, false, 0.0);
        }
        assert!(gui.open, "click opened the panel");
        assert!((gui.slide - 1.0).abs() < 0.01, "settled out: {}", gui.slide);

        // Grab the handle (now at the panel's left edge) and push it
        // most of the way home: releases closed.
        let hr = gui.handle_rect(800, 600);
        gui.set_cursor(hr.x + 2.0, hr.y + 10.0);
        assert!(gui.on_press_at(800, 600), "grab consumed");
        gui.set_cursor(790.0, hr.y + 10.0);
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(gui.slide < 0.2, "panel tracked the drag: {}", gui.slide);
        gui.on_release();
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(!gui.open, "released near home: closed");
    }

    #[test]
    fn placement_buttons_arm_place_and_cancel() {
        let mut gui = open_gui();
        let s = state();
        let hit = layout(&s, &LaunchDraft::default(), 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Button { id, hit, .. } if *id == ButtonId::Burst => Some(*hit),
                _ => None,
            })
            .expect("burst button");

        let click = |gui: &mut Gui, s: &PanelState| {
            gui.set_cursor(hit.x + 2.0, hit.y + 2.0);
            gui.input.pressed = true;
            gui.tick(0.016, s, 800, 600, false, 0.0);
            gui.input.released = true;
            gui.tick(0.016, s, 800, 600, false, 0.0)
        };

        // Click arms without emitting; the arena click places at the
        // cursor and disarms.
        let cmds = click(&mut gui, &s);
        assert!(cmds.is_empty(), "arming emits nothing");
        assert!(gui.is_armed());
        let placed = gui.place_armed(150.0, 220.0);
        assert!(
            matches!(placed, Some(PanelCommand::SpawnBurst(x, y))
                if (x - 150.0).abs() < 1e-9 && (y - 220.0).abs() < 1e-9),
            "placed at the arena click"
        );
        assert!(!gui.is_armed(), "placing disarms");

        // A second press cancels instead of double-arming.
        click(&mut gui, &s);
        assert!(gui.is_armed());
        click(&mut gui, &s);
        assert!(!gui.is_armed(), "second press cancels");

        // Esc-style disarm and closing the panel both drop the tool.
        click(&mut gui, &s);
        gui.disarm();
        assert!(!gui.is_armed());
        // Hiding the panel keeps the tool: arm, hide, place under
        // where the panel sat.
        click(&mut gui, &s);
        gui.open = false;
        gui.slide = 0.0;
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(gui.is_armed(), "armed tool survives hiding the panel");
        let placed = gui.place_armed(760.0, 300.0);
        assert!(
            matches!(placed, Some(PanelCommand::SpawnBurst(x, _)) if (x - 760.0).abs() < 1e-9),
            "placed on ground the panel had covered"
        );
        assert!(!gui.is_armed());
    }

    #[test]
    fn launch_sliders_edit_the_draft_and_relaunch_emits_it() {
        let mut gui = open_gui();
        let s = state();
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!((gui.launch.particle_size - 1.5).abs() < 1e-9, "seeded");

        // Drag the size slider to its right end: the draft changes,
        // nothing is emitted.
        let track = find_slider_track(&s, SliderId::LaunchSize);
        gui.set_cursor(track.x + track.w - 1.0, track.y + 2.0);
        gui.input.pressed = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(cmds.is_empty(), "draft sliders emit nothing");
        assert!((gui.launch.particle_size - 10.0).abs() < 1e-9);
        gui.input.released = true;
        gui.tick(0.016, &s, 800, 600, false, 0.0);

        // Cycle the preset once, then apply: the command carries the
        // draft.
        let click_button = |gui: &mut Gui, s: &PanelState, id: ButtonId| {
            let hit = layout(s, &gui.launch, 800.0 - PANEL_WIDTH, 0.0)
                .0
                .iter()
                .find_map(|laid| match &laid.item {
                    Item::Button { id: bid, hit, .. } if *bid == id => Some(*hit),
                    _ => None,
                })
                .expect("button in layout");
            gui.set_cursor(hit.x + 2.0, hit.y + 2.0);
            gui.input.pressed = true;
            gui.tick(0.016, s, 800, 600, false, 0.0);
            gui.input.released = true;
            gui.tick(0.016, s, 800, 600, false, 0.0)
        };
        click_button(&mut gui, &s, ButtonId::CyclePreset);
        assert_eq!(preset_names()[gui.launch.preset_idx], "fireworks");
        let cmds = click_button(&mut gui, &s, ButtonId::Relaunch);
        assert!(
            cmds.iter().any(|c| matches!(c, PanelCommand::Relaunch {
                preset: Some(p),
                particle_size,
                min_particles: None,
                ..
            } if p == "fireworks"
                && matches!(particle_size, Some(v) if (v - 10.0).abs() < 1e-9))),
            "relaunch carries the draft"
        );
    }

    #[test]
    fn emitter_buttons_map_to_their_commands() {
        let mut gui = Gui::new();
        gui.armed = Some(ButtonId::PlaceEmitter);
        assert!(matches!(
            gui.place_armed(120.0, 340.0),
            Some(PanelCommand::PlaceEmitter(x, y)) if (x - 120.0).abs() < 1e-9 && (y - 340.0).abs() < 1e-9
        ));
        assert!(matches!(
            button_command(ButtonId::ClearEmitters, &PanelState::default()),
            PanelCommand::Plain(Command::ClearEmitters)
        ));
        assert_eq!(
            placement_label(ButtonId::PlaceEmitter),
            "click to place emitter (aims at center)"
        );
    }

    /// Press and release a button by id, returning the release tick's
    /// commands.
    fn click_button(gui: &mut Gui, s: &PanelState, id: ButtonId) -> Vec<PanelCommand> {
        let hit = layout(s, &gui.launch, 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Button { id: bid, hit, .. } if *bid == id => Some(*hit),
                _ => None,
            })
            .expect("button in layout");
        gui.set_cursor(hit.x + 2.0, hit.y + 2.0);
        gui.input.pressed = true;
        gui.tick(0.016, s, 800, 600, false, 0.0);
        gui.input.released = true;
        gui.tick(0.016, s, 800, 600, false, 0.0)
    }

    fn emitter_selected() -> PanelState {
        PanelState {
            selection: Some(PanelSelection::Emitter {
                id: 3,
                rate: 2.0,
                cap: 12,
                angle_deg: 90.0,
            }),
            ..state()
        }
    }

    fn stroke_selected() -> PanelState {
        PanelState {
            selection: Some(PanelSelection::Stroke {
                id: 5,
                segments: 4,
                note: WallNote::Auto,
            }),
            ..state()
        }
    }

    #[test]
    fn inspector_rows_appear_only_with_a_selection() {
        let has = |s: &PanelState, want_slider: Option<SliderId>, want_button: Option<ButtonId>| {
            let (items, _) = layout(s, &LaunchDraft::default(), 800.0 - PANEL_WIDTH, 0.0);
            items.iter().any(|laid| match &laid.item {
                Item::Slider { id, .. } => Some(*id) == want_slider,
                Item::Button { id, .. } => Some(*id) == want_button,
                _ => false,
            })
        };
        let none = state();
        assert!(!has(&none, Some(SliderId::EmitterRate), None));
        assert!(!has(&none, None, Some(ButtonId::DeleteSelected)));
        assert!(!has(&none, None, Some(ButtonId::CycleStrokeNote)));
        assert!(
            has(&none, None, Some(ButtonId::Select)),
            "tool always shown"
        );

        let emitter = emitter_selected();
        assert!(has(&emitter, Some(SliderId::EmitterRate), None));
        assert!(has(&emitter, Some(SliderId::EmitterCap), None));
        assert!(has(&emitter, None, Some(ButtonId::ReAim)));
        assert!(has(&emitter, None, Some(ButtonId::DeleteSelected)));
        assert!(!has(&emitter, None, Some(ButtonId::CycleStrokeNote)));

        let stroke = stroke_selected();
        assert!(!has(&stroke, Some(SliderId::EmitterRate), None));
        assert!(has(&stroke, None, Some(ButtonId::CycleStrokeNote)));
        assert!(has(&stroke, None, Some(ButtonId::DeleteSelected)));
        assert!(!has(&stroke, None, Some(ButtonId::ReAim)));
    }

    #[test]
    fn rate_drag_emits_an_id_carrying_command() {
        let mut gui = open_gui();
        let s = emitter_selected();
        let track = find_slider_track(&s, SliderId::EmitterRate);
        gui.set_cursor(track.x + track.w / 2.0, track.y + 2.0);
        gui.input.pressed = true;
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        // A live drag tracks the cursor past the track end: clamps to
        // the top of the range.
        gui.set_cursor(track.x + track.w + 20.0, track.y + 2.0);
        let cmds = gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(
            cmds.iter().any(|c| matches!(c,
                PanelCommand::SetEmitterRate(3, v) if (v - 20.0).abs() < 1e-9)),
            "right end of the rate track, addressed to the selected id"
        );
        // The selection dying mid-drag stops the stream, no panic.
        gui.set_cursor(track.x + track.w / 2.0, track.y + 2.0);
        let cmds = gui.tick(0.016, &state(), 800, 600, false, 0.0);
        assert!(cmds.is_empty(), "no selection, no command");
    }

    #[test]
    fn delete_button_emits_the_selected_kind() {
        let mut gui = open_gui();
        let cmds = click_button(&mut gui, &emitter_selected(), ButtonId::DeleteSelected);
        assert!(
            cmds.iter()
                .any(|c| matches!(c, PanelCommand::DeleteEmitter(3))),
            "emitter selection deletes the emitter"
        );
        let cmds = click_button(&mut gui, &stroke_selected(), ButtonId::DeleteSelected);
        assert!(
            cmds.iter()
                .any(|c| matches!(c, PanelCommand::DeleteStroke(5))),
            "stroke selection deletes the stroke"
        );
    }

    #[test]
    fn note_button_emits_the_cycle_command() {
        let mut gui = open_gui();
        let cmds = click_button(&mut gui, &stroke_selected(), ButtonId::CycleStrokeNote);
        assert!(
            cmds.iter()
                .any(|c| matches!(c, PanelCommand::CycleStrokeNote(5)))
        );
    }

    #[test]
    fn select_tool_arms_and_emits_select_at() {
        let mut gui = open_gui();
        let cmds = click_button(&mut gui, &state(), ButtonId::Select);
        assert!(cmds.is_empty(), "arming emits nothing");
        assert!(gui.is_armed());
        let placed = gui.place_armed(240.0, 180.0);
        assert!(matches!(placed, Some(PanelCommand::SelectAt(x, y))
                if (x - 240.0).abs() < 1e-9 && (y - 180.0).abs() < 1e-9));
        assert!(!gui.is_armed());
    }

    #[test]
    fn reaim_carries_the_arm_time_id() {
        let mut gui = open_gui();
        let cmds = click_button(&mut gui, &emitter_selected(), ButtonId::ReAim);
        assert!(cmds.is_empty(), "arming emits nothing");
        assert!(gui.is_armed());
        assert_eq!(gui.reaim_target, Some(3), "target captured at arm time");
        // Even if the selection has changed by click time, the armed
        // tool aims the emitter it was armed for.
        let placed = gui.place_armed(400.0, 300.0);
        assert!(matches!(placed, Some(PanelCommand::AimEmitterAt(3, x, y))
                if (x - 400.0).abs() < 1e-9 && (y - 300.0).abs() < 1e-9));
        assert_eq!(gui.reaim_target, None, "target consumed");

        // Esc-style disarm clears the captured target too.
        click_button(&mut gui, &emitter_selected(), ButtonId::ReAim);
        assert_eq!(gui.reaim_target, Some(3));
        gui.disarm();
        assert_eq!(gui.reaim_target, None);
    }

    #[test]
    fn preset_cycle_names_are_real_builtins() {
        use clap::ValueEnum;
        for name in &preset_names()[1..] {
            assert!(
                crate::presets::Preset::from_str(name, true).is_ok(),
                "panel preset '{name}' is not a built-in"
            );
        }
    }

    #[test]
    fn detents_snap_and_shift_bypasses() {
        // Linear 0..100 mapping: the track-space radius (2.5%) reaches
        // 2.5 value units.
        const LINEAR: &[(f64, f64)] = &[(0.0, 0.0), (100.0, 1.0)];
        assert!((snap_to_detents(0.4, &[0.0], LINEAR, false)).abs() < 1e-12);
        assert!((snap_to_detents(2.4, &[0.0], LINEAR, false)).abs() < 1e-12);
        let free = snap_to_detents(2.4, &[0.0], LINEAR, true);
        assert!((free - 2.4).abs() < 1e-12, "shift bypasses the snap");
        let out_of_reach = snap_to_detents(4.0, &[0.0], LINEAR, false);
        assert!((out_of_reach - 4.0).abs() < 1e-12);
    }

    #[test]
    fn gravity_mapping_magnifies_the_everyday_band() {
        // Round trips through the piecewise mapping.
        for v in [
            -1000.0, -400.0, -100.0, -30.0, 0.0, 10.0, 50.0, 100.0, 700.0,
        ] {
            let t = value_to_t(GRAVITY_POINTS, v);
            let back = t_to_value(GRAVITY_POINTS, t);
            assert!((back - v).abs() < 1e-9, "roundtrip {v}: {back}");
        }
        // The everyday band owns 60% of the track...
        assert!((value_to_t(GRAVITY_POINTS, -100.0) - 0.2).abs() < 1e-12);
        assert!((value_to_t(GRAVITY_POINTS, 100.0) - 0.8).abs() < 1e-12);
        // ...so 10% steps sit well clear of the detent radius.
        let step = value_to_t(GRAVITY_POINTS, 10.0) - value_to_t(GRAVITY_POINTS, 0.0);
        assert!(
            step > DETENT_FRACTION,
            "10% gravity must not be swallowed by the 0 detent: {step}"
        );
    }

    #[test]
    fn every_ten_percent_gravity_stop_is_reachable() {
        // The regression behind the magnified band: dragging across the
        // track must be able to produce each 10% stop from -100 to 100,
        // not jump 0 -> 100.
        let s = state();
        let track = find_slider_track(&s, SliderId::Gravity);
        let mut reachable = std::collections::HashSet::new();
        let mut x = track.x;
        while x <= track.x + track.w {
            let raw = slider_value_at(x, &track, GRAVITY_POINTS);
            let snapped = snap_to_detents(raw, &[0.0, 100.0], GRAVITY_POINTS, false);
            #[allow(clippy::cast_possible_truncation)]
            let cmd = ((snapped / 10.0).round() as i64) * 10;
            if (-100..=100).contains(&cmd) {
                reachable.insert(cmd);
            }
            x += 0.5;
        }
        for stop in (-100..=100).step_by(10) {
            assert!(
                reachable.contains(&stop),
                "{stop}% gravity unreachable by drag"
            );
        }
    }

    #[test]
    fn slider_maps_cursor_and_clamps() {
        let track = Rect {
            x: 100.0,
            y: 0.0,
            w: 100.0,
            h: 5.0,
        };
        assert!((slider_value_at(100.0, &track, ELASTICITY_POINTS)).abs() < 1e-12);
        assert!((slider_value_at(200.0, &track, ELASTICITY_POINTS) - 1.5).abs() < 1e-12);
        assert!((slider_value_at(400.0, &track, ELASTICITY_POINTS) - 1.5).abs() < 1e-12);
        assert!((slider_value_at(150.0, &track, ELASTICITY_POINTS) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn gravity_drag_snaps_to_the_zero_detent() {
        let mut gui = open_gui();
        let s = state();
        let track = find_slider_track(&s, SliderId::Gravity);
        // Press just left of the track's midpoint: raw value near -20,
        // inside the detent radius (3% of 2000 = 60) around 0.
        gui.set_cursor(track.x + track.w * 0.49, track.y + 2.0);
        gui.input.pressed = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(
            cmds.iter()
                .any(|c| matches!(c, PanelCommand::SetGravity(0))),
            "snapped to the zero detent"
        );
    }

    #[test]
    fn toggle_press_emits_its_command() {
        let mut gui = open_gui();
        let s = state();
        let row = layout(&s, &LaunchDraft::default(), 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Toggle { id, .. } if *id == ToggleId::Matter => Some(laid.row),
                _ => None,
            })
            .expect("matter toggle");
        gui.set_cursor(row.x + 4.0, row.y + row.h / 2.0);
        gui.input.pressed = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(
            cmds.iter()
                .any(|c| matches!(c, PanelCommand::Plain(Command::ToggleMatter))),
            "toggle emitted"
        );
    }

    #[test]
    fn buttons_fire_on_release_over_the_button() {
        let mut gui = open_gui();
        let s = state();
        let row = layout(&s, &LaunchDraft::default(), 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Button { id, hit, .. } if *id == ButtonId::Reset => Some(*hit),
                _ => None,
            })
            .expect("reset button");
        gui.set_cursor(row.x + 2.0, row.y + 2.0);
        gui.input.pressed = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(cmds.is_empty(), "no fire on press");
        // Release off the button cancels.
        gui.set_cursor(0.0, 0.0);
        gui.input.released = true;
        assert!(gui.tick(0.016, &s, 800, 600, false, 0.0).is_empty());
        // Press and release over it fires.
        gui.set_cursor(row.x + 2.0, row.y + 2.0);
        gui.input.pressed = true;
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        gui.input.released = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(
            cmds.iter()
                .any(|c| matches!(c, PanelCommand::Plain(Command::Reset))),
            "fired on release"
        );
    }

    #[test]
    fn closed_panel_consumes_nothing_and_scroll_clamps() {
        let mut gui = Gui::new();
        gui.set_cursor(790.0, 300.0);
        assert!(!gui.wants_pointer(800), "closed panel leaves the sim alone");

        let mut gui = open_gui();
        let s = state();
        gui.set_cursor(700.0, 300.0);
        assert!(gui.wants_pointer(800));
        // Scroll far past the end: clamped to the content overflow.
        gui.input.wheel = 1000.0;
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(
            gui.scroll <= (gui.content_height - 600.0).max(0.0) + 1e-9,
            "scroll clamped: {} vs content {}",
            gui.scroll,
            gui.content_height
        );
        // And back up: never negative.
        gui.input.wheel = -10_000.0;
        gui.tick(0.016, &s, 800, 600, false, 0.0);
        assert!(gui.scroll.abs() < 1e-9, "scroll floor at zero");
    }
}
