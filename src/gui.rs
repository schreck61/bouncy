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
use crate::sim::Polarity;
use crate::text::draw_text;

/// Panel width in simulation pixels.
pub const PANEL_WIDTH: f64 = 200.0;
/// Inner padding between the panel edge and its content.
const PAD: f64 = 10.0;
/// Snap radius for slider detents, as a fraction of the slider's range.
const DETENT_FRACTION: f64 = 0.03;
/// Scroll wheel speed, pixels per line.
const WHEEL_STEP: f64 = 24.0;
/// Slide animation rate (1/s): the exponential approach constant for the
/// panel's settle — critically-damped feel, no bounce.
const SLIDE_RATE: f64 = 14.0;

/// Identity of a value slider.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SliderId {
    Gravity,
    ParticleElasticity,
    WallElasticity,
    TimeScale,
    ExplosionThreshold,
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
    Mute,
}

/// Identity of a push button.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ButtonId {
    PauseResume,
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
    ClearWells,
    ClearWalls,
    ExportScene,
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
    pub muted: bool,
    pub spawn_mode: String,
    pub color_mode: String,
    pub hud: String,
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
        min: f64,
        max: f64,
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

    /// The panel's left edge for the current slide progress.
    fn panel_x(&self, width: u32) -> f64 {
        f64::from(width) - PANEL_WIDTH * ease(self.slide)
    }

    /// Whether the panel currently owns the pointer: over the visible
    /// panel, or mid-interaction (a drag must not leak to the sim even
    /// if the cursor strays off the panel).
    pub fn wants_pointer(&self, width: u32) -> bool {
        if self.dragging.is_some() || self.pressed_button.is_some() {
            return true;
        }
        self.slide > 0.0 && self.cursor.0 >= self.panel_x(width)
    }

    /// A mouse press: returns true when the panel consumes it.
    pub fn on_press(&mut self, width: u32) -> bool {
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
    ) -> Vec<PanelCommand> {
        // Exponential approach: critically-damped settle, no bounce.
        let target = if self.open { 1.0 } else { 0.0 };
        self.slide += (target - self.slide) * (1.0 - (-SLIDE_RATE * dt.max(0.0)).exp());
        if (self.slide - target).abs() < 0.001 {
            self.slide = target;
        }

        let mut commands = Vec::new();
        if self.slide <= 0.0 {
            // Hidden: drop stale interaction state and swallow edges.
            self.dragging = None;
            self.pressed_button = None;
            self.hover = None;
            self.input = PanelInput::default();
            return commands;
        }

        let (layout, content_height) = layout(state, self.panel_x(width), self.scroll);
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
                    min,
                    max,
                    value,
                    detents,
                    track,
                    ..
                } = &laid.item
                {
                    if *id != drag {
                        continue;
                    }
                    let raw = slider_value_at(cx, track, *min, *max);
                    let snapped = snap_to_detents(raw, detents, *max - *min, shift);
                    if (snapped - value).abs() > 1e-9 {
                        commands.push(slider_command(drag, snapped));
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
                    commands.push(button_command(pressed, state, width, height));
                }
            }
        }

        commands
    }

    /// Draw the panel over the finished frame. Uses the same layout the
    /// tick resolved interactions against.
    pub fn draw(&self, frame: &mut [u8], state: &PanelState, width: u32, height: u32) {
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

        let (layout, _) = layout(state, x0, self.scroll);
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
                min,
                max,
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
                    let dx = track.x + (d - min) / (max - min) * track.w;
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
                let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
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
                let (fill, border) = if *on {
                    ([90, 140, 220], [120, 165, 235])
                } else {
                    ([40, 44, 52], [90, 96, 108])
                };
                fill_rect(frame, width, height, *hit, fill, 255);
                stroke_rect(frame, width, height, *hit, border);
                if *on {
                    let dot = Rect {
                        x: hit.x + hit.w - hit.h + 3.0,
                        y: hit.y + 3.0,
                        w: hit.h - 6.0,
                        h: hit.h - 6.0,
                    };
                    fill_rect(frame, width, height, dot, [240, 244, 250], 255);
                } else {
                    let dot = Rect {
                        x: hit.x + 3.0,
                        y: hit.y + 3.0,
                        w: hit.h - 6.0,
                        h: hit.h - 6.0,
                    };
                    fill_rect(frame, width, height, dot, [140, 146, 158], 255);
                }
            }
            Item::Button { id, label, hit } => {
                let pressed = self.pressed_button == Some(*id);
                let bg = if pressed {
                    [28, 32, 40]
                } else if hovered {
                    [56, 62, 74]
                } else {
                    [42, 47, 57]
                };
                fill_rect(frame, width, height, *hit, bg, 240);
                stroke_rect(frame, width, height, *hit, [95, 102, 116]);
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
fn layout(state: &PanelState, panel_x: f64, scroll: f64) -> (Vec<Laid>, f64) {
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
        Item::Readout(format!("{} wells   {} walls", state.wells, state.walls)),
        17.0,
        &mut y,
    );

    let pause_label = if state.paused { "Resume" } else { "Pause" };
    button_row(
        &mut out,
        &[
            (ButtonId::PauseResume, pause_label),
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
        -1000.0,
        1000.0,
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
        0.0,
        1.5,
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
        0.0,
        1.5,
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
        0.1,
        4.0,
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
        0.0,
        1000.0,
        f64::from(state.explosion_threshold),
        &[0.0],
        if state.explosion_threshold == 0 {
            "off".to_string()
        } else {
            format!("{}/s births", state.explosion_threshold)
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

    push_item(
        &mut out,
        x,
        w,
        Item::Header("actions (at center)"),
        24.0,
        &mut y,
    );
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
    min: f64,
    max: f64,
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
            min,
            max,
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

/// Smoothstep easing for the slide animation's spatial mapping.
fn ease(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Slider value for a cursor x position over a track.
fn slider_value_at(cx: f64, track: &Rect, min: f64, max: f64) -> f64 {
    let t = ((cx - track.x) / track.w).clamp(0.0, 1.0);
    min + t * (max - min)
}

/// Snap a value to the nearest detent within reach, unless Shift is
/// held (fine control bypasses the snap).
fn snap_to_detents(value: f64, detents: &[f64], range: f64, shift: bool) -> f64 {
    if shift {
        return value;
    }
    let radius = range * DETENT_FRACTION;
    detents
        .iter()
        .copied()
        .find(|d| (value - d).abs() <= radius)
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
    }
}

/// The command a button click emits. Placement actions land at the
/// arena center (cursor-follow placement is future work).
fn button_command(id: ButtonId, state: &PanelState, width: u32, height: u32) -> PanelCommand {
    let (cx, cy) = (f64::from(width) / 2.0, f64::from(height) / 2.0);
    match id {
        ButtonId::PauseResume => PanelCommand::SetPaused(!state.paused),
        ButtonId::Reset => PanelCommand::Plain(Command::Reset),
        ButtonId::CycleSpawn => PanelCommand::Plain(Command::CycleSpawnMode),
        ButtonId::CycleColor => PanelCommand::Plain(Command::CycleColorMode),
        ButtonId::CycleHud => PanelCommand::Plain(Command::CycleHud),
        ButtonId::Burst => PanelCommand::SpawnBurst(cx, cy),
        ButtonId::Comet => PanelCommand::LaunchComet(cx, cy),
        ButtonId::Explode => PanelCommand::TriggerExplosion(cx, cy),
        ButtonId::Screenshot => PanelCommand::Plain(Command::Screenshot),
        ButtonId::PinWell => PanelCommand::PinWell(cx, cy, Polarity::Attract),
        ButtonId::PinRepeller => PanelCommand::PinWell(cx, cy, Polarity::Repel),
        ButtonId::ClearWells => PanelCommand::Plain(Command::ClearWells),
        ButtonId::ClearWalls => PanelCommand::Plain(Command::ClearWalls),
        ButtonId::ExportScene => PanelCommand::Plain(Command::ExportScene),
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
            ..PanelState::default()
        }
    }

    /// The rect of a control, straight from the layout.
    fn find_slider_track(s: &PanelState, id: SliderId) -> Rect {
        layout(s, 800.0 - PANEL_WIDTH, 0.0)
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
        let (layout, _) = layout(&s, 800.0 - PANEL_WIDTH, 0.0);
        let reset = layout
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Button { id, hit, .. } if *id == ButtonId::Reset => Some(*hit),
                _ => None,
            })
            .expect("reset");
        gui.set_cursor(reset.x + 4.0, reset.y + 4.0);
        let _ = gui.tick(0.016, &s, w, h, false);
        gui.draw(&mut frame, &s, w, h);

        let path = std::path::Path::new("target/panel-preview.png");
        crate::render::write_png(path, &frame, w, h).expect("png written");
        println!("panel preview written to {}", path.display());
    }

    #[test]
    fn detents_snap_and_shift_bypasses() {
        assert!((snap_to_detents(0.4, &[0.0], 100.0, false)).abs() < 1e-12);
        assert!((snap_to_detents(2.9, &[0.0], 100.0, false)).abs() < 1e-12);
        let free = snap_to_detents(2.9, &[0.0], 100.0, true);
        assert!((free - 2.9).abs() < 1e-12, "shift bypasses the snap");
        let out_of_reach = snap_to_detents(4.0, &[0.0], 100.0, false);
        assert!((out_of_reach - 4.0).abs() < 1e-12);
    }

    #[test]
    fn slider_maps_cursor_and_clamps() {
        let track = Rect {
            x: 100.0,
            y: 0.0,
            w: 100.0,
            h: 5.0,
        };
        assert!((slider_value_at(100.0, &track, 0.0, 1.5)).abs() < 1e-12);
        assert!((slider_value_at(200.0, &track, 0.0, 1.5) - 1.5).abs() < 1e-12);
        assert!((slider_value_at(400.0, &track, 0.0, 1.5) - 1.5).abs() < 1e-12);
        assert!((slider_value_at(150.0, &track, 0.0, 1.5) - 0.75).abs() < 1e-12);
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
        let cmds = gui.tick(0.016, &s, 800, 600, false);
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
        let row = layout(&s, 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Toggle { id, .. } if *id == ToggleId::Matter => Some(laid.row),
                _ => None,
            })
            .expect("matter toggle");
        gui.set_cursor(row.x + 4.0, row.y + row.h / 2.0);
        gui.input.pressed = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false);
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
        let row = layout(&s, 800.0 - PANEL_WIDTH, 0.0)
            .0
            .iter()
            .find_map(|laid| match &laid.item {
                Item::Button { id, hit, .. } if *id == ButtonId::Reset => Some(*hit),
                _ => None,
            })
            .expect("reset button");
        gui.set_cursor(row.x + 2.0, row.y + 2.0);
        gui.input.pressed = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false);
        assert!(cmds.is_empty(), "no fire on press");
        // Release off the button cancels.
        gui.set_cursor(0.0, 0.0);
        gui.input.released = true;
        assert!(gui.tick(0.016, &s, 800, 600, false).is_empty());
        // Press and release over it fires.
        gui.set_cursor(row.x + 2.0, row.y + 2.0);
        gui.input.pressed = true;
        gui.tick(0.016, &s, 800, 600, false);
        gui.input.released = true;
        let cmds = gui.tick(0.016, &s, 800, 600, false);
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
        gui.tick(0.016, &s, 800, 600, false);
        assert!(
            gui.scroll <= (gui.content_height - 600.0).max(0.0) + 1e-9,
            "scroll clamped: {} vs content {}",
            gui.scroll,
            gui.content_height
        );
        // And back up: never negative.
        gui.input.wheel = -10_000.0;
        gui.tick(0.016, &s, 800, 600, false);
        assert!(gui.scroll.abs() < 1e-9, "scroll floor at zero");
    }
}
