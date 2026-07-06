// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Application shell: windowing, input handling, HUD, audio dispatch, and
//! the winit event loop glue around the headless [`Simulation`] core.

use crate::audio::Audio;
use crate::config::{ColorMode, Config};
use crate::render::{
    create_render_context, dim_rect, fade_frame, kaleidoscope_frame, render_explosion,
    render_particles, render_segments, render_wells, RenderContext,
};
use crate::sim::{Simulation, Well, MAX_PINNED_WELLS, MAX_WALL_SEGMENTS};
use crate::text::{draw_text, draw_text_centered, measure_text};
use std::rc::Rc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

/// Frames of physics skipped at startup to let the GPU initialize.
const WARMUP_FRAMES: u32 = 3;
/// Runtime gravity adjustment step (percent) for Up/Down arrows.
const GRAVITY_STEP: i32 = 10;
/// Runtime elasticity adjustment step for arrow and bracket keys.
const ELASTICITY_STEP: f64 = 0.05;
/// Runtime explosion-threshold adjustment step for the -/= keys.
const THRESHOLD_STEP: usize = 5;
const THRESHOLD_MAX: usize = 1000;
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
}

/// Convert physical pixels to logical pixels given a scale factor.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn physical_to_logical(physical: u32, scale_factor: f64) -> u32 {
    (f64::from(physical) / scale_factor) as u32
}

/// Main application state: I/O around the simulation core.
pub struct App {
    // Configuration (kept to construct the simulation once dimensions are known)
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
    frame_count: u64,
    fps_timer: Instant,
    warmup_frames: u32,
    current_fps: f64,

    // Interaction state
    paused: bool,
    step_once: bool,
    hud_mode: HudMode,
    cursor_x: f64,
    cursor_y: f64,
    /// Cursor gravity well: 0 = off, 1 = attract (G held), -1 = repel (Shift+G).
    well_direction: i8,
    /// Wall drawing (V held): where the next segment starts, if active.
    wall_anchor: Option<(f64, f64)>,
    shift_down: bool,
    time_scale: f64,
    /// Bullet-time choreography: when the current slowdown started, if one
    /// is active. Purely presentational — the simulation only sees a
    /// smaller dt.
    bullet_time_start: Option<Instant>,
    /// Bullet time on explosions enabled (--bullet-time opts in).
    bullet_time_enabled: bool,
    /// Cursor auto-hide state: what visibility we last set on the window.
    cursor_visible: bool,
    last_cursor_move: Instant,
    window_focused: bool,
    cursor_inside: bool,

    // Render failure tracking (transient surface loss should not crash)
    consecutive_render_failures: u32,
}

impl App {
    /// Create a new App with the given configuration.
    pub fn new(config: Config) -> Self {
        App {
            trails: config.trails,
            color_mode: config.color_mode,
            kaleidoscope: config.kaleidoscope,
            verbose: config.verbose,
            bullet_time_enabled: config.bullet_time,
            audio: Audio::new(config.mute, config.music),
            sim: None,
            config,
            render: None,
            scale_factor: 1.0,
            last_time: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            warmup_frames: WARMUP_FRAMES,
            current_fps: 0.0,
            paused: false,
            step_once: false,
            hud_mode: HudMode::Hidden,
            cursor_x: 0.0,
            cursor_y: 0.0,
            well_direction: 0,
            wall_anchor: None,
            shift_down: false,
            time_scale: 1.0,
            bullet_time_start: None,
            cursor_visible: true,
            last_cursor_move: Instant::now(),
            window_focused: true,
            cursor_inside: false,
            consecutive_render_failures: 0,
        }
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.render.as_ref().map(|r| (r.width(), r.height()))
    }

    /// The gravity well input for this frame, if held.
    fn well(&self) -> Option<Well> {
        (self.well_direction != 0).then_some(Well {
            x: self.cursor_x,
            y: self.cursor_y,
            direction: self.well_direction,
        })
    }

    /// Update FPS counter and (in verbose mode) print statistics.
    fn update_fps_counter(&mut self) {
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            // Precision loss acceptable: frame_count is small relative to f64 mantissa
            #[allow(clippy::cast_precision_loss)]
            let fps = self.frame_count as f64 / elapsed;
            self.current_fps = fps;
            if self.verbose {
                let count = self.sim.as_ref().map_or(0, Simulation::particle_count);
                println!("FPS: {fps:.1}, Particles: {count}");
            }
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }

    /// Advance the simulation and dispatch resulting audio.
    fn simulate(&mut self, dt: f64, now: Instant) {
        let well = self.well();
        let Some(ref mut sim) = self.sim else {
            return;
        };
        let events = sim.step(dt, now, well);

        if events.max_collision_energy > 0.0 {
            #[allow(clippy::cast_possible_truncation)]
            self.audio
                .play_ping(events.max_collision_energy, events.collision_pan as f32);
        }
        if events.explosion_started {
            self.audio.play_explosion();
            self.start_bullet_time(now);
        }
    }

    /// Enter bullet time (an explosion ring just started), if enabled. A
    /// ring starting during another ring's ramp restarts the dip.
    fn start_bullet_time(&mut self, now: Instant) {
        if self.bullet_time_enabled {
            self.bullet_time_start = Some(now);
        }
    }

    /// This frame's bullet-time multiplier on the time scale, expiring the
    /// effect once the ramp back to full speed completes.
    fn bullet_time_multiplier(&mut self, now: Instant) -> f64 {
        let Some(start) = self.bullet_time_start else {
            return 1.0;
        };
        let elapsed = now.duration_since(start).as_secs_f64();
        if elapsed >= BULLET_TIME_HOLD_SECS + BULLET_TIME_RAMP_SECS {
            self.bullet_time_start = None;
            return 1.0;
        }
        bullet_time_factor(elapsed)
    }

    /// Build the HUD overlay text for the current mode.
    fn hud_lines(&self, sim: &Simulation) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(ref preset) = self.config.preset {
            lines.push(format!("Preset: {preset}"));
        }
        lines.extend([
            format!("FPS: {:.1}", self.current_fps),
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
                    "Explosions: at {}/s spawn rate  (-/=)",
                    sim.explosion_threshold
                )
            },
            format!("Spawn: {}  (B)", sim.spawn_mode.label()),
            format!(
                "Matter: {}  (X)   Flow: {}  (F)",
                if sim.matter { "on" } else { "off" },
                if sim.flow { "on" } else { "off" },
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
                "Music: {}  (S)   Kaleidoscope: {}  (K)",
                if self.audio.is_music() { "on" } else { "off" },
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
        if self.well_direction > 0 {
            flags.push("WELL: ATTRACT");
        } else if self.well_direction < 0 {
            flags.push("WELL: REPEL");
        }
        if self.bullet_time_start.is_some() {
            flags.push("BULLET TIME");
        }
        if !flags.is_empty() {
            lines.push(flags.join("  "));
        }

        if self.hud_mode == HudMode::StatsAndKeys {
            lines.push(String::new());
            for key_line in [
                "P pause   N step   R reset   M mute",
                "T trails   C colors   B spawn mode",
                "X matter (fusion/fission)   F flow field",
                "S musical pings   K kaleidoscope",
                "G hold: gravity well (Shift+G repels)",
                "W pin well (Shift+W repel, Shift+R clear)",
                "V hold+drag: draw walls (Shift+V clears)",
                "Click: burst   Right-click: explosion",
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

        let Some(ref mut render) = self.render else {
            return;
        };

        let trails = self.trails;
        let color_mode = self.color_mode;
        let kaleidoscope = self.kaleidoscope;
        let stopped = sim.stopped();
        let paused = self.paused;

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
            render_segments(frame, sim.wall_segments(), width, height);
            render_particles(frame, sim.particles(), width, height, color_mode);
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
        });

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
        let idle = self.last_cursor_move.elapsed().as_secs_f64();
        let desired = cursor_should_be_visible(
            self.window_focused,
            self.cursor_inside,
            self.well_direction != 0 || self.wall_anchor.is_some(),
            idle,
        );
        if desired != self.cursor_visible {
            if let Some(ref render) = self.render {
                render.window().set_cursor_visible(desired);
            }
            self.cursor_visible = desired;
        }
    }

    fn update_and_render(&mut self) {
        self.update_cursor_visibility();
        let Some((width, height)) = self.dimensions() else {
            return;
        };
        let now = Instant::now();

        // Warmup frames for GPU initialization
        if self.warmup_frames > 0 {
            self.warmup_frames -= 1;
            self.last_time = now;
            self.fps_timer = now;
            self.frame_count = 0;
            self.render_frame(width, height);
            return;
        }

        let dt = now.duration_since(self.last_time).as_secs_f64().min(0.05);
        self.last_time = now;

        let time_scale = self.time_scale * self.bullet_time_multiplier(now);
        if !self.paused {
            self.simulate(dt * time_scale, now);
        } else if self.step_once {
            // Frame-step (N while paused): advance one fixed-size step.
            self.simulate(FRAME_STEP_DT * time_scale, now);
        }
        self.step_once = false;

        self.render_frame(width, height);
        self.update_fps_counter();
    }

    /// Handle a key press or release.
    fn handle_key(
        &mut self,
        key_code: KeyCode,
        state: ElementState,
        repeat: bool,
        event_loop: &ActiveEventLoop,
    ) {
        if state == ElementState::Released {
            if key_code == KeyCode::KeyG {
                self.well_direction = 0;
            }
            if key_code == KeyCode::KeyV {
                self.wall_anchor = None;
            }
            return;
        }

        match key_code {
            KeyCode::Space | KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
            KeyCode::KeyP if !repeat => {
                self.paused = !self.paused;
                println!("{}", if self.paused { "Paused" } else { "Resumed" });
            }
            KeyCode::KeyN if self.paused => self.step_once = true,
            KeyCode::KeyR if !repeat => {
                if let Some(ref mut sim) = self.sim {
                    if self.shift_down {
                        let cleared = sim.clear_wells();
                        println!("Cleared {cleared} pinned wells");
                    } else {
                        println!("Simulation reset");
                        sim.reset();
                        self.paused = false;
                    }
                }
            }
            KeyCode::KeyM if !repeat => {
                let muted = self.audio.toggle_mute();
                println!("Audio {}", if muted { "muted" } else { "unmuted" });
            }
            KeyCode::KeyH if !repeat => self.hud_mode = self.hud_mode.next(),
            KeyCode::KeyT if !repeat => {
                self.trails = !self.trails;
                println!("Trails {}", if self.trails { "on" } else { "off" });
            }
            KeyCode::KeyC if !repeat => {
                self.color_mode = match self.color_mode {
                    ColorMode::Solid => ColorMode::Velocity,
                    ColorMode::Velocity => ColorMode::Solid,
                };
            }
            KeyCode::KeyB if !repeat => {
                if let Some(ref mut sim) = self.sim {
                    sim.spawn_mode = sim.spawn_mode.next();
                    println!("Spawn mode: {}", sim.spawn_mode.label());
                }
            }
            KeyCode::KeyX if !repeat => {
                if let Some(ref mut sim) = self.sim {
                    sim.matter = !sim.matter;
                    println!("Matter mechanics {}", if sim.matter { "on" } else { "off" });
                }
            }
            KeyCode::KeyF if !repeat => {
                if let Some(ref mut sim) = self.sim {
                    sim.flow = !sim.flow;
                    println!("Flow field {}", if sim.flow { "on" } else { "off" });
                    if sim.flow {
                        // The flow is about to move particles.
                        sim.wake();
                    }
                }
            }
            KeyCode::KeyS if !repeat => {
                let music = self.audio.toggle_music();
                println!(
                    "Musical pings {}",
                    if music { "on (pentatonic)" } else { "off" }
                );
            }
            KeyCode::KeyK if !repeat => {
                self.kaleidoscope = !self.kaleidoscope;
                println!(
                    "Kaleidoscope {}",
                    if self.kaleidoscope { "on" } else { "off" }
                );
            }
            KeyCode::KeyG => {
                self.well_direction = if self.shift_down { -1 } else { 1 };
                // The well moves particles; leave the stopped state if set.
                if let Some(ref mut sim) = self.sim {
                    sim.wake();
                }
            }
            KeyCode::KeyV if !repeat => {
                if self.shift_down {
                    if let Some(ref mut sim) = self.sim {
                        let cleared = sim.clear_wall_segments();
                        println!("Cleared {cleared} wall segments");
                    }
                } else {
                    // Start drawing: segments are added as the cursor moves.
                    self.wall_anchor = Some((self.cursor_x, self.cursor_y));
                }
            }
            KeyCode::KeyW if !repeat => {
                if let Some(ref mut sim) = self.sim {
                    let direction = if self.shift_down { -1 } else { 1 };
                    if sim.pin_well(self.cursor_x, self.cursor_y, direction) {
                        println!(
                            "Pinned {} well at ({:.0}, {:.0}); {} total (Shift+R clears)",
                            if direction > 0 {
                                "attracting"
                            } else {
                                "repelling"
                            },
                            self.cursor_x,
                            self.cursor_y,
                            sim.pinned_wells().len()
                        );
                    } else {
                        println!("Pinned well limit reached ({MAX_PINNED_WELLS})");
                    }
                }
            }
            KeyCode::ArrowUp => {
                if let Some(ref mut sim) = self.sim {
                    sim.gravity_percent = (sim.gravity_percent + GRAVITY_STEP).min(1000);
                    println!("Gravity: {}%", sim.gravity_percent);
                }
            }
            KeyCode::ArrowDown => {
                if let Some(ref mut sim) = self.sim {
                    sim.gravity_percent = (sim.gravity_percent - GRAVITY_STEP).max(-1000);
                    println!("Gravity: {}%", sim.gravity_percent);
                }
            }
            KeyCode::ArrowRight => {
                if let Some(ref mut sim) = self.sim {
                    sim.particle_elasticity = (sim.particle_elasticity + ELASTICITY_STEP).min(1.5);
                    println!("Particle elasticity: {:.2}", sim.particle_elasticity);
                }
            }
            KeyCode::ArrowLeft => {
                if let Some(ref mut sim) = self.sim {
                    sim.particle_elasticity = (sim.particle_elasticity - ELASTICITY_STEP).max(0.0);
                    println!("Particle elasticity: {:.2}", sim.particle_elasticity);
                }
            }
            KeyCode::BracketRight => {
                if let Some(ref mut sim) = self.sim {
                    sim.wall_elasticity = (sim.wall_elasticity + ELASTICITY_STEP).min(1.5);
                    println!("Wall elasticity: {:.2}", sim.wall_elasticity);
                }
            }
            KeyCode::BracketLeft => {
                if let Some(ref mut sim) = self.sim {
                    sim.wall_elasticity = (sim.wall_elasticity - ELASTICITY_STEP).max(0.0);
                    println!("Wall elasticity: {:.2}", sim.wall_elasticity);
                }
            }
            KeyCode::Period => self.adjust_time_scale(TIME_SCALE_STEP),
            KeyCode::Comma => self.adjust_time_scale(-TIME_SCALE_STEP),
            KeyCode::Equal => {
                if let Some(ref mut sim) = self.sim {
                    sim.explosion_threshold =
                        (sim.explosion_threshold + THRESHOLD_STEP).min(THRESHOLD_MAX);
                    println!("Explosion threshold: {}/s", sim.explosion_threshold);
                }
            }
            KeyCode::Minus => {
                if let Some(ref mut sim) = self.sim {
                    sim.explosion_threshold =
                        sim.explosion_threshold.saturating_sub(THRESHOLD_STEP);
                    if sim.explosion_threshold == 0 {
                        println!("Explosion threshold: off");
                    } else {
                        println!("Explosion threshold: {}/s", sim.explosion_threshold);
                    }
                }
            }
            _ => {}
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
    fn extend_wall(&mut self) {
        let Some((ax, ay)) = self.wall_anchor else {
            return;
        };
        let (dx, dy) = (self.cursor_x - ax, self.cursor_y - ay);
        if dx * dx + dy * dy < WALL_SEGMENT_MIN_LENGTH * WALL_SEGMENT_MIN_LENGTH {
            return;
        }
        let Some(ref mut sim) = self.sim else {
            return;
        };
        if sim.add_wall_segment(ax, ay, self.cursor_x, self.cursor_y) {
            self.wall_anchor = Some((self.cursor_x, self.cursor_y));
        } else {
            // Stop the stroke at the cap instead of retrying every motion.
            println!("Wall segment limit reached ({MAX_WALL_SEGMENTS})");
            self.wall_anchor = None;
        }
    }

    /// Handle a mouse button press at the current cursor position.
    fn handle_mouse(&mut self, button: MouseButton) {
        let (x, y) = (self.cursor_x, self.cursor_y);
        let Some(ref mut sim) = self.sim else {
            return;
        };
        if button == MouseButton::Left {
            sim.spawn_burst(x, y);
        } else if button == MouseButton::Right && sim.trigger_manual_explosion(x, y) {
            println!("Explosion triggered at cursor");
            self.audio.play_explosion();
            self.start_bullet_time(Instant::now());
        }
    }
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
        let width = physical_to_logical(physical_size.width, scale_factor);
        let height = physical_to_logical(physical_size.height, scale_factor);
        self.scale_factor = scale_factor;

        println!(
            "Window: {}x{} physical, {}x{} logical, scale={}",
            physical_size.width, physical_size.height, width, height, scale_factor
        );

        self.sim = Some(Simulation::new(&self.config, width, height));
        self.last_time = Instant::now();
        self.fps_timer = Instant::now();
        self.frame_count = 0;

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
                    (self.cursor_x, self.cursor_y) =
                        render.window_pos_to_sim(position.x, position.y);
                }
                self.cursor_inside = true;
                self.last_cursor_move = Instant::now();
                self.extend_wall();
            }
            WindowEvent::CursorEntered { .. } => {
                self.cursor_inside = true;
                self.last_cursor_move = Instant::now();
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_inside = false;
            }
            WindowEvent::Focused(focused) => {
                self.window_focused = focused;
                if focused {
                    // Fresh focus should not instantly hide the cursor.
                    self.last_cursor_move = Instant::now();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                self.handle_mouse(button);
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

    #[test]
    fn time_scale_steps_retrace_to_the_starting_point() {
        // Regression: the old multiplicative step (x1.25) could never
        // land back on 1.0 after clamping at either end of the range.
        let config = Config::try_resolve_from(["bouncy"]).unwrap();
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
}
