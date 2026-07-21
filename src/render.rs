// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Frame rendering: GPU (pixels/wgpu) and CPU (softbuffer) backends plus the
//! drawing routines for particles, explosions, and effects.

use crate::color::{color_component, hsv_to_rgba};
use crate::config::ColorMode;
use crate::explosion::{EXPLOSION_RING_WIDTH, Explosion};
use crate::physics::{Particle, Segment};
use crate::sim::{Polarity, Well};
#[cfg(not(target_arch = "wasm32"))]
use ouroboros::self_referencing;
#[cfg(not(target_arch = "wasm32"))]
use pixels::{Pixels, SurfaceTexture};
#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroU32;
use std::rc::Rc;
use winit::window::Window;

/// Convert f64 coordinate to signed pixel position (truncates toward zero).
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn coord_to_pixel(v: f64) -> i32 {
    v as i32
}

/// Convert f64 to unsigned dimension, clamping negative values to 0.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn coord_to_pixel_unsigned(v: f64) -> u32 {
    v.max(0.0) as u32
}

/// Convert f32 color component (0.0-255.0) to u8.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn color_component_f32(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Write `color` at `(x, y)` if it lies inside the frame: the single home
/// for the bounds check and index arithmetic every drawing routine needs.
#[inline]
#[allow(clippy::cast_sign_loss)]
fn put_pixel(frame: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
    if x >= 0 && (x as u32) < width && y >= 0 && (y as u32) < height {
        let idx = ((y as u32) as usize * width as usize + (x as u32) as usize) * 4;
        frame[idx..idx + 4].copy_from_slice(&color);
    }
}

/// Fade the frame toward black, leaving motion trails.
/// Multiplies every channel by ~7/8 using integer math.
pub fn fade_frame(frame: &mut [u8]) {
    for byte in frame.iter_mut() {
        *byte = ((u16::from(*byte) * 224) >> 8) as u8;
    }
}

/// Darken a rectangle of the frame in place (a semi-transparent black
/// overlay), clipped to the frame bounds. `rect` is `(x, y, w, h)`; `keep`
/// is the numerator of the retained brightness out of 256, e.g. 77 keeps
/// ~30%.
pub fn dim_rect(frame: &mut [u8], width: u32, height: u32, rect: (u32, u32, u32, u32), keep: u16) {
    let (x0, y0, w, h) = rect;
    // Clamp the origin too: an x0 beyond the frame would invert the row
    // slice bounds below and panic.
    let x0 = x0.min(width);
    let x1 = x0.saturating_add(w).min(width);
    let y1 = y0.saturating_add(h).min(height);
    for y in y0.min(height)..y1 {
        let row = (y as usize * width as usize + x0 as usize) * 4;
        let row_end = (y as usize * width as usize + x1 as usize) * 4;
        for byte in &mut frame[row..row_end] {
            *byte = ((u16::from(*byte) * keep) >> 8) as u8;
        }
    }
}

/// Color of a particle under the given color mode. `top_speed` is the
/// simulation's configured initial speed: the hue scale spans rest (blue)
/// to 1.5x that speed (red), so slow, gentle presets get the full palette
/// instead of rendering uniformly blue.
fn particle_color(particle: &Particle, color_mode: ColorMode, top_speed: f64) -> [u8; 4] {
    match color_mode {
        ColorMode::Solid => particle.color,
        ColorMode::Velocity => {
            // Map speed to hue: 240 (blue, slow) down to 0 (red, fast).
            let t = (particle.speed() / (top_speed * 1.5)).clamp(0.0, 1.0);
            hsv_to_rgba(240.0 * (1.0 - t), 0.8, 1.0)
        }
    }
}

/// Render all particles as filled squares/discs of their own radius.
pub fn render_particles(
    frame: &mut [u8],
    particles: &[Particle],
    width: u32,
    height: u32,
    color_mode: ColorMode,
    top_speed: f64,
) {
    for particle in particles {
        // The radius rounds to the nearest integer: radius 1 draws a 3x3
        // square, while the default 1.5 rounds up to a 13-pixel disc.
        #[allow(clippy::cast_possible_truncation)]
        let r = (particle.radius.round() as i32).max(1);
        let disc = r > 1;
        let r_sq = r * r;
        let cx = coord_to_pixel(particle.x);
        let cy = coord_to_pixel(particle.y);
        let color = particle_color(particle, color_mode, top_speed);

        for dy in -r..=r {
            for dx in -r..=r {
                if disc && dx * dx + dy * dy > r_sq {
                    continue;
                }
                put_pixel(frame, width, height, cx + dx, cy + dy, color);
            }
        }
    }
}

/// Color of drawn wall segments: warm sandstone, distinct from both the
/// well markers and the random bright particle palette.
const WALL_COLOR: [u8; 4] = [225, 195, 130, 255];
/// A chimed wall flares toward this hot white of the same sandstone hue.
const WALL_FLASH_COLOR: [u8; 4] = [255, 246, 214, 255];

/// Draw the drawn wall segments as 1-pixel lines, clipped to the frame.
/// `flash` holds per-segment chime-flash intensities (0.0 = resting
/// color, 1.0 = full flare), index-aligned with `segments`; segments
/// beyond its length render at rest.
pub fn render_segments(
    frame: &mut [u8],
    segments: &[Segment],
    flash: &[f32],
    width: u32,
    height: u32,
) {
    for (i, seg) in segments.iter().enumerate() {
        let intensity = flash.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let color = if intensity > 0.0 {
            let mut c = WALL_COLOR;
            for (ch, flare) in c.iter_mut().zip(WALL_FLASH_COLOR) {
                let base = f32::from(*ch);
                // Channel math stays within u8 range by construction.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let mixed = (base + (f32::from(flare) - base) * intensity).round() as u8;
                *ch = mixed;
            }
            c
        } else {
            WALL_COLOR
        };
        draw_line(
            frame,
            width,
            height,
            (seg.x1, seg.y1),
            (seg.x2, seg.y2),
            color,
        );
    }
}

/// Bresenham line between two points, clipped per pixel to the frame.
fn draw_line(
    frame: &mut [u8],
    width: u32,
    height: u32,
    from: (f64, f64),
    to: (f64, f64),
    color: [u8; 4],
) {
    let (mut x, mut y) = (coord_to_pixel(from.0), coord_to_pixel(from.1));
    let (x_end, y_end) = (coord_to_pixel(to.0), coord_to_pixel(to.1));
    let dx = (x_end - x).abs();
    let dy = -(y_end - y).abs();
    let sx = if x < x_end { 1 } else { -1 };
    let sy = if y < y_end { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put_pixel(frame, width, height, x, y, color);
        if x == x_end && y == y_end {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Mirror the frame 4-fold about the screen center (kaleidoscope): the
/// top-left quadrant reflects into the other three. Runs as a post-process
/// after all drawing but before the HUD, so overlay text stays readable.
/// For odd dimensions the center row/column belong to the source quadrant.
pub fn kaleidoscope_frame(frame: &mut [u8], width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let row_bytes = w * 4;
    // Reflect the left half of each top-half row onto its right half...
    for y in 0..h.div_ceil(2) {
        let row = y * row_bytes;
        for x in 0..w / 2 {
            let src = row + x * 4;
            let dst = row + (w - 1 - x) * 4;
            frame.copy_within(src..src + 4, dst);
        }
    }
    // ...then reflect the (now symmetric) top half onto the bottom.
    for y in 0..h / 2 {
        let src = y * row_bytes;
        let dst = (h - 1 - y) * row_bytes;
        frame.copy_within(src..src + row_bytes, dst);
    }
}

/// Radius of the pinned-well marker ring in pixels.
const WELL_MARKER_RADIUS: i32 = 7;
/// Marker color for attracting wells (cool cyan: pulls inward).
const WELL_ATTRACT_COLOR: [u8; 4] = [120, 220, 255, 255];
/// Marker color for repelling wells (hot orange: pushes outward).
const WELL_REPEL_COLOR: [u8; 4] = [255, 140, 80, 255];

/// Draw each pinned well as a small ring with a center dot: cyan for
/// attractors, orange for repellers.
pub fn render_wells(frame: &mut [u8], wells: &[Well], width: u32, height: u32) {
    let r = WELL_MARKER_RADIUS;
    for well in wells {
        let color = match well.polarity {
            Polarity::Attract => WELL_ATTRACT_COLOR,
            Polarity::Repel => WELL_REPEL_COLOR,
        };
        let cx = coord_to_pixel(well.x);
        let cy = coord_to_pixel(well.y);
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                let on_ring = d2 <= r * r && d2 >= (r - 1) * (r - 1);
                let on_dot = d2 <= 1;
                if !(on_ring || on_dot) {
                    continue;
                }
                let px = cx + dx;
                let py = cy + dy;

                // Bounds check: px/py are valid pixel coordinates after this check
                #[allow(clippy::cast_sign_loss)]
                if px >= 0 && (px as u32) < width && py >= 0 && (py as u32) < height {
                    let idx = ((py as u32) as usize * width as usize + (px as u32) as usize) * 4;
                    frame[idx..idx + 4].copy_from_slice(&color);
                }
            }
        }
    }
}

/// Render the explosion as an expanding orange ring.
pub fn render_explosion(frame: &mut [u8], exp: &Explosion, width: u32, height: u32) {
    let inner_radius = (exp.radius - EXPLOSION_RING_WIDTH).max(0.0);
    let outer_radius = exp.radius;

    // Calculate bounding box for the ring (clamped to screen bounds)
    let min_x = coord_to_pixel_unsigned(exp.x - outer_radius);
    let max_x = coord_to_pixel_unsigned((exp.x + outer_radius).ceil()).min(width);
    let min_y = coord_to_pixel_unsigned(exp.y - outer_radius);
    let max_y = coord_to_pixel_unsigned((exp.y + outer_radius).ceil()).min(height);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = f64::from(x) - exp.x;
            let dy = f64::from(y) - exp.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist >= inner_radius && dist <= outer_radius {
                let intensity = 1.0 - ((dist - inner_radius) / EXPLOSION_RING_WIDTH).abs();
                let brightness = color_component(intensity * 255.0);
                let idx = (y as usize * width as usize + x as usize) * 4;
                frame[idx] = brightness;
                frame[idx + 1] = color_component_f32(f32::from(brightness) * 0.6);
                frame[idx + 2] = color_component_f32(f32::from(brightness) * 0.2);
                frame[idx + 3] = 255;
            }
        }
    }
}

/// Write an RGBA frame to `path` as an 8-bit RGB PNG. The alpha channel
/// is dropped: trails fade it below 255, which viewers would render as
/// transparency instead of the black the simulation shows.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_png(
    path: &std::path::Path,
    frame: &[u8],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("cannot create '{}': {e}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("PNG header: {e}"))?;
    let rgb: Vec<u8> = frame
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    writer
        .write_image_data(&rgb)
        .map_err(|e| format!("PNG data: {e}"))?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
/// GPU render context using ouroboros for safe self-referential struct.
/// Pixels borrows from Window, so they must be in the same struct.
/// Uses Rc<Window> to allow sharing the window with fallback logic.
#[self_referencing]
pub struct GpuRenderContext {
    window: Rc<Window>,
    width: u32,
    height: u32,
    #[borrows(window)]
    #[covariant]
    pixels: Pixels<'this>,
}

#[cfg(not(target_arch = "wasm32"))]
/// CPU render context using softbuffer (no self-reference needed).
pub struct CpuRenderContext {
    window: Rc<Window>,
    width: u32,           // Logical width (for simulation/rendering)
    height: u32,          // Logical height (for simulation/rendering)
    physical_width: u32,  // Physical width (for softbuffer surface)
    physical_height: u32, // Physical height (for softbuffer surface)
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    buffer: Vec<u8>, // RGBA buffer for rendering functions (logical size)
    // Precomputed physical->logical pixel maps for nearest-neighbor scaling.
    x_map: Vec<usize>,
    y_map: Vec<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
/// Render context abstraction supporting both GPU and CPU backends.
pub enum RenderContext {
    Gpu(Box<GpuRenderContext>),
    Cpu(CpuRenderContext),
}

#[cfg(not(target_arch = "wasm32"))]
impl RenderContext {
    /// Get the window reference for requesting redraws.
    pub fn window(&self) -> &Window {
        match self {
            RenderContext::Gpu(ctx) => ctx.borrow_window(),
            RenderContext::Cpu(ctx) => &ctx.window,
        }
    }

    /// Get the logical width.
    pub fn width(&self) -> u32 {
        match self {
            RenderContext::Gpu(ctx) => *ctx.borrow_width(),
            RenderContext::Cpu(ctx) => ctx.width,
        }
    }

    /// Get the logical height.
    pub fn height(&self) -> u32 {
        match self {
            RenderContext::Gpu(ctx) => *ctx.borrow_height(),
            RenderContext::Cpu(ctx) => ctx.height,
        }
    }

    /// Get a mutable reference to the RGBA frame buffer and call a function with it.
    pub fn with_frame<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        match self {
            RenderContext::Gpu(ctx) => ctx.with_pixels_mut(|pixels| f(pixels.frame_mut())),
            RenderContext::Cpu(ctx) => f(ctx.buffer.as_mut_slice()),
        }
    }

    /// Present the frame to the screen.
    pub fn present(&mut self) -> Result<(), String> {
        match self {
            RenderContext::Gpu(ctx) => {
                ctx.with_pixels_mut(|pixels| pixels.render().map_err(|e| e.to_string()))
            }
            RenderContext::Cpu(ctx) => {
                let mut sb_buffer = ctx.surface.buffer_mut().map_err(|e| e.to_string())?;
                let logical_width = ctx.width as usize;
                let physical_width = ctx.physical_width as usize;
                let physical_height = ctx.physical_height as usize;

                if physical_width == logical_width && physical_height == ctx.height as usize {
                    // Fast path: no scaling, just RGBA -> 0RGB conversion.
                    for (dst, src) in sb_buffer.iter_mut().zip(ctx.buffer.chunks_exact(4)) {
                        *dst = (u32::from(src[0]) << 16)
                            | (u32::from(src[1]) << 8)
                            | u32::from(src[2]);
                    }
                } else {
                    // Nearest-neighbor scale using the precomputed pixel maps.
                    for py in 0..physical_height {
                        let src_row = &ctx.buffer[ctx.y_map[py] * logical_width * 4..];
                        let dst_row = &mut sb_buffer[py * physical_width..][..physical_width];
                        for (dst, &lx) in dst_row.iter_mut().zip(&ctx.x_map) {
                            let src = &src_row[lx * 4..lx * 4 + 3];
                            *dst = (u32::from(src[0]) << 16)
                                | (u32::from(src[1]) << 8)
                                | u32::from(src[2]);
                        }
                    }
                }

                sb_buffer.present().map_err(|e| e.to_string())?;
                Ok(())
            }
        }
    }

    /// Map a cursor position (physical window pixels) to simulation
    /// coordinates, according to how each backend actually presents the
    /// frame. The GPU renderer (pixels) scales the buffer by an *integer*
    /// factor and centers it, so at fractional scale factors (Windows
    /// 125%/150%) the frame is letterboxed — converting the cursor with
    /// the window scale factor instead put interactions inches away from
    /// the pointer. Asking the renderer to invert its own scaling matrix
    /// is correct by construction. The CPU renderer stretches the frame
    /// edge-to-edge, so its mapping is a plain ratio.
    pub fn window_pos_to_sim(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            RenderContext::Gpu(ctx) => {
                let pixels = ctx.borrow_pixels();
                #[allow(clippy::cast_possible_truncation)]
                let (px, py) = pixels
                    .window_pos_to_pixel((x as f32, y as f32))
                    .unwrap_or_else(|outside| pixels.clamp_pixel_pos(outside));
                #[allow(clippy::cast_precision_loss)]
                (px as f64, py as f64)
            }
            RenderContext::Cpu(ctx) => (
                stretch_axis(x, ctx.width, ctx.physical_width),
                stretch_axis(y, ctx.height, ctx.physical_height),
            ),
        }
    }

    /// Resize the surface (for GPU backend only, when window size changes).
    pub fn resize_surface(&mut self, width: u32, height: u32) {
        if let RenderContext::Gpu(ctx) = self {
            ctx.with_pixels_mut(|pixels| {
                let _ = pixels.resize_surface(width, height);
            });
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Create a GPU render context using ouroboros for safe self-referential struct.
fn try_create_gpu_context(
    window: &Rc<Window>,
    width: u32,
    height: u32,
) -> Result<GpuRenderContext, pixels::Error> {
    GpuRenderContextTryBuilder {
        window: Rc::clone(window),
        width,
        height,
        #[allow(clippy::borrowed_box)]
        pixels_builder: |win: &Rc<Window>| {
            let size = win.inner_size();
            let surface_texture = SurfaceTexture::new(size.width, size.height, win.as_ref());
            Pixels::new(width, height, surface_texture)
        },
    }
    .try_build()
}

#[cfg(not(target_arch = "wasm32"))]
/// Create a CPU render context using softbuffer as fallback.
/// `width` and `height` are logical dimensions for the simulation.
/// `physical_width` and `physical_height` are the actual surface dimensions.
fn create_cpu_context(
    window: Rc<Window>,
    width: u32,
    height: u32,
    physical_width: u32,
    physical_height: u32,
) -> CpuRenderContext {
    let context =
        softbuffer::Context::new(Rc::clone(&window)).expect("Failed to create softbuffer context");
    let mut surface = softbuffer::Surface::new(&context, Rc::clone(&window))
        .expect("Failed to create softbuffer surface");
    // Resize to physical dimensions - softbuffer works with actual pixels
    surface
        .resize(
            NonZeroU32::new(physical_width).expect("Width must be > 0"),
            NonZeroU32::new(physical_height).expect("Height must be > 0"),
        )
        .expect("Failed to resize softbuffer surface");
    // Render buffer uses logical dimensions
    let buffer = vec![0u8; (width as usize) * (height as usize) * 4];
    let x_map = scale_map(physical_width, width);
    let y_map = scale_map(physical_height, height);
    CpuRenderContext {
        window,
        width,
        height,
        physical_width,
        physical_height,
        surface,
        buffer,
        x_map,
        y_map,
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Map a physical window coordinate onto one axis of a `logical`-sized
/// frame stretched across `physical` pixels (the CPU renderer's fill).
fn stretch_axis(pos: f64, logical: u32, physical: u32) -> f64 {
    pos * f64::from(logical) / f64::from(physical)
}

#[cfg(not(target_arch = "wasm32"))]
/// Precompute the physical->logical nearest-neighbor index map for one axis.
fn scale_map(physical: u32, logical: u32) -> Vec<usize> {
    (0..physical as usize)
        .map(|p| p * logical as usize / physical as usize)
        .collect()
}

/// Web render context: the same RGBA buffer the drawing routines already
/// target, blitted into a 2D canvas with `putImageData`. The canvas
/// backing store is kept at the simulation's logical size; CSS scales the
/// element for display (letterboxed via `object-fit: contain`), so
/// presentation cost is the browser's, not ours.
#[cfg(target_arch = "wasm32")]
pub struct RenderContext {
    window: Rc<Window>,
    width: u32,
    height: u32,
    buffer: Vec<u8>,
    canvas: web_sys::HtmlCanvasElement,
    ctx2d: web_sys::CanvasRenderingContext2d,
    /// A JS-side copy of the frame. `ImageData` cannot wrap a view into
    /// wasm memory when that memory is shared (the multi-threaded
    /// build), so every present copies the frame into this ordinary
    /// `Uint8ClampedArray`, which `image_data` wraps once per size.
    js_frame: js_sys::Uint8ClampedArray,
    image_data: web_sys::ImageData,
}

/// Build the persistent JS frame array + `ImageData` pair for a size.
#[cfg(target_arch = "wasm32")]
fn js_frame_pair(width: u32, height: u32) -> (js_sys::Uint8ClampedArray, web_sys::ImageData) {
    let js_frame = js_sys::Uint8ClampedArray::new_with_length(width * height * 4);
    let image_data =
        web_sys::ImageData::new_with_js_u8_clamped_array_and_sh(&js_frame, width, height)
            .expect("ImageData creation failed");
    (js_frame, image_data)
}

#[cfg(target_arch = "wasm32")]
impl RenderContext {
    /// The winit window (canvas wrapper) this context renders into.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Frame width in simulation (logical) pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in simulation (logical) pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Run the drawing closure over the RGBA frame buffer.
    pub fn with_frame<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        f(self.buffer.as_mut_slice())
    }

    /// Blit the frame to the canvas: copy into the JS-side array (shared
    /// wasm memory cannot back an `ImageData`) and `putImageData`.
    pub fn present(&mut self) -> Result<(), String> {
        self.js_frame.copy_from(&self.buffer);
        self.ctx2d
            .put_image_data(&self.image_data, 0.0, 0.0)
            .map_err(|_| "putImageData failed".to_string())
    }

    /// Map a cursor position to simulation coordinates. winit's web
    /// backend reports positions in physical pixels (CSS pixels times
    /// devicePixelRatio) relative to the canvas; the canvas is displayed
    /// at its CSS size while the backing store stays at simulation size,
    /// so the mapping is CSS position scaled by the display ratio.
    pub fn window_pos_to_sim(&self, x: f64, y: f64) -> (f64, f64) {
        let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
        let rect = self.canvas.get_bounding_client_rect();
        let (css_w, css_h) = (rect.width().max(1.0), rect.height().max(1.0));
        // object-fit: contain letterboxes the frame inside the element;
        // compute the drawn frame's offset and scale within the box.
        let scale = (css_w / f64::from(self.width)).min(css_h / f64::from(self.height));
        let off_x = (css_w - f64::from(self.width) * scale) / 2.0;
        let off_y = (css_h - f64::from(self.height) * scale) / 2.0;
        let sim_x = (x / dpr - off_x) / scale;
        let sim_y = (y / dpr - off_y) / scale;
        (
            sim_x.clamp(0.0, f64::from(self.width)),
            sim_y.clamp(0.0, f64::from(self.height)),
        )
    }

    /// Window-surface resizes are presentation-only on the web (CSS
    /// scales the canvas element); the backing store follows the
    /// simulation size, not the window.
    pub fn resize_surface(&mut self, _width: u32, _height: u32) {}

    /// Resize the frame to a new simulation size: reallocate the buffer,
    /// the JS-side frame, and the canvas backing store (live-resize
    /// support).
    pub fn resize_sim(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.buffer = vec![0u8; (width as usize) * (height as usize) * 4];
        (self.js_frame, self.image_data) = js_frame_pair(width, height);
        self.canvas.set_width(width);
        self.canvas.set_height(height);
    }
}

/// Create the web render context over the window's canvas. Signature
/// mirrors the native constructor so the shell code is target-agnostic;
/// the physical dimensions and CPU flag are meaningless here.
#[cfg(target_arch = "wasm32")]
pub fn create_render_context(
    window: &Rc<Window>,
    width: u32,
    height: u32,
    _physical_width: u32,
    _physical_height: u32,
    _force_cpu: bool,
) -> RenderContext {
    use winit::platform::web::WindowExtWebSys;

    let canvas = window.canvas().expect("window has no canvas");
    // Backing store at simulation size; CSS (web/style.css) scales the
    // element itself.
    canvas.set_width(width);
    canvas.set_height(height);
    let ctx2d = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .expect("2d context unavailable")
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .expect("2d context has unexpected type");
    let buffer = vec![0u8; (width as usize) * (height as usize) * 4];
    let (js_frame, image_data) = js_frame_pair(width, height);
    RenderContext {
        window: Rc::clone(window),
        width,
        height,
        buffer,
        canvas,
        ctx2d,
        js_frame,
        image_data,
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(not(target_arch = "wasm32"))]
/// Create render context, trying GPU first with CPU fallback (unless `force_cpu` is set).
/// `width` and `height` are logical dimensions.
/// `physical_width` and `physical_height` are the actual surface dimensions.
pub fn create_render_context(
    window: &Rc<Window>,
    width: u32,
    height: u32,
    physical_width: u32,
    physical_height: u32,
    force_cpu: bool,
) -> RenderContext {
    if force_cpu {
        println!("Rendering: CPU (softbuffer) [forced]");
        let cpu_ctx = create_cpu_context(
            Rc::clone(window),
            width,
            height,
            physical_width,
            physical_height,
        );
        window.request_redraw();
        return RenderContext::Cpu(cpu_ctx);
    }

    match try_create_gpu_context(window, width, height) {
        Ok(gpu_ctx) => {
            println!("Rendering: GPU (pixels/wgpu)");
            gpu_ctx.borrow_window().request_redraw();
            RenderContext::Gpu(Box::new(gpu_ctx))
        }
        Err(_gpu_error) => {
            println!("GPU unavailable, using CPU rendering");
            let cpu_ctx = create_cpu_context(
                Rc::clone(window),
                width,
                height,
                physical_width,
                physical_height,
            );
            println!("Rendering: CPU (softbuffer)");
            window.request_redraw();
            RenderContext::Cpu(cpu_ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn stretch_axis_maps_full_physical_range_onto_logical() {
        // Regression context: the friend's 4K-at-150% setup (3840 physical,
        // 2560 logical). The CPU renderer stretches edge-to-edge, so the
        // mapping is a plain ratio; the GPU path instead asks pixels to
        // invert its own (integer-scaled, centered) matrix.
        assert_eq!(stretch_axis(0.0, 2560, 3840), 0.0);
        assert_eq!(stretch_axis(1920.0, 2560, 3840), 1280.0, "center to center");
        assert_eq!(stretch_axis(3840.0, 2560, 3840), 2560.0);
        // Identity when logical == physical (scale factor 1).
        assert_eq!(stretch_axis(123.0, 500, 500), 123.0);
    }

    #[test]
    fn scale_map_covers_full_logical_range() {
        let map = scale_map(200, 100);
        assert_eq!(map.len(), 200);
        assert_eq!(map[0], 0);
        assert_eq!(*map.last().unwrap(), 99);
        assert!(map.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn scale_map_identity_when_equal() {
        let map = scale_map(100, 100);
        assert!(map.iter().enumerate().all(|(i, &v)| i == v));
    }

    #[test]
    fn dim_rect_darkens_only_inside_and_clips() {
        let (width, height) = (10u32, 10u32);
        let mut frame = vec![200u8; (width * height * 4) as usize];
        // Rectangle extends past the frame edge: must clip, not panic.
        dim_rect(&mut frame, width, height, (5, 5, 100, 100), 77);

        let px = |x: u32, y: u32| frame[((y * width + x) * 4) as usize];
        assert_eq!(px(0, 0), 200, "outside must be untouched");
        assert_eq!(px(4, 5), 200, "left of rect must be untouched");
        assert!(px(5, 5) < 200 / 2, "inside must be darkened");
        assert!(px(9, 9) < 200 / 2, "clipped corner must be darkened");
    }

    #[test]
    fn fade_frame_darkens_toward_black() {
        let mut frame = vec![255u8, 128, 10, 255];
        fade_frame(&mut frame);
        assert!(frame[0] < 255);
        assert!(frame[1] < 128);
        // Repeated fading reaches zero.
        for _ in 0..100 {
            fade_frame(&mut frame);
        }
        assert_eq!(frame, vec![0, 0, 0, 0]);
    }

    #[test]
    fn render_particles_stays_in_bounds() {
        let (width, height) = (50u32, 40u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        let mut rng = StdRng::seed_from_u64(9);
        // Particles at the extreme corners must not panic.
        let mut particles = vec![
            Particle::new_at_position(&mut rng, 0.0, 0.0, 1.5, 600.0),
            Particle::new_at_position(&mut rng, 49.9, 39.9, 1.5, 600.0),
            Particle::new_at_position(&mut rng, -5.0, -5.0, 1.5, 600.0),
        ];
        particles[0].color = [1, 2, 3, 255];
        for radius in [1.5, 5.0, 10.0] {
            for p in &mut particles {
                p.radius = radius;
            }
            render_particles(
                &mut frame,
                &particles,
                width,
                height,
                ColorMode::Solid,
                600.0,
            );
            render_particles(
                &mut frame,
                &particles,
                width,
                height,
                ColorMode::Velocity,
                600.0,
            );
        }
        assert!(frame.iter().any(|&b| b > 0));
    }

    #[test]
    fn flashed_segments_flare_toward_white() {
        let (width, height) = (50u32, 40u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        let segments = [
            Segment {
                x1: 5.0,
                y1: 10.0,
                x2: 15.0,
                y2: 10.0,
            },
            Segment {
                x1: 5.0,
                y1: 20.0,
                x2: 15.0,
                y2: 20.0,
            },
        ];
        // Segment 0 at full flare, segment 1 resting.
        render_segments(&mut frame, &segments, &[1.0, 0.0], width, height);
        let px = |x: u32, y: u32| {
            let i = ((y * width + x) * 4) as usize;
            [frame[i], frame[i + 1], frame[i + 2], frame[i + 3]]
        };
        assert_eq!(px(10, 10), WALL_FLASH_COLOR, "flashed bar flares");
        assert_eq!(px(10, 20), WALL_COLOR, "resting bar keeps its color");
        // A missing flash entry is a resting wall, never a panic.
        render_segments(&mut frame, &segments, &[0.5], width, height);
    }

    #[test]
    fn render_segments_draws_lines_and_clips() {
        let (width, height) = (50u32, 40u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        let segments = [
            Segment {
                x1: 5.0,
                y1: 10.0,
                x2: 15.0,
                y2: 10.0,
            },
            // Runs off both ends of the frame: must clip, not panic.
            Segment {
                x1: -10.0,
                y1: -10.0,
                x2: 70.0,
                y2: 60.0,
            },
        ];
        render_segments(&mut frame, &segments, &[], width, height);

        let px = |x: u32, y: u32| frame[((y * width + x) * 4) as usize];
        assert_eq!(px(5, 10), WALL_COLOR[0], "line start drawn");
        assert_eq!(px(10, 10), WALL_COLOR[0], "line middle drawn");
        assert_eq!(px(15, 10), WALL_COLOR[0], "line end drawn");
        assert_eq!(px(10, 9), 0, "above the line untouched");
        // The clipped diagonal still leaves its on-screen trace: it must
        // cross every row of the frame somewhere.
        for y in 0..height {
            assert!(
                (0..width).any(|x| px(x, y) == WALL_COLOR[0]),
                "diagonal missing from row {y}"
            );
        }
    }

    #[test]
    fn kaleidoscope_mirrors_four_fold_and_handles_odd_sizes() {
        // Even size: a marked pixel appears in all four quadrants and
        // stale content in the other quadrants is overwritten.
        {
            let (w, h) = (6u32, 4u32);
            let mut frame = vec![0u8; (w * h * 4) as usize];
            let idx = |x: u32, y: u32| ((y * w + x) * 4) as usize;
            frame[idx(1, 1)] = 200;
            frame[idx(4, 2)] = 99; // stale pixel in the bottom-right quadrant
            kaleidoscope_frame(&mut frame, w, h);
            assert_eq!(frame[idx(1, 1)], 200, "source pixel survives");
            assert_eq!(frame[idx(4, 1)], 200, "mirrored horizontally");
            assert_eq!(frame[idx(1, 2)], 200, "mirrored vertically");
            assert_eq!(frame[idx(4, 2)], 200, "stale content overwritten");
        }

        // Odd size: the center row/column belong to the source quadrant.
        {
            let (w, h) = (5u32, 5u32);
            let mut frame = vec![0u8; (w * h * 4) as usize];
            let idx = |x: u32, y: u32| ((y * w + x) * 4) as usize;
            frame[idx(2, 2)] = 150; // dead center
            frame[idx(0, 2)] = 70; // center row, left edge
            kaleidoscope_frame(&mut frame, w, h);
            assert_eq!(frame[idx(2, 2)], 150, "center pixel survives");
            assert_eq!(frame[idx(4, 2)], 70, "center row mirrors horizontally");
            assert_eq!(frame[idx(0, 2)], 70, "left edge untouched");
        }
    }

    #[test]
    fn write_png_roundtrips_the_frame() {
        let (width, height) = (3u32, 2u32);
        // Distinct per-pixel colors; alpha varies to prove it is dropped.
        // Wrapping is the point: distinct byte values per position.
        #[allow(clippy::cast_possible_truncation)]
        let frame: Vec<u8> = (0..width * height * 4).map(|i| (i * 7) as u8).collect();
        let path = std::env::temp_dir().join(format!(
            "bouncy-test-{}-{:?}.png",
            std::process::id(),
            std::thread::current().id()
        ));
        write_png(&path, &frame, width, height).unwrap();

        let decoder = png::Decoder::new(std::fs::File::open(&path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!((info.width, info.height), (width, height));
        assert_eq!(info.color_type, png::ColorType::Rgb);
        let expected: Vec<u8> = frame
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect();
        assert_eq!(&buf[..expected.len()], &expected[..]);
    }

    #[test]
    fn velocity_colors_span_the_configured_speed_range() {
        // Regression: hue used to normalize against the INITIAL_VELOCITY
        // constant (600), so slow presets rendered uniformly blue. A
        // particle at 1.5x the *configured* top speed must render red no
        // matter how slow that configuration is.
        let (width, height) = (20u32, 20u32);
        let mut rng = StdRng::seed_from_u64(2);
        let center = |frame: &[u8]| {
            let idx = ((10 * width + 10) * 4) as usize;
            [frame[idx], frame[idx + 1], frame[idx + 2]]
        };

        let mut fast = Particle::new_at_position(&mut rng, 10.0, 10.0, 2.0, 600.0);
        fast.vx = 90.0; // 1.5x of a gentle 60 px/s setting
        fast.vy = 0.0;
        let mut frame = vec![0u8; (width * height * 4) as usize];
        render_particles(
            &mut frame,
            &[fast],
            width,
            height,
            ColorMode::Velocity,
            60.0,
        );
        let [r, _, b] = center(&frame);
        assert!(r > 200 && b < 100, "top of the range is red: r={r} b={b}");

        let mut slow = Particle::new_at_position(&mut rng, 10.0, 10.0, 2.0, 600.0);
        slow.vx = 0.0;
        slow.vy = 0.0;
        let mut frame = vec![0u8; (width * height * 4) as usize];
        render_particles(
            &mut frame,
            &[slow],
            width,
            height,
            ColorMode::Velocity,
            60.0,
        );
        let [r, _, b] = center(&frame);
        assert!(b > 200 && r < 100, "rest is blue: r={r} b={b}");
    }

    #[test]
    fn render_wells_draws_markers_and_clips() {
        let (width, height) = (50u32, 40u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        let wells = [
            Well {
                x: 25.0,
                y: 20.0,
                polarity: Polarity::Attract,
            },
            Well {
                x: 0.0,
                y: 0.0,
                polarity: Polarity::Repel,
            }, // clipped at the corner
            Well {
                x: -20.0,
                y: 100.0,
                polarity: Polarity::Attract,
            }, // fully off-screen
        ];
        render_wells(&mut frame, &wells, width, height);

        let px = |x: u32, y: u32| {
            let idx = ((y * width + x) * 4) as usize;
            [frame[idx], frame[idx + 1], frame[idx + 2], frame[idx + 3]]
        };
        // Attractor: cyan on the ring and at the center dot, hollow between.
        assert_eq!(px(25, 13), WELL_ATTRACT_COLOR, "top of the ring");
        assert_eq!(px(25, 20), WELL_ATTRACT_COLOR, "center dot");
        assert_eq!(px(25, 16), [0, 0, 0, 0], "ring interior stays hollow");
        // Repeller: its ring reaches into the frame despite clipping.
        assert_eq!(px(7, 0), WELL_REPEL_COLOR, "clipped repeller still drawn");
    }

    #[test]
    fn render_explosion_stays_in_bounds() {
        let (width, height) = (50u32, 40u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        let mut rng = StdRng::seed_from_u64(9);
        let mut particles: Vec<Particle> = Vec::new();
        let mut exp =
            crate::explosion::Explosion::new(&mut rng, 25.0, 20.0, 1000.0, &mut particles, 1.0, 0);
        // Expand well past the frame bounds; drawing must clip.
        for _ in 0..100 {
            exp.update(0.016);
        }
        render_explosion(&mut frame, &exp, width, height);
    }
}
