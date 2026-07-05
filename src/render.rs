// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Frame rendering: GPU (pixels/wgpu) and CPU (softbuffer) backends plus the
//! drawing routines for particles, explosions, and effects.

use crate::config::ColorMode;
use crate::explosion::{Explosion, EXPLOSION_RING_WIDTH};
use crate::physics::{color_component, hsv_to_rgba, Particle, INITIAL_VELOCITY};
use crate::sim::Well;
use ouroboros::self_referencing;
use pixels::{Pixels, SurfaceTexture};
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

/// Color of a particle under the given color mode.
fn particle_color(particle: &Particle, color_mode: ColorMode) -> [u8; 4] {
    match color_mode {
        ColorMode::Solid => particle.color,
        ColorMode::Velocity => {
            // Map speed to hue: 240 (blue, slow) down to 0 (red, fast).
            let t = (particle.speed() / (INITIAL_VELOCITY * 1.5)).clamp(0.0, 1.0);
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
) {
    for particle in particles {
        // Radius ~1.5 draws the classic 3x3 square; larger radii draw discs.
        #[allow(clippy::cast_possible_truncation)]
        let r = (particle.radius.round() as i32).max(1);
        let disc = r > 1;
        let r_sq = r * r;
        let cx = coord_to_pixel(particle.x);
        let cy = coord_to_pixel(particle.y);
        let color = particle_color(particle, color_mode);

        for dy in -r..=r {
            for dx in -r..=r {
                if disc && dx * dx + dy * dy > r_sq {
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
        let color = if well.direction >= 0 {
            WELL_ATTRACT_COLOR
        } else {
            WELL_REPEL_COLOR
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

/// Render context abstraction supporting both GPU and CPU backends.
pub enum RenderContext {
    Gpu(Box<GpuRenderContext>),
    Cpu(CpuRenderContext),
}

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

    /// Resize the surface (for GPU backend only, when window size changes).
    pub fn resize_surface(&mut self, width: u32, height: u32) {
        if let RenderContext::Gpu(ctx) = self {
            ctx.with_pixels_mut(|pixels| {
                let _ = pixels.resize_surface(width, height);
            });
        }
    }
}

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

/// Precompute the physical->logical nearest-neighbor index map for one axis.
fn scale_map(physical: u32, logical: u32) -> Vec<usize> {
    (0..physical as usize)
        .map(|p| p * logical as usize / physical as usize)
        .collect()
}

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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
            render_particles(&mut frame, &particles, width, height, ColorMode::Solid);
            render_particles(&mut frame, &particles, width, height, ColorMode::Velocity);
        }
        assert!(frame.iter().any(|&b| b > 0));
    }

    #[test]
    fn render_wells_draws_markers_and_clips() {
        let (width, height) = (50u32, 40u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        let wells = [
            Well {
                x: 25.0,
                y: 20.0,
                direction: 1,
            },
            Well {
                x: 0.0,
                y: 0.0,
                direction: -1,
            }, // clipped at the corner
            Well {
                x: -20.0,
                y: 100.0,
                direction: 1,
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
