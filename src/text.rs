// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Bitmap text rendering onto RGBA frames using an embedded font.

use ab_glyph::{point, Font, FontRef, Glyph, PxScale, ScaleFont};
use std::sync::OnceLock;

/// Embedded font data (Liberation Sans Bold - SIL Open Font License).
const FONT_DATA: &[u8] = include_bytes!("../assets/LiberationSans-Bold.ttf");

/// Lazily-initialized cached font to avoid parsing on every render.
fn get_font() -> &'static FontRef<'static> {
    static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
    FONT.get_or_init(|| FontRef::try_from_slice(FONT_DATA).expect("Failed to load embedded font"))
}

/// Lay out `text` at `font_size`, returning positioned glyphs (relative to a
/// `(0, ascent)` baseline origin) and the pixel dimensions of the text block.
fn layout(text: &str, font_size: f32) -> (Vec<Glyph>, f32, f32) {
    let font = get_font();
    let scale = PxScale::from(font_size);
    let scaled = font.as_scaled(scale);

    let mut glyphs = Vec::with_capacity(text.len());
    let mut caret_x = 0.0f32;
    let mut previous = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(prev) = previous {
            caret_x += scaled.kern(prev, id);
        }
        glyphs.push(id.with_scale_and_position(scale, point(caret_x, scaled.ascent())));
        caret_x += scaled.h_advance(id);
        previous = Some(id);
    }

    (glyphs, caret_x, scaled.ascent() - scaled.descent())
}

/// Measure the pixel dimensions of `text` at `font_size`.
pub fn measure_text(text: &str, font_size: f32) -> (f32, f32) {
    let (_, w, h) = layout(text, font_size);
    (w, h)
}

/// Draw `text` with its top-left corner at `origin`, alpha-blending glyph
/// coverage over the existing frame contents.
pub fn draw_text(
    frame: &mut [u8],
    width: u32,
    height: u32,
    text: &str,
    font_size: f32,
    origin: (f32, f32),
    color: [u8; 3],
) {
    let font = get_font();
    let (glyphs, _, _) = layout(text, font_size);

    for glyph in glyphs {
        let mut glyph = glyph;
        glyph.position.x += origin.0;
        glyph.position.y += origin.1;
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, coverage| {
            // Glyph-relative coords offset by the outline bounds give the
            // frame pixel position; glyph extents are far below i32::MAX.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let px = bounds.min.x as i32 + gx as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let py = bounds.min.y as i32 + gy as i32;

            #[allow(clippy::cast_sign_loss)]
            if px >= 0 && (px as u32) < width && py >= 0 && (py as u32) < height {
                let idx = ((py as u32) * width + (px as u32)) as usize * 4;
                let alpha = u16::from(crate::physics::color_component(f64::from(
                    coverage.clamp(0.0, 1.0) * 255.0,
                )));
                // Standard source-over blend; results always fit in u8.
                #[allow(clippy::cast_possible_truncation)]
                for (channel, &c) in color.iter().enumerate() {
                    let dst = u16::from(frame[idx + channel]);
                    frame[idx + channel] =
                        ((u16::from(c) * alpha + dst * (255 - alpha)) / 255) as u8;
                }
                frame[idx + 3] = 255;
            }
        });
    }
}

/// Draw `text` horizontally and vertically centered on the frame.
pub fn draw_text_centered(
    frame: &mut [u8],
    width: u32,
    height: u32,
    text: &str,
    font_size: f32,
    color: [u8; 3],
) {
    let (text_width, text_height) = measure_text(text, font_size);
    #[allow(clippy::cast_precision_loss)]
    let origin_x = ((width as f32 - text_width) / 2.0).max(0.0);
    #[allow(clippy::cast_precision_loss)]
    let origin_y = ((height as f32 - text_height) / 2.0).max(0.0);
    draw_text(
        frame,
        width,
        height,
        text,
        font_size,
        (origin_x, origin_y),
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_nonzero_dimensions() {
        let (w, h) = measure_text("STOPPED", 72.0);
        assert!(w > 0.0);
        assert!(h > 0.0);
        let (w2, _) = measure_text("STOPPED STOPPED", 72.0);
        assert!(w2 > w);
    }

    #[test]
    fn draws_pixels_into_frame() {
        let (width, height) = (200u32, 100u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        draw_text_centered(&mut frame, width, height, "Hi", 40.0, [255, 100, 100]);
        assert!(frame.iter().any(|&b| b > 0), "expected some pixels drawn");
    }

    #[test]
    fn clips_text_larger_than_frame() {
        let (width, height) = (10u32, 10u32);
        let mut frame = vec![0u8; (width * height * 4) as usize];
        // Must not panic even though the text is far larger than the frame.
        draw_text_centered(&mut frame, width, height, "STOPPED", 72.0, [255, 255, 255]);
        draw_text(
            &mut frame,
            width,
            height,
            "X",
            40.0,
            (-5.0, -5.0),
            [255, 255, 255],
        );
    }

    #[test]
    fn blends_over_background() {
        let (width, height) = (100u32, 60u32);
        // White background: text pixels should stay within u8 without wrap.
        let mut frame = vec![255u8; (width * height * 4) as usize];
        draw_text_centered(&mut frame, width, height, "O", 40.0, [0, 0, 0]);
        assert!(frame.iter().any(|&b| b < 255), "expected darkened pixels");
    }
}
