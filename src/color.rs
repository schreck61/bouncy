// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Color conversion helpers: a leaf module shared by particle creation
//! (random bright colors), rendering (velocity hues, explosion glow), and
//! text blending.

use rand::Rng;

/// Convert f64 color component (0.0-255.0) to u8, clamped to valid range.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn color_component(v: f64) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Convert an HSV hue angle to its sector (0-5) for color calculation.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hue_sector(hue: f64) -> u32 {
    (hue / 60.0) as u32
}

/// Convert an HSV color to an RGBA pixel.
pub fn hsv_to_rgba(hue: f64, saturation: f64, value: f64) -> [u8; 4] {
    let chroma = value * saturation;
    let secondary = chroma * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let match_value = value - chroma;

    let (red, green, blue) = match hue_sector(hue) {
        0 => (chroma, secondary, 0.0),
        1 => (secondary, chroma, 0.0),
        2 => (0.0, chroma, secondary),
        3 => (0.0, secondary, chroma),
        4 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };

    [
        color_component((red + match_value) * 255.0),
        color_component((green + match_value) * 255.0),
        color_component((blue + match_value) * 255.0),
        255,
    ]
}

/// Generate a random bright color using HSV color space.
pub fn random_bright_color(rng: &mut impl Rng) -> [u8; 4] {
    hsv_to_rgba(rng.random_range(0.0..360.0), 0.4, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn hsv_produces_valid_bright_colors() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            let c = random_bright_color(&mut rng);
            assert_eq!(c[3], 255);
            // With value=1.0 and saturation=0.4, max channel is always 255.
            assert_eq!(*c.iter().take(3).max().unwrap(), 255);
        }
    }
}
