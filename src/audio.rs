// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Synthesized audio: collision pings and explosion rumbles.
//!
//! Audio is optional — if no output device is available the simulation runs
//! silently. Pings are pre-generated at startup into a small set of pitch
//! buckets so playback allocates nothing per collision, and are panned
//! left/right based on where on screen the collision happened.

use crate::physics::COLLISION_ENERGY_NORMALIZER;
use rand::Rng;
use rodio::buffer::SamplesBuffer;
use rodio::source::ChannelVolume;
use rodio::{OutputStream, OutputStreamBuilder};

pub const AUDIO_SAMPLE_RATE: u32 = 44100;
const PING_DURATION_MS: u64 = 80;
const PING_MIN_FREQ: f32 = 300.0;
const PING_MAX_FREQ: f32 = 1500.0;
const PING_BUCKETS: usize = 16;
const EXPLOSION_DURATION_MS: u64 = 800;

/// Generate a ping sound with exponential decay at the given frequency.
fn generate_ping(frequency: f32) -> SamplesBuffer {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * PING_DURATION_MS / 1000) as usize;
    // u32 sample rate fits in f32 mantissa (44100 << 2^23)
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            // Precision loss from usize->f32 is acceptable for audio timing
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let envelope = (-t * 20.0).exp();
            let wave = (2.0 * std::f32::consts::PI * frequency * t).sin();
            wave * envelope * 0.3
        })
        .collect();

    SamplesBuffer::new(1, AUDIO_SAMPLE_RATE, samples)
}

/// Generate a low-frequency rumble sound for explosions.
fn generate_explosion() -> SamplesBuffer {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * EXPLOSION_DURATION_MS / 1000) as usize;
    // u32 sample rate fits in f32 mantissa (44100 << 2^23)
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;
    let mut rng = rand::rng();

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            // Precision loss from usize->f32 is acceptable for audio timing
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let envelope = if t < 0.05 {
                t / 0.05
            } else {
                (-((t - 0.05) * 3.0)).exp()
            };

            let rumble = (2.0 * std::f32::consts::PI * 60.0 * t).sin() * 0.4
                + (2.0 * std::f32::consts::PI * 80.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 40.0 * t).sin() * 0.3;

            let noise: f32 = rng.random_range(-1.0..1.0) * 0.5;

            (rumble + noise) * envelope * 0.6
        })
        .collect();

    SamplesBuffer::new(1, AUDIO_SAMPLE_RATE, samples)
}

/// Pick the pre-generated ping bucket for a normalized (0.0-1.0) energy.
fn ping_bucket(energy_normalized: f32) -> usize {
    // Bucket count is small; truncation is the intended floor behavior.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let bucket = (energy_normalized.clamp(0.0, 1.0) * (PING_BUCKETS - 1) as f32).round() as usize;
    bucket.min(PING_BUCKETS - 1)
}

/// Constant-power stereo gains for a pan position (0.0 = left, 1.0 = right).
fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
    (pan.cos(), pan.sin())
}

/// Minimum collision energy (closing speed) that makes an audible ping.
const PING_MIN_ENERGY: f64 = 40.0;

/// Soft contacts are inaudible.
fn ping_audible(energy: f64) -> bool {
    energy >= PING_MIN_ENERGY
}

/// Audio output system. Holds the output stream (if a device is available),
/// the mute state, and pre-generated ping buffers.
pub struct Audio {
    stream: Option<OutputStream>,
    muted: bool,
    pings: Vec<SamplesBuffer>,
}

impl Audio {
    /// Open the default audio device, degrading to silence if unavailable.
    pub fn new(muted: bool) -> Self {
        let stream = match OutputStreamBuilder::open_default_stream() {
            Ok(mut stream) => {
                stream.log_on_drop(false);
                Some(stream)
            }
            Err(e) => {
                eprintln!("Audio unavailable ({e}); running silently");
                None
            }
        };

        #[allow(clippy::cast_precision_loss)]
        let pings = (0..PING_BUCKETS)
            .map(|i| {
                let t = i as f32 / (PING_BUCKETS - 1) as f32;
                generate_ping(PING_MIN_FREQ + t * (PING_MAX_FREQ - PING_MIN_FREQ))
            })
            .collect();

        Audio {
            stream,
            muted,
            pings,
        }
    }

    /// Toggle mute; returns the new muted state.
    pub fn toggle_mute(&mut self) -> bool {
        self.muted = !self.muted;
        self.muted
    }

    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Play a collision ping. Pitch follows collision energy; the sound is
    /// panned by `pan` (0.0 = left edge of screen, 1.0 = right edge).
    /// Contacts below the audibility floor are silent — a drifting field of
    /// grazing particles should not produce a constant ticking.
    pub fn play_ping(&self, energy: f64, pan: f32) {
        if !ping_audible(energy) {
            return;
        }
        let Some(stream) = self.stream.as_ref().filter(|_| !self.muted) else {
            return;
        };
        // f64->f32 truncation is acceptable for audio frequency calculation
        #[allow(clippy::cast_possible_truncation)]
        let energy_normalized = (energy / COLLISION_ENERGY_NORMALIZER).clamp(0.0, 1.0) as f32;
        let ping = self.pings[ping_bucket(energy_normalized)].clone();
        let (left, right) = pan_gains(pan);
        stream
            .mixer()
            .add(ChannelVolume::new(ping, vec![left, right]));
    }

    /// Play the explosion rumble sound.
    pub fn play_explosion(&self) {
        let Some(stream) = self.stream.as_ref().filter(|_| !self.muted) else {
            return;
        };
        stream.mixer().add(generate_explosion());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_has_expected_length_and_amplitude() {
        let ping = generate_ping(440.0);
        let samples: Vec<f32> = ping.collect();
        assert_eq!(samples.len(), 44100 * 80 / 1000);
        assert!(samples.iter().all(|s| s.abs() <= 1.0));
        assert!(samples.iter().any(|s| s.abs() > 0.01));
    }

    #[test]
    fn explosion_has_expected_length_and_amplitude() {
        let explosion = generate_explosion();
        let samples: Vec<f32> = explosion.collect();
        assert_eq!(samples.len(), 44100 * 800 / 1000);
        assert!(samples.iter().all(|s| s.abs() <= 1.2));
    }

    #[test]
    fn ping_buckets_cover_range() {
        assert_eq!(ping_bucket(0.0), 0);
        assert_eq!(ping_bucket(1.0), PING_BUCKETS - 1);
        assert_eq!(ping_bucket(2.0), PING_BUCKETS - 1);
        assert_eq!(ping_bucket(-1.0), 0);
    }

    #[test]
    fn soft_contacts_are_inaudible() {
        assert!(!ping_audible(0.0));
        assert!(!ping_audible(PING_MIN_ENERGY - 1.0));
        assert!(ping_audible(PING_MIN_ENERGY));
        assert!(ping_audible(500.0));
    }

    #[test]
    fn pan_gains_are_constant_power() {
        for pan in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let (l, r) = pan_gains(pan);
            assert!((l * l + r * r - 1.0).abs() < 1e-5);
        }
        let (l, r) = pan_gains(0.0);
        assert!(l > 0.99 && r < 0.01);
        let (l, r) = pan_gains(1.0);
        assert!(l < 0.01 && r > 0.99);
    }
}
