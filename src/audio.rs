// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Synthesized audio: collision pings and explosion rumbles.
//!
//! Audio is optional — if no output device is available the simulation runs
//! silently. Pings are pre-generated at startup into a small set of pitch
//! buckets so playback allocates nothing per collision, and are panned
//! left/right based on where on screen the collision happened. Musical mode
//! (--music / S) swaps the linear pitch buckets for pentatonic scale
//! degrees, turning collision showers into wind-chime melodies.

#![cfg_attr(target_arch = "wasm32", allow(clippy::unused_self))]

// The web demo runs silently: browsers gate AudioContext behind a user
// gesture and rodio's device model does not map cleanly onto that, so the
// wasm build stubs the audio API instead - the same public surface, no
// sound. This extends the existing philosophy (no output device: run
// silently) to a platform where the "device" needs ceremony.
#[cfg(target_arch = "wasm32")]
pub struct Audio {
    muted: bool,
    music: bool,
}

#[cfg(target_arch = "wasm32")]
impl Audio {
    pub fn new(muted: bool, music: bool) -> Self {
        Audio { muted, music }
    }
    pub fn toggle_mute(&mut self) -> bool {
        self.muted = !self.muted;
        self.muted
    }
    pub fn is_muted(&self) -> bool {
        self.muted
    }
    pub fn toggle_music(&mut self) -> bool {
        self.music = !self.music;
        self.music
    }
    pub fn is_music(&self) -> bool {
        self.music
    }
    pub fn play_ping(&self, _energy: f64, _pan: f32) {}
    pub fn play_explosion(&self) {}
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
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

    // Musical mode (--music / S key): pings snap to a major-pentatonic scale,
    // with the collision energy picking the scale degree.
    /// Semitone offsets of the scale degrees, spanning two octaves up from the
    /// root and closing on the double octave.
    const PENTATONIC_SEMITONES: [f32; 11] =
        [0.0, 2.0, 4.0, 7.0, 9.0, 12.0, 14.0, 16.0, 19.0, 21.0, 24.0];
    /// Root note of the scale (C4). Two octaves up lands on C6 (~1046 Hz),
    /// comfortably inside the continuous map's 300-1500 Hz band.
    const MUSIC_ROOT_FREQ: f32 = 261.63;

    /// Equal-temperament frequency of the scale degree at `index`.
    fn note_frequency(index: usize) -> f32 {
        MUSIC_ROOT_FREQ * (PENTATONIC_SEMITONES[index] / 12.0).exp2()
    }

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
        let duration_samples =
            (u64::from(AUDIO_SAMPLE_RATE) * EXPLOSION_DURATION_MS / 1000) as usize;
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

    /// Map a normalized (0.0-1.0) energy onto one of `count` pre-generated
    /// buffers: linear pitch buckets normally, scale degrees in musical mode.
    fn energy_index(energy_normalized: f32, count: usize) -> usize {
        // Counts are small; truncation is the intended rounding behavior.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let index = (energy_normalized.clamp(0.0, 1.0) * (count - 1) as f32).round() as usize;
        index.min(count - 1)
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
    /// the mute and musical-mode states, and pre-generated ping buffers for
    /// both modes (so toggling at runtime is instant and allocation-free).
    pub struct Audio {
        stream: Option<OutputStream>,
        muted: bool,
        music: bool,
        pings: Vec<SamplesBuffer>,
        notes: Vec<SamplesBuffer>,
        explosion: SamplesBuffer,
    }

    impl Audio {
        /// Open the default audio device, degrading to silence if unavailable.
        pub fn new(muted: bool, music: bool) -> Self {
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
            let notes = (0..PENTATONIC_SEMITONES.len())
                .map(|i| generate_ping(note_frequency(i)))
                .collect();

            Audio {
                stream,
                muted,
                music,
                pings,
                notes,
                // Pre-generated like the pings; the rumble's noise component is
                // identical across explosions, which is imperceptible under the
                // oscillator mix.
                explosion: generate_explosion(),
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

        /// Toggle musical mode; returns the new state.
        pub fn toggle_music(&mut self) -> bool {
            self.music = !self.music;
            self.music
        }

        pub fn is_music(&self) -> bool {
            self.music
        }

        /// Play a collision ping. Pitch follows collision energy — continuously
        /// in the default mode, snapped to a pentatonic scale degree in musical
        /// mode — and the sound is panned by `pan` (0.0 = left edge of screen,
        /// 1.0 = right edge). Contacts below the audibility floor are silent —
        /// a drifting field of grazing particles should not produce a constant
        /// ticking.
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
            let buffers = if self.music { &self.notes } else { &self.pings };
            let ping = buffers[energy_index(energy_normalized, buffers.len())].clone();
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
            stream.mixer().add(self.explosion.clone());
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
        fn energy_index_covers_the_full_range_for_both_modes() {
            for count in [PING_BUCKETS, PENTATONIC_SEMITONES.len()] {
                assert_eq!(energy_index(0.0, count), 0);
                assert_eq!(energy_index(1.0, count), count - 1);
                assert_eq!(energy_index(2.0, count), count - 1, "clamped above");
                assert_eq!(energy_index(-1.0, count), 0, "clamped below");
                let mid = energy_index(0.5, count);
                assert!(
                    mid > 0 && mid < count - 1,
                    "mid energy picks a middle index"
                );
            }
        }

        #[test]
        fn pentatonic_notes_span_two_octaves_and_stay_on_scale() {
            let first = note_frequency(0);
            let last = note_frequency(PENTATONIC_SEMITONES.len() - 1);
            assert!((first - MUSIC_ROOT_FREQ).abs() < 1e-3, "starts at the root");
            assert!(
                (last - MUSIC_ROOT_FREQ * 4.0).abs() < 1e-2,
                "ends two octaves up"
            );

            for (i, &expected_semitones) in PENTATONIC_SEMITONES.iter().enumerate() {
                let f = note_frequency(i);
                // Each note is an exact equal-temperament scale degree.
                let semitones = 12.0 * (f / MUSIC_ROOT_FREQ).log2();
                assert!(
                    (semitones - expected_semitones).abs() < 1e-3,
                    "degree {i} is {semitones} semitones"
                );
                if i > 0 {
                    assert!(f > note_frequency(i - 1), "scale ascends");
                }
            }
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
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
