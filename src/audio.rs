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
//!
//! The synthesis itself is pure math shared by two backends: rodio on
//! native targets, and `WebAudio` on wasm — where the engine can only start
//! inside a user gesture (browser autoplay policy), so it initializes
//! lazily via `web::WebHandle::enable_audio` and plays silence until then.

use crate::config::ChimeTimbre;
use crate::physics::COLLISION_ENERGY_NORMALIZER;
use rand::Rng;

/// Sample rate (Hz) all buffers are synthesized at, shared by both
/// backends so the same sample math yields the same pitches.
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
pub(crate) const PENTATONIC_SEMITONES: [f32; 11] =
    [0.0, 2.0, 4.0, 7.0, 9.0, 12.0, 14.0, 16.0, 19.0, 21.0, 24.0];
/// Root note of the scale (C4). Two octaves up lands on C6 (~1046 Hz),
/// comfortably inside the continuous map's 300-1500 Hz band.
const MUSIC_ROOT_FREQ: f32 = 261.63;
/// Number of scale degrees in the notes palette — the shared pitch range
/// of musical pings and wall chimes (scene files address degrees by index).
pub const NOTE_COUNT: usize = PENTATONIC_SEMITONES.len();

/// Equal-temperament frequency of the scale degree at `index`.
fn note_frequency(index: usize) -> f32 {
    MUSIC_ROOT_FREQ * (PENTATONIC_SEMITONES[index] / 12.0).exp2()
}

/// Synthesize a ping with exponential decay at the given frequency.
fn synth_ping(frequency: f32) -> Vec<f32> {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * PING_DURATION_MS / 1000) as usize;
    // u32 sample rate fits in f32 mantissa (44100 << 2^23)
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;

    (0..duration_samples)
        .map(|i| {
            // Precision loss from usize->f32 is acceptable for audio timing
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let envelope = (-t * 20.0).exp();
            let wave = (2.0 * std::f32::consts::PI * frequency * t).sin();
            wave * envelope * 0.3
        })
        .collect()
}

/// Synthesize the low-frequency explosion rumble.
fn synth_explosion() -> Vec<f32> {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * EXPLOSION_DURATION_MS / 1000) as usize;
    // u32 sample rate fits in f32 mantissa (44100 << 2^23)
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;
    let mut rng = rand::rng();

    (0..duration_samples)
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
        .collect()
}

/// Woody bar strike: fundamental plus the double-octave partial marimba
/// bars are tuned for, the partial dying much faster than the body.
fn synth_marimba(frequency: f32) -> Vec<f32> {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * 250 / 1000) as usize;
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;
    (0..duration_samples)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let tau = 2.0 * std::f32::consts::PI * t;
            let body = (tau * frequency).sin() * (-t * 11.0).exp();
            let partial = (tau * frequency * 4.0).sin() * 0.45 * (-t * 32.0).exp();
            (body + partial) * 0.32
        })
        .collect()
}

/// Plucked string via Karplus-Strong: a noise burst circulating a
/// lowpassed delay line the length of one period. Bright attack, the
/// harmonics decaying fastest — the classic cheap pluck.
fn synth_pluck(frequency: f32) -> Vec<f32> {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * 600 / 1000) as usize;
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let period = ((AUDIO_SAMPLE_RATE as f32 / frequency).round() as usize).max(2);
    let mut rng = rand::rng();
    let mut line: Vec<f32> = (0..period).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut out = Vec::with_capacity(duration_samples);
    let mut i = 0usize;
    for _ in 0..duration_samples {
        let next = (i + 1) % period;
        let sample = line[i];
        // Averaging two taps lowpasses the loop; 0.996 sets the ring time.
        line[i] = 0.996 * 0.5 * (line[i] + line[next]);
        out.push(sample * 0.28);
        i = next;
    }
    out
}

/// Pitched drum: the fundamental swept down from nearly double the
/// pitch (integrated phase, so the sweep is click-free) over a breath
/// of attack noise.
fn synth_drum(frequency: f32) -> Vec<f32> {
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * 300 / 1000) as usize;
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;
    let mut rng = rand::rng();
    let mut phase = 0.0f32;
    (0..duration_samples)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let sweep = frequency * (1.0 + 0.8 * (-t * 24.0).exp());
            phase += 2.0 * std::f32::consts::PI * sweep / sample_rate_f;
            let body = phase.sin() * (-t * 9.0).exp();
            let breath: f32 = rng.random_range(-1.0..1.0) * 0.35 * (-t * 60.0).exp();
            (body + breath) * 0.34
        })
        .collect()
}

/// Struck bell: a few inharmonic partials (ratios from real bell
/// spectra), the higher ones dying faster.
fn synth_bell(frequency: f32) -> Vec<f32> {
    const PARTIALS: [(f32, f32, f32); 4] = [
        (1.0, 1.0, 6.0),
        (2.76, 0.55, 10.0),
        (5.40, 0.30, 16.0),
        (8.93, 0.15, 24.0),
    ];
    let duration_samples = (u64::from(AUDIO_SAMPLE_RATE) * 400 / 1000) as usize;
    #[allow(clippy::cast_precision_loss)]
    let sample_rate_f = AUDIO_SAMPLE_RATE as f32;
    (0..duration_samples)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate_f;
            let tau = 2.0 * std::f32::consts::PI * t;
            let sum: f32 = PARTIALS
                .iter()
                .map(|&(ratio, amp, decay)| {
                    (tau * frequency * ratio).sin() * amp * (-t * decay).exp()
                })
                .sum();
            sum * 0.18
        })
        .collect()
}

/// The chime palette for a timbre: eleven pentatonic degrees in the
/// requested voice. `Chime` is the original ping, so the default
/// sounds exactly as it always has.
fn synth_chime_palette(timbre: ChimeTimbre) -> Vec<Vec<f32>> {
    let synth: fn(f32) -> Vec<f32> = match timbre {
        ChimeTimbre::Chime => synth_ping,
        ChimeTimbre::Marimba => synth_marimba,
        ChimeTimbre::Pluck => synth_pluck,
        ChimeTimbre::Drum => synth_drum,
        ChimeTimbre::Bell => synth_bell,
    };
    (0..PENTATONIC_SEMITONES.len())
        .map(|i| synth(note_frequency(i)))
        .collect()
}

/// The two ping palettes: linear pitch buckets and pentatonic degrees.
fn synth_ping_palettes() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    #[allow(clippy::cast_precision_loss)]
    let pings = (0..PING_BUCKETS)
        .map(|i| {
            let t = i as f32 / (PING_BUCKETS - 1) as f32;
            synth_ping(PING_MIN_FREQ + t * (PING_MAX_FREQ - PING_MIN_FREQ))
        })
        .collect();
    let notes = (0..PENTATONIC_SEMITONES.len())
        .map(|i| synth_ping(note_frequency(i)))
        .collect();
    (pings, notes)
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

/// Normalized 0-1 energy for bucket selection.
#[allow(clippy::cast_possible_truncation)]
fn normalize_energy(energy: f64) -> f32 {
    (energy / COLLISION_ENERGY_NORMALIZER).clamp(0.0, 1.0) as f32
}

/// Constant-power stereo gains for a pan position (0.0 = left, 1.0 = right).
#[cfg(not(target_arch = "wasm32"))]
fn pan_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
    (pan.cos(), pan.sin())
}

/// Chime volume for an impact: a 0.3 floor keeps gentle rolls audible
/// while hard strikes ring out at full gain. Crate-visible because MIDI
/// velocity rides the same curve — by construction, not by copy.
pub(crate) fn note_gain(energy: f64) -> f32 {
    0.3 + 0.7 * normalize_energy(energy)
}

/// Particle-ping gain for a 0-100 percent volume setting.
fn ping_gain(percent: i32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let gain = percent.clamp(0, 100) as f32 / 100.0;
    gain
}

/// The 0-100 percent value a stored ping gain corresponds to.
fn ping_percent(gain: f32) -> i32 {
    #[allow(clippy::cast_possible_truncation)]
    let percent = (gain * 100.0).round() as i32;
    percent.clamp(0, 100)
}

/// Minimum collision energy (closing speed) that makes an audible ping.
/// Public so the simulation can pre-filter wall-chime events with the
/// same floor the backends enforce (pure data, no device dependency).
pub const PING_MIN_ENERGY: f64 = 40.0;

/// Soft contacts are inaudible.
fn ping_audible(energy: f64) -> bool {
    energy >= PING_MIN_ENERGY
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{
        AUDIO_SAMPLE_RATE, ChimeTimbre, energy_index, normalize_energy, note_gain, pan_gains,
        ping_audible, ping_gain, ping_percent, synth_chime_palette, synth_explosion,
        synth_ping_palettes,
    };
    use rodio::buffer::SamplesBuffer;
    use rodio::source::ChannelVolume;
    use rodio::{OutputStream, OutputStreamBuilder};

    fn buffer(samples: Vec<f32>) -> SamplesBuffer {
        SamplesBuffer::new(1, AUDIO_SAMPLE_RATE, samples)
    }

    /// Audio output system. Holds the output stream (if a device is
    /// available), the mute and musical-mode states, and pre-generated ping
    /// buffers for both modes (so toggling at runtime is instant and
    /// allocation-free).
    pub struct Audio {
        stream: Option<OutputStream>,
        muted: bool,
        music: bool,
        /// Particle-ping gain (0.0-1.0); chimes and rumbles ignore it.
        ping_volume: f32,
        pings: Vec<SamplesBuffer>,
        notes: Vec<SamplesBuffer>,
        /// Wall-chime notes in the launch-selected timbre; pings keep
        /// their own voice above.
        chimes: Vec<SamplesBuffer>,
        explosion: SamplesBuffer,
    }

    impl Audio {
        /// Open the default audio device, degrading to silence if unavailable.
        /// A muted start never touches the device at all — the stream is
        /// opened lazily on first unmute, so `--mute` runs (and unit tests)
        /// stay off the audio hardware entirely.
        pub fn new(muted: bool, music: bool, timbre: ChimeTimbre, ping_volume: i32) -> Self {
            let stream = if muted { None } else { Self::open_stream() };

            let (pings, notes) = synth_ping_palettes();
            Audio {
                stream,
                muted,
                music,
                ping_volume: ping_gain(ping_volume),
                pings: pings.into_iter().map(buffer).collect(),
                notes: notes.into_iter().map(buffer).collect(),
                chimes: synth_chime_palette(timbre)
                    .into_iter()
                    .map(buffer)
                    .collect(),
                // Pre-generated like the pings; the rumble's noise component is
                // identical across explosions, which is imperceptible under the
                // oscillator mix.
                explosion: buffer(synth_explosion()),
            }
        }

        fn open_stream() -> Option<OutputStream> {
            match OutputStreamBuilder::open_default_stream() {
                Ok(mut stream) => {
                    stream.log_on_drop(false);
                    Some(stream)
                }
                Err(e) => {
                    eprintln!("Audio unavailable ({e}); running silently");
                    None
                }
            }
        }

        /// Toggle mute; returns the new muted state. Unmuting opens the
        /// device on first use (and retries if it was unavailable before).
        pub fn toggle_mute(&mut self) -> bool {
            self.muted = !self.muted;
            if !self.muted && self.stream.is_none() {
                self.stream = Self::open_stream();
            }
            self.muted
        }

        /// Whether audio is currently muted.
        pub fn is_muted(&self) -> bool {
            self.muted
        }

        /// Toggle musical mode; returns the new state.
        pub fn toggle_music(&mut self) -> bool {
            self.music = !self.music;
            self.music
        }

        /// Whether musical (pentatonic) mode is on.
        pub fn is_music(&self) -> bool {
            self.music
        }

        /// Set the particle-ping volume from a 0-100 percent value.
        pub fn set_ping_volume(&mut self, percent: i32) {
            self.ping_volume = ping_gain(percent);
        }

        /// The particle-ping volume as a 0-100 percent value.
        pub fn ping_volume_percent(&self) -> i32 {
            ping_percent(self.ping_volume)
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
            if self.ping_volume <= 0.0 {
                return;
            }
            let energy_normalized = normalize_energy(energy);
            let buffers = if self.music { &self.notes } else { &self.pings };
            let ping = buffers[energy_index(energy_normalized, buffers.len())].clone();
            let (left, right) = pan_gains(pan);
            stream.mixer().add(ChannelVolume::new(
                ping,
                vec![left * self.ping_volume, right * self.ping_volume],
            ));
        }

        /// Play scale degree `note` from the chime palette directly,
        /// with volume scaled by impact energy. Ignores the music toggle:
        /// wall-chime pitch is geometry, not energy, and chimes stay on
        /// the scale so random walls sound musical.
        pub fn play_note(&self, note: usize, energy: f64, pan: f32) {
            if !ping_audible(energy) {
                return;
            }
            let Some(stream) = self.stream.as_ref().filter(|_| !self.muted) else {
                return;
            };
            let buffer = self.chimes[note.min(self.chimes.len() - 1)].clone();
            let gain = note_gain(energy);
            let (left, right) = pan_gains(pan);
            stream
                .mixer()
                .add(ChannelVolume::new(buffer, vec![left * gain, right * gain]));
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
        use super::Audio;

        /// Regression: a muted Audio must never open the output device.
        /// Parallel test-suite Apps grabbing real WASAPI streams crashed
        /// the Windows CI runner with `STATUS_ACCESS_VIOLATION`.
        #[test]
        fn ping_volume_round_trips_and_clamps() {
            let mut audio = Audio::new(true, false, super::ChimeTimbre::Chime, 50);
            assert_eq!(audio.ping_volume_percent(), 50);
            audio.set_ping_volume(0);
            assert_eq!(audio.ping_volume_percent(), 0);
            audio.set_ping_volume(250);
            assert_eq!(audio.ping_volume_percent(), 100, "clamped high");
            audio.set_ping_volume(-5);
            assert_eq!(audio.ping_volume_percent(), 0, "clamped low");
        }

        #[test]
        fn muted_construction_stays_off_the_audio_device() {
            let audio = Audio::new(true, false, super::ChimeTimbre::Chime, 100);
            assert!(audio.muted);
            assert!(
                audio.stream.is_none(),
                "muted Audio must not hold an output stream"
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::Audio;

/// `WebAudio` backend. The engine (context + decoded buffers) is a
/// thread-local singleton created by [`web_enable`] — which the page must
/// call from inside a user gesture, per browser autoplay policy. Until
/// then every play call is silently dropped, extending the "no device:
/// run silently" philosophy to a platform where the device needs consent.
#[cfg(target_arch = "wasm32")]
mod web {
    use super::{
        AUDIO_SAMPLE_RATE, ChimeTimbre, energy_index, normalize_energy, note_gain, ping_audible,
        ping_gain, ping_percent, synth_chime_palette, synth_explosion, synth_ping_palettes,
    };
    use std::cell::{Cell, RefCell};
    use web_sys::{AudioBuffer, AudioContext, AudioContextOptions};

    struct Engine {
        ctx: AudioContext,
        pings: Vec<AudioBuffer>,
        notes: Vec<AudioBuffer>,
        /// Wall-chime notes in the launch-selected timbre.
        chimes: Vec<AudioBuffer>,
        explosion: AudioBuffer,
    }

    thread_local! {
        static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
        /// The timbre the engine should synthesize with, parked here by
        /// [`Audio::new`]: [`web_enable`] runs inside a user-gesture
        /// callback with no path back to the config.
        static TIMBRE: Cell<ChimeTimbre> = const { Cell::new(ChimeTimbre::Chime) };
    }

    // 44,100 is exactly representable in f32; the length casts are
    // bounded by the sub-second buffer sizes.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn make_buffer(ctx: &AudioContext, samples: &[f32]) -> Option<AudioBuffer> {
        let buffer = ctx
            .create_buffer(1, samples.len() as u32, AUDIO_SAMPLE_RATE as f32)
            .ok()?;
        // web-sys's typed copy_to_channel passes a view into wasm memory,
        // which the WebIDL rejects when that memory is shared (the
        // multi-threaded build) — the same restriction the Canvas2D
        // backend hit with ImageData. Stage through an ordinary JS
        // Float32Array (TypedArray.set accepts shared sources) and call
        // copyToChannel with it dynamically.
        let staged = js_sys::Float32Array::new_with_length(samples.len() as u32);
        staged.copy_from(samples);
        let method = js_sys::Reflect::get(&buffer, &"copyToChannel".into()).ok()?;
        let method: js_sys::Function = wasm_bindgen::JsCast::dyn_into(method).ok()?;
        method.call2(&buffer, &staged, &0.into()).ok()?;
        Some(buffer)
    }

    /// Create (or resume) the `WebAudio` engine. Must be called from a user
    /// gesture the first time; returns whether the engine is ready.
    pub fn web_enable() -> bool {
        ENGINE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some(engine) = slot.as_ref() {
                let _ = engine.ctx.resume();
                return true;
            }
            let options = AudioContextOptions::new();
            #[allow(clippy::cast_precision_loss)]
            options.set_sample_rate(AUDIO_SAMPLE_RATE as f32);
            let Ok(ctx) = AudioContext::new_with_context_options(&options) else {
                return false;
            };
            let _ = ctx.resume();
            let (pings, notes) = synth_ping_palettes();
            let chime_palette = synth_chime_palette(TIMBRE.get());
            let build = |v: Vec<Vec<f32>>| -> Option<Vec<AudioBuffer>> {
                v.into_iter().map(|s| make_buffer(&ctx, &s)).collect()
            };
            let (Some(pings), Some(notes), Some(chimes), Some(explosion)) = (
                build(pings),
                build(notes),
                build(chime_palette),
                make_buffer(&ctx, &synth_explosion()),
            ) else {
                return false;
            };
            *slot = Some(Engine {
                ctx,
                pings,
                notes,
                chimes,
                explosion,
            });
            true
        })
    }

    /// Whether the engine has been created (the page's sound button label).
    pub fn web_ready() -> bool {
        ENGINE.with(|slot| slot.borrow().is_some())
    }

    /// Play the selected buffer panned to `pan` (0 = left, 1 = right;
    /// `WebAudio`'s `StereoPannerNode` applies the equal-power law itself)
    /// at `gain` (1.0 = full volume).
    fn play(select: impl Fn(&Engine) -> &AudioBuffer, pan: f32, gain: f32) {
        ENGINE.with(|slot| {
            let slot = slot.borrow();
            let Some(engine) = slot.as_ref() else { return };
            let Ok(source) = engine.ctx.create_buffer_source() else {
                return;
            };
            source.set_buffer(Some(select(engine)));
            let Ok(volume) = engine.ctx.create_gain() else {
                return;
            };
            volume.gain().set_value(gain);
            let Ok(panner) = engine.ctx.create_stereo_panner() else {
                return;
            };
            panner.pan().set_value((pan.clamp(0.0, 1.0) * 2.0) - 1.0);
            let _ = source.connect_with_audio_node(&volume);
            let _ = volume.connect_with_audio_node(&panner);
            let _ = panner.connect_with_audio_node(&engine.ctx.destination());
            let _ = source.start();
        });
    }

    /// Same public surface as the native backend; mute/music state lives
    /// here, the device lives in the thread-local engine.
    pub struct Audio {
        muted: bool,
        music: bool,
        /// Particle-ping gain (0.0-1.0); chimes and rumbles ignore it.
        ping_volume: f32,
    }

    impl Audio {
        /// State-only construction; the `WebAudio` engine itself is
        /// created later by [`super::web_enable`] inside a user gesture,
        /// picking up the timbre parked here.
        pub fn new(muted: bool, music: bool, timbre: ChimeTimbre, ping_volume: i32) -> Self {
            TIMBRE.set(timbre);
            Audio {
                muted,
                music,
                ping_volume: ping_gain(ping_volume),
            }
        }

        /// Toggle mute; returns the new muted state.
        pub fn toggle_mute(&mut self) -> bool {
            self.muted = !self.muted;
            self.muted
        }

        /// Whether audio is currently muted.
        pub fn is_muted(&self) -> bool {
            self.muted
        }

        /// Toggle musical mode; returns the new state.
        pub fn toggle_music(&mut self) -> bool {
            self.music = !self.music;
            self.music
        }

        /// Whether musical (pentatonic) mode is on.
        pub fn is_music(&self) -> bool {
            self.music
        }

        /// Set the particle-ping volume from a 0-100 percent value.
        pub fn set_ping_volume(&mut self, percent: i32) {
            self.ping_volume = ping_gain(percent);
        }

        /// The particle-ping volume as a 0-100 percent value.
        pub fn ping_volume_percent(&self) -> i32 {
            ping_percent(self.ping_volume)
        }

        /// Play a collision ping (see the native backend for the pitch and
        /// pan semantics); dropped silently until the engine exists.
        pub fn play_ping(&self, energy: f64, pan: f32) {
            if self.muted || self.ping_volume <= 0.0 || !ping_audible(energy) {
                return;
            }
            let music = self.music;
            let energy_normalized = normalize_energy(energy);
            play(
                move |engine| {
                    let buffers = if music { &engine.notes } else { &engine.pings };
                    &buffers[energy_index(energy_normalized, buffers.len())]
                },
                pan,
                self.ping_volume,
            );
        }

        /// Play scale degree `note` from the pentatonic palette directly,
        /// with volume scaled by impact energy (see the native backend).
        pub fn play_note(&self, note: usize, energy: f64, pan: f32) {
            if self.muted || !ping_audible(energy) {
                return;
            }
            play(
                move |engine| &engine.chimes[note.min(engine.chimes.len() - 1)],
                pan,
                note_gain(energy),
            );
        }

        /// Play the explosion rumble; dropped silently until the engine
        /// exists.
        pub fn play_explosion(&self) {
            if self.muted {
                return;
            }
            play(|engine| &engine.explosion, 0.5, 1.0);
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::{Audio, web_enable, web_ready};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn ping_has_expected_length_and_amplitude() {
        let samples = synth_ping(440.0);
        assert_eq!(samples.len(), 44100 * 80 / 1000);
        assert!(samples.iter().all(|s| s.abs() <= 1.0));
        assert!(samples.iter().any(|s| s.abs() > 0.01));
    }

    #[test]
    fn explosion_has_expected_length_and_amplitude() {
        let samples = synth_explosion();
        assert_eq!(samples.len(), 44100 * 800 / 1000);
        assert!(samples.iter().all(|s| s.abs() <= 1.2));
        assert!(samples.iter().any(|s| s.abs() > 0.05));
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

    #[test]
    fn pentatonic_notes_span_two_octaves_and_stay_on_scale() {
        let first = note_frequency(0);
        let last = note_frequency(PENTATONIC_SEMITONES.len() - 1);
        assert!((first - MUSIC_ROOT_FREQ).abs() < 1e-3, "starts at the root");
        assert!(
            (last - MUSIC_ROOT_FREQ * 4.0).abs() < 1e-2,
            "ends two octaves up"
        );
        // Scene files and the sim address degrees through this constant.
        assert_eq!(NOTE_COUNT, PENTATONIC_SEMITONES.len());
        assert_eq!(NOTE_COUNT, 11);
    }

    #[test]
    fn every_timbre_synthesizes_bounded_audible_notes() {
        for timbre in [
            ChimeTimbre::Chime,
            ChimeTimbre::Marimba,
            ChimeTimbre::Pluck,
            ChimeTimbre::Drum,
            ChimeTimbre::Bell,
        ] {
            let palette = synth_chime_palette(timbre);
            assert_eq!(palette.len(), NOTE_COUNT, "{timbre:?}");
            for (i, samples) in palette.iter().enumerate() {
                assert!(!samples.is_empty(), "{timbre:?} note {i}");
                assert!(
                    samples.iter().all(|s| s.abs() <= 1.2),
                    "{timbre:?} note {i} clips"
                );
                assert!(
                    samples.iter().any(|s| s.abs() > 0.01),
                    "{timbre:?} note {i} is silent"
                );
                // Voices ring out rather than ending on a cliff.
                let tail = &samples[samples.len() - 100..];
                assert!(
                    tail.iter().all(|s| s.abs() < 0.2),
                    "{timbre:?} note {i} ends abruptly"
                );
            }
        }
    }

    #[test]
    fn timbres_have_distinct_ring_times() {
        // The pluck rings much longer than the original chime — the
        // audible core of "sounds like an instrument, not a music box".
        let ring = |samples: &[f32]| samples.iter().rposition(|s| s.abs() > 0.02).unwrap_or(0);
        let chime = ring(&synth_ping(note_frequency(0)));
        let pluck = ring(&synth_pluck(note_frequency(0)));
        let marimba = ring(&synth_marimba(note_frequency(0)));
        assert!(pluck > chime * 3, "pluck must ring: {pluck} vs {chime}");
        assert!(marimba > chime, "marimba outlasts the ping");
    }

    #[test]
    fn note_gain_is_monotonic_and_bounded() {
        let mut prev = 0.0f32;
        for step in 0..=20 {
            let energy = f64::from(step) * (COLLISION_ENERGY_NORMALIZER / 10.0);
            let gain = note_gain(energy);
            assert!((0.3..=1.0).contains(&gain), "gain {gain} out of range");
            assert!(gain >= prev, "gain must not decrease with energy");
            prev = gain;
        }
        assert!((note_gain(0.0) - 0.3).abs() < 1e-6, "floor at zero energy");
        assert!(
            (note_gain(COLLISION_ENERGY_NORMALIZER * 2.0) - 1.0).abs() < 1e-6,
            "full gain past the normalizer"
        );
    }
}
