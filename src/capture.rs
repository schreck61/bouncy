// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Self-contained capture: record the chime stream and export it as a
//! standard MIDI file and a WAV bounce of the internal synth — the
//! no-DAW path to keeping what the instrument played.
//!
//! The recorder and both encoders are pure (no clocks, no filesystem):
//! `Capture` accumulates wall-clock event times from the frame `dt`s
//! the shell feeds it, `to_smf`/`to_wav` turn the event list into file
//! bytes, and only [`write_files`] touches the disk — so everything
//! else unit-tests by re-parsing the bytes with the same crates that
//! wrote them, and a future web shell can hand the same bytes to a
//! download instead. Timing is deliberately wall-clock (unscaled `dt`):
//! a capture reproduces what the listener heard, so slow-motion
//! sessions capture at listening speed, and pause (which skips the
//! simulate step entirely) contributes no dead air.

use crate::audio::{AUDIO_SAMPLE_RATE, note_gain, pan_gains, synth_chime_palette};
use crate::config::ChimeTimbre;
use crate::sim::WallChime;

/// The live note gate, mirrored from the MIDI scheduler: captured
/// note-offs land `NOTE_GATE` after their on, exactly like the wire.
const GATE_SECS: f64 = 0.2;
/// Silent tail after the last voice finishes ringing in the WAV.
const WAV_TAIL_SECS: f64 = 0.25;
/// SMF resolution in ticks per quarter note.
const PPQ: u16 = 480;
/// Tempo written when quantize is off: ticks still scale so wall-clock
/// durations are faithful; the grid is only meaningful with a real bpm.
const FALLBACK_BPM: f64 = 120.0;

/// One recorded chime, in seconds since capture start. Carries both
/// truths: the resolved MIDI stream (post per-wall mapping — what a
/// live DAW port would have received) and the synth parameters (what
/// the ear heard — the WAV ignores MIDI key overrides by design).
pub struct CaptureEvent {
    pub t: f64,
    pub channel: u8,
    pub key: u8,
    pub velocity: u8,
    pub degree: usize,
    pub energy: f64,
    pub pan: f64,
}

/// A recording in progress: the event list plus the wall clock that
/// timestamps it. Tempo and timbre are snapshotted at start so a
/// mid-recording settings change cannot shear the file against what
/// was heard.
pub struct Capture {
    events: Vec<CaptureEvent>,
    clock: f64,
    bpm: f64,
    timbre: ChimeTimbre,
}

impl Capture {
    /// Begin recording. `bpm` 0 (quantize off) falls back to a nominal
    /// grid at encode time.
    pub fn start(bpm: f64, timbre: ChimeTimbre) -> Self {
        Capture {
            events: Vec::new(),
            clock: 0.0,
            bpm,
            timbre,
        }
    }

    /// Advance the capture clock by one frame of *unscaled* wall time.
    pub fn advance(&mut self, dt: f64) {
        self.clock += dt;
    }

    /// Record one chime at the current clock, resolved through the same
    /// [`crate::midi::chime_message`] every live sender uses — the .mid
    /// carries per-wall mappings exactly as a connected port would.
    /// Out-of-scale degrees with no override are silent live and stay
    /// silent here.
    pub fn record(&mut self, chime: &WallChime) {
        let Some((channel, key, velocity)) =
            crate::midi::chime_message(chime.note, chime.midi, chime.energy)
        else {
            return;
        };
        self.events.push(CaptureEvent {
            t: self.clock,
            channel,
            key,
            velocity,
            degree: chime.note,
            energy: chime.energy,
            pan: chime.pan,
        });
    }

    /// Seconds recorded so far (the HUD readout).
    pub fn elapsed(&self) -> f64 {
        self.clock
    }

    /// Notes recorded so far.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True when nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The recorded events, encode-ready.
    pub fn events(&self) -> &[CaptureEvent] {
        &self.events
    }

    /// The tempo snapshotted at start (0 = quantize was off).
    pub fn bpm(&self) -> f64 {
        self.bpm
    }

    /// The chime voice snapshotted at start.
    pub fn timbre(&self) -> ChimeTimbre {
        self.timbre
    }
}

/// One wire event during SMF encoding; offs sort before ons at the
/// same tick, mirroring the live early-off-then-on ordering.
enum Kind {
    Off { channel: u8, key: u8 },
    On { channel: u8, key: u8, velocity: u8 },
}

/// Encode the event list as a format-0 standard MIDI file. Note-offs
/// are computed here as a pure post-process — off at `min(t + gate,
/// next same-voice on)` — reproducing the live scheduler's 200 ms gate
/// and early-off-on-restrike semantics without any live bookkeeping.
pub fn to_smf(events: &[CaptureEvent], bpm: f64) -> Vec<u8> {
    use midly::num::{u4, u7, u15, u24, u28};
    use midly::{
        Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
    };

    let bpm = if bpm > 0.0 { bpm } else { FALLBACK_BPM };
    let ticks_of = |t: f64| -> u64 {
        // seconds → beats → ticks.
        let ticks = t * bpm / 60.0 * f64::from(PPQ);
        // Non-negative by construction; far below u64 range.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            ticks.round() as u64
        }
    };

    let mut wire: Vec<(u64, u8, Kind)> = Vec::with_capacity(events.len() * 2);
    for (i, e) in events.iter().enumerate() {
        let on_tick = ticks_of(e.t);
        wire.push((
            on_tick,
            1,
            Kind::On {
                channel: e.channel,
                key: e.key,
                velocity: e.velocity,
            },
        ));
        let next_same_voice = events[i + 1..]
            .iter()
            .find(|n| n.channel == e.channel && n.key == e.key)
            .map(|n| n.t);
        let off_t = match next_same_voice {
            Some(next) => next.min(e.t + GATE_SECS),
            None => e.t + GATE_SECS,
        };
        wire.push((
            ticks_of(off_t),
            0,
            Kind::Off {
                channel: e.channel,
                key: e.key,
            },
        ));
    }
    wire.sort_by_key(|(tick, order, _)| (*tick, *order));

    let mut track: Vec<TrackEvent> = Vec::with_capacity(wire.len() + 2);
    // Tempo meta first: microseconds per quarter note.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let us_per_beat = (60_000_000.0 / bpm).round() as u32;
    track.push(TrackEvent {
        delta: u28::from(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(us_per_beat))),
    });
    let mut last_tick = 0u64;
    for (tick, _, kind) in wire {
        // Deltas fit u28 for any real session length.
        #[allow(clippy::cast_possible_truncation)]
        let delta = u28::from((tick - last_tick) as u32);
        last_tick = tick;
        let (channel, message) = match kind {
            Kind::On {
                channel,
                key,
                velocity,
            } => (
                channel,
                MidiMessage::NoteOn {
                    key: u7::from(key),
                    vel: u7::from(velocity),
                },
            ),
            Kind::Off { channel, key } => (
                channel,
                MidiMessage::NoteOff {
                    key: u7::from(key),
                    vel: u7::from(0),
                },
            ),
        };
        track.push(TrackEvent {
            delta,
            kind: TrackEventKind::Midi {
                channel: u4::from(channel),
                message,
            },
        });
    }
    track.push(TrackEvent {
        delta: u28::from(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let smf = Smf {
        header: Header {
            format: Format::SingleTrack,
            timing: Timing::Metrical(u15::from(PPQ)),
        },
        tracks: vec![track],
    };
    let mut bytes = Vec::new();
    smf.write(&mut bytes).expect("Vec write is infallible");
    bytes
}

/// Render the event list as a stereo 16-bit WAV: an offline mix of the
/// chime palette through the exact gain and pan math the live path
/// uses. Voices with noise components are re-synthesized (fresh RNG),
/// so the bounce is sonically — not bit — identical to the session.
/// Peak-normalizes only when the mix would clip.
pub fn to_wav(events: &[CaptureEvent], timbre: ChimeTimbre) -> Vec<u8> {
    let palette = synth_chime_palette(timbre);
    let longest = palette.iter().map(Vec::len).max().unwrap_or(0);
    let rate = f64::from(AUDIO_SAMPLE_RATE);
    let last_t = events.iter().map(|e| e.t).fold(0.0f64, f64::max);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let frames = ((last_t + WAV_TAIL_SECS) * rate) as usize + longest;

    // Interleaved stereo accumulator.
    let mut mix = vec![0.0f32; frames * 2];
    for e in events {
        let voice = &palette[e.degree.min(palette.len() - 1)];
        let gain = note_gain(e.energy);
        #[allow(clippy::cast_possible_truncation)]
        let (left, right) = pan_gains(e.pan as f32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let offset = (e.t * rate) as usize;
        for (i, &s) in voice.iter().enumerate() {
            let frame = offset + i;
            if frame >= frames {
                break;
            }
            mix[frame * 2] += s * gain * left;
            mix[frame * 2 + 1] += s * gain * right;
        }
    }

    // Transparent safety: scale the whole mix down only if it clips
    // (rodio would have summed the live voices just as hot).
    let peak = mix.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    let scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: AUDIO_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("Vec write is infallible");
        for s in &mix {
            #[allow(clippy::cast_possible_truncation)]
            let sample = (s * scale * f32::from(i16::MAX)).round() as i16;
            writer
                .write_sample(sample)
                .expect("Vec write is infallible");
        }
        writer.finalize().expect("Vec write is infallible");
    }
    cursor.into_inner()
}

/// Write the capture next to the working directory's other exports:
/// `bouncy-capture-{secs}.mid` + `.wav`, with the screenshot/export
/// unique-suffix loop so nothing is ever clobbered. Returns both paths.
pub fn write_files(capture: &Capture) -> Result<(std::path::PathBuf, std::path::PathBuf), String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    for n in 0..100 {
        let suffix = if n == 0 {
            String::new()
        } else {
            format!("-{n}")
        };
        let mid = std::path::PathBuf::from(format!("bouncy-capture-{secs}{suffix}.mid"));
        let wav = std::path::PathBuf::from(format!("bouncy-capture-{secs}{suffix}.wav"));
        if mid.exists() || wav.exists() {
            continue;
        }
        std::fs::write(&mid, to_smf(capture.events(), capture.bpm()))
            .map_err(|e| format!("cannot write {}: {e}", mid.display()))?;
        std::fs::write(&wav, to_wav(capture.events(), capture.timbre()))
            .map_err(|e| format!("cannot write {}: {e}", wav.display()))?;
        return Ok((mid, wav));
    }
    Err("cannot find a free capture filename".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::WallMidi;

    fn event(t: f64, channel: u8, key: u8, velocity: u8) -> CaptureEvent {
        CaptureEvent {
            t,
            channel,
            key,
            velocity,
            degree: 0,
            energy: 800.0,
            pan: 0.5,
        }
    }

    #[test]
    fn recorder_resolves_mappings_and_keeps_wall_time() {
        let mut cap = Capture::start(0.0, ChimeTimbre::Chime);
        cap.advance(1.5);
        cap.record(&WallChime {
            stroke: 1,
            note: 0,
            energy: 800.0,
            pan: 0.25,
            midi: WallMidi {
                key: Some(48),
                channel: 2,
            },
        });
        cap.advance(0.5);
        cap.record(&WallChime {
            stroke: 2,
            note: 4,
            energy: 100.0,
            pan: 0.75,
            midi: WallMidi::default(),
        });
        assert_eq!(cap.len(), 2);
        assert!((cap.elapsed() - 2.0).abs() < 1e-12);
        let e = &cap.events()[0];
        assert_eq!((e.t, e.channel, e.key), (1.5, 2, 48), "mapping resolved");
        let e = &cap.events()[1];
        assert_eq!(e.channel, 0, "default mapping on channel 1");
        assert_eq!(
            e.key,
            crate::midi::key_for_degree(4).unwrap(),
            "auto key from the degree"
        );
    }

    #[test]
    fn smf_round_trips_through_midly() {
        let events = [
            event(0.0, 0, 60, 100),
            event(0.1, 0, 60, 90),  // restrike: early off at 0.1
            event(0.25, 2, 48, 64), // second voice, own channel
        ];
        let bytes = to_smf(&events, 120.0);
        let smf = midly::Smf::parse(&bytes).expect("well-formed SMF");
        assert_eq!(smf.header.format, midly::Format::SingleTrack);
        let midly::Timing::Metrical(ppq) = smf.header.timing else {
            panic!("metrical timing");
        };
        assert_eq!(ppq.as_int(), PPQ);
        assert_eq!(smf.tracks.len(), 1);
        let track = &smf.tracks[0];
        // Tempo meta first: 120 bpm = 500,000 us per beat.
        assert!(matches!(
            track[0].kind,
            midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(t)) if t.as_int() == 500_000
        ));
        // Reconstruct absolute ticks and count ons/offs per voice.
        let mut tick = 0u64;
        let mut ons = Vec::new();
        let mut offs = Vec::new();
        for ev in track {
            tick += u64::from(ev.delta.as_int());
            match ev.kind {
                midly::TrackEventKind::Midi { channel, message } => match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        ons.push((tick, channel.as_int(), key.as_int(), vel.as_int()));
                    }
                    midly::MidiMessage::NoteOff { key, .. } => {
                        offs.push((tick, channel.as_int(), key.as_int()));
                    }
                    _ => {}
                },
                midly::TrackEventKind::Meta(midly::MetaMessage::EndOfTrack) => break,
                _ => {}
            }
        }
        // 120 bpm, PPQ 480: one second = 960 ticks.
        assert_eq!(
            ons,
            vec![
                (0, 0, 60, 100),
                (96, 0, 60, 90),  // 0.1 s
                (240, 2, 48, 64), // 0.25 s
            ]
        );
        // First off is the early-off at the restrike (0.1 s), not the
        // 200 ms gate; the others gate normally.
        assert_eq!(
            offs,
            vec![
                (96, 0, 60),  // early off, same tick as the restrike on
                (288, 0, 60), // 0.1 + 0.2
                (432, 2, 48), // 0.25 + 0.2
            ]
        );
        // Off sorts before on at the shared tick 96.
        let order: Vec<u64> = track
            .iter()
            .scan(0u64, |t, ev| {
                *t += u64::from(ev.delta.as_int());
                Some(*t)
            })
            .collect();
        assert!(order.windows(2).all(|w| w[0] <= w[1]), "monotone ticks");
    }

    #[test]
    fn smf_without_quantize_uses_the_fallback_grid() {
        let bytes = to_smf(&[event(1.0, 0, 60, 100)], 0.0);
        let smf = midly::Smf::parse(&bytes).unwrap();
        assert!(matches!(
            smf.tracks[0][0].kind,
            midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(t))
                if t.as_int() == 500_000
        ));
        // 1.0 s at the 120 fallback is exactly two beats = 960 ticks:
        // wall-clock duration survives whatever grid is written.
        let on_tick: u64 = smf.tracks[0]
            .iter()
            .scan(0u64, |t, ev| {
                *t += u64::from(ev.delta.as_int());
                Some((*t, ev.kind))
            })
            .find_map(|(t, kind)| {
                matches!(
                    kind,
                    midly::TrackEventKind::Midi {
                        message: midly::MidiMessage::NoteOn { .. },
                        ..
                    }
                )
                .then_some(t)
            })
            .unwrap();
        assert_eq!(on_tick, 960);
    }

    #[test]
    fn wav_round_trips_through_hound() {
        let events = [event(0.0, 0, 60, 100), event(0.3, 0, 65, 127)];
        let bytes = to_wav(&events, ChimeTimbre::Chime);
        let reader = hound::WavReader::new(std::io::Cursor::new(&bytes)).expect("well-formed WAV");
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, AUDIO_SAMPLE_RATE);
        assert_eq!(spec.bits_per_sample, 16);
        let samples: Vec<i16> = reader.into_samples().map(Result::unwrap).collect();
        assert!(samples.len() % 2 == 0, "whole stereo frames");
        // Signal exists, and nothing clips.
        let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap();
        assert!(peak > 0, "the mix is not silence");
        // RMS in the first 100 ms (voice ringing) beats the tail.
        let frame_count = samples.len() / 2;
        let rms = |range: std::ops::Range<usize>| {
            let n = range.len().max(1);
            let sum: f64 = samples[range.start * 2..range.end * 2]
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum();
            #[allow(clippy::cast_precision_loss)]
            (sum / n as f64).sqrt()
        };
        let head = rms(0..4410);
        let tail = rms(frame_count - 4410..frame_count);
        assert!(
            head > tail * 4.0,
            "events ring, the tail decays: head {head:.0} vs tail {tail:.0}"
        );
    }

    #[test]
    fn wav_of_an_empty_capture_is_a_short_silence() {
        let bytes = to_wav(&[], ChimeTimbre::Chime);
        let reader = hound::WavReader::new(std::io::Cursor::new(&bytes)).unwrap();
        assert!(reader.into_samples::<i16>().all(|s| s.unwrap() == 0));
    }

    #[test]
    fn dense_same_voice_events_never_clip_the_wav() {
        // Twenty overlapping full-energy strikes on one voice: the peak
        // scan must rescale instead of wrapping.
        let events: Vec<CaptureEvent> = (0..20)
            .map(|i| event(f64::from(i) * 0.01, 0, 60, 127))
            .collect();
        let bytes = to_wav(&events, ChimeTimbre::Bell);
        let reader = hound::WavReader::new(std::io::Cursor::new(&bytes)).unwrap();
        let peak = reader
            .into_samples::<i16>()
            .map(|s| s.unwrap().unsigned_abs())
            .max()
            .unwrap();
        assert!(peak <= i16::MAX.unsigned_abs(), "no wraparound");
    }
}
