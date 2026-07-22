// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Native MIDI out: wall-chime strikes forwarded to an external MIDI
//! port (a DAW or hardware synth). Chimes only — collision pings stay
//! synth-side ambience — and the mapping is fixed this release: the
//! pentatonic degree becomes a key in C-major pentatonic from middle C,
//! velocity follows the impact energy on the same curve the audible
//! chime gain uses, everything on channel 1.
//!
//! The message logic lives in a pure `Scheduler` that turns strikes
//! into raw 3-byte messages and owns the pending note-offs, so all of
//! it unit-tests without a port; the `MidiOut` shell (Stage 2) owns the
//! actual `midir` connection and writes whatever the scheduler returns.
//!
//! MIDI is deliberately independent of the audio mute: silencing the
//! local synth while driving a DAW is a primary use of this feature.

use crate::audio::{NOTE_COUNT, PENTATONIC_SEMITONES, note_gain};
use std::time::{Duration, Instant};

/// How long a chime note is held before its note-off. Chime-like: long
/// enough for a DAW envelope to speak, short enough that dense scenes
/// don't stack sustains. Note-offs are drained once per frame, so they
/// quantize to frame boundaries (~16 ms of jitter on this gate) — that
/// is deliberate; a dedicated timing thread would buy nothing audible.
pub const NOTE_GATE: Duration = Duration::from_millis(200);

/// Zero-based MIDI channel the chimes speak on (channel 1 on the wire).
pub const MIDI_CHANNEL: u8 = 0;

/// MIDI key of the scale root: middle C, matching the synth's C4 root.
const ROOT_KEY: u8 = 60;

/// The MIDI key for a pentatonic scale degree, or None past the scale.
pub fn key_for_degree(degree: usize) -> Option<u8> {
    if degree >= NOTE_COUNT {
        return None;
    }
    // The semitone table holds whole numbers well inside u8 range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(ROOT_KEY + PENTATONIC_SEMITONES[degree] as u8)
}

/// Impact energy as MIDI velocity: the audible chime-gain curve itself
/// (`note_gain`'s 0.3 floor keeps gentle rolls speaking), scaled to the
/// wire range and clamped to 1..=127 — never a 0, which the wire would
/// read as a note-off.
pub fn velocity_for_energy(energy: f64) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = (note_gain(energy) * 127.0).round() as u8;
    v.clamp(1, 127)
}

/// Pure note lifecycle: turns strikes into raw messages and owns the
/// pending note-offs. No I/O — the caller writes the returned bytes.
#[derive(Default)]
pub(crate) struct Scheduler {
    /// Pending note-offs as (due time, key), in insertion order (which
    /// is due order: the gate is constant).
    pending: Vec<(Instant, u8)>,
}

impl Scheduler {
    /// Messages for a chime strike: a note-on for the degree, preceded
    /// by an early note-off when the same key is already ringing (a
    /// re-strike must never stack two note-ons on one key). Returns
    /// nothing for a degree outside the scale.
    pub(crate) fn note_on(&mut self, degree: usize, energy: f64, now: Instant) -> Vec<[u8; 3]> {
        let Some(key) = key_for_degree(degree) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(2);
        if let Some(i) = self.pending.iter().position(|(_, k)| *k == key) {
            self.pending.remove(i);
            out.push(note_off(key));
        }
        out.push([0x90 | MIDI_CHANNEL, key, velocity_for_energy(energy)]);
        self.pending.push((now + NOTE_GATE, key));
        out
    }

    /// Note-offs that have come due, in due order.
    pub(crate) fn flush(&mut self, now: Instant) -> Vec<[u8; 3]> {
        let mut out = Vec::new();
        self.pending.retain(|(due, key)| {
            if *due <= now {
                out.push(note_off(*key));
                false
            } else {
                true
            }
        });
        out
    }

    /// Every pending note-off at once — the disconnect/drop path. The
    /// caller follows with CC 123 (all notes off) for belt and braces.
    pub(crate) fn drain_all(&mut self) -> Vec<[u8; 3]> {
        self.pending
            .drain(..)
            .map(|(_, key)| note_off(key))
            .collect()
    }
}

/// CC 123 "all notes off" for the chime channel: sent on connect (a
/// previous run killed without cleanup may have left notes ringing)
/// and after the drop drain.
pub(crate) const ALL_NOTES_OFF: [u8; 3] = [0xB0 | MIDI_CHANNEL, 123, 0];

/// The wire form of releasing `key`: status 0x80 (note-off), velocity 0.
fn note_off(key: u8) -> [u8; 3] {
    [0x80 | MIDI_CHANNEL, key, 0]
}

/// Every port's display name, in port order, with a stable fallback for
/// ports whose name the backend cannot read.
fn port_names(out: &midir::MidiOutput) -> Vec<String> {
    out.ports()
        .iter()
        .map(|p| {
            out.port_name(p)
                .unwrap_or_else(|_| "(unnamed port)".to_string())
        })
        .collect()
}

/// A live connection to one MIDI output port: the thin I/O shell around
/// the pure `Scheduler`. Send errors are logged once and swallowed —
/// a yanked USB cable must never panic a frame.
pub struct MidiOut {
    conn: midir::MidiOutputConnection,
    scheduler: Scheduler,
    /// The connected port's full name, for the HUD and logs.
    pub port_name: String,
    /// A send has failed since connect (log once, then stay quiet).
    send_failed: bool,
}

impl MidiOut {
    /// Names of every MIDI output port, in port order. Empty when the
    /// backend fails to initialize (headless CI) — callers print their
    /// own "none found" message.
    pub fn ports() -> Vec<String> {
        let Ok(out) = midir::MidiOutput::new("bouncy") else {
            return Vec::new();
        };
        port_names(&out)
    }

    /// Connect to the port matching `query`: an all-digits query picks
    /// by index (the `--list-midi-ports` numbering), anything else
    /// matches case-insensitively on a name substring. The connection
    /// opens with CC 123 (all notes off) — a previous run killed
    /// without cleanup may have left notes ringing.
    pub fn connect(query: &str) -> Result<MidiOut, String> {
        let out = midir::MidiOutput::new("bouncy").map_err(|e| format!("MIDI init: {e}"))?;
        let ports = out.ports();
        if ports.is_empty() {
            return Err("no MIDI output ports found".to_string());
        }
        let names = port_names(&out);
        // The parse is the digits test: numeric queries pick by the
        // --list-midi-ports index, anything else matches a name.
        let index = if let Ok(i) = query.parse::<usize>() {
            if i >= ports.len() {
                return Err(format!("port index {i} out of range (0..{})", ports.len()));
            }
            i
        } else {
            let needle = query.to_lowercase();
            names
                .iter()
                .position(|n| n.to_lowercase().contains(&needle))
                .ok_or_else(|| {
                    format!("no MIDI output port matching '{query}' (see --list-midi-ports)")
                })?
        };
        let conn = out
            .connect(&ports[index], "bouncy chimes")
            .map_err(|e| format!("MIDI connect: {e}"))?;
        let mut midi = MidiOut {
            conn,
            scheduler: Scheduler::default(),
            port_name: names[index].clone(),
            send_failed: false,
        };
        midi.send(ALL_NOTES_OFF);
        Ok(midi)
    }

    /// Forward one chime strike (a note-on now, its note-off scheduled
    /// one [`NOTE_GATE`] later).
    pub fn note_on(&mut self, degree: usize, energy: f64, now: Instant) {
        for msg in self.scheduler.note_on(degree, energy, now) {
            self.send(msg);
        }
    }

    /// Send the note-offs that have come due. Called once per frame —
    /// note-offs quantize to frame boundaries by design (see
    /// [`NOTE_GATE`]).
    pub fn flush(&mut self, now: Instant) {
        for msg in self.scheduler.flush(now) {
            self.send(msg);
        }
    }

    fn send(&mut self, msg: [u8; 3]) {
        if let Err(e) = self.conn.send(&msg)
            && !self.send_failed
        {
            self.send_failed = true;
            eprintln!("MIDI send failed ({e}); further errors suppressed");
        }
    }
}

impl Drop for MidiOut {
    /// A quit must never leave a DAW droning: release everything still
    /// ringing, then belt-and-braces CC 123.
    fn drop(&mut self) {
        for msg in self.scheduler.drain_all() {
            let _ = self.conn.send(&msg);
        }
        let _ = self.conn.send(&ALL_NOTES_OFF);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_mirror_the_synth_scale() {
        // Cross-check against the audio table itself, not a copy.
        for (degree, semis) in PENTATONIC_SEMITONES.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let expected = 60 + *semis as u8;
            assert_eq!(key_for_degree(degree), Some(expected), "degree {degree}");
        }
        assert_eq!(key_for_degree(NOTE_COUNT), None, "past the scale");
        assert_eq!(key_for_degree(usize::MAX), None);
        // Root and ceiling as spot values: C4 and C6.
        assert_eq!(key_for_degree(0), Some(60));
        assert_eq!(key_for_degree(10), Some(84));
    }

    #[test]
    fn velocity_follows_the_audible_gain_curve() {
        assert_eq!(velocity_for_energy(0.0), 38, "0.3 floor of 127");
        assert_eq!(velocity_for_energy(800.0), 127, "full scale");
        assert_eq!(velocity_for_energy(1e9), 127, "clamped above");
        assert_eq!(velocity_for_energy(-5.0), 38, "negative clamps to floor");
        assert!(velocity_for_energy(0.0) > 0, "never a wire note-off");
        let half = velocity_for_energy(400.0);
        assert!(half > 38 && half < 127, "monotone midpoint: {half}");
    }

    #[test]
    fn strike_emits_note_on_and_schedules_the_off() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        let msgs = s.note_on(0, 800.0, now);
        assert_eq!(msgs, vec![[0x90, 60, 127]]);
        assert!(s.flush(now).is_empty(), "gate not yet elapsed");
        assert!(
            s.flush(now + Duration::from_millis(199)).is_empty(),
            "one millisecond short of the 200 ms gate"
        );
        assert_eq!(
            s.flush(now + NOTE_GATE),
            vec![[0x80, 60, 0]],
            "off lands at the gate"
        );
        assert!(s.flush(now + NOTE_GATE).is_empty(), "off sends once");
    }

    #[test]
    fn offs_drain_in_due_order_across_keys() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(0, 100.0, now);
        s.note_on(3, 100.0, now + Duration::from_millis(50));
        s.note_on(7, 100.0, now + Duration::from_millis(100));
        let offs = s.flush(now + NOTE_GATE + Duration::from_millis(60));
        let keys: Vec<u8> = offs.iter().map(|m| m[1]).collect();
        assert_eq!(
            keys,
            vec![key_for_degree(0).unwrap(), key_for_degree(3).unwrap()],
            "due ones only, in due order"
        );
        assert_eq!(
            s.flush(now + Duration::from_secs(1)).len(),
            1,
            "the third follows when due"
        );
    }

    #[test]
    fn same_key_restrike_sends_the_early_off_first() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(2, 100.0, now);
        let msgs = s.note_on(2, 800.0, now + Duration::from_millis(50));
        let key = key_for_degree(2).unwrap();
        assert_eq!(
            msgs,
            vec![[0x80, key, 0], [0x90, key, 127]],
            "early off precedes the new on"
        );
        // Exactly one pending off remains, at the re-strike's gate.
        assert!(s.flush(now + NOTE_GATE).is_empty(), "old deadline is gone");
        assert_eq!(
            s.flush(now + Duration::from_millis(50) + NOTE_GATE),
            vec![[0x80, key, 0]]
        );
    }

    #[test]
    fn drain_all_empties_everything() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(0, 100.0, now);
        s.note_on(5, 100.0, now);
        let offs = s.drain_all();
        assert_eq!(offs.len(), 2);
        assert!(offs.iter().all(|m| m[0] == 0x80 && m[2] == 0));
        assert!(s.flush(now + Duration::from_secs(10)).is_empty());
        assert_eq!(ALL_NOTES_OFF, [0xB0, 123, 0]);
    }

    #[test]
    fn out_of_scale_degree_is_silent() {
        let mut s = Scheduler::default();
        assert!(s.note_on(NOTE_COUNT, 800.0, Instant::now()).is_empty());
        assert!(s.pending.is_empty(), "nothing scheduled either");
    }

    // Never open a real port in tests: CI runners have none, and a
    // developer machine's DAW must not receive test notes.

    #[test]
    fn ports_never_panics() {
        let _ = MidiOut::ports();
    }

    /// `unwrap_err` needs Debug on the Ok side; a connection must also
    /// never be treated as printable — hence the match.
    fn expect_err(result: Result<MidiOut, String>) -> String {
        match result {
            Err(e) => e,
            Ok(_) => panic!("must not connect in tests"),
        }
    }

    #[test]
    fn connect_to_a_missing_port_errors_cleanly() {
        // Three environments, three failure shapes: a machine with
        // ports names the query, a machine without ports says so, and
        // a headless Linux runner (no /dev/snd/seq) fails at backend
        // init. All must error cleanly; none may connect.
        let err = expect_err(MidiOut::connect("no-such-port-zzz"));
        assert!(
            err.contains("no-such-port-zzz")
                || err.contains("no MIDI output ports")
                || err.contains("MIDI init"),
            "error names the query, the empty-port state, or the backend: {err}"
        );
        let err = expect_err(MidiOut::connect("9999"));
        assert!(
            err.contains("out of range")
                || err.contains("no MIDI output ports")
                || err.contains("MIDI init"),
            "index errors are specific: {err}"
        );
    }
}
