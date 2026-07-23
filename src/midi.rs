// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! MIDI out: wall-chime strikes forwarded to an external MIDI port (a
//! DAW or hardware synth — natively over `midir`, in the browser over
//! `WebMIDI`). Chimes only — collision pings stay synth-side ambience.
//! By default the pentatonic degree becomes a key in C-major pentatonic
//! from middle C on channel 1; a wall's [`WallMidi`] mapping can pin a
//! fixed key and/or a different channel per stroke. Velocity follows
//! the impact energy on the same curve the audible chime gain uses.
//!
//! The message logic lives in a pure, cross-target `Scheduler` that
//! turns resolved (channel, key, velocity) strikes into raw 3-byte
//! messages and owns the pending note-offs, so all of it unit-tests
//! without a port; the native `MidiOut` shell owns the actual `midir`
//! connection and writes whatever the scheduler returns.
//!
//! MIDI is deliberately independent of the audio mute: silencing the
//! local synth while driving a DAW is a primary use of this feature.

use crate::audio::{NOTE_COUNT, PENTATONIC_SEMITONES, note_gain};
use crate::presets::WallMidi;
// std::time on native; performance.now() on wasm, like the sim clock.
use web_time::{Duration, Instant};

/// How long a chime note is held before its note-off. Chime-like: long
/// enough for a DAW envelope to speak, short enough that dense scenes
/// don't stack sustains. Note-offs are drained once per frame, so they
/// quantize to frame boundaries (~16 ms of jitter on this gate) — that
/// is deliberate; a dedicated timing thread would buy nothing audible.
pub const NOTE_GATE: Duration = Duration::from_millis(200);

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

/// Human name of a MIDI key for inspector labels: `"60 (C4)"`. Octave
/// numbering follows the middle-C-equals-C4 convention the pentatonic
/// mapping's root uses.
pub fn key_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = i32::from(key) / 12 - 1;
    let name = NAMES[usize::from(key % 12)];
    format!("{key} ({name}{octave})")
}

/// Resolve one chime to wire terms: `(channel, key, velocity)`. The
/// wall's fixed [`WallMidi::key`] wins when set; otherwise the degree
/// maps through the pentatonic table, and `None` comes back only for
/// an out-of-scale degree with no override. Every sender — the native
/// port, the browser port, and the capture recorder — resolves through
/// here, so a wall's mapping means the same thing everywhere.
pub fn chime_message(degree: usize, midi: WallMidi, energy: f64) -> Option<(u8, u8, u8)> {
    let key = match midi.key {
        Some(key) => key,
        None => key_for_degree(degree)?,
    };
    Some((midi.channel, key, velocity_for_energy(energy)))
}

/// Pure note lifecycle: turns resolved strikes into raw messages and
/// owns the pending note-offs. No I/O — the caller writes the returned
/// bytes. Voices are keyed by (channel, key): the same key on two
/// channels is two independent notes (two DAW tracks).
#[derive(Default)]
pub(crate) struct Scheduler {
    /// Pending note-offs as (due time, channel, key), in insertion
    /// order (which is due order: the gate is constant).
    pending: Vec<(Instant, u8, u8)>,
    /// Bitmask of channels that have carried any note this connection,
    /// for targeted CC 123 on disconnect.
    used: u16,
}

impl Scheduler {
    /// Messages for a chime strike: a note-on, preceded by an early
    /// note-off when the same (channel, key) voice is already ringing
    /// (a re-strike must never stack two note-ons on one voice).
    pub(crate) fn note_on(
        &mut self,
        channel: u8,
        key: u8,
        velocity: u8,
        now: Instant,
    ) -> Vec<[u8; 3]> {
        let mut out = Vec::with_capacity(2);
        if let Some(i) = self
            .pending
            .iter()
            .position(|(_, c, k)| *c == channel && *k == key)
        {
            self.pending.remove(i);
            out.push(note_off(channel, key));
        }
        out.push([0x90 | channel, key, velocity]);
        self.pending.push((now + NOTE_GATE, channel, key));
        self.used |= 1 << channel;
        out
    }

    /// Note-offs that have come due, in due order.
    pub(crate) fn flush(&mut self, now: Instant) -> Vec<[u8; 3]> {
        let mut out = Vec::new();
        self.pending.retain(|(due, channel, key)| {
            if *due <= now {
                out.push(note_off(*channel, *key));
                false
            } else {
                true
            }
        });
        out
    }

    /// Every pending note-off at once — the disconnect path: the native
    /// drop and the browser's port switch. The caller follows with
    /// CC 123 (all notes off) for belt and braces.
    pub(crate) fn drain_all(&mut self) -> Vec<[u8; 3]> {
        self.pending
            .drain(..)
            .map(|(_, channel, key)| note_off(channel, key))
            .collect()
    }

    /// Channels that have carried any note this connection, as a
    /// bitmask (bit n = zero-based channel n).
    pub(crate) fn used_channels(&self) -> u16 {
        self.used
    }
}

/// CC 123 "all notes off" for one channel: sent on connect (broadcast —
/// a previous run killed without cleanup may have left notes ringing on
/// any channel) and after the drop drain (targeted by `used_channels`).
pub(crate) fn all_notes_off(channel: u8) -> [u8; 3] {
    [0xB0 | channel, 123, 0]
}

/// The wire form of releasing a voice: status 0x80 (note-off),
/// velocity 0.
fn note_off(channel: u8, key: u8) -> [u8; 3] {
    [0x80 | channel, key, 0]
}

/// Every port's display name, in port order, with a stable fallback for
/// ports whose name the backend cannot read.
#[cfg(not(target_arch = "wasm32"))]
fn port_names(out: &midir::MidiOutput) -> Vec<String> {
    out.ports()
        .iter()
        .map(|p| {
            out.port_name(p)
                .unwrap_or_else(|_| "(unnamed port)".to_string())
        })
        .collect()
}

/// The next stop on the panel's port cycle: none → each port in order →
/// none. A connected port that has vanished from the list (unplugged)
/// restarts the cycle at the first port. Pure — the caller enumerates
/// and connects.
#[cfg(not(target_arch = "wasm32"))]
pub fn next_port(current: Option<&str>, ports: &[String]) -> Option<String> {
    let first = ports.first()?;
    match current {
        None => Some(first.clone()),
        Some(name) => match ports.iter().position(|p| p == name) {
            Some(i) => ports.get(i + 1).cloned(),
            None => Some(first.clone()),
        },
    }
}

/// A live connection to one MIDI output port: the thin I/O shell around
/// the pure `Scheduler`. Send errors are logged once and swallowed —
/// a yanked USB cable must never panic a frame.
#[cfg(not(target_arch = "wasm32"))]
pub struct MidiOut {
    conn: midir::MidiOutputConnection,
    scheduler: Scheduler,
    /// The connected port's full name, for the HUD and logs.
    pub port_name: String,
    /// A send has failed since connect (log once, then stay quiet).
    send_failed: bool,
}

#[cfg(not(target_arch = "wasm32"))]
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
        // Broadcast: the predecessor's channels are unknowable.
        for channel in 0..16 {
            midi.send(all_notes_off(channel));
        }
        Ok(midi)
    }

    /// Forward one resolved chime strike (a note-on now, its note-off
    /// scheduled one [`NOTE_GATE`] later). Resolve with
    /// [`chime_message`] first.
    pub fn note_on(&mut self, channel: u8, key: u8, velocity: u8, now: Instant) {
        for msg in self.scheduler.note_on(channel, key, velocity, now) {
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

#[cfg(not(target_arch = "wasm32"))]
impl Drop for MidiOut {
    /// A quit must never leave a DAW droning: release everything still
    /// ringing, then belt-and-braces CC 123 on every channel that ever
    /// carried a note — plus channel 1 unconditionally, preserving the
    /// pre-mapping behavior verbatim for default configurations.
    fn drop(&mut self) {
        for msg in self.scheduler.drain_all() {
            let _ = self.conn.send(&msg);
        }
        let used = self.scheduler.used_channels() | 1;
        for channel in 0..16 {
            if used & (1 << channel) != 0 {
                let _ = self.conn.send(&all_notes_off(channel));
            }
        }
    }
}

/// The browser MIDI shell: `WebMIDI` (Chromium-first) driven Rust-side,
/// mirroring the `WebAudio` engine's shape — a thread-local holding the
/// live output plus the same pure [`Scheduler`] the native shell uses.
/// `requestMIDIAccess` is promise-based and permission-gated, so
/// enabling is asynchronous: [`web::enable`] kicks off the request and
/// the state machine flips to ready (or failed) frames later, observed
/// through the polled snapshot.
#[cfg(target_arch = "wasm32")]
pub(crate) mod web {
    use super::{Scheduler, all_notes_off};
    use std::cell::RefCell;
    use wasm_bindgen::{JsCast, JsValue};
    use web_time::Instant;

    enum State {
        /// Never asked (the page hasn't clicked Enable MIDI).
        Idle,
        /// Permission request in flight.
        Pending,
        /// Connected to an output (the first available at enable; the
        /// panel dropdown can switch afterward).
        Ready {
            /// Retained past the connect so the dropdown can re-
            /// enumerate ports and switch without a fresh permission
            /// request.
            access: web_sys::MidiAccess,
            output: web_sys::MidiOutput,
            scheduler: Scheduler,
            send_failed: bool,
        },
        /// No API, no ports, or permission denied — terminal until the
        /// user clicks again.
        Failed,
    }

    thread_local! {
        static MIDI: RefCell<State> = const { RefCell::new(State::Idle) };
    }

    /// Kick off the permission request (idempotent while pending or
    /// ready; a failed state retries — the user clicked again on
    /// purpose). Resolution lands asynchronously.
    pub(crate) fn enable() {
        let should_request = MIDI.with(|m| {
            let mut m = m.borrow_mut();
            match *m {
                State::Idle | State::Failed => {
                    *m = State::Pending;
                    true
                }
                State::Pending | State::Ready { .. } => false,
            }
        });
        if should_request {
            wasm_bindgen_futures::spawn_local(request());
        }
    }

    /// The async half: request access, take the first output port.
    async fn request() {
        let outcome = try_request().await;
        MIDI.with(|m| {
            *m.borrow_mut() = match outcome {
                Ok((access, output)) => {
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "MIDI out: connected to '{}'",
                        port_label(&output)
                    )));
                    // The predecessor's channels are unknowable —
                    // broadcast, like the native connect.
                    for channel in 0..16 {
                        let _ = send_bytes(&output, all_notes_off(channel));
                    }
                    State::Ready {
                        access,
                        output,
                        scheduler: Scheduler::default(),
                        send_failed: false,
                    }
                }
                Err(e) => {
                    web_sys::console::warn_1(&JsValue::from_str(&format!("MIDI unavailable: {e}")));
                    State::Failed
                }
            };
        });
    }

    async fn try_request() -> Result<(web_sys::MidiAccess, web_sys::MidiOutput), String> {
        let navigator = web_sys::window().ok_or("no window")?.navigator();
        let promise = navigator
            .request_midi_access()
            .map_err(|_| "Web MIDI API not available in this browser")?;
        let access = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|_| "permission denied")?
            .dyn_into::<web_sys::MidiAccess>()
            .map_err(|_| "unexpected MIDIAccess shape")?;
        let first = output_list(&access)
            .into_iter()
            .next()
            .ok_or("no MIDI output ports found")?;
        Ok((access, first))
    }

    /// Every output port, in the browser's enumeration order (stable
    /// within a session — the dropdown indexes into this).
    /// `MIDIOutputMap` is a read-only maplike: iterate its values via the
    /// JS Map interface, taking ports without knowing their
    /// browser-assigned ids.
    fn output_list(access: &web_sys::MidiAccess) -> Vec<web_sys::MidiOutput> {
        let outputs: js_sys::Map = access.outputs().unchecked_into();
        js_sys::try_iter(&outputs.values())
            .ok()
            .flatten()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|v| v.dyn_into::<web_sys::MidiOutput>().ok())
            .collect()
    }

    /// A port's display name, with the same fallback the native shell
    /// uses for unnamed ports.
    fn port_label(output: &web_sys::MidiOutput) -> String {
        output.name().unwrap_or_else(|| "(unnamed port)".into())
    }

    /// Names of every output port, in enumeration order — empty unless
    /// ready (the dropdown only shows after a successful enable).
    pub(crate) fn ports() -> Vec<String> {
        MIDI.with(|m| match *m.borrow() {
            State::Ready { ref access, .. } => output_list(access).iter().map(port_label).collect(),
            _ => Vec::new(),
        })
    }

    /// The connected port's name, if ready.
    pub(crate) fn current_port() -> Option<String> {
        MIDI.with(|m| match *m.borrow() {
            State::Ready { ref output, .. } => Some(port_label(output)),
            _ => None,
        })
    }

    /// Switch the live connection to the port at `index` in the current
    /// enumeration. The old port is silenced first (drain + CC 123 on
    /// every used channel), the new one starts with a fresh scheduler —
    /// the browser twin of the native panel's port cycle. False when
    /// not ready or the index is stale.
    pub(crate) fn select_port(index: u32) -> bool {
        MIDI.with(|m| {
            let mut state = m.borrow_mut();
            let State::Ready {
                ref access,
                ref mut output,
                ref mut scheduler,
                ref mut send_failed,
            } = *state
            else {
                return false;
            };
            let Some(next) = output_list(access).into_iter().nth(index as usize) else {
                return false;
            };
            if next.id() == output.id() {
                return true;
            }
            // Silence the old port before the swap: pending note-offs,
            // then CC 123 on every channel that carried a note.
            for msg in scheduler.drain_all() {
                let _ = send_bytes(output, msg);
            }
            let used = scheduler.used_channels();
            for channel in 0..16 {
                if used & (1 << channel) != 0 {
                    let _ = send_bytes(output, all_notes_off(channel));
                }
            }
            // The new port opens like every fresh connect: broadcast
            // CC 123 (its previous tenant's channels are unknowable).
            for channel in 0..16 {
                let _ = send_bytes(&next, all_notes_off(channel));
            }
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "MIDI out: switched to '{}'",
                port_label(&next)
            )));
            *output = next;
            *scheduler = Scheduler::default();
            *send_failed = false;
            true
        })
    }

    fn send_bytes(output: &web_sys::MidiOutput, msg: [u8; 3]) -> Result<(), JsValue> {
        let data = js_sys::Array::of3(
            &JsValue::from(msg[0]),
            &JsValue::from(msg[1]),
            &JsValue::from(msg[2]),
        );
        output.send(&data)
    }

    /// A port is connected and ready to send.
    pub(crate) fn ready() -> bool {
        MIDI.with(|m| matches!(*m.borrow(), State::Ready { .. }))
    }

    /// The last enable attempt failed (drives the page's error banner).
    pub(crate) fn failed() -> bool {
        MIDI.with(|m| matches!(*m.borrow(), State::Failed))
    }

    /// Forward one resolved chime strike, exactly like the native
    /// `MidiOut::note_on`. A no-op unless ready.
    pub(crate) fn note_on(channel: u8, key: u8, velocity: u8, now: Instant) {
        MIDI.with(|m| {
            if let State::Ready {
                ref output,
                ref mut scheduler,
                ref mut send_failed,
                ..
            } = *m.borrow_mut()
            {
                for msg in scheduler.note_on(channel, key, velocity, now) {
                    if send_bytes(output, msg).is_err() && !*send_failed {
                        *send_failed = true;
                        web_sys::console::warn_1(&JsValue::from_str(
                            "MIDI send failed; further errors suppressed",
                        ));
                    }
                }
            }
        });
    }

    /// Send the note-offs that have come due. Called once per frame.
    pub(crate) fn flush(now: Instant) {
        MIDI.with(|m| {
            if let State::Ready {
                ref output,
                ref mut scheduler,
                ..
            } = *m.borrow_mut()
            {
                for msg in scheduler.flush(now) {
                    let _ = send_bytes(output, msg);
                }
            }
        });
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

    /// The 1.12 call shape: resolve a bare degree at default mapping and
    /// feed the scheduler — the equivalence baseline for the refactor.
    fn strike(s: &mut Scheduler, degree: usize, energy: f64, now: Instant) -> Vec<[u8; 3]> {
        match chime_message(degree, WallMidi::default(), energy) {
            Some((ch, key, vel)) => s.note_on(ch, key, vel, now),
            None => Vec::new(),
        }
    }

    #[test]
    fn default_mapping_reproduces_the_1_12_byte_streams() {
        // The exact wire bytes the 1.12/1.13 releases produced for a
        // default configuration: channel 0 status bytes, pentatonic
        // keys, gain-curve velocities.
        let mut s = Scheduler::default();
        let now = Instant::now();
        assert_eq!(strike(&mut s, 0, 800.0, now), vec![[0x90, 60, 127]]);
        assert_eq!(
            s.flush(now + NOTE_GATE),
            vec![[0x80, 60, 0]],
            "off lands at the gate on channel 1"
        );
        assert_eq!(strike(&mut s, 10, 0.0, now), vec![[0x90, 84, 38]]);
        assert_eq!(s.used_channels(), 1, "only channel 1 ever used");
    }

    #[test]
    fn key_names_speak_daw() {
        assert_eq!(key_name(60), "60 (C4)", "middle C");
        assert_eq!(key_name(0), "0 (C-1)", "the MIDI floor");
        assert_eq!(key_name(127), "127 (G9)", "the MIDI ceiling");
        assert_eq!(key_name(69), "69 (A4)", "concert A");
        assert_eq!(key_name(61), "61 (C#4)");
    }

    #[test]
    fn chime_message_resolves_override_auto_and_out_of_scale() {
        let auto = WallMidi::default();
        assert_eq!(chime_message(0, auto, 800.0), Some((0, 60, 127)));
        let mapped = WallMidi {
            key: Some(48),
            channel: 9,
        };
        assert_eq!(
            chime_message(3, mapped, 800.0),
            Some((9, 48, 127)),
            "fixed key wins over the degree, channel carried"
        );
        assert_eq!(
            chime_message(NOTE_COUNT, mapped, 800.0),
            Some((9, 48, 127)),
            "an override even rescues an out-of-scale degree"
        );
        assert_eq!(
            chime_message(NOTE_COUNT, auto, 800.0),
            None,
            "no override, out of scale: silent"
        );
    }

    #[test]
    fn strike_emits_note_on_and_schedules_the_off() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        let msgs = s.note_on(0, 60, 127, now);
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
    fn offs_drain_in_due_order_across_voices() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(0, 60, 100, now);
        s.note_on(2, 65, 100, now + Duration::from_millis(50));
        s.note_on(0, 72, 100, now + Duration::from_millis(100));
        let offs = s.flush(now + NOTE_GATE + Duration::from_millis(60));
        assert_eq!(
            offs,
            vec![[0x80, 60, 0], [0x82, 65, 0]],
            "due ones only, in due order, each on its own channel"
        );
        assert_eq!(
            s.flush(now + Duration::from_secs(1)),
            vec![[0x80, 72, 0]],
            "the third follows when due"
        );
    }

    #[test]
    fn same_voice_restrike_sends_the_early_off_first() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(4, 67, 50, now);
        let msgs = s.note_on(4, 67, 127, now + Duration::from_millis(50));
        assert_eq!(
            msgs,
            vec![[0x84, 67, 0], [0x94, 67, 127]],
            "early off precedes the new on, both on the voice's channel"
        );
        // Exactly one pending off remains, at the re-strike's gate.
        assert!(s.flush(now + NOTE_GATE).is_empty(), "old deadline is gone");
        assert_eq!(
            s.flush(now + Duration::from_millis(50) + NOTE_GATE),
            vec![[0x84, 67, 0]]
        );
    }

    #[test]
    fn same_key_on_two_channels_is_two_independent_voices() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(0, 60, 100, now);
        let msgs = s.note_on(5, 60, 100, now + Duration::from_millis(50));
        assert_eq!(
            msgs,
            vec![[0x95, 60, 100]],
            "no early off: channel 6's C4 is not channel 1's C4"
        );
        let offs = s.flush(now + Duration::from_secs(1));
        assert_eq!(offs, vec![[0x80, 60, 0], [0x85, 60, 0]]);
        assert_eq!(s.used_channels(), 0b10_0001, "channels 1 and 6 marked");
    }

    #[test]
    fn drain_all_empties_everything() {
        let mut s = Scheduler::default();
        let now = Instant::now();
        s.note_on(0, 60, 100, now);
        s.note_on(3, 69, 100, now);
        let offs = s.drain_all();
        assert_eq!(offs, vec![[0x80, 60, 0], [0x83, 69, 0]]);
        assert!(s.flush(now + Duration::from_secs(10)).is_empty());
        assert_eq!(all_notes_off(0), [0xB0, 123, 0], "the 1.12 bytes");
        assert_eq!(all_notes_off(15), [0xBF, 123, 0]);
    }

    #[test]
    fn out_of_scale_degree_is_silent_without_an_override() {
        let mut s = Scheduler::default();
        assert!(strike(&mut s, NOTE_COUNT, 800.0, Instant::now()).is_empty());
        assert!(s.pending.is_empty(), "nothing scheduled either");
        assert_eq!(s.used_channels(), 0, "no channel marked");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn next_port_cycles_none_through_every_port_and_back() {
        let ports = vec!["IAC Bus 1".to_string(), "Synth USB".to_string()];
        assert_eq!(next_port(None, &ports), Some("IAC Bus 1".to_string()));
        assert_eq!(
            next_port(Some("IAC Bus 1"), &ports),
            Some("Synth USB".to_string())
        );
        assert_eq!(
            next_port(Some("Synth USB"), &ports),
            None,
            "past the last port the cycle returns to none"
        );
        // A single port toggles none <-> that port.
        let one = vec!["IAC Bus 1".to_string()];
        assert_eq!(next_port(None, &one), Some("IAC Bus 1".to_string()));
        assert_eq!(next_port(Some("IAC Bus 1"), &one), None);
        // No ports: nowhere to go, connected or not.
        assert_eq!(next_port(None, &[]), None);
        assert_eq!(next_port(Some("IAC Bus 1"), &[]), None);
        // The connected port vanished (unplugged): restart at the first.
        assert_eq!(
            next_port(Some("Gone USB"), &ports),
            Some("IAC Bus 1".to_string())
        );
    }

    // Never open a real port in tests: CI runners have none, and a
    // developer machine's DAW must not receive test notes.

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn ports_never_panics() {
        let _ = MidiOut::ports();
    }

    /// `unwrap_err` needs Debug on the Ok side; a connection must also
    /// never be treated as printable — hence the match.
    #[cfg(not(target_arch = "wasm32"))]
    fn expect_err(result: Result<MidiOut, String>) -> String {
        match result {
            Err(e) => e,
            Ok(_) => panic!("must not connect in tests"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
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
