// Bouncy web demo: loader + control panel. The panel is a thin veneer
// over the wasm WebHandle, which is itself a veneer over the same
// Command dispatch the keyboard uses. State flows one way: the panel
// sends commands, and every readout/control reflects the per-frame
// snapshot the simulation publishes.

const $ = (id) => document.getElementById(id);

function fail(message) {
  const el = $("error");
  el.textContent = String(message);
  el.hidden = false;
}

// Surface silent failures (wasm traps inside rAF callbacks never reach
// the console API) on the page itself.
window.addEventListener("error", (e) => {
  fail(`${e.message}\n@ ${e.filename}:${e.lineno}`);
});
window.addEventListener("unhandledrejection", (e) => {
  fail(`unhandled rejection: ${e.reason}`);
});

// Cache-busting: index.html loads this file as controls.js?v=<crate
// version> (a drift test pins the tag to the Rust version), and that
// tag is forwarded to the wasm module chain below. Without it, browsers
// heuristically cache pkg/bouncy.js and the .wasm for months, and a
// returning visitor keeps running a stale simulation after a deploy.
const BUNDLE_V = new URL(import.meta.url).searchParams.get("v") ?? "dev";
const versioned = (path) => `${path}?v=${BUNDLE_V}`;

// Load the single-threaded bundle by default. A multi-threaded bundle
// (pkg-mt/) is used when the page is cross-origin isolated and the
// bundle exists (?st forces single-threaded); see web/README.md.
// Returns { mod, initPool }: the thread pool initializes after the app
// is up, since rayon is only consulted above the population thresholds.
async function loadBouncy() {
  const forceSt = new URLSearchParams(location.search).has("st");
  if (globalThis.crossOriginIsolated && !forceSt) {
    try {
      const mt = await import(versioned("./pkg-mt/bouncy.js"));
      await mt.default({
        module_or_path: new URL(versioned("pkg-mt/bouncy_bg.wasm"), location.href),
      });
      const threads = navigator.hardwareConcurrency ?? 4;
      const initPool = async () => {
        await mt.initThreadPool(threads);
        $("threads-note").textContent = `Multi-threaded build: ${threads} threads.`;
      };
      return { mod: mt, initPool };
    } catch {
      // No MT bundle deployed (or it failed to load): fall through.
    }
  }
  const st = await import(versioned("./pkg/bouncy.js"));
  await st.default({
    module_or_path: new URL(versioned("pkg/bouncy_bg.wasm"), location.href),
  });
  $("threads-note").textContent = globalThis.crossOriginIsolated && !forceSt
    ? "Single-threaded build (no multi-threaded bundle deployed)."
    : "Single-threaded build (multi-threading needs cross-origin isolation).";
  return { mod: st, initPool: null };
}

// ---- Placement tools ---------------------------------------------------
// Position-shaped actions arm a one-shot tool: the button lights up, a
// chip narrates it, and the next click on the canvas places the action
// there. The overlay swallows that click before winit can turn it into
// a burst; Esc or a second press on the button cancels. The keyboard
// path (hover the canvas, press W etc.) is unchanged.
let armed = null; // { id, label, act(x, y) }

function disarmTool() {
  if (!armed) return;
  $(armed.id).classList.remove("armed");
  armed = null;
  $("tool-overlay").hidden = true;
  $("chip-tool").hidden = true;
}

function armTool(id, spec) {
  if (armed && armed.id === id) {
    disarmTool(); // second press cancels
    return;
  }
  disarmTool();
  armed = { id, ...spec };
  $(id).classList.add("armed");
  $("tool-overlay").hidden = false;
  const chip = $("chip-tool");
  chip.textContent = spec.drag
    ? `placing ${spec.label} (Esc cancels)`
    : `placing ${spec.label} — click the canvas (Esc cancels)`;
  chip.hidden = false;
}

// Map a click's CSS position to simulation coordinates. The canvas
// backing store stays at simulation size while object-fit: contain
// letterboxes it in the element — this is the CSS-space twin of
// render.rs::window_pos_to_sim, using the snapshot's arena size.
function clickToSim(e) {
  const s = latest;
  if (!s.width || !s.height) return null;
  const rect = $("bouncy").getBoundingClientRect();
  const scale = Math.min(rect.width / s.width, rect.height / s.height);
  const offX = (rect.width - s.width * scale) / 2;
  const offY = (rect.height - s.height * scale) / 2;
  const x = (e.clientX - rect.left - offX) / scale;
  const y = (e.clientY - rect.top - offY) / scale;
  return [
    Math.min(Math.max(x, 0), s.width),
    Math.min(Math.max(y, 0), s.height),
  ];
}

function bind(handle) {
  const canvas = $("bouncy");
  $("btn-pause").onclick = () => handle.set_paused?.(!latest.paused);
  $("btn-step").onclick = () => handle.step_frame?.();
  $("btn-reset").onclick = () => handle.reset?.();

  const tools = {
    "btn-burst": { label: "burst", act: (x, y) => handle.spawn_burst?.(x, y) },
    "btn-comet": { label: "comet", act: (x, y) => handle.launch_comet?.(x, y) },
    "btn-well": { label: "well", act: (x, y) => handle.pin_well?.(x, y, false) },
    "btn-repel": {
      label: "repeller",
      act: (x, y) => handle.pin_well?.(x, y, true),
    },
    "btn-emitter": {
      label: "emitter",
      act: (x, y) => handle.place_emitter?.(x, y),
    },
    "btn-explode": {
      label: "explosion",
      act: (x, y) => handle.trigger_explosion?.(x, y),
    },
    // Inspector tools. Optional-chained: a stale cached wasm bundle
    // without these exports degrades to a no-op click.
    "btn-select": {
      label: "selection (click an emitter or wall)",
      act: (x, y) => handle.select_at?.(x, y),
    },
    "btn-reaim": {
      label: "emitter aim",
      act: (x, y) => {
        if (latest.selection_id != null) {
          handle.aim_emitter_at?.(latest.selection_id, x, y);
        }
      },
    },
    // The one drag-shaped tool: no act — the overlay's pointer
    // handlers below own it. Works on touch, where held-V cannot.
    "btn-wall": {
      label: "wall — drag on the canvas",
      drag: true,
    },
  };
  for (const [id, spec] of Object.entries(tools)) {
    $(id).onclick = () => armTool(id, spec);
  }
  $("tool-overlay").onclick = (e) => {
    // Drag tools are handled by the pointer listeners below; the
    // click that follows their pointerup must not disarm-and-miss.
    if (armed && armed.drag) return;
    const pos = armed && clickToSim(e);
    if (pos) armed.act(...pos);
    disarmTool();
  };

  // The wall tool's drag: press anchors, each move past the same 12 px
  // threshold the native V-drag uses emits a segment (the first opens
  // a stroke, the rest chain onto it), release ends the stroke and
  // disarms. Pointer events cover mouse and touch alike.
  let wallAnchor = null;
  let wallExtend = false;
  const overlay = $("tool-overlay");
  overlay.addEventListener("pointerdown", (e) => {
    if (!armed || !armed.drag) return;
    overlay.setPointerCapture(e.pointerId);
    wallAnchor = clickToSim(e);
    wallExtend = false;
  });
  overlay.addEventListener("pointermove", (e) => {
    if (!wallAnchor || !armed) return;
    const pos = clickToSim(e);
    if (!pos) return;
    const [x1, y1] = wallAnchor;
    const [x2, y2] = pos;
    if (Math.hypot(x2 - x1, y2 - y1) < 12) return;
    handle.draw_wall?.(x1, y1, x2, y2, wallExtend);
    wallAnchor = pos;
    wallExtend = true;
  });
  overlay.addEventListener("pointerup", (e) => {
    if (!wallAnchor) return;
    // The release point completes the stroke: a fast drag (or a touch
    // flick) can cross the whole arena between move events, and the
    // last reach must not be dropped.
    const pos = clickToSim(e);
    if (pos) {
      const [x1, y1] = wallAnchor;
      const [x2, y2] = pos;
      if (Math.hypot(x2 - x1, y2 - y1) >= 12) {
        handle.draw_wall?.(x1, y1, x2, y2, wallExtend);
      }
    }
    wallAnchor = null;
    disarmTool();
  });
  // Capture phase: cancel the tool before winit's canvas listener can
  // see the Escape (which would otherwise stop the whole event loop).
  window.addEventListener(
    "keydown",
    (e) => {
      if (armed && e.key === "Escape") {
        disarmTool();
        e.stopPropagation();
        e.preventDefault();
      }
    },
    true,
  );

  // Enable MIDI: first click asks for permission (gesture-gated, like
  // Enable sound); once ready, later clicks toggle sending.
  $("midi-port").onchange = (e) => {
    // Index into the snapshot's midi_ports enumeration (browser port
    // order is stable within a session).
    handle.set_midi_port?.(e.target.selectedIndex);
  };
  $("btn-midi").onclick = () => {
    if (latest.midi_ready) {
      handle.toggle_midi?.();
    } else {
      handle.enable_midi?.();
    }
  };

  $("btn-clear-wells").onclick = () => handle.clear_wells?.();
  $("btn-clear-walls").onclick = () => handle.clear_walls?.();
  $("btn-clear-emitters").onclick = () => handle.clear_emitters?.();

  // Inspector actions, all addressed to the currently selected id.
  $("btn-deselect").onclick = () => handle.deselect?.();
  $("btn-delete-emitter").onclick = () => {
    if (latest.selection_id != null) handle.delete_emitter?.(latest.selection_id);
  };
  $("btn-delete-stroke").onclick = () => {
    if (latest.selection_id != null) handle.delete_stroke?.(latest.selection_id);
  };
  $("btn-note").onclick = () => {
    if (latest.selection_id != null) {
      handle.cycle_stroke_note?.(latest.selection_id);
    }
  };
  $("btn-gate").onclick = () => {
    if (latest.selection_id != null) {
      handle.cycle_stroke_gate?.(latest.selection_id);
    }
  };
  $("btn-pass").onclick = () => {
    if (latest.selection_id != null) {
      handle.cycle_stroke_pass?.(latest.selection_id);
    }
  };
  $("btn-enote").onclick = () => {
    if (latest.selection_id != null) {
      handle.cycle_emitter_note?.(latest.selection_id);
    }
  };
  $("in-erate").oninput = (e) => {
    if (latest.selection_id != null) {
      handle.set_emitter_rate?.(latest.selection_id, Number(e.target.value));
    }
  };
  $("in-ecap").oninput = (e) => {
    if (latest.selection_id != null) {
      handle.set_emitter_cap?.(latest.selection_id, Number(e.target.value));
    }
  };
  $("in-midikey").oninput = (e) => {
    if (latest.selection_id != null) {
      handle.set_stroke_midi_key?.(latest.selection_id, Number(e.target.value));
    }
  };
  $("in-midich").oninput = (e) => {
    if (latest.selection_id != null) {
      handle.set_stroke_midi_channel?.(latest.selection_id, Number(e.target.value));
    }
  };
  $("btn-color").onclick = () => handle.cycle_color_mode?.();
  $("btn-spawn").onclick = () => handle.cycle_spawn_mode?.();
  $("btn-hud").onclick = () => handle.cycle_hud?.();
  $("btn-fullscreen").onclick = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else $("stage").requestFullscreen?.();
  };

  $("in-gravity").oninput = (e) => handle.set_gravity?.(Number(e.target.value));
  $("in-pelastic").oninput = (e) =>
    handle.set_particle_elasticity?.(Number(e.target.value));
  $("in-welastic").oninput = (e) =>
    handle.set_wall_elasticity?.(Number(e.target.value));
  $("in-time").oninput = (e) => handle.set_time_scale?.(Number(e.target.value));
  $("in-threshold").oninput = (e) =>
    handle.set_explosion_threshold?.(Number(e.target.value));
  $("in-pings").oninput = (e) => handle.set_ping_volume?.(Number(e.target.value));
  // Optional-chained like the inspector controls: a stale cached wasm
  // bundle without the setters degrades to a no-op.
  $("in-bpm").oninput = (e) => handle.set_bpm?.(Number(e.target.value));
  $("btn-beat-div").onclick = () => {
    const next = { 1: 2, 2: 4, 4: 8, 8: 1 }[latest.beat_div ?? 4] ?? 4;
    handle.set_beat_div?.(next);
  };

  $("tg-matter").onchange = () => handle.toggle_matter?.();
  $("tg-flow").onchange = () => handle.toggle_flow?.();
  $("tg-selfgrav").onchange = () => handle.toggle_self_gravity?.();
  $("tg-trails").onchange = () => handle.toggle_trails?.();
  $("tg-kaleido").onchange = () => handle.toggle_kaleidoscope?.();
  $("tg-music").onchange = (e) => handle.set_music?.(e.target.checked);
  $("tg-chimes").onchange = () => handle.toggle_wall_chimes?.();

  // First click creates the WebAudio engine (must happen synchronously
  // inside the gesture, per autoplay policy) and unmutes; afterwards the
  // button is a plain mute toggle.
  $("btn-sound").onclick = () => {
    if (!latest.audio_ready) {
      if (!handle.enable_audio?.()) {
        fail("WebAudio unavailable in this browser");
      }
    } else {
      handle.set_muted?.(!latest.muted);
    }
  };

  $("btn-shot").onclick = () => {
    canvas.toBlob((blob) => {
      if (blob) download(blob, `bouncy-${Date.now()}.png`);
    });
  };
  $("btn-scene").onclick = () => {
    const toml = handle.scene_toml?.();
    if (toml) {
      download(new Blob([toml], { type: "application/toml" }),
               `bouncy-scene-${Date.now()}.toml`);
    }
  };
  $("btn-share").onclick = async () => {
    await navigator.clipboard.writeText(shareUrl());
    const btn = $("btn-share");
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = "Copy share link"), 1200);
  };

  $("panel-toggle").onclick = () => $("panel").classList.toggle("open");

  // Hotkeys (V-drag walls, D-select, Y, ...) arrive through winit's
  // listeners on the canvas, which need DOM focus — where focus starts
  // on load, and where every panel click yanks it away. Clicking the
  // arena to get it back would burst, so instead the panel hands focus
  // straight back after each interaction: the panel is a veneer, never
  // the focus owner (matching the native shell, where the panel cannot
  // hold focus at all).
  const refocusCanvas = () => $("bouncy").focus({ preventScroll: true });
  $("panel").addEventListener("pointerup", (e) => {
    // Not for <select>s: yanking focus off a select dismisses its
    // native popup the instant it opens. Focus returns on 'change'
    // below instead, once a choice has actually been made.
    if (e.target.closest("select")) return;
    setTimeout(refocusCanvas, 0);
  });
  $("panel").addEventListener("change", (e) => {
    if (e.target.matches("select")) setTimeout(refocusCanvas, 0);
  });
  $("panel-toggle").addEventListener("pointerup", () => setTimeout(refocusCanvas, 0));
  canvas.addEventListener("pointerdown", () => canvas.focus());

  // Live resize: the arena tracks the canvas's CSS size (debounced so a
  // drag-resize doesn't thrash the O(n) rescale). Between debounces the
  // CSS letterboxing keeps the old frame displayed correctly.
  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const w = Math.round(canvas.clientWidth);
      const h = Math.round(canvas.clientHeight);
      if (w > 0 && h > 0 && (w !== latest.width || h !== latest.height)) {
        handle.resize?.(w, h);
      }
    }, 250);
  }).observe(canvas);
}

// Launch options: construction-time parameters (the native command
// line's role). Applying rebuilds the URL — preserving any parameters
// typed by hand — and reloads; the CLI parser validates on the way back
// in, surfacing any error in the panel.
function bindLaunchOptions(mod) {
  const params = new URLSearchParams(location.search);
  const sel = $("lo-preset");
  for (const name of mod.preset_names()) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    sel.appendChild(option);
  }
  const fields = [["lo-size", "particle-size"], ["lo-speed", "initial-speed"],
                  ["lo-seed", "seed"], ["lo-min", "min-particles"]];
  sel.value = params.get("preset") ?? "";
  for (const [id, key] of fields) $(id).value = params.get(key) ?? "";

  $("lo-apply").onclick = () => {
    // The most recent deliberate act wins. Preset unchanged: start from
    // the share link's parameters (the launch URL plus the session's
    // touched settings), so adjustments travel — exactly like the
    // native panel's relaunch. Preset changed: the new bundle should
    // take precedence over tweaks made while exploring the old one, so
    // start from a clean slate keeping only the loader's `st`.
    const current = new URLSearchParams(location.search);
    const presetChanged = (sel.value || "") !== (current.get("preset") ?? "");
    const p = presetChanged
      ? new URLSearchParams(current.has("st") ? "st" : "")
      : new URLSearchParams(new URL(shareUrl()).search);
    const set = (key, value) => (value ? p.set(key, value) : p.delete(key));
    set("preset", sel.value);
    for (const [id, key] of fields) set(key, $(id).value.trim());
    location.search = p.toString();
  };
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

let latest = { paused: false };
// The configuration the session started with (first real snapshot):
// the share link carries the launch URL plus only what changed since.
let initialState = null;

// Build a URL that reproduces this session: start from the launch
// parameters (so ?preset=...&seed=... survive), then overlay every
// runtime setting the user changed during play — later parameters win
// in the CLI parser, exactly like typing them after a preset. Booleans
// switched OFF are best-effort: the parameter is removed, but a value
// baked into a preset cannot be negated (the CLI has the same limit).
function shareUrl() {
  const p = new URLSearchParams(location.search);
  p.delete("cb");
  const s = latest;
  const init = initialState ?? s;

  const numeric = [
    ["bpm", s.bpm ?? 0, init.bpm ?? 0, String],
    ["beat-div", s.beat_div ?? 4, init.beat_div ?? 4, String],
    ["ping-volume", s.ping_volume, init.ping_volume, String],
    ["gravity", s.gravity, init.gravity, String],
    ["particle-elasticity", s.particle_elasticity, init.particle_elasticity,
     (v) => v.toFixed(2)],
    ["wall-elasticity", s.wall_elasticity, init.wall_elasticity,
     (v) => v.toFixed(2)],
    ["explosion-threshold", s.explosion_threshold, init.explosion_threshold,
     String],
    ["spawn-mode", s.spawn_mode, init.spawn_mode, String],
    ["color-mode", s.color_mode, init.color_mode, String],
  ];
  for (const [key, now, was, fmt] of numeric) {
    if (now !== was) p.set(key, fmt(now));
  }
  const flags = [
    ["matter", s.matter, init.matter],
    ["flow", s.flow, init.flow],
    ["self-gravity", s.self_gravity, init.self_gravity],
    ["trails", s.trails, init.trails],
    ["kaleidoscope", s.kaleidoscope, init.kaleidoscope],
    ["music", s.music, init.music],
    ["wall-chimes", s.wall_chimes, init.wall_chimes],
    ["mute", s.muted, init.muted],
  ];
  for (const [key, now, was] of flags) {
    if (now !== was) {
      if (now) p.set(key, "");
      else p.delete(key);
    }
  }
  const query = p.toString().replace(/=(?=&|$)/g, "");
  return `${location.origin}${location.pathname}${query ? "?" + query : ""}`;
}
// Console access for testing and tinkering.
globalThis.bouncyShareUrl = shareUrl;

// The selection id the inspector last showed: slider *inputs* are
// (re)written only when this changes — outputs update every frame, but
// rewriting a range input mid-drag would fight the user's thumb.
let lastSelId = null;
// The MIDI-failure banner fires once per failure, not every poll.
let midiFailedShown = false;
// The port list the dropdown last showed (as JSON): options are
// rebuilt only when the enumeration changes — rewriting a <select>
// every frame would fight the user's open menu.
let lastMidiPorts = null;

// Reflect the snapshot into the panel. Sliders follow the simulation
// unless the user is mid-drag (their element has focus).
// Write a text value only when it changed: reflect() runs every
// animation frame, and unconditional textContent writes — even
// same-string ones — replace text nodes and dirty layout inside the
// scrolling panel. During an active scroll that forced layout work
// competes with the same rAF budget the simulation runs on.
const textCache = new Map();
function setText(id, value) {
  if (textCache.get(id) === value) return;
  textCache.set(id, value);
  $(id).textContent = value;
}

function reflect(s) {
  latest = s;

  // Inspector: `??` guards degrade a stale wasm bundle (no selection
  // fields) to a permanently hidden inspector, never an error.
  const kind = s.selection_kind ?? null;
  $("inspector").hidden = kind === null;
  $("insp-emitter").hidden = kind !== "emitter";
  $("insp-stroke").hidden = kind !== "stroke";
  if (kind === "emitter") {
    setText("insp-emitter-label", `Emitter #${s.selection_id} — aim ${Math.round(s.selection_angle)}°`);
    setText("out-erate", `${s.selection_rate.toFixed(1)}/s`);
    setText("out-ecap", `${s.selection_cap} live`);
    if (s.selection_id !== lastSelId) {
      $("in-erate").value = s.selection_rate;
      $("in-ecap").value = s.selection_cap;
    }
    // Pre-1.13 bundle: no emitter-note field yet.
    setText("btn-enote", `Note: ${s.selection_emitter_note ?? "…"}`);
  } else if (kind === "stroke") {
    const n = s.selection_segments;
    setText("insp-stroke-label", `Wall #${s.selection_id} — ${n} segment${n === 1 ? "" : "s"}`);
    setText("btn-note", `Note: ${s.selection_note}`);
    // Pre-1.13 bundle: no filter fields yet.
    setText("btn-gate", `Gate: ${s.selection_gate ?? "…"}`);
    setText("btn-pass", `Pass: ${s.selection_pass ?? "…"}`);
    // Pre-1.14 bundle: no MIDI mapping fields — the sliders stay hidden.
    const hasMidi = s.selection_midi_key != null;
    $("insp-midi").hidden = !hasMidi;
    if (hasMidi) {
      setText("out-midikey", s.selection_midi_key);
      setText("out-midich", `${s.selection_midi_channel}`);
      if (s.selection_id !== lastSelId) {
        $("in-midikey").value = s.selection_midi_key === "auto"
          ? -1
          : parseInt(s.selection_midi_key, 10);
        $("in-midich").value = s.selection_midi_channel;
      }
    }
  }
  lastSelId = kind === null ? null : s.selection_id;
  setText("ro-fps", s.fps.toFixed(0));
  setText("ro-particles", s.particles.toLocaleString());
  setText("ro-cap", `of ${s.max_particles.toLocaleString()}`);
  setText("ro-births", `${s.birth_rate}`);
  setText("ro-size", `${s.width}×${s.height}`);
  setText("btn-pause", s.paused ? "Resume" : "Pause");

  // Status chips: the running/paused/stopped chip is always visible;
  // the rest appear only while their state holds.
  const state = $("chip-state");
  setText("chip-state", s.stopped ? "stopped" : s.paused ? "paused" : "running");
  const chipClass = `chip ${s.stopped ? "stop" : s.paused ? "pause" : "run"}`;
  if (state.className !== chipClass) state.className = chipClass;
  $("chip-muted").hidden = !s.muted;
  $("chip-exploding").hidden = !s.exploding;

  // Enable MIDI shows only where the API exists AND the bundle
  // publishes the fields (a stale wasm keeps it hidden); its label
  // tracks the async permission state.
  const midiCapable =
    typeof navigator.requestMIDIAccess === "function" &&
    s.midi_ready !== undefined;
  $("btn-midi").hidden = !midiCapable;
  if (midiCapable) {
    setText("btn-midi", s.midi_ready
      ? (s.midi_enabled ? "MIDI on — pause" : "MIDI paused — send")
      : "Enable MIDI");
    if (s.midi_failed && !midiFailedShown) {
      midiFailedShown = true;
      fail("MIDI unavailable or permission denied");
    }
  }

  // Port dropdown: revealed once connected, on bundles that publish
  // the port list (`??` guards a stale wasm). Selection follows the
  // live connection unless the user has the menu focused.
  const midiPorts = s.midi_ports ?? [];
  const portSel = $("midi-port");
  portSel.hidden = !midiCapable || !s.midi_ready || midiPorts.length === 0;
  if (!portSel.hidden) {
    const key = JSON.stringify(midiPorts);
    if (key !== lastMidiPorts) {
      lastMidiPorts = key;
      portSel.replaceChildren(...midiPorts.map((name) => new Option(name)));
    }
    if (document.activeElement !== portSel && s.midi_port != null) {
      const current = midiPorts.indexOf(s.midi_port);
      if (current >= 0 && portSel.selectedIndex !== current) {
        portSel.selectedIndex = current;
      }
    }
  }

  // Cycle buttons and clear buttons show where they currently stand.
  setText("val-spawn", s.spawn_mode);
  setText("val-color", s.color_mode);
  setText("val-hud", s.hud ?? "…"); // pre-1.3.2 bundle: no field yet
  setText("btn-clear-wells", s.wells > 0 ? `Clear wells (${s.wells})` : "Clear wells");
  setText("btn-clear-walls", s.walls > 0 ? `Clear walls (${s.walls})` : "Clear walls");
  setText("btn-clear-emitters", s.emitters > 0 ? `Clear emitters (${s.emitters})` : "Clear emitters");

  const follow = (id, value, format) => {
    const el = $(`in-${id}`);
    if (document.activeElement !== el) el.value = value;
    setText(`out-${id}`, format);
  };
  follow("gravity", s.gravity, `${s.gravity}%`);
  follow("pelastic", s.particle_elasticity, s.particle_elasticity.toFixed(2));
  follow("welastic", s.wall_elasticity, s.wall_elasticity.toFixed(2));
  follow("time", s.time_scale, `${s.time_scale.toFixed(2)}x`);
  follow("threshold", s.explosion_threshold,
         s.explosion_threshold === 0 ? "off" : `${s.explosion_threshold}/s`);
  follow("pings", s.ping_volume,
         s.ping_volume === 0 ? "silent" : `${s.ping_volume}%`);
  // `??` guards: a stale wasm bundle without the fields reads as
  // quantize off on the default grid, never NaN.
  const bpm = s.bpm ?? 0;
  follow("bpm", bpm, bpm === 0 ? "off" : `${bpm.toFixed(0)} bpm`);
  setText("val-beat-div", `1/${s.beat_div ?? 4}`);

  $("tg-matter").checked = s.matter;
  $("tg-flow").checked = s.flow;
  $("tg-selfgrav").checked = s.self_gravity;
  $("tg-trails").checked = s.trails;
  $("tg-kaleido").checked = s.kaleidoscope;
  $("tg-music").checked = s.music;
  $("tg-chimes").checked = s.wall_chimes;
  setText("btn-sound", !s.audio_ready
    ? "Enable sound"
    : s.muted ? "Muted — unmute" : "Sound on — mute");
}

(async () => {
  try {
    const { mod, initPool } = await loadBouncy();
    // Stamp the header with the wasm module's own crate version
    // (optional-chained so a stale cached bundle without the export
    // still loads).
    $("version").textContent = mod.version ? `v${mod.version()}` : "";
    // Loader-only parameters are not CLI options; strip them before the
    // query reaches the config parser (st: force single-threaded;
    // cb: cache-buster).
    const params = new URLSearchParams(location.search);
    params.delete("st");
    params.delete("cb");
    // The preset chip comes from the launch URL: presets are
    // construction-time state, so the snapshot has no field for them.
    const preset = params.get("preset");
    if (preset) {
      $("chip-preset").textContent = `preset: ${preset}`;
      $("chip-preset").hidden = false;
    }
    const handle = new mod.WebHandle(params.toString());
    // Console access for tinkering: bouncyHandle.set_gravity(-500) etc.
    // (Not `bouncy`: the canvas id already claims that DOM global.)
    globalThis.bouncyHandle = handle;
    bind(handle);
    bindLaunchOptions(mod);
    // Poll at ~30 Hz, not every frame: handle.state() serializes the
    // full snapshot out of wasm into a fresh JS object, and doing that
    // at display rate is pure allocation churn — readouts updating
    // 30 times a second read as continuous, and the reclaimed frame
    // budget belongs to the simulation.
    const POLL_MS = 33;
    let lastPoll = 0;
    const poll = (t) => {
      if (t - lastPoll >= POLL_MS) {
        lastPoll = t;
        const s = handle.state();
        if (s) {
          // width > 0 distinguishes a published snapshot from the
          // default.
          if (!initialState && s.width > 0) initialState = s;
          reflect(s);
        }
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
    if (initPool) {
      $("threads-note").textContent = "Starting thread pool...";
      await initPool();
    }
  } catch (e) {
    fail(e);
    console.error(e);
  }
})();
