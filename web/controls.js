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

// Load the single-threaded bundle by default. A multi-threaded bundle
// (pkg-mt/) is used when the page is cross-origin isolated and the
// bundle exists (?st forces single-threaded); see web/README.md.
// Returns { mod, initPool }: the thread pool initializes after the app
// is up, since rayon is only consulted above the population thresholds.
async function loadBouncy() {
  const forceSt = new URLSearchParams(location.search).has("st");
  if (globalThis.crossOriginIsolated && !forceSt) {
    try {
      const mt = await import("./pkg-mt/bouncy.js");
      await mt.default();
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
  const st = await import("./pkg/bouncy.js");
  await st.default();
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
  chip.textContent = `placing ${spec.label} — click the canvas (Esc cancels)`;
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
  $("btn-pause").onclick = () => handle.set_paused(!latest.paused);
  $("btn-step").onclick = () => handle.step_frame();
  $("btn-reset").onclick = () => handle.reset();

  const tools = {
    "btn-burst": { label: "burst", act: (x, y) => handle.spawn_burst(x, y) },
    "btn-comet": { label: "comet", act: (x, y) => handle.launch_comet(x, y) },
    "btn-well": { label: "well", act: (x, y) => handle.pin_well(x, y, false) },
    "btn-repel": {
      label: "repeller",
      act: (x, y) => handle.pin_well(x, y, true),
    },
    "btn-explode": {
      label: "explosion",
      act: (x, y) => handle.trigger_explosion(x, y),
    },
  };
  for (const [id, spec] of Object.entries(tools)) {
    $(id).onclick = () => armTool(id, spec);
  }
  $("tool-overlay").onclick = (e) => {
    const pos = armed && clickToSim(e);
    if (pos) armed.act(...pos);
    disarmTool();
  };
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

  $("btn-clear-wells").onclick = () => handle.clear_wells();
  $("btn-clear-walls").onclick = () => handle.clear_walls();
  $("btn-color").onclick = () => handle.cycle_color_mode();
  $("btn-spawn").onclick = () => handle.cycle_spawn_mode();
  $("btn-hud").onclick = () => handle.cycle_hud();
  $("btn-fullscreen").onclick = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else $("stage").requestFullscreen?.();
  };

  $("in-gravity").oninput = (e) => handle.set_gravity(Number(e.target.value));
  $("in-pelastic").oninput = (e) =>
    handle.set_particle_elasticity(Number(e.target.value));
  $("in-welastic").oninput = (e) =>
    handle.set_wall_elasticity(Number(e.target.value));
  $("in-time").oninput = (e) => handle.set_time_scale(Number(e.target.value));
  $("in-threshold").oninput = (e) =>
    handle.set_explosion_threshold(Number(e.target.value));

  $("tg-matter").onchange = () => handle.toggle_matter();
  $("tg-flow").onchange = () => handle.toggle_flow();
  $("tg-selfgrav").onchange = () => handle.toggle_self_gravity();
  $("tg-trails").onchange = () => handle.toggle_trails();
  $("tg-kaleido").onchange = () => handle.toggle_kaleidoscope();
  $("tg-music").onchange = (e) => handle.set_music(e.target.checked);

  // First click creates the WebAudio engine (must happen synchronously
  // inside the gesture, per autoplay policy) and unmutes; afterwards the
  // button is a plain mute toggle.
  $("btn-sound").onclick = () => {
    if (!latest.audio_ready) {
      if (!handle.enable_audio()) {
        fail("WebAudio unavailable in this browser");
      }
    } else {
      handle.set_muted(!latest.muted);
    }
  };

  $("btn-shot").onclick = () => {
    canvas.toBlob((blob) => {
      if (blob) download(blob, `bouncy-${Date.now()}.png`);
    });
  };
  $("btn-scene").onclick = () => {
    const toml = handle.scene_toml();
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
        handle.resize(w, h);
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
    const p = new URLSearchParams(location.search);
    p.delete("cb");
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

// Reflect the snapshot into the panel. Sliders follow the simulation
// unless the user is mid-drag (their element has focus).
function reflect(s) {
  latest = s;
  $("ro-fps").textContent = s.fps.toFixed(0);
  $("ro-particles").textContent = s.particles.toLocaleString();
  $("ro-cap").textContent = `of ${s.max_particles.toLocaleString()}`;
  $("ro-births").textContent = `${s.birth_rate}`;
  $("ro-size").textContent = `${s.width}×${s.height}`;
  $("btn-pause").textContent = s.paused ? "Resume" : "Pause";

  // Status chips: the running/paused/stopped chip is always visible;
  // the rest appear only while their state holds.
  const state = $("chip-state");
  state.textContent = s.stopped ? "stopped" : s.paused ? "paused" : "running";
  state.className = `chip ${s.stopped ? "stop" : s.paused ? "pause" : "run"}`;
  $("chip-muted").hidden = !s.muted;
  $("chip-exploding").hidden = !s.exploding;

  // Cycle buttons and clear buttons show where they currently stand.
  $("val-spawn").textContent = s.spawn_mode;
  $("val-color").textContent = s.color_mode;
  $("val-hud").textContent = s.hud ?? "…"; // pre-1.3.2 bundle: no field yet
  $("btn-clear-wells").textContent =
    s.wells > 0 ? `Clear wells (${s.wells})` : "Clear wells";
  $("btn-clear-walls").textContent =
    s.walls > 0 ? `Clear walls (${s.walls})` : "Clear walls";

  const follow = (id, value, format) => {
    const el = $(`in-${id}`);
    if (document.activeElement !== el) el.value = value;
    $(`out-${id}`).textContent = format;
  };
  follow("gravity", s.gravity, `${s.gravity}%`);
  follow("pelastic", s.particle_elasticity, s.particle_elasticity.toFixed(2));
  follow("welastic", s.wall_elasticity, s.wall_elasticity.toFixed(2));
  follow("time", s.time_scale, `${s.time_scale.toFixed(2)}x`);
  follow("threshold", s.explosion_threshold,
         s.explosion_threshold === 0 ? "off" : `${s.explosion_threshold}/s`);

  $("tg-matter").checked = s.matter;
  $("tg-flow").checked = s.flow;
  $("tg-selfgrav").checked = s.self_gravity;
  $("tg-trails").checked = s.trails;
  $("tg-kaleido").checked = s.kaleidoscope;
  $("tg-music").checked = s.music;
  $("btn-sound").textContent = !s.audio_ready
    ? "Enable sound"
    : s.muted ? "Muted — unmute" : "Sound on — mute";
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
    const poll = () => {
      const s = handle.state();
      if (s) {
        // width > 0 distinguishes a published snapshot from the default.
        if (!initialState && s.width > 0) initialState = s;
        reflect(s);
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
