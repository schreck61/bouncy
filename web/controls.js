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

function bind(handle) {
  const canvas = $("bouncy");
  const center = () => {
    const s = handle.state();
    return [s.width / 2, s.height / 2];
  };

  $("btn-pause").onclick = () => handle.set_paused(!latest.paused);
  $("btn-step").onclick = () => handle.step_frame();
  $("btn-reset").onclick = () => handle.reset();
  $("btn-explode").onclick = () => handle.trigger_explosion(...center());
  $("btn-burst").onclick = () => handle.spawn_burst(...center());
  $("btn-comet").onclick = () => handle.launch_comet(...center());
  $("btn-well").onclick = () => handle.pin_well(...center(), false);
  $("btn-repel").onclick = () => handle.pin_well(...center(), true);
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
  $("ro-particles").textContent = `${s.particles}`;
  $("ro-particles").title = `cap ${s.max_particles}`;
  $("ro-births").textContent = `${s.birth_rate}`;
  $("ro-size").textContent = `${s.width}×${s.height}`;
  $("btn-pause").textContent = s.paused ? "Resume" : "Pause";

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
    : s.muted ? "Unmute" : "Mute";
}

(async () => {
  try {
    const { mod, initPool } = await loadBouncy();
    // Loader-only parameters are not CLI options; strip them before the
    // query reaches the config parser (st: force single-threaded;
    // cb: cache-buster).
    const params = new URLSearchParams(location.search);
    params.delete("st");
    params.delete("cb");
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
