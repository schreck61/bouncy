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

// Load the single-threaded bundle by default. A multi-threaded bundle
// (pkg-mt/) is used when the page is cross-origin isolated and the
// bundle exists; see web/README.md.
async function loadBouncy() {
  if (globalThis.crossOriginIsolated) {
    try {
      const mt = await import("./pkg-mt/bouncy.js");
      await mt.default();
      await mt.initThreadPool(navigator.hardwareConcurrency ?? 4);
      $("threads-note").textContent =
        `Multi-threaded build: ${navigator.hardwareConcurrency ?? "?"} threads.`;
      return mt;
    } catch {
      // No MT bundle deployed (or it failed to init): fall through.
    }
  }
  const st = await import("./pkg/bouncy.js");
  await st.default();
  $("threads-note").textContent =
    "Single-threaded build (multi-threading needs cross-origin isolation).";
  return st;
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
    const s = latest;
    const params = new URLSearchParams();
    if (s.gravity !== 100) params.set("gravity", s.gravity);
    if (s.particle_elasticity !== 1) {
      params.set("particle-elasticity", s.particle_elasticity.toFixed(2));
    }
    if (s.wall_elasticity !== 1) {
      params.set("wall-elasticity", s.wall_elasticity.toFixed(2));
    }
    if (s.explosion_threshold !== 30) {
      params.set("explosion-threshold", s.explosion_threshold);
    }
    for (const [key, on] of [["matter", s.matter], ["flow", s.flow],
                             ["self-gravity", s.self_gravity],
                             ["trails", s.trails],
                             ["kaleidoscope", s.kaleidoscope]]) {
      if (on) params.set(key, "");
    }
    const url = `${location.origin}${location.pathname}?${params}`
      .replace(/=(?=&|$)/g, "");
    await navigator.clipboard.writeText(url);
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

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

let latest = { paused: false };

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
}

(async () => {
  try {
    const mod = await loadBouncy();
    const handle = new mod.WebHandle(location.search);
    bind(handle);
    const poll = () => {
      const s = handle.state();
      if (s) reflect(s);
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  } catch (e) {
    fail(e);
    console.error(e);
  }
})();
