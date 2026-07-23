// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Bouncy: a particle-physics toy. The library crate exists so the same
//! simulation core can back two shells: the native binary (`main.rs`) and
//! the WebAssembly demo (`web.rs`). Everything interesting lives in the
//! modules; see the README for the architecture tour.

pub mod app;
pub mod audio;
pub mod color;
pub mod config;
pub mod explosion;
pub mod physics;
pub mod presets;
pub mod render;
pub mod sim;
pub mod text;

#[cfg(not(target_arch = "wasm32"))]
pub mod capture;
#[cfg(not(target_arch = "wasm32"))]
pub mod gui;
// Cross-target since 1.14: the pure message half (Scheduler,
// chime_message) compiles everywhere; the midir port shell stays
// native and the WebMIDI shell wasm, cfg-gated inside the module.
pub mod midi;
// Cross-target like midi: pure data structures shared by both shells'
// frame loops; the clocks live in the shells.
pub mod perf;
#[cfg(target_arch = "wasm32")]
pub mod web;

/// Run the native application: resolve configuration, then hand control
/// to the winit event loop. Kept here (not in `main.rs`) so the binary
/// stays a two-line shim over the library.
#[cfg(not(target_arch = "wasm32"))]
pub fn run() {
    use winit::event_loop::{ControlFlow, EventLoop};

    let config = config::Config::resolve();

    if config.list_presets {
        presets::print_list(config.presets_file.as_deref());
        return;
    }

    if config.list_midi_ports {
        let ports = midi::MidiOut::ports();
        if ports.is_empty() {
            println!("No MIDI output ports found");
        } else {
            for (i, name) in ports.iter().enumerate() {
                println!("  [{i}] {name}");
            }
        }
        return;
    }

    config.print_summary();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new(config);
    let _ = event_loop.run_app(&mut app);
}
