// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Entry point: parse the command line and run the winit event loop.

mod app;
mod audio;
mod color;
mod config;
mod explosion;
mod physics;
mod presets;
mod render;
mod sim;
mod text;

use app::App;
use config::Config;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    let config = Config::resolve();

    if config.list_presets {
        presets::print_list(config.presets_file.as_deref());
        return;
    }

    config.print_summary();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    let _ = event_loop.run_app(&mut app);
}
