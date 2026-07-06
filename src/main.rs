// Copyright (c) 2026 James O. Schreckengast
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

mod app;
mod audio;
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

    // Print configuration summary
    if let Some(ref preset) = config.preset {
        println!("Preset: {preset}");
    }
    println!("Spawn mode: {}", config.effective_spawn_mode().label());
    if config.matter {
        println!("Matter mechanics: fusion/fission enabled");
    }
    if config.flow {
        println!("Flow field: enabled");
    }
    if config.wells > 0 {
        println!("Pinned wells: {}", config.wells);
    }
    if config.music {
        println!("Musical pings: pentatonic scale");
    }
    if config.kaleidoscope {
        println!("Kaleidoscope: enabled");
    }
    if config.gravity != 100 {
        println!("Gravity: {}%", config.gravity);
    }
    if (config.wall_elasticity - 1.0).abs() > f64::EPSILON {
        println!("Wall elasticity: {}", config.wall_elasticity);
    }
    if (config.particle_elasticity - 1.0).abs() > f64::EPSILON {
        println!("Particle elasticity: {}", config.particle_elasticity);
    }
    if let Some(seed) = config.seed {
        println!("Random seed: {seed}");
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    let _ = event_loop.run_app(&mut app);
}
