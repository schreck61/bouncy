// Copyright (c) 2026 James O. Schreckengast
// Licensed under the MIT License. See LICENSE for details.

//! Per-frame performance instrumentation behind `--perf`: the shell
//! times each frame phase (simulate, raster, present, publish) and the
//! physics layer counts rayon fork-join dispatches; a rolling window
//! turns the samples into the mean/p95/max lines the HUD shows.
//!
//! Everything here is pure data — no clocks, no I/O — so it unit-tests
//! headless. The shell reads `web_time::Instant` (`performance.now()`
//! on the web: ~5 µs resolution on a cross-origin-isolated page, ~20 µs
//! on Firefox — fine for millisecond-scale phases) and pushes one
//! [`FrameSample`] per frame. The window accumulates Rust-side every
//! frame, so a 30 Hz page poll — or a backgrounded tab that only runs
//! frames in bursts — still sees every spike in the p95/max columns.

/// Frames per rolling window: two seconds at 60 fps — long enough for
/// p95 to mean something, short enough to react while A/B-ing knobs.
const WINDOW: usize = 120;

/// Rolling per-phase sample window (last `WINDOW` = 120 frames).
pub struct PhaseWindow {
    /// Millisecond samples, a ring: `idx` is the next write slot.
    samples: [f32; WINDOW],
    idx: usize,
    len: usize,
}

impl Default for PhaseWindow {
    fn default() -> Self {
        PhaseWindow {
            samples: [0.0; WINDOW],
            idx: 0,
            len: 0,
        }
    }
}

impl PhaseWindow {
    /// Record one frame's value (milliseconds, or any unit — the
    /// dispatch window stores counts).
    pub fn push(&mut self, value: f64) {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.samples[self.idx] = value as f32;
        }
        self.idx = (self.idx + 1) % WINDOW;
        self.len = (self.len + 1).min(WINDOW);
    }

    /// Mean over the live window (0 while empty).
    pub fn mean(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let sum: f64 = self.samples[..self.len].iter().map(|&s| f64::from(s)).sum();
        // len is at most WINDOW (120) — exact in f64.
        #[allow(clippy::cast_precision_loss)]
        {
            sum / self.len as f64
        }
    }

    /// 95th percentile over the live window (0 while empty). Sorts a
    /// copy — at most `WINDOW` elements, and only when the HUD is
    /// actually being drawn.
    pub fn p95(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let mut live = self.samples[..self.len].to_vec();
        live.sort_by(f32::total_cmp);
        f64::from(live[(self.len - 1) * 95 / 100])
    }

    /// Maximum over the live window (0 while empty).
    pub fn max(&self) -> f64 {
        self.samples[..self.len]
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .into()
    }
}

/// One frame's phase timings plus the rayon dispatch count, recorded
/// by the shell at the end of `update_and_render`.
#[derive(Default)]
pub struct FrameSample {
    /// Simulation step (physics, collisions, chimes).
    pub simulate_ms: f64,
    /// CPU rasterization into the frame buffer (`with_frame`).
    pub raster_ms: f64,
    /// Buffer-to-screen: GPU present natively, the Canvas2D/WebGL blit
    /// on the web.
    pub present_ms: f64,
    /// Web snapshot publish (0 on native).
    pub publish_ms: f64,
    /// Rayon fork-join dispatches this frame
    /// (`physics::take_par_dispatches`).
    pub dispatches: u32,
}

/// The `--perf` recorder the App owns: one window per phase, formatted
/// into HUD lines on demand.
#[derive(Default)]
pub struct PerfRecorder {
    simulate: PhaseWindow,
    raster: PhaseWindow,
    present: PhaseWindow,
    publish: PhaseWindow,
    dispatches: PhaseWindow,
    last_dispatches: u32,
}

impl PerfRecorder {
    /// Fold one frame's sample into the windows.
    pub fn record(&mut self, s: &FrameSample) {
        self.simulate.push(s.simulate_ms);
        self.raster.push(s.raster_ms);
        self.present.push(s.present_ms);
        self.publish.push(s.publish_ms);
        self.dispatches.push(f64::from(s.dispatches));
        self.last_dispatches = s.dispatches;
    }

    /// The HUD block: a header plus one line per phase and the
    /// dispatch line, in fixed order (tests pin the prefixes).
    pub fn lines(&self) -> Vec<String> {
        let phase = |name: &str, w: &PhaseWindow| {
            format!(
                "{name:<8}{:>6.2} /{:>6.2} /{:>6.2}",
                w.mean(),
                w.p95(),
                w.max()
            )
        };
        vec![
            "perf (120f)  mean / p95 / max ms".to_string(),
            phase("sim", &self.simulate),
            phase("raster", &self.raster),
            phase("present", &self.present),
            phase("publish", &self.publish),
            format!(
                "rayon    {:>6.1} /{:>6.0} /{:>6.0}   (last: {})",
                self.dispatches.mean(),
                self.dispatches.p95(),
                self.dispatches.max(),
                self.last_dispatches
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_stats_on_known_samples() {
        let mut w = PhaseWindow::default();
        assert_eq!(w.mean(), 0.0, "empty window is all zeros");
        assert_eq!(w.p95(), 0.0);
        assert_eq!(w.max(), 0.0);
        for v in [1.0, 2.0, 3.0, 4.0] {
            w.push(v);
        }
        assert!((w.mean() - 2.5).abs() < 1e-6);
        assert!((w.max() - 4.0).abs() < 1e-6);
        // p95 of 4 samples: index (4-1)*95/100 = 2 → the third sorted.
        assert!((w.p95() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn window_wraps_past_capacity() {
        let mut w = PhaseWindow::default();
        // 120 zeros, then 120 ones: the window must forget the zeros.
        for _ in 0..WINDOW {
            w.push(0.0);
        }
        for _ in 0..WINDOW {
            w.push(1.0);
        }
        assert!((w.mean() - 1.0).abs() < 1e-6, "old samples evicted");
        assert!((w.p95() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn p95_excludes_a_single_spike_max_keeps_it() {
        let mut w = PhaseWindow::default();
        for _ in 0..(WINDOW - 1) {
            w.push(0.5);
        }
        w.push(50.0);
        assert!((w.p95() - 0.5).abs() < 1e-6, "one spike in 120 is past p95");
        assert!((w.max() - 50.0).abs() < 1e-6, "max pins the spike");
        assert!(w.mean() > 0.5 && w.mean() < 1.0);
    }

    #[test]
    fn recorder_lines_have_fixed_shape() {
        let mut r = PerfRecorder::default();
        r.record(&FrameSample {
            simulate_ms: 3.0,
            raster_ms: 1.0,
            present_ms: 0.5,
            publish_ms: 0.1,
            dispatches: 4,
        });
        let lines = r.lines();
        assert_eq!(lines.len(), 6, "header + four phases + rayon");
        assert!(lines[0].starts_with("perf"));
        assert!(lines[1].starts_with("sim"));
        assert!(lines[2].starts_with("raster"));
        assert!(lines[3].starts_with("present"));
        assert!(lines[4].starts_with("publish"));
        assert!(lines[5].starts_with("rayon"));
        assert!(lines[5].ends_with("(last: 4)"));
    }
}
