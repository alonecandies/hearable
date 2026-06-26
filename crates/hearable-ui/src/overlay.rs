//! egui/eframe translucent always-on-top caption overlay (the `ui` feature). macOS + X11;
//! the Wayland layer-shell path is Phase 3. Rendering only — display state lives in
//! [`crate::CaptionView`].

use crate::model::CaptionView;
use eframe::egui;
use hearable_core::CaptionEvent;
use std::sync::mpsc::Receiver;

/// The overlay application: drains caption events into the view-model and renders them.
pub struct CaptionOverlay {
    view: CaptionView,
    rx: Receiver<CaptionEvent>,
}

impl CaptionOverlay {
    pub fn new(rx: Receiver<CaptionEvent>, max_lines: usize) -> Self {
        Self {
            view: CaptionView::new(max_lines),
            rx,
        }
    }
}

impl eframe::App for CaptionOverlay {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0] // fully transparent window background
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(ev) = self.rx.try_recv() {
            self.view.push(&ev);
        }

        let panel = egui::Frame::default()
            .fill(egui::Color32::from_black_alpha(209)) // ~0.82 opacity, WCAG-AAA on white text
            .inner_margin(egui::Margin::same(12));
        egui::CentralPanel::default().frame(panel).show(ctx, |ui| {
            for line in self.view.visible() {
                let accent = egui::Color32::from_rgb(line.color[0], line.color[1], line.color[2]);
                let text_color = if line.is_partial {
                    egui::Color32::from_white_alpha(166)
                } else {
                    egui::Color32::WHITE
                };
                ui.horizontal(|ui| {
                    // Colored speaker tag doubles as the non-color (textual) differentiator.
                    ui.label(egui::RichText::new(&line.speaker).color(accent).strong());
                    ui.label(egui::RichText::new(&line.text).color(text_color).size(22.0));
                });
            }
        });

        ctx.request_repaint(); // keep polling the channel for new captions
    }
}

/// Launch the overlay window, consuming caption events from `rx`. Blocks until it closes.
pub fn run_overlay(rx: Receiver<CaptionEvent>, max_lines: usize) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("hearable")
            .with_inner_size([720.0, 200.0])
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top()
            .with_mouse_passthrough(true),
        ..Default::default()
    };
    eframe::run_native(
        "hearable",
        options,
        Box::new(move |_cc| Ok(Box::new(CaptionOverlay::new(rx, max_lines)))),
    )
}
