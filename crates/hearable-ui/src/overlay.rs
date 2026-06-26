//! egui/eframe translucent always-on-top caption overlay (the `ui` feature). macOS + X11;
//! the Wayland layer-shell path is Phase 3. Rendering only — display state lives in
//! [`crate::CaptionView`].

use crate::model::CaptionView;
use eframe::egui;
use hearable_core::{CaptionEvent, ClusterId, UiCommand};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};

/// The overlay application: drains caption events into the view-model and renders them,
/// offering an inline "name" field for each as-yet-unnamed speaker cluster.
pub struct CaptionOverlay {
    view: CaptionView,
    rx: Receiver<CaptionEvent>,
    commands: Sender<UiCommand>,
    /// In-progress name text, keyed by cluster id.
    name_edits: HashMap<u64, String>,
}

impl CaptionOverlay {
    pub fn new(rx: Receiver<CaptionEvent>, commands: Sender<UiCommand>, max_lines: usize) -> Self {
        Self {
            view: CaptionView::new(max_lines),
            rx,
            commands,
            name_edits: HashMap::new(),
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
        let lines = self.view.visible();

        let panel = egui::Frame::default()
            .fill(egui::Color32::from_black_alpha(209)) // ~0.82 opacity, WCAG-AAA with white text
            .inner_margin(egui::Margin::same(12));
        egui::CentralPanel::default().frame(panel).show(ctx, |ui| {
            for line in &lines {
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
                    if let Some(cid) = line.cluster_id {
                        let mut submit: Option<String> = None;
                        {
                            let entry = self.name_edits.entry(cid).or_default();
                            ui.add(
                                egui::TextEdit::singleline(entry)
                                    .hint_text("name")
                                    .desired_width(90.0),
                            );
                            if ui.button("Name").clicked() && !entry.trim().is_empty() {
                                submit = Some(entry.trim().to_string());
                            }
                        }
                        if let Some(name) = submit {
                            let _ = self.commands.send(UiCommand::NameSpeaker {
                                cluster_id: ClusterId(cid),
                                name,
                            });
                            self.name_edits.remove(&cid);
                        }
                    }
                });
            }
        });

        ctx.request_repaint(); // keep polling the channel for new captions
    }
}

/// Launch the overlay window. Consumes caption events from `rx` and forwards naming actions
/// on `commands`. Blocks until the window is closed.
pub fn run_overlay(
    rx: Receiver<CaptionEvent>,
    commands: Sender<UiCommand>,
    max_lines: usize,
) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("hearable")
            .with_inner_size([720.0, 200.0])
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top(),
        // NOTE: mouse passthrough is intentionally OFF so the inline "Name" controls are
        // clickable. A hotkey to toggle passthrough (captions-only vs interactive) is a
        // planned refinement.
        ..Default::default()
    };
    eframe::run_native(
        "hearable",
        options,
        Box::new(move |_cc| Ok(Box::new(CaptionOverlay::new(rx, commands, max_lines)))),
    )
}
