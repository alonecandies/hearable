//! The caption overlay: a pure, testable view-model plus a channel-backed [`CaptionSink`],
//! with the egui rendering behind the `ui` feature.

pub mod model;
pub mod sink;

pub use model::{CaptionView, DisplayLine, SPEAKER_PALETTE};
pub use sink::ChannelCaptionSink;

#[cfg(feature = "ui")]
pub mod overlay;
#[cfg(feature = "ui")]
pub use overlay::{run_overlay, CaptionOverlay};
