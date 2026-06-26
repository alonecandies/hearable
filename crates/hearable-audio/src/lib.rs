//! Audio capture and voice-activity segmentation.
//!
//! Phase 0 ships the [`WavAudioSource`] fixture replayer and the deterministic
//! [`EnergySegmenter`]. Phase 1 adds the real `cpal` capture path (with `rtrb` + `rubato`)
//! and the sherpa-onnx Silero VAD, both behind the same traits.

pub mod segmenter;
pub mod wav_source;

pub use segmenter::{EnergySegmenter, SegConfig};
pub use wav_source::WavAudioSource;
