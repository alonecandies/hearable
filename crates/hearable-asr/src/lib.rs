//! Speech-to-text engines behind the [`hearable_core::AsrEngine`] trait.
//!
//! Phase 0 ships [`MockAsrEngine`]. Phase 1+ add the sherpa-onnx `SenseVoiceEngine`
//! (multilingual, utterance-chunk), the streaming Zipformer engine, and the cloud adapter.

pub mod mock;

pub use mock::MockAsrEngine;

#[cfg(feature = "sherpa")]
pub mod sense_voice;
#[cfg(feature = "sherpa")]
pub use sense_voice::{SenseVoiceEngine, SenseVoicePaths};
