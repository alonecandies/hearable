//! Speech-to-text engines behind the [`hearable_core::AsrEngine`] trait.
//!
//! Phase 0 ships [`MockAsrEngine`]. Phase 1+ add the sherpa-onnx `SenseVoiceEngine`
//! (multilingual, utterance-chunk), the streaming Zipformer engine, and the cloud adapter.

pub mod mock;

pub use mock::MockAsrEngine;

#[cfg(feature = "sherpa")]
pub mod lang_id;
#[cfg(feature = "sherpa")]
pub mod sense_voice;
#[cfg(feature = "sherpa")]
pub mod whisper;
#[cfg(feature = "sherpa")]
pub use lang_id::{LanguageId, LanguageIdPaths};
#[cfg(feature = "sherpa")]
pub use sense_voice::{SenseVoiceEngine, SenseVoicePaths};
#[cfg(feature = "sherpa")]
pub use whisper::{WhisperEngine, WhisperPaths};
