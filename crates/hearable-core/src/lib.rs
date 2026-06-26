//! Shared types, traits, configuration, and error handling for `hearable`.
//!
//! This crate is the dependency leaf of the workspace: the domain crates
//! (`hearable-audio`, `hearable-asr`, `hearable-speaker`, `hearable-store`) depend on it
//! and never on each other. Pipeline orchestration lives in the `hearable` binary.

pub mod config;
pub mod error;
pub mod testutil;
pub mod traits;
pub mod types;

pub use config::{LanguageMode, Profile, Retention, Settings};
pub use error::{Error, Result};
pub use traits::{
    AsrEngine, AudioSource, CaptionSink, EmbeddingExtractor, Identifier, ProfileStore, VadSegmenter,
};
pub use types::{
    AsrCaps, CaptionEvent, ClusterId, Embedding, SpeakerLabel, TranscriptResult, Utterance,
    UtteranceId,
};
