use crate::{
    AsrCaps, CaptionEvent, ClusterId, Embedding, Profile, Result, SpeakerLabel, TranscriptResult,
    Utterance,
};

/// A source of 16 kHz mono f32 frames (the real microphone in Phase 1, a WAV fixture in tests).
///
/// `start` drives frames into `on_frame` until the source is exhausted or [`AudioSource::stop`]
/// is called.
pub trait AudioSource {
    fn start(&mut self, on_frame: &mut dyn FnMut(&[f32])) -> Result<()>;
    fn stop(&mut self);
}

/// Splits a frame stream into discrete [`Utterance`]s.
pub trait VadSegmenter {
    /// Feed one frame; returns any utterances completed by this frame.
    fn push(&mut self, frame: &[f32]) -> Vec<Utterance>;
    /// End of stream: emit any in-progress utterance.
    fn flush(&mut self) -> Vec<Utterance>;
}

/// Speech-to-text engine. Implementations: on-device offline/streaming, cloud, and mock.
pub trait AsrEngine: Send {
    fn capabilities(&self) -> AsrCaps;
    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult>;
}

/// Extracts a speaker embedding from an utterance.
pub trait EmbeddingExtractor: Send {
    fn embed(&mut self, utt: &Utterance) -> Result<Embedding>;
}

/// Assigns each embedding to a known person or an anonymous cluster, and supports
/// promoting a cluster to a named profile.
pub trait Identifier {
    fn identify(&mut self, e: &Embedding) -> SpeakerLabel;
    /// Identify with awareness of the utterance length, so short (less reliable) utterances
    /// can use a relaxed matching threshold. The default ignores duration.
    fn identify_with_duration(&mut self, e: &Embedding, _duration_secs: f32) -> SpeakerLabel {
        self.identify(e)
    }
    fn promote(&mut self, cluster: ClusterId, name: &str) -> Result<()>;
}

/// Persists named speaker profiles (embeddings only — never raw audio).
pub trait ProfileStore {
    fn load_profiles(&self) -> Result<Vec<Profile>>;
    fn upsert_profile(&self, name: &str, embeddings: &[Embedding]) -> Result<()>;
}

/// Receives caption events for display (the overlay in Phase 1, a buffer in tests).
pub trait CaptionSink: Send {
    fn emit(&mut self, ev: CaptionEvent);
}
