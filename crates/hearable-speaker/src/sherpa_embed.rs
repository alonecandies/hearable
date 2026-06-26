//! Speaker-embedding extraction via the official `sherpa-onnx` crate (the `sherpa` feature),
//! e.g. the ERes2NetV2 model. Implements [`EmbeddingExtractor`]; the embedding it produces
//! feeds the same [`crate::LeaderClusterIdentifier`] used in Phase 0.

use hearable_core::{Embedding, EmbeddingExtractor, Error, Result, Utterance};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

const SAMPLE_RATE: i32 = 16_000;

pub struct SherpaEmbeddingExtractor {
    extractor: SpeakerEmbeddingExtractor,
}

impl SherpaEmbeddingExtractor {
    pub fn new(model: &str, num_threads: i32) -> Result<Self> {
        let cfg = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string()),
            num_threads: num_threads.max(1),
            debug: false,
            provider: Some("cpu".into()),
        };
        let extractor = SpeakerEmbeddingExtractor::create(&cfg)
            .ok_or_else(|| Error::Speaker("failed to create speaker embedding extractor".into()))?;
        Ok(Self { extractor })
    }

    /// The embedding dimension reported by the loaded model (read it; don't assume 192/256).
    pub fn dim(&self) -> i32 {
        self.extractor.dim()
    }
}

impl EmbeddingExtractor for SherpaEmbeddingExtractor {
    fn embed(&mut self, utt: &Utterance) -> Result<Embedding> {
        let stream = self
            .extractor
            .create_stream()
            .ok_or_else(|| Error::Speaker("failed to create embedding stream".into()))?;
        stream.accept_waveform(SAMPLE_RATE, &utt.pcm16k);
        stream.input_finished();
        let v = self
            .extractor
            .compute(&stream)
            .ok_or_else(|| Error::Speaker("embedding computation failed".into()))?;
        Ok(Embedding(v))
    }
}
