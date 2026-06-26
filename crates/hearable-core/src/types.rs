use serde::{Deserialize, Serialize};

/// Monotonic identifier assigned to each detected utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UtteranceId(pub u64);

/// Identifier for an anonymous speaker cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub u64);

/// A segment of speech, resampled to 16 kHz mono f32.
#[derive(Debug, Clone)]
pub struct Utterance {
    pub id: UtteranceId,
    pub pcm16k: Vec<f32>,
    pub t0_ms: u64,
    pub t1_ms: u64,
}

impl Utterance {
    /// Duration of the utterance in seconds, derived from sample count at 16 kHz.
    pub fn duration_secs(&self) -> f32 {
        self.pcm16k.len() as f32 / 16_000.0
    }
}

/// A speaker embedding (voice fingerprint).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Cosine similarity in `[-1, 1]`; returns `0.0` if either vector is zero-length
    /// or zero-magnitude (never `NaN`).
    pub fn cosine(&self, other: &Embedding) -> f32 {
        let (a, b) = (&self.0, &other.0);
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = na * nb;
        if denom > 0.0 {
            dot / denom
        } else {
            0.0
        }
    }
}

/// The result of transcribing one utterance (or an interim streaming hypothesis).
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptResult {
    pub text: String,
    pub lang: Option<String>,
    pub confidence: f32,
    pub is_final: bool,
}

/// Capabilities an [`crate::AsrEngine`] advertises, so the pipeline can adapt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrCaps {
    pub streaming: bool,
    pub multilingual: bool,
    pub auto_detect: bool,
}

/// Who a caption is attributed to.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerLabel {
    Known { name: String, score: f32 },
    Unknown { cluster_id: ClusterId, score: f32 },
}

/// A caption ready for display: text plus speaker attribution and timing.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionEvent {
    pub utt_id: UtteranceId,
    pub text: String,
    pub lang: Option<String>,
    pub speaker: SpeakerLabel,
    pub t0_ms: u64,
    pub t1_ms: u64,
    pub is_final: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let a = Embedding(vec![1.0, 2.0, 3.0]);
        assert!((a.cosine(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        let a = Embedding(vec![1.0, 0.0]);
        let b = Embedding(vec![0.0, 1.0]);
        assert!(a.cosine(&b).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_zero_vector_is_zero_not_nan() {
        let a = Embedding(vec![0.0, 0.0]);
        let b = Embedding(vec![1.0, 1.0]);
        assert_eq!(a.cosine(&b), 0.0);
    }
}
