use hearable_core::{AsrCaps, AsrEngine, Result, TranscriptResult, Utterance};
use std::collections::HashMap;

/// Deterministic ASR engine for tests and the Phase 0 mock pipeline: maps an utterance id
/// to a fixed transcript. Unmapped utterances transcribe to the empty string.
pub struct MockAsrEngine {
    map: HashMap<u64, String>,
}

impl MockAsrEngine {
    pub fn new<S: Into<String>>(map: HashMap<u64, S>) -> Self {
        Self {
            map: map.into_iter().map(|(k, v)| (k, v.into())).collect(),
        }
    }
}

impl AsrEngine for MockAsrEngine {
    fn capabilities(&self) -> AsrCaps {
        AsrCaps {
            streaming: false,
            multilingual: true,
            auto_detect: true,
        }
    }

    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult> {
        let text = self.map.get(&utt.id.0).cloned().unwrap_or_default();
        Ok(TranscriptResult {
            text,
            lang: Some("en".into()),
            confidence: 1.0,
            is_final: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{AsrEngine, Utterance, UtteranceId};

    fn utt(id: u64) -> Utterance {
        Utterance {
            id: UtteranceId(id),
            pcm16k: vec![],
            t0_ms: 0,
            t1_ms: 0,
        }
    }

    #[test]
    fn returns_mapped_text() {
        let mut e = MockAsrEngine::new(HashMap::from([(7u64, "hello".to_string())]));
        let r = e.transcribe(&utt(7)).unwrap();
        assert_eq!(r.text, "hello");
        assert!(r.is_final);
    }

    #[test]
    fn unmapped_is_empty() {
        let mut e = MockAsrEngine::new(HashMap::<u64, String>::new());
        assert_eq!(e.transcribe(&utt(1)).unwrap().text, "");
    }

    #[test]
    fn advertises_multilingual_auto_detect() {
        let e = MockAsrEngine::new(HashMap::<u64, String>::new());
        let caps = e.capabilities();
        assert!(caps.multilingual && caps.auto_detect && !caps.streaming);
    }
}
