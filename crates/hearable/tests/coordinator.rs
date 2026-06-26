//! Threaded coordinator integration tests: end-to-end across real worker threads using the
//! deterministic mocks (no models, no mic).

use hearable::coordinator::run_threaded;
use hearable_asr::MockAsrEngine;
use hearable_audio::{EnergySegmenter, SegConfig, WavAudioSource};
use hearable_core::testutil::SharedCaptionSink;
use hearable_core::{
    AsrCaps, AsrEngine, Embedding, EmbeddingExtractor, Result, SpeakerLabel, TranscriptResult,
    Utterance,
};
use hearable_speaker::{ClusterConfig, LeaderClusterIdentifier};
use std::collections::HashMap;
use std::path::Path;

struct MockEmbed {
    map: HashMap<u64, Embedding>,
}
impl EmbeddingExtractor for MockEmbed {
    fn embed(&mut self, utt: &Utterance) -> Result<Embedding> {
        Ok(self
            .map
            .get(&utt.id.0)
            .cloned()
            .unwrap_or(Embedding(vec![0.0, 0.0, 1.0])))
    }
}

/// An ASR engine that sleeps before returning, to force the inference thread behind the
/// capture thread and exercise the drop-oldest path.
struct SlowAsr {
    delay_ms: u64,
}
impl AsrEngine for SlowAsr {
    fn capabilities(&self) -> AsrCaps {
        AsrCaps {
            streaming: false,
            multilingual: true,
            auto_detect: true,
        }
    }
    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult> {
        std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
        Ok(TranscriptResult {
            text: format!("utt-{}", utt.id.0),
            lang: Some("en".into()),
            confidence: 1.0,
            is_final: true,
        })
    }
}

/// Write `spans` voiced segments (each followed by silence) as a 16 kHz mono i16 WAV.
fn write_multi_span_wav(path: &Path, spans: usize) {
    let sr = 16_000usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    let mut push = |secs: f32, amp: f32| {
        for n in 0..(secs * sr as f32) as usize {
            w.write_sample((amp * (n as f32 * 0.1).sin() * 30_000.0) as i16)
                .unwrap();
        }
    };
    for _ in 0..spans {
        push(0.2, 1.0);
        push(0.2, 0.0);
    }
    w.finalize().unwrap();
}

#[test]
fn threaded_pipeline_happy_path_two_speakers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.wav");
    write_multi_span_wav(&path, 2);

    let source = WavAudioSource::open(&path).unwrap();
    let seg = EnergySegmenter::new(SegConfig::default());
    let asr = MockAsrEngine::new(HashMap::from([(0u64, "hello"), (1u64, "world")]));
    let embed = MockEmbed {
        map: HashMap::from([
            (0u64, Embedding(vec![1.0, 0.0, 0.0])),
            (1u64, Embedding(vec![0.0, 1.0, 0.0])),
        ]),
    };
    let id = LeaderClusterIdentifier::new(ClusterConfig::default(), vec![]);
    let sink = SharedCaptionSink::new();

    let outcome = run_threaded(source, seg, asr, embed, id, sink.clone(), 8);

    let evs = sink.snapshot();
    assert_eq!(evs.len(), 2, "expected 2 caption events");
    assert_eq!(evs[0].text, "hello");
    assert_eq!(evs[1].text, "world");
    assert_eq!(outcome.dropped, 0, "no drops with ample capacity");

    let c0 = match &evs[0].speaker {
        SpeakerLabel::Unknown { cluster_id, .. } => cluster_id.0,
        other => panic!("expected Unknown, got {other:?}"),
    };
    let c1 = match &evs[1].speaker {
        SpeakerLabel::Unknown { cluster_id, .. } => cluster_id.0,
        other => panic!("expected Unknown, got {other:?}"),
    };
    assert_ne!(c0, c1, "distinct embeddings -> distinct clusters");
}

#[test]
fn threaded_pipeline_conserves_utterances_under_overload() {
    // 20 spans -> 20 utterances; a slow ASR + small capacity forces drop-oldest.
    // Invariant: every produced utterance is either emitted or dropped (never lost).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.wav");
    let spans = 20;
    write_multi_span_wav(&path, spans);

    let source = WavAudioSource::open(&path).unwrap();
    let seg = EnergySegmenter::new(SegConfig::default());
    let asr = SlowAsr { delay_ms: 2 };
    let embed = MockEmbed {
        map: HashMap::new(),
    };
    let id = LeaderClusterIdentifier::new(ClusterConfig::default(), vec![]);
    let sink = SharedCaptionSink::new();

    let outcome = run_threaded(source, seg, asr, embed, id, sink.clone(), 4);

    let emitted = sink.snapshot().len() as u64;
    assert_eq!(
        emitted + outcome.dropped,
        spans as u64,
        "every utterance is either emitted ({emitted}) or dropped ({})",
        outcome.dropped
    );
    assert!(emitted >= 1, "at least one utterance should be processed");
}
