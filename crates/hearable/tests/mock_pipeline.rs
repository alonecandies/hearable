//! End-to-end test of the trait seams: WAV source -> segmenter -> ASR + embedding ->
//! identifier -> caption sink, all with deterministic mocks (no models, no mic).

use hearable::pipeline::run_pipeline;
use hearable_asr::MockAsrEngine;
use hearable_audio::{EnergySegmenter, SegConfig, WavAudioSource};
use hearable_core::testutil::BufferCaptionSink;
use hearable_core::{Embedding, EmbeddingExtractor, Result, SpeakerLabel, Utterance};
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

/// Two voiced spans separated by silence (mirrors the segmenter's own fixture), as 16 kHz
/// mono i16 WAV.
fn write_two_speaker_wav(path: &Path) {
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
            let s = (amp * (n as f32 * 0.1).sin() * 30_000.0) as i16;
            w.write_sample(s).unwrap();
        }
    };
    push(0.3, 1.0);
    push(0.3, 0.0);
    push(0.3, 1.0);
    w.finalize().unwrap();
}

#[test]
fn mock_pipeline_produces_two_captions_with_distinct_speakers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.wav");
    write_two_speaker_wav(&path);

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
    let mut sink = BufferCaptionSink::new();

    run_pipeline(source, seg, asr, embed, id, &mut sink).unwrap();

    let evs = sink.events();
    assert_eq!(evs.len(), 2, "expected 2 caption events, got {}", evs.len());
    assert_eq!(evs[0].text, "hello");
    assert_eq!(evs[1].text, "world");

    let c0 = match &evs[0].speaker {
        SpeakerLabel::Unknown { cluster_id, .. } => cluster_id.0,
        other => panic!("expected Unknown, got {other:?}"),
    };
    let c1 = match &evs[1].speaker {
        SpeakerLabel::Unknown { cluster_id, .. } => cluster_id.0,
        other => panic!("expected Unknown, got {other:?}"),
    };
    assert_ne!(
        c0, c1,
        "distinct embeddings must yield distinct speaker clusters"
    );
}
