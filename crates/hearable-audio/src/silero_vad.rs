//! Silero VAD utterance segmentation via the official `sherpa-onnx` crate (the `sherpa`
//! feature). Implements [`VadSegmenter`] exactly like the Phase-0 [`crate::EnergySegmenter`],
//! so the pipeline swaps one for the other with no other changes.

use hearable_core::{Error, Result, Utterance, UtteranceId, VadSegmenter};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

/// Silero v5 expects fixed 512-sample windows at 16 kHz.
const WINDOW: usize = 512;
const SAMPLE_RATE: i32 = 16_000;

/// Configuration for the Silero VAD (defaults tuned for an accessibility false-positive budget).
pub struct SileroVadConfig {
    pub model: String,
    pub threshold: f32,
    pub min_silence_secs: f32,
    pub min_speech_secs: f32,
    pub max_speech_secs: f32,
}

impl SileroVadConfig {
    pub fn with_model(model: impl Into<String>) -> Self {
        // Sensitivity tuned from a first real run ("sometimes doesn't hear me"): a lower
        // threshold and shorter min-speech catch quieter / briefer utterances.
        Self {
            model: model.into(),
            threshold: 0.35,
            min_silence_secs: 0.25,
            min_speech_secs: 0.12,
            max_speech_secs: 12.0,
        }
    }
}

pub struct SileroVad {
    vad: VoiceActivityDetector,
    buf: Vec<f32>,
    next_id: u64,
}

impl SileroVad {
    pub fn new(cfg: &SileroVadConfig) -> Result<Self> {
        let silero = SileroVadModelConfig {
            model: Some(cfg.model.clone()),
            threshold: cfg.threshold,
            min_silence_duration: cfg.min_silence_secs,
            min_speech_duration: cfg.min_speech_secs,
            max_speech_duration: cfg.max_speech_secs,
            ..Default::default()
        };
        let config = VadModelConfig {
            silero_vad: silero,
            sample_rate: SAMPLE_RATE,
            num_threads: 1,
            provider: Some("cpu".into()),
            ..Default::default()
        };
        let vad = VoiceActivityDetector::create(&config, 30.0)
            .ok_or_else(|| Error::Audio("failed to create Silero VAD".into()))?;
        Ok(Self {
            vad,
            buf: Vec::new(),
            next_id: 0,
        })
    }

    fn drain_segments(&mut self) -> Vec<Utterance> {
        let mut out = Vec::new();
        while let Some(seg) = self.vad.front() {
            let start = seg.start().max(0) as u64;
            let n = seg.n().max(0) as u64;
            let id = UtteranceId(self.next_id);
            self.next_id += 1;
            out.push(Utterance {
                id,
                pcm16k: seg.samples().to_vec(),
                t0_ms: start * 1000 / SAMPLE_RATE as u64,
                t1_ms: (start + n) * 1000 / SAMPLE_RATE as u64,
            });
            self.vad.pop();
        }
        out
    }
}

impl VadSegmenter for SileroVad {
    fn push(&mut self, frame: &[f32]) -> Vec<Utterance> {
        self.buf.extend_from_slice(frame);
        let mut out = Vec::new();
        while self.buf.len() >= WINDOW {
            let window: Vec<f32> = self.buf.drain(..WINDOW).collect();
            self.vad.accept_waveform(&window);
            out.extend(self.drain_segments());
        }
        out
    }

    fn flush(&mut self) -> Vec<Utterance> {
        if !self.buf.is_empty() {
            let mut window = std::mem::take(&mut self.buf);
            window.resize(WINDOW, 0.0);
            self.vad.accept_waveform(&window);
        }
        self.vad.flush();
        self.drain_segments()
    }
}
