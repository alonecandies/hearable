//! Offline Whisper ASR via the official `sherpa-onnx` crate (the `sherpa` feature).
//!
//! Whisper covers far more languages than SenseVoice — including Vietnamese — so this is the
//! engine to use when SenseVoice's zh/en/ja/ko/yue set is too narrow. Heavier than SenseVoice;
//! pick a model size that fits the target machine (tiny/base/small for CPU, medium/large for
//! quality).

use hearable_core::{AsrCaps, AsrEngine, Error, Result, TranscriptResult, Utterance};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig};

/// Filesystem paths to the Whisper model artifacts.
pub struct WhisperPaths {
    pub encoder: String,
    pub decoder: String,
    pub tokens: String,
}

/// Utterance-chunk Whisper ASR engine.
pub struct WhisperEngine {
    recognizer: OfflineRecognizer,
    sample_rate: i32,
    /// Forced language (e.g. "vi"); `None` lets Whisper auto-detect.
    language: Option<String>,
}

impl WhisperEngine {
    /// `language` = `Some("vi")` to force Vietnamese, or `None` to auto-detect.
    pub fn new(paths: &WhisperPaths, language: Option<String>, num_threads: i32) -> Result<Self> {
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(paths.encoder.clone()),
            decoder: Some(paths.decoder.clone()),
            language: language.clone(),
            task: Some("transcribe".into()),
            tail_paddings: -1,
            enable_token_timestamps: false,
            enable_segment_timestamps: false,
        };
        config.model_config.tokens = Some(paths.tokens.clone());
        config.model_config.provider = Some("cpu".into());
        config.model_config.num_threads = num_threads.max(1);

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| Error::Asr("failed to create Whisper recognizer".into()))?;
        Ok(Self {
            recognizer,
            sample_rate: 16_000,
            language,
        })
    }
}

impl AsrEngine for WhisperEngine {
    fn capabilities(&self) -> AsrCaps {
        AsrCaps {
            streaming: false,
            multilingual: true,
            auto_detect: self.language.is_none(),
        }
    }

    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(self.sample_rate, &utt.pcm16k);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| Error::Asr("Whisper produced no result".into()))?;
        Ok(TranscriptResult {
            text: result.text,
            lang: self.language.clone(),
            confidence: 1.0,
            is_final: true,
        })
    }
}
