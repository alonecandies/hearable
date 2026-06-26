//! On-device multilingual ASR via the official `sherpa-onnx` SenseVoice model.
//!
//! Enabled by the `sherpa` feature. SenseVoice's result carries no language field, so the
//! detected language is filled by a separate language-ID pass (see the spec's LID routing);
//! this engine reports `lang: None` and leaves routing to the pipeline.

use hearable_core::{AsrCaps, AsrEngine, Error, Result, TranscriptResult, Utterance};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig};

/// Filesystem paths to the SenseVoice model artifacts.
pub struct SenseVoicePaths {
    pub model: String,
    pub tokens: String,
}

/// Utterance-chunk multilingual ASR engine.
pub struct SenseVoiceEngine {
    recognizer: OfflineRecognizer,
    sample_rate: i32,
}

impl SenseVoiceEngine {
    pub fn new(paths: &SenseVoicePaths, num_threads: i32) -> Result<Self> {
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
            model: Some(paths.model.clone()),
            language: Some("auto".into()),
            use_itn: true,
        };
        config.model_config.tokens = Some(paths.tokens.clone());
        config.model_config.provider = Some("cpu".into());
        config.model_config.num_threads = num_threads.max(1);

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| Error::Asr("failed to create SenseVoice recognizer".into()))?;
        Ok(Self {
            recognizer,
            sample_rate: 16_000,
        })
    }
}

impl AsrEngine for SenseVoiceEngine {
    fn capabilities(&self) -> AsrCaps {
        AsrCaps {
            streaming: false,
            multilingual: true,
            auto_detect: true,
        }
    }

    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(self.sample_rate, &utt.pcm16k);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| Error::Asr("SenseVoice produced no result".into()))?;
        Ok(TranscriptResult {
            text: result.text,
            lang: None,
            confidence: 1.0,
            is_final: true,
        })
    }
}
