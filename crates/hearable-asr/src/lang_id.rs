//! Spoken-language identification via sherpa-onnx's Whisper-based detector (the `sherpa`
//! feature). SenseVoice's recognizer result carries no language, so this fills it in.

use hearable_core::{Error, Result, Utterance};
use sherpa_onnx::{
    SpokenLanguageIdentification, SpokenLanguageIdentificationConfig,
    SpokenLanguageIdentificationWhisperConfig,
};

/// Paths to the Whisper encoder/decoder used for language detection (e.g. whisper-tiny).
pub struct LanguageIdPaths {
    pub encoder: String,
    pub decoder: String,
}

/// Detects the spoken language of an utterance.
pub struct LanguageId {
    slid: SpokenLanguageIdentification,
    sample_rate: i32,
}

impl LanguageId {
    pub fn new(paths: &LanguageIdPaths, num_threads: i32) -> Result<Self> {
        let cfg = SpokenLanguageIdentificationConfig {
            whisper: SpokenLanguageIdentificationWhisperConfig {
                encoder: Some(paths.encoder.clone()),
                decoder: Some(paths.decoder.clone()),
                tail_paddings: -1,
            },
            num_threads: num_threads.max(1),
            provider: Some("cpu".into()),
            debug: false,
        };
        let slid = SpokenLanguageIdentification::create(&cfg)
            .ok_or_else(|| Error::Asr("failed to create language identifier".into()))?;
        Ok(Self {
            slid,
            sample_rate: 16_000,
        })
    }

    /// Best-effort language code (e.g. "en", "zh"); `None` if detection fails.
    pub fn detect(&self, utt: &Utterance) -> Option<String> {
        let stream = self.slid.create_stream();
        stream.accept_waveform(self.sample_rate, &utt.pcm16k);
        self.slid.compute(&stream).map(|r| r.lang)
    }
}
