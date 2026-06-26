use hearable_core::{Utterance, UtteranceId, VadSegmenter};

/// Tuning for the energy-based segmenter.
#[derive(Debug, Clone)]
pub struct SegConfig {
    pub sample_rate: u32,
    /// RMS above this marks a frame as speech.
    pub rms_threshold: f32,
    /// Trailing silence required to close an utterance.
    pub min_silence_ms: u64,
    /// Minimum speech length for an utterance to be emitted (drops blips).
    pub min_speech_ms: u64,
}

impl Default for SegConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            rms_threshold: 0.05,
            min_silence_ms: 150,
            min_speech_ms: 100,
        }
    }
}

/// A deterministic RMS-threshold speech/silence state machine that splits a frame stream
/// into utterances. This is a Phase 0 stand-in for the sherpa-onnx Silero VAD (Phase 1),
/// behind the same [`VadSegmenter`] interface so the pipeline is unaffected by the swap.
pub struct EnergySegmenter {
    cfg: SegConfig,
    in_speech: bool,
    buf: Vec<f32>,
    silence_run: usize,
    samples_seen: u64,
    seg_start: u64,
    next_id: u64,
}

impl EnergySegmenter {
    pub fn new(cfg: SegConfig) -> Self {
        Self {
            cfg,
            in_speech: false,
            buf: Vec::new(),
            silence_run: 0,
            samples_seen: 0,
            seg_start: 0,
            next_id: 0,
        }
    }

    fn ms(&self, samples: u64) -> u64 {
        samples * 1000 / self.cfg.sample_rate as u64
    }

    fn min_silence_samples(&self) -> usize {
        (self.cfg.min_silence_ms * self.cfg.sample_rate as u64 / 1000) as usize
    }

    fn min_speech_samples(&self) -> u64 {
        self.cfg.min_speech_ms * self.cfg.sample_rate as u64 / 1000
    }

    fn emit(&mut self) -> Option<Utterance> {
        if self.buf.is_empty() {
            return None;
        }
        let len = self.buf.len() as u64;
        if len < self.min_speech_samples() {
            self.buf.clear();
            return None;
        }
        let id = UtteranceId(self.next_id);
        self.next_id += 1;
        Some(Utterance {
            id,
            pcm16k: std::mem::take(&mut self.buf),
            t0_ms: self.ms(self.seg_start),
            t1_ms: self.ms(self.seg_start + len),
        })
    }
}

impl VadSegmenter for EnergySegmenter {
    fn push(&mut self, frame: &[f32]) -> Vec<Utterance> {
        let mut out = Vec::new();
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len().max(1) as f32).sqrt();
        let voiced = rms >= self.cfg.rms_threshold;
        if voiced {
            if !self.in_speech {
                self.in_speech = true;
                self.seg_start = self.samples_seen;
            }
            self.silence_run = 0;
            self.buf.extend_from_slice(frame);
        } else if self.in_speech {
            // Keep trailing audio until the hangover trips, then close the utterance.
            self.buf.extend_from_slice(frame);
            self.silence_run += frame.len();
            if self.silence_run >= self.min_silence_samples() {
                self.in_speech = false;
                self.silence_run = 0;
                if let Some(u) = self.emit() {
                    out.push(u);
                }
            }
        }
        self.samples_seen += frame.len() as u64;
        out
    }

    fn flush(&mut self) -> Vec<Utterance> {
        let mut out = Vec::new();
        if self.in_speech {
            self.in_speech = false;
            if let Some(u) = self.emit() {
                out.push(u);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::VadSegmenter;

    // 16 kHz: 0.3s speech, 0.3s silence, 0.3s speech => 2 utterances.
    fn signal() -> Vec<f32> {
        let sr = 16_000usize;
        let mut v = Vec::new();
        let mut push = |secs: f32, amp: f32| {
            for n in 0..(secs * sr as f32) as usize {
                v.push(amp * (n as f32 * 0.1).sin());
            }
        };
        push(0.3, 0.5);
        push(0.3, 0.0);
        push(0.3, 0.5);
        v
    }

    #[test]
    fn splits_two_speech_spans() {
        let mut seg = EnergySegmenter::new(SegConfig::default());
        let sig = signal();
        let mut utts = Vec::new();
        for chunk in sig.chunks(512) {
            utts.extend(seg.push(chunk));
        }
        utts.extend(seg.flush());
        assert_eq!(utts.len(), 2, "expected 2 utterances, got {}", utts.len());
        assert!(!utts[0].pcm16k.is_empty());
        assert_eq!(utts[0].id.0, 0);
        assert_eq!(utts[1].id.0, 1);
    }

    #[test]
    fn pure_silence_yields_nothing() {
        let mut seg = EnergySegmenter::new(SegConfig::default());
        for chunk in vec![0.0f32; 16_000].chunks(512) {
            assert!(seg.push(chunk).is_empty());
        }
        assert!(seg.flush().is_empty());
    }
}
