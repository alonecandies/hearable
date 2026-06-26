use hearable_core::{AudioSource, Error, Result};
use std::path::Path;

/// An [`AudioSource`] backed by a 16 kHz WAV file. Multi-channel files are downmixed to mono;
/// non-16 kHz files are rejected (this is a fixture replayer, not a resampler). Used to replay
/// fixtures deterministically in tests.
pub struct WavAudioSource {
    samples: Vec<f32>,
    frame_len: usize,
    stop: bool,
}

impl WavAudioSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path).map_err(|e| Error::Audio(e.to_string()))?;
        let spec = reader.spec();
        if spec.sample_rate != 16_000 {
            return Err(Error::Audio(format!(
                "WavAudioSource expects 16 kHz audio, got {} Hz",
                spec.sample_rate
            )));
        }
        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / max)
                    .collect()
            }
        };
        // Downmix interleaved channels to mono by averaging each frame.
        let channels = spec.channels.max(1) as usize;
        let samples = if channels == 1 {
            interleaved
        } else {
            interleaved
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        Ok(Self {
            samples,
            frame_len: 512,
            stop: false,
        })
    }

    pub fn with_frame_len(mut self, n: usize) -> Self {
        self.frame_len = n.max(1);
        self
    }
}

impl AudioSource for WavAudioSource {
    fn start(&mut self, on_frame: &mut dyn FnMut(&[f32])) -> Result<()> {
        self.stop = false;
        for chunk in self.samples.chunks(self.frame_len) {
            if self.stop {
                break;
            }
            on_frame(chunk);
        }
        Ok(())
    }

    fn stop(&mut self) {
        self.stop = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_all_samples_as_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for n in 0..1000i16 {
            w.write_sample(n).unwrap();
        }
        w.finalize().unwrap();

        let mut src = WavAudioSource::open(&path).unwrap().with_frame_len(256);
        let mut total = 0usize;
        src.start(&mut |f| total += f.len()).unwrap();
        assert_eq!(total, 1000);
    }

    #[test]
    fn rejects_non_16k() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hi.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        w.write_sample(0i16).unwrap();
        w.finalize().unwrap();
        assert!(WavAudioSource::open(&path).is_err());
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("st.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..100 {
            w.write_sample(1000i16).unwrap(); // L
            w.write_sample(3000i16).unwrap(); // R
        }
        w.finalize().unwrap();

        let mut src = WavAudioSource::open(&path).unwrap().with_frame_len(512);
        let mut samples = Vec::new();
        src.start(&mut |f| samples.extend_from_slice(f)).unwrap();
        assert_eq!(samples.len(), 100, "stereo -> mono halves the sample count");
        // Each mono sample is the average of L (1000) and R (3000) = 2000, normalized by 2^15.
        let expected = 2000.0 / 32768.0;
        assert!((samples[0] - expected).abs() < 1e-4);
    }
}
