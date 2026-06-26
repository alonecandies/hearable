use hearable_core::{AudioSource, Error, Result};
use std::path::Path;

/// An [`AudioSource`] backed by a WAV file (assumed 16 kHz mono). Used to replay fixtures
/// deterministically in tests; the real `cpal` microphone source arrives in Phase 1.
pub struct WavAudioSource {
    samples: Vec<f32>,
    frame_len: usize,
    stop: bool,
}

impl WavAudioSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader =
            hound::WavReader::open(path).map_err(|e| Error::Audio(e.to_string()))?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
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
}
