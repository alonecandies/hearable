use audioadapter_buffers::direct::InterleavedSlice;
use hearable_core::{Error, Result};
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

/// All downstream ML (VAD, ASR, embeddings) expects 16 kHz mono f32.
const TARGET_RATE: usize = 16_000;
/// Input frames consumed per rubato `process_into_buffer` call (FixedAsync::Input).
const CHUNK: usize = 1024;

/// Streaming resampler to 16 kHz mono f32. Accepts arbitrary-length input pushes, buffering
/// internally and emitting resampled output as whole chunks become available. When the input
/// is already 16 kHz it is a zero-cost identity pass (no rubato instance is created).
pub struct Resampler16k {
    inner: Option<Async<f32>>,
    chunk: usize,
    in_buf: Vec<f32>,
    out_scratch: Vec<f32>,
}

impl Resampler16k {
    /// Build a resampler from `input_rate` Hz mono to 16 kHz mono.
    pub fn new(input_rate: u32) -> Result<Self> {
        if input_rate as usize == TARGET_RATE || input_rate == 0 {
            return Ok(Self {
                inner: None,
                chunk: 0,
                in_buf: Vec::new(),
                out_scratch: Vec::new(),
            });
        }
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::Blackman2,
        };
        let ratio = TARGET_RATE as f64 / input_rate as f64;
        let inner = Async::<f32>::new_sinc(ratio, 1.1, &params, CHUNK, 1, FixedAsync::Input)
            .map_err(|e| Error::Audio(format!("resampler init: {e}")))?;
        let out_scratch = vec![0.0; inner.output_frames_max()];
        Ok(Self {
            inner: Some(inner),
            chunk: CHUNK,
            in_buf: Vec::new(),
            out_scratch,
        })
    }

    /// True when input is already 16 kHz (pushes pass through unchanged).
    pub fn is_identity(&self) -> bool {
        self.inner.is_none()
    }

    /// Feed input samples; returns whatever 16 kHz output is ready.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.inner.is_none() {
            return Ok(input.to_vec());
        }
        self.in_buf.extend_from_slice(input);
        let mut out = Vec::new();
        while self.in_buf.len() >= self.chunk {
            let rs = self.inner.as_mut().unwrap();
            let written =
                resample_chunk(rs, &self.in_buf[..self.chunk], &mut self.out_scratch, None)?;
            out.extend_from_slice(&self.out_scratch[..written]);
            self.in_buf.drain(..self.chunk);
        }
        Ok(out)
    }

    /// Flush the final partial chunk (call at end of stream).
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        if self.inner.is_none() || self.in_buf.is_empty() {
            return Ok(Vec::new());
        }
        let valid = self.in_buf.len();
        self.in_buf.resize(self.chunk, 0.0); // pad; partial_len marks the real count
        let rs = self.inner.as_mut().unwrap();
        let written = resample_chunk(
            rs,
            &self.in_buf[..self.chunk],
            &mut self.out_scratch,
            Some(valid),
        )?;
        self.in_buf.clear();
        Ok(self.out_scratch[..written].to_vec())
    }
}

fn resample_chunk(
    rs: &mut Async<f32>,
    input: &[f32],
    out_scratch: &mut [f32],
    partial_len: Option<usize>,
) -> Result<usize> {
    let in_adapter = InterleavedSlice::new(input, 1, input.len())
        .map_err(|e| Error::Audio(format!("resample input adapter: {e:?}")))?;
    let out_len = out_scratch.len();
    let mut out_adapter = InterleavedSlice::new_mut(out_scratch, 1, out_len)
        .map_err(|e| Error::Audio(format!("resample output adapter: {e:?}")))?;
    let indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len,
    };
    let (_read, written) = rs
        .process_into_buffer(&in_adapter, &mut out_adapter, Some(&indexing))
        .map_err(|e| Error::Audio(format!("resample: {e}")))?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passes_through_at_16k() {
        let mut r = Resampler16k::new(16_000).unwrap();
        assert!(r.is_identity());
        let input = vec![0.1, -0.2, 0.3, 0.4];
        assert_eq!(r.push(&input).unwrap(), input);
        assert!(r.flush().unwrap().is_empty());
    }

    #[test]
    fn downsamples_48k_to_roughly_one_third() {
        let mut r = Resampler16k::new(48_000).unwrap();
        assert!(!r.is_identity());
        // 1 second of a 48 kHz sine -> ~16000 output samples.
        let input: Vec<f32> = (0..48_000)
            .map(|n| (n as f32 * 2.0 * std::f32::consts::PI * 440.0 / 48_000.0).sin())
            .collect();
        let mut out = r.push(&input).unwrap();
        out.extend(r.flush().unwrap());
        assert!(
            (14_000..=17_000).contains(&out.len()),
            "expected ~16000 samples, got {}",
            out.len()
        );
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
