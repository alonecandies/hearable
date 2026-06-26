//! Real microphone capture via `cpal` (the `mic` feature).
//!
//! The realtime audio callback only downmixes to mono and pushes into a wait-free `rtrb`
//! ring buffer (no allocation, locking, or logging). [`AudioSource::start`] runs on the
//! caller's thread: it drains the ring, resamples to 16 kHz via [`Resampler16k`], and hands
//! frames to `on_frame` until [`AudioSource::stop`] is signalled.

use crate::Resampler16k;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample as _;
use hearable_core::{AudioSource, Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct MicAudioSource {
    stop: Arc<AtomicBool>,
}

impl Default for MicAudioSource {
    fn default() -> Self {
        Self::new()
    }
}

impl MicAudioSource {
    pub fn new() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A handle that can be flipped to stop capture from another thread.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    channels: usize,
    mut producer: rtrb::Producer<f32>,
    err_flag: Arc<AtomicBool>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    let data_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // Realtime-safe: downmix to mono and push; drop on full (never block the audio thread).
        for frame in data.chunks(channels.max(1)) {
            let mut acc = 0.0f32;
            for s in frame {
                acc += f32::from_sample(*s);
            }
            let _ = producer.push(acc / channels.max(1) as f32);
        }
    };
    let err_cb = move |_e| err_flag.store(true, Ordering::Relaxed);
    device
        .build_input_stream(config, data_cb, err_cb, None)
        .map_err(|e| Error::Audio(format!("build input stream: {e}")))
}

impl AudioSource for MicAudioSource {
    fn start(&mut self, on_frame: &mut dyn FnMut(&[f32])) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| Error::Audio("no default input device".into()))?;
        let supported = device
            .default_input_config()
            .map_err(|e| Error::Audio(format!("default input config: {e}")))?;
        let sample_format = supported.sample_format();
        let channels = supported.channels() as usize;
        let in_rate = supported.sample_rate();
        let config: cpal::StreamConfig = supported.into();

        let cap = (in_rate as usize * 2).max(8192); // ~2 s of mono headroom
        let (producer, mut consumer) = rtrb::RingBuffer::<f32>::new(cap);
        let err_flag = Arc::new(AtomicBool::new(false));

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                build_stream::<f32>(&device, config, channels, producer, err_flag.clone())?
            }
            cpal::SampleFormat::I16 => {
                build_stream::<i16>(&device, config, channels, producer, err_flag.clone())?
            }
            cpal::SampleFormat::U16 => {
                build_stream::<u16>(&device, config, channels, producer, err_flag.clone())?
            }
            other => {
                return Err(Error::Audio(format!(
                    "unsupported sample format: {other:?}"
                )))
            }
        };
        stream
            .play()
            .map_err(|e| Error::Audio(format!("play stream: {e}")))?;

        let mut resampler = Resampler16k::new(in_rate)?;
        let mut scratch: Vec<f32> = Vec::with_capacity(cap);
        self.stop.store(false, Ordering::Relaxed);

        while !self.stop.load(Ordering::Relaxed) {
            if err_flag.load(Ordering::Relaxed) {
                return Err(Error::Audio("input stream error (device changed?)".into()));
            }
            scratch.clear();
            while let Ok(s) = consumer.pop() {
                scratch.push(s);
            }
            if scratch.is_empty() {
                std::thread::sleep(Duration::from_millis(8));
                continue;
            }
            let frames = resampler.push(&scratch)?;
            if !frames.is_empty() {
                on_frame(&frames);
            }
        }

        let tail = resampler.flush()?;
        if !tail.is_empty() {
            on_frame(&tail);
        }
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
