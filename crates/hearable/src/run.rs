//! Live pipeline wiring (the `live` feature = `sherpa` + `mic` + `ui`).
//!
//! Builds the real engines, loads saved speaker profiles, runs the threaded coordinator on a
//! worker thread, and drives the egui overlay on the main thread (required by the OS window
//! system). Interactive speaker naming (clicking a "Speaker N" tag) needs the identifier
//! shared behind a mutex with a UI callback channel — a noted follow-up; this wires the
//! capture → caption → overlay display path.

use hearable::run_threaded;
use hearable_asr::{SenseVoiceEngine, SenseVoicePaths};
use hearable_audio::{MicAudioSource, SileroVad, SileroVadConfig};
use hearable_core::{Error, ProfileStore, Result, Settings};
use hearable_speaker::{ClusterConfig, LeaderClusterIdentifier, SherpaEmbeddingExtractor};
use hearable_store::SqliteProfileStore;
use hearable_ui::{run_overlay, ChannelCaptionSink};
use std::path::Path;
use std::sync::atomic::Ordering;

pub fn run(models: &Path) -> Result<()> {
    let _settings = Settings::load()?;

    let sense_model = models.join("model.int8.onnx");
    let tokens = models.join("tokens.txt");
    let vad_model = models.join("silero_vad.onnx");
    let embed_model = models.join("3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx");

    let db_path = match directories::ProjectDirs::from("app", "krystal", "hearable") {
        Some(dirs) => {
            std::fs::create_dir_all(dirs.data_dir())?;
            dirs.data_dir().join("hearable.db")
        }
        None => std::path::PathBuf::from("hearable.db"),
    };

    let vad = SileroVad::new(&SileroVadConfig::with_model(path_str(&vad_model)?))?;
    let asr = SenseVoiceEngine::new(
        &SenseVoicePaths {
            model: path_str(&sense_model)?.to_string(),
            tokens: path_str(&tokens)?.to_string(),
        },
        2,
    )?;
    let embed = SherpaEmbeddingExtractor::new(path_str(&embed_model)?, 1)?;

    let store = SqliteProfileStore::open(&db_path)?;
    let profiles = store.load_profiles()?;
    let identifier = LeaderClusterIdentifier::new(ClusterConfig::default(), profiles);

    let (tx, rx) = std::sync::mpsc::channel();
    let sink = ChannelCaptionSink::new(tx);

    let mic = MicAudioSource::new();
    let stop = mic.stop_handle();

    // Pipeline on a worker thread (blocks on the always-on mic stream).
    let worker = std::thread::spawn(move || {
        let _ = run_threaded(mic, vad, asr, embed, identifier, sink, 4);
    });

    // Overlay on the main thread; blocks until the window is closed.
    let overlay_result = run_overlay(rx, 3);

    stop.store(true, Ordering::Relaxed);
    let _ = worker.join();
    overlay_result.map_err(|e| Error::Audio(format!("overlay: {e}")))?;
    Ok(())
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| Error::Config(format!("non-UTF8 model path: {}", p.display())))
}
