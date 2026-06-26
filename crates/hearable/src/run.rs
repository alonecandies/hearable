//! Live pipeline wiring (the `live` feature = `sherpa` + `mic` + `ui`).
//!
//! Builds the real engines, loads saved speaker profiles, runs the threaded coordinator on a
//! worker thread, and drives the egui overlay on the main thread (required by the OS window
//! system). A command-handler thread applies the overlay's "name this speaker" actions to the
//! shared identifier and persists the resulting profile.

use hearable::run_threaded;
use hearable_asr::{LanguageId, LanguageIdPaths, SenseVoiceEngine, SenseVoicePaths};
use hearable_audio::{MicAudioSource, SileroVad, SileroVadConfig};
use hearable_core::{Error, Identifier, ProfileStore, Result, Settings, UiCommand};
use hearable_speaker::{ClusterConfig, LeaderClusterIdentifier, SherpaEmbeddingExtractor};
use hearable_store::SqliteProfileStore;
use hearable_ui::{run_overlay, ChannelCaptionSink};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

pub fn run(models: &Path) -> Result<()> {
    let settings = Settings::load()?;

    // First run: persist settings (privacy-first defaults) so the choice sticks. An
    // interactive retention prompt is a planned onboarding refinement.
    if let Some(dirs) = directories::ProjectDirs::from("app", "krystal", "hearable") {
        let cfg_path = dirs.config_dir().join("config.toml");
        if !cfg_path.exists() {
            let _ = settings.save_to(&cfg_path);
        }
    }

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
    // Attach the language-ID pass only if the Whisper LID models are present.
    let lid_encoder = models.join("whisper-encoder.onnx");
    let lid_decoder = models.join("whisper-decoder.onnx");
    let asr = if lid_encoder.exists() && lid_decoder.exists() {
        let lid = LanguageId::new(
            &LanguageIdPaths {
                encoder: path_str(&lid_encoder)?.to_string(),
                decoder: path_str(&lid_decoder)?.to_string(),
            },
            1,
        )?;
        asr.with_language_id(lid)
    } else {
        asr
    };
    let embed = SherpaEmbeddingExtractor::new(path_str(&embed_model)?, 1)?;

    let store = SqliteProfileStore::open(&db_path)?;
    let profiles = store.load_profiles()?;
    let identifier = Arc::new(Mutex::new(LeaderClusterIdentifier::new(
        ClusterConfig::default(),
        profiles,
    )));

    // Caption channel: pipeline sink -> overlay. Command channel: overlay -> handler.
    let (caption_tx, caption_rx) = std::sync::mpsc::channel();
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<UiCommand>();
    let sink = ChannelCaptionSink::new(caption_tx);

    let mic = MicAudioSource::new();
    let stop = mic.stop_handle();

    // Pipeline on a worker thread (blocks on the always-on mic stream).
    let pipeline_id = Arc::clone(&identifier);
    let worker = std::thread::spawn(move || {
        let _ = run_threaded(mic, vad, asr, embed, pipeline_id, sink, 4);
    });

    // Command handler: name a speaker -> promote in the identifier -> persist the profile.
    let handler_id = Arc::clone(&identifier);
    let handler = std::thread::spawn(move || {
        for cmd in cmd_rx {
            let UiCommand::NameSpeaker { cluster_id, name } = cmd;
            let centroid = {
                let mut id = handler_id.lock().unwrap();
                if id.promote(cluster_id, &name).is_err() {
                    continue;
                }
                id.centroid_of(cluster_id)
            };
            if let Some(c) = centroid {
                let _ = store.upsert_profile(&name, &[c]);
            }
        }
    });

    // Overlay on the main thread; blocks until the window is closed.
    let overlay_result = run_overlay(caption_rx, cmd_tx, 3);

    // Window closed -> stop capture; dropping cmd_tx ends the handler.
    stop.store(true, Ordering::Relaxed);
    let _ = worker.join();
    let _ = handler.join();
    overlay_result.map_err(|e| Error::Audio(format!("overlay: {e}")))?;
    Ok(())
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| Error::Config(format!("non-UTF8 model path: {}", p.display())))
}
