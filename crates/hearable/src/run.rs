//! Live pipeline wiring (the `live` feature = `sherpa` + `mic` + `ui`).
//!
//! Builds the real engines, loads saved speaker profiles, runs the threaded coordinator on a
//! worker thread, and drives the egui overlay on the main thread (required by the OS window
//! system). A command-handler thread applies the overlay's "name this speaker" actions to the
//! shared identifier and persists the resulting profile.

use hearable::run_threaded;
use hearable_asr::{
    LanguageId, LanguageIdPaths, SenseVoiceEngine, SenseVoicePaths, WhisperEngine, WhisperPaths,
};
use hearable_audio::{MicAudioSource, SileroVad, SileroVadConfig};
use hearable_core::{AsrEngine, Error, Identifier, ProfileStore, Result, Settings, UiCommand};
use hearable_speaker::{ClusterConfig, LeaderClusterIdentifier, SherpaEmbeddingExtractor};
use hearable_store::SqliteProfileStore;
use hearable_ui::{run_overlay, ChannelCaptionSink};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

pub fn run(models: &Path, engine: &str, language: Option<String>) -> Result<()> {
    let settings = Settings::load()?;

    // First run: persist settings (privacy-first defaults) to the same path load() reads, so
    // the choice sticks. An interactive retention prompt is a planned onboarding refinement.
    if let Some(cfg_path) = Settings::config_path() {
        if !cfg_path.exists() {
            if let Err(e) = settings.save_to(&cfg_path) {
                eprintln!(
                    "[hearable] could not persist settings to {}: {e}",
                    cfg_path.display()
                );
            }
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

    require(&vad_model)?;
    require(&embed_model)?;
    let vad = SileroVad::new(&SileroVadConfig::with_model(path_str(&vad_model)?))?;

    // Select the ASR engine. Whisper covers Vietnamese and many more languages; SenseVoice is
    // faster but limited to zh/en/ja/ko/yue.
    let asr: Box<dyn AsrEngine> = match engine {
        "whisper" => {
            let (enc, dec, tok) = (
                models.join("whisper-encoder.onnx"),
                models.join("whisper-decoder.onnx"),
                models.join("whisper-tokens.txt"),
            );
            require(&enc)?;
            require(&dec)?;
            require(&tok)?;
            Box::new(WhisperEngine::new(
                &WhisperPaths {
                    encoder: path_str(&enc)?.to_string(),
                    decoder: path_str(&dec)?.to_string(),
                    tokens: path_str(&tok)?.to_string(),
                },
                language,
                2,
            )?)
        }
        _ => {
            require(&sense_model)?;
            require(&tokens)?;
            let sv = SenseVoiceEngine::new(
                &SenseVoicePaths {
                    model: path_str(&sense_model)?.to_string(),
                    tokens: path_str(&tokens)?.to_string(),
                },
                2,
            )?;
            // Attach the language-ID pass only if the Whisper LID models are present.
            let lid_encoder = models.join("whisper-encoder.onnx");
            let lid_decoder = models.join("whisper-decoder.onnx");
            if lid_encoder.exists() && lid_decoder.exists() {
                let lid = LanguageId::new(
                    &LanguageIdPaths {
                        encoder: path_str(&lid_encoder)?.to_string(),
                        decoder: path_str(&lid_decoder)?.to_string(),
                    },
                    1,
                )?;
                Box::new(sv.with_language_id(lid))
            } else {
                Box::new(sv)
            }
        }
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
        let outcome = run_threaded(mic, vad, asr, embed, pipeline_id, sink, 4);
        if let Some(err) = outcome.capture_error {
            eprintln!("[hearable] audio capture stopped: {err}");
        }
        if outcome.inference_errors > 0 {
            eprintln!(
                "[hearable] {} utterance(s) skipped due to inference errors",
                outcome.inference_errors
            );
        }
    });

    // Command handler: name a speaker -> promote in the identifier -> persist the profile.
    let handler_id = Arc::clone(&identifier);
    let handler = std::thread::spawn(move || {
        for cmd in cmd_rx {
            let UiCommand::NameSpeaker { cluster_id, name } = cmd;
            let centroid = {
                let mut id = handler_id.lock().unwrap_or_else(|p| p.into_inner());
                if id.promote(cluster_id, &name).is_err() {
                    continue;
                }
                id.centroid_of(cluster_id)
            };
            if let Some(c) = centroid {
                if let Err(e) = store.upsert_profile(&name, &[c]) {
                    eprintln!("[hearable] failed to save speaker profile '{name}': {e}");
                }
            }
        }
    });

    // Overlay on the main thread; blocks until the window is closed. Keep 500 lines of
    // scrollable history.
    let overlay_result = run_overlay(caption_rx, cmd_tx, 500);

    // Window closed -> stop capture; dropping cmd_tx ends the handler.
    stop.store(true, Ordering::Relaxed);
    if worker.join().is_err() {
        eprintln!("[hearable] pipeline thread panicked");
    }
    if handler.join().is_err() {
        eprintln!("[hearable] speaker-naming thread panicked");
    }
    overlay_result.map_err(|e| Error::Audio(format!("overlay: {e}")))?;
    Ok(())
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| Error::Config(format!("non-UTF8 model path: {}", p.display())))
}

fn require(p: &Path) -> Result<()> {
    if p.exists() {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "missing model file: {}\nDownload the models first:  ./scripts/fetch-models.sh <dir>",
            p.display()
        )))
    }
}
