use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "hearable",
    version,
    about = "Real-time captioning + speaker identification overlay for Deaf/HoH users"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Print the resolved settings.
    Config,
    /// Start live captioning: microphone -> VAD -> ASR -> speaker ID -> overlay.
    /// Requires a build with `--features live` (sherpa + mic + ui).
    Run {
        /// Directory containing the model files (SenseVoice, Silero VAD, ERes2NetV2, tokens).
        #[arg(long)]
        models: PathBuf,
        /// ASR engine: "sensevoice" (zh/en/ja/ko/yue) or "whisper" (broad multilingual incl. Vietnamese).
        #[arg(long, default_value = "sensevoice")]
        engine: String,
        /// Force a language code (e.g. "vi"); default = auto-detect.
        #[arg(long)]
        language: Option<String>,
    },
}
