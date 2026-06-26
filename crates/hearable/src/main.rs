mod cli;
#[cfg(all(feature = "sherpa", feature = "mic", feature = "ui"))]
mod run;

use clap::Parser;
use cli::{Cli, Command};
use hearable_core::Settings;
use std::path::Path;

fn main() -> hearable_core::Result<()> {
    let args = Cli::parse();
    match args.command {
        Some(Command::Config) | None => {
            let settings = Settings::load()?;
            println!("{settings:?}");
        }
        Some(Command::Run { models }) => run_command(&models)?,
    }
    Ok(())
}

#[cfg(all(feature = "sherpa", feature = "mic", feature = "ui"))]
fn run_command(models: &Path) -> hearable_core::Result<()> {
    run::run(models)
}

#[cfg(not(all(feature = "sherpa", feature = "mic", feature = "ui")))]
fn run_command(_models: &Path) -> hearable_core::Result<()> {
    eprintln!(
        "This build was compiled without the live features. \
         Rebuild with:\n    cargo build --release --features live"
    );
    Ok(())
}
