mod cli;

use clap::Parser;
use cli::{Cli, Command};
use hearable_core::Settings;

fn main() -> hearable_core::Result<()> {
    let args = Cli::parse();
    match args.command {
        Some(Command::Config) | None => {
            let settings = Settings::load()?;
            println!("{settings:?}");
        }
    }
    Ok(())
}
