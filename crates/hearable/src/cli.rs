use clap::{Parser, Subcommand};

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
    /// Print the resolved settings (Phase 0 stand-in for the overlay run loop).
    Config,
}
