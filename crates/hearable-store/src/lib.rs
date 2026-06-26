//! Local persistence: speaker profiles (Phase 0), plus the model downloader and the
//! optional encrypted transcript history added in later phases.

pub mod profile_store;
pub mod schema;

pub use profile_store::SqliteProfileStore;

#[cfg(feature = "download")]
pub mod model_manager;
#[cfg(feature = "download")]
pub use model_manager::{ModelManager, ModelSpec};
