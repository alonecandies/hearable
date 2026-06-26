use crate::{Embedding, Result};
use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// How much of what is heard is kept. Defaults to the privacy-first option.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Retention {
    /// Rolling in-memory window only; nothing written to disk.
    Ephemeral,
    /// Encrypted local transcript history.
    Persistent,
}

/// Language handling: auto-detect (multilingual) or a single fixed language.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "lang")]
pub enum LanguageMode {
    Auto,
    Fixed(String),
}

/// User-facing settings, layered from defaults -> config.toml -> `HEARABLE_*` env.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub retention: Retention,
    pub language_mode: LanguageMode,
    /// Word-by-word streaming mode (only valid with a fixed language).
    pub streaming: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            retention: Retention::Ephemeral,
            language_mode: LanguageMode::Auto,
            streaming: false,
        }
    }
}

impl Settings {
    /// Load settings: compiled defaults, overlaid by `config.toml` if present,
    /// overlaid by `HEARABLE_*` environment variables.
    pub fn load() -> Result<Settings> {
        Figment::from(Serialized::defaults(Settings::default()))
            .merge(Toml::file("config.toml"))
            .merge(Env::prefixed("HEARABLE_"))
            .extract()
            .map_err(|e| crate::Error::Config(e.to_string()))
    }

    /// Persist settings as TOML (creating parent directories). Used to record the user's
    /// first-run retention choice.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let toml = toml::to_string_pretty(self).map_err(|e| crate::Error::Config(e.to_string()))?;
        std::fs::write(path, toml)?;
        Ok(())
    }
}

/// A persisted named speaker. Stores embeddings (voice fingerprints) only.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub name: String,
    pub embeddings: Vec<Embedding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_privacy_first() {
        let s = Settings::default();
        assert!(matches!(s.retention, Retention::Ephemeral));
        assert!(matches!(s.language_mode, LanguageMode::Auto));
        assert!(!s.streaming);
    }

    #[test]
    fn save_to_then_reload_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");
        let s = Settings {
            retention: Retention::Persistent,
            language_mode: LanguageMode::Auto,
            streaming: true,
        };
        s.save_to(&path).unwrap();

        let loaded: Settings = Figment::from(Serialized::defaults(Settings::default()))
            .merge(Toml::file(&path))
            .extract()
            .unwrap();
        assert!(loaded.streaming);
        assert!(matches!(loaded.retention, Retention::Persistent));
    }

    #[test]
    // `Jail::expect_with` forces a closure returning `Result<_, figment::Error>`, and
    // `figment::Error` is large; this is test-only and outside our control.
    #[allow(clippy::result_large_err)]
    fn env_overrides_streaming() {
        figment::Jail::expect_with(|jail| {
            jail.set_env("HEARABLE_STREAMING", "true");
            let s = Settings::load().unwrap();
            assert!(s.streaming);
            Ok(())
        });
    }
}
