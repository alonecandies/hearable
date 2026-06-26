# hearable Phase 0 (Scaffold & Testable Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `hearable` Cargo workspace with the shared type/trait surface, the pure-logic core (online speaker clustering + identification, profile store, config), mock engines, and a mock end-to-end pipeline — all green in CI with **no microphone, GPU, or model download**.

**Architecture:** A 6-crate workspace plus a binary. `hearable-core` is the dependency leaf: it owns the shared data types and every cross-crate trait, so the domain crates (`-audio`, `-asr`, `-speaker`, `-store`) depend only on it and never on each other. Pipeline orchestration lives in the `hearable` binary. This resolves the dependency cycle implied by the spec's looser "core = orchestration + types" wording (a deliberate, scoped refinement of spec §3.1/§3.2).

**Tech Stack:** Rust (edition 2021, rust-version 1.85), `thiserror`, `serde`, `figment`+`toml`, `rusqlite` (bundled), `hound`, `clap`; dev: `insta`, `proptest`, `rstest`, `tempfile`. **No native ML/audio/GUI crates in Phase 0** — `sherpa-onnx`, `cpal`, `rubato`, `egui` arrive in Phase 1.

## Global Constraints

- `rust-version = "1.85"`, `edition = "2021"` for every crate (copied verbatim into `[workspace.package]`).
- **No dependency on the `ort` crate** anywhere (spec §2): ONNX Runtime is reached only through `sherpa-onnx`, added in Phase 1.
- **Do not depend on the deprecated `sherpa-rs`** crate; Phase 1 uses the official `sherpa-onnx` crate (spec §2.1).
- Default `cargo test` must pass with **no microphone, GPU, or network model download** (spec §1.6, §8). Any live-audio/native code is added behind a Cargo feature that CI does not enable.
- Profiles persist **voice embeddings only, never raw audio** (spec §6).
- Workspace deps are declared once in `[workspace.dependencies]` and referenced with `.workspace = true` (DRY).
- Every task ends green and is committed.

---

## File structure (created across Phase 0)

```
Cargo.toml                         # [workspace]
rustfmt.toml, .gitignore
.github/workflows/ci.yml
crates/
  hearable-core/   src/{lib,types,traits,config,error,testutil}.rs
  hearable-speaker/src/{lib,identify}.rs
  hearable-store/  src/{lib,profile_store,schema}.rs
  hearable-asr/    src/{lib,mock}.rs
  hearable-audio/  src/{lib,wav_source,segmenter}.rs
  hearable/        src/{main,cli,pipeline}.rs   tests/mock_pipeline.rs
```

(`hearable-ui` is intentionally deferred to Phase 1, when windowing/egui enters.)

---

### Task 0.1: Workspace skeleton + CI

**Files:**
- Create: `Cargo.toml`, `rustfmt.toml`, `.gitignore`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: the `[workspace.package]` and `[workspace.dependencies]` tables every later task references with `.workspace = true`.

- [ ] **Step 1: Create the workspace root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.package]
version = "0.0.0"
edition = "2021"
rust-version = "1.85"
license = "MIT"
authors = ["Long Hoang <long.hoang@krystal.app>"]
repository = "https://github.com/alonecandies/hearable"

[workspace.dependencies]
thiserror = "2"
serde = { version = "1", features = ["derive"] }
figment = { version = "0.10", features = ["toml", "env"] }
toml = "0.8"
rusqlite = { version = "0.40", features = ["bundled", "blob"] }
hound = "3.5"
clap = { version = "4", features = ["derive"] }
# dev
insta = { version = "1.48", features = ["yaml", "redactions"] }
proptest = "1.11"
rstest = "0.23"
tempfile = "3"
# intra-workspace
hearable-core = { path = "crates/hearable-core" }
hearable-speaker = { path = "crates/hearable-speaker" }
hearable-store = { path = "crates/hearable-store" }
hearable-asr = { path = "crates/hearable-asr" }
hearable-audio = { path = "crates/hearable-audio" }
```

- [ ] **Step 2: Create `rustfmt.toml` and `.gitignore`**

```toml
# rustfmt.toml
edition = "2021"
```
```gitignore
# .gitignore
/target
**/*.rs.bk
.DS_Store
*.profraw
```

- [ ] **Step 3: Create CI `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@1.85
        with: { components: rustfmt, clippy }
      - run: cargo fmt --all --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo test --workspace
```

- [ ] **Step 4: Verify the empty workspace builds**

Run: `cargo build --workspace`
Expected: `Finished` (no members yet → trivially succeeds, or after first crate exists).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml rustfmt.toml .gitignore .github/
git commit -m "build: scaffold hearable cargo workspace + CI"
```

---

### Task 0.2: `hearable-core` shared types

**Files:**
- Create: `crates/hearable-core/Cargo.toml`, `crates/hearable-core/src/lib.rs`, `src/types.rs`, `src/error.rs`
- Test: in-module `#[cfg(test)]` in `src/types.rs`

**Interfaces:**
- Produces (every other crate consumes these):
  - `UtteranceId(pub u64)`, `ClusterId(pub u64)`
  - `Utterance { id: UtteranceId, pcm16k: Vec<f32>, t0_ms: u64, t1_ms: u64 }`
  - `Embedding(pub Vec<f32>)` with `fn cosine(&self, other: &Embedding) -> f32`
  - `TranscriptResult { text: String, lang: Option<String>, confidence: f32, is_final: bool }`
  - `AsrCaps { streaming: bool, multilingual: bool, auto_detect: bool }`
  - `SpeakerLabel { Known { name: String, score: f32 }, Unknown { cluster_id: ClusterId, score: f32 } }`
  - `CaptionEvent { utt_id: UtteranceId, text: String, lang: Option<String>, speaker: SpeakerLabel, t0_ms: u64, t1_ms: u64, is_final: bool }`
  - `enum Error` + `type Result<T> = std::result::Result<T, Error>`

- [ ] **Step 1: Create `crates/hearable-core/Cargo.toml`**

```toml
[package]
name = "hearable-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true
serde.workspace = true
```

- [ ] **Step 2: Write the failing cosine test in `src/types.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cosine_of_identical_is_one() {
        let a = Embedding(vec![1.0, 2.0, 3.0]);
        assert!((a.cosine(&a) - 1.0).abs() < 1e-6);
    }
    #[test]
    fn cosine_of_orthogonal_is_zero() {
        let a = Embedding(vec![1.0, 0.0]);
        let b = Embedding(vec![0.0, 1.0]);
        assert!(a.cosine(&b).abs() < 1e-6);
    }
    #[test]
    fn cosine_of_zero_vector_is_zero_not_nan() {
        let a = Embedding(vec![0.0, 0.0]);
        let b = Embedding(vec![1.0, 1.0]);
        assert_eq!(a.cosine(&b), 0.0);
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p hearable-core`
Expected: FAIL — `Embedding` / `cosine` not found.

- [ ] **Step 4: Implement `src/types.rs`**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UtteranceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub u64);

#[derive(Debug, Clone)]
pub struct Utterance {
    pub id: UtteranceId,
    pub pcm16k: Vec<f32>,
    pub t0_ms: u64,
    pub t1_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Cosine similarity in [-1, 1]; returns 0.0 if either vector is zero-length.
    pub fn cosine(&self, other: &Embedding) -> f32 {
        let (a, b) = (&self.0, &other.0);
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptResult {
    pub text: String,
    pub lang: Option<String>,
    pub confidence: f32,
    pub is_final: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrCaps {
    pub streaming: bool,
    pub multilingual: bool,
    pub auto_detect: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerLabel {
    Known { name: String, score: f32 },
    Unknown { cluster_id: ClusterId, score: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptionEvent {
    pub utt_id: UtteranceId,
    pub text: String,
    pub lang: Option<String>,
    pub speaker: SpeakerLabel,
    pub t0_ms: u64,
    pub t1_ms: u64,
    pub is_final: bool,
}
```

- [ ] **Step 5: Implement `src/error.rs` and `src/lib.rs`**

```rust
// src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("audio error: {0}")]
    Audio(String),
    #[error("asr error: {0}")]
    Asr(String),
    #[error("store error: {0}")]
    Store(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
```
```rust
// src/lib.rs
pub mod error;
pub mod types;
pub use error::{Error, Result};
pub use types::*;
```

- [ ] **Step 6: Run tests, then commit**

Run: `cargo test -p hearable-core` → Expected: PASS (3 tests).
```bash
git add crates/hearable-core
git commit -m "feat(core): shared types (Utterance, Embedding, CaptionEvent) + Error"
```

---

### Task 0.3: `hearable-core` traits, config, and test sink

**Files:**
- Create: `crates/hearable-core/src/traits.rs`, `src/config.rs`, `src/testutil.rs`
- Modify: `crates/hearable-core/src/lib.rs` (add modules), `Cargo.toml` (add figment/toml)
- Test: `#[cfg(test)]` in `src/config.rs`

**Interfaces:**
- Consumes: `Utterance`, `Embedding`, `TranscriptResult`, `AsrCaps`, `SpeakerLabel`, `CaptionEvent`, `Result` (Task 0.2).
- Produces:
  - traits `AudioSource`, `VadSegmenter`, `AsrEngine`, `EmbeddingExtractor`, `Identifier`, `ProfileStore`, `CaptionSink` (exact signatures below)
  - `Settings { retention: Retention, language_mode: LanguageMode, streaming: bool }` with `Settings::load() -> Result<Settings>` and `Default`
  - `enum Retention { Ephemeral, Persistent }`, `enum LanguageMode { Auto, Fixed(String) }`
  - `BufferCaptionSink` (test util implementing `CaptionSink`, exposing `events() -> &[CaptionEvent]`)

- [ ] **Step 1: Add figment/toml to `hearable-core/Cargo.toml`**

```toml
[dependencies]
thiserror.workspace = true
serde.workspace = true
figment.workspace = true
toml.workspace = true
```

- [ ] **Step 2: Write `src/traits.rs`**

```rust
use crate::{AsrCaps, CaptionEvent, ClusterId, Embedding, Result, SpeakerLabel, TranscriptResult, Utterance};

/// A source of 16 kHz mono f32 frames (real mic in Phase 1, WAV fixture in tests).
pub trait AudioSource {
    fn start(&mut self, on_frame: &mut dyn FnMut(&[f32])) -> Result<()>;
    fn stop(&mut self);
}

/// Splits a frame stream into discrete utterances.
pub trait VadSegmenter {
    fn push(&mut self, frame: &[f32]) -> Vec<Utterance>;
    fn flush(&mut self) -> Vec<Utterance>;
}

pub trait AsrEngine: Send {
    fn capabilities(&self) -> AsrCaps;
    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult>;
}

pub trait EmbeddingExtractor: Send {
    fn embed(&mut self, utt: &Utterance) -> Result<Embedding>;
}

pub trait Identifier {
    fn identify(&mut self, e: &Embedding) -> SpeakerLabel;
    fn promote(&mut self, cluster: ClusterId, name: &str) -> Result<()>;
}

pub trait ProfileStore {
    fn load_profiles(&self) -> Result<Vec<crate::config::Profile>>;
    fn upsert_profile(&self, name: &str, embeddings: &[Embedding]) -> Result<()>;
}

pub trait CaptionSink: Send {
    fn emit(&mut self, ev: CaptionEvent);
}
```

- [ ] **Step 3: Write the failing config test in `src/config.rs`**

```rust
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
    fn env_overrides_streaming() {
        figment::Jail::expect_with(|jail| {
            jail.set_env("HEARABLE_STREAMING", "true");
            let s = Settings::load().unwrap();
            assert!(s.streaming);
            Ok(())
        });
    }
}
```

- [ ] **Step 4: Run to verify it fails**

Run: `cargo test -p hearable-core config` → Expected: FAIL (`Settings` undefined).

- [ ] **Step 5: Implement `src/config.rs`**

```rust
use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;
use serde::{Deserialize, Serialize};
use crate::{Embedding, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Retention { Ephemeral, Persistent }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "lang")]
pub enum LanguageMode { Auto, Fixed(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub retention: Retention,
    pub language_mode: LanguageMode,
    pub streaming: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { retention: Retention::Ephemeral, language_mode: LanguageMode::Auto, streaming: false }
    }
}

impl Settings {
    /// defaults -> config.toml (if present) -> HEARABLE_* env overrides.
    pub fn load() -> Result<Settings> {
        Figment::from(Serialized::defaults(Settings::default()))
            .merge(Toml::file("config.toml"))
            .merge(Env::prefixed("HEARABLE_"))
            .extract()
            .map_err(|e| crate::Error::Config(e.to_string()))
    }
}

/// A persisted named speaker (embeddings only — never raw audio).
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub name: String,
    pub embeddings: Vec<Embedding>,
}
```

- [ ] **Step 6: Write `src/testutil.rs` (BufferCaptionSink)**

```rust
use crate::{CaptionEvent, CaptionSink};

#[derive(Default)]
pub struct BufferCaptionSink { events: Vec<CaptionEvent> }
impl BufferCaptionSink {
    pub fn new() -> Self { Self::default() }
    pub fn events(&self) -> &[CaptionEvent] { &self.events }
}
impl CaptionSink for BufferCaptionSink {
    fn emit(&mut self, ev: CaptionEvent) { self.events.push(ev); }
}
```

- [ ] **Step 7: Wire modules in `src/lib.rs`, run tests, commit**

```rust
pub mod config;
pub mod error;
pub mod testutil;
pub mod traits;
pub mod types;
pub use config::{LanguageMode, Profile, Retention, Settings};
pub use error::{Error, Result};
pub use traits::*;
pub use types::*;
```
Run: `cargo test -p hearable-core` → Expected: PASS.
```bash
git add crates/hearable-core
git commit -m "feat(core): cross-crate traits, figment Settings, BufferCaptionSink"
```

---

### Task 0.4: `hearable-speaker` online leader-cluster identifier

**Files:**
- Create: `crates/hearable-speaker/Cargo.toml`, `src/lib.rs`, `src/identify.rs`
- Test: `#[cfg(test)]` in `src/identify.rs` (unit + proptest)

**Interfaces:**
- Consumes: `Embedding`, `ClusterId`, `SpeakerLabel`, `Identifier`, `Profile`, `Result` (core).
- Produces: `LeaderClusterIdentifier::new(cfg: ClusterConfig, profiles: Vec<Profile>) -> Self` implementing `Identifier`; `ClusterConfig { threshold: f32, short_threshold: f32, short_secs: f32, ema_alpha: f32 }` with `Default` (0.70 / 0.62 / 1.5 / 0.05 per spec §4.4).

- [ ] **Step 1: Create `Cargo.toml`**

```toml
[package]
name = "hearable-speaker"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
hearable-core.workspace = true

[dev-dependencies]
proptest.workspace = true
```

- [ ] **Step 2: Write failing tests in `src/identify.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{Embedding, SpeakerLabel};

    fn id_default() -> LeaderClusterIdentifier {
        LeaderClusterIdentifier::new(ClusterConfig::default(), vec![])
    }

    #[test]
    fn first_embedding_creates_unknown_cluster() {
        let mut id = id_default();
        match id.identify(&Embedding(vec![1.0, 0.0, 0.0])) {
            SpeakerLabel::Unknown { cluster_id, .. } => assert_eq!(cluster_id.0, 0),
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn similar_embedding_joins_same_cluster() {
        let mut id = id_default();
        let a = id.identify(&Embedding(vec![1.0, 0.0, 0.0]));
        let b = id.identify(&Embedding(vec![0.98, 0.02, 0.0]));
        assert_eq!(cluster_of(&a), cluster_of(&b));
    }

    #[test]
    fn dissimilar_embedding_creates_new_cluster() {
        let mut id = id_default();
        let a = id.identify(&Embedding(vec![1.0, 0.0, 0.0]));
        let b = id.identify(&Embedding(vec![0.0, 1.0, 0.0]));
        assert_ne!(cluster_of(&a), cluster_of(&b));
    }

    #[test]
    fn known_profile_matches_by_name() {
        use hearable_core::Profile;
        let prof = Profile { name: "Mom".into(), embeddings: vec![Embedding(vec![1.0, 0.0, 0.0])] };
        let mut id = LeaderClusterIdentifier::new(ClusterConfig::default(), vec![prof]);
        match id.identify(&Embedding(vec![0.97, 0.0, 0.0])) {
            SpeakerLabel::Known { name, .. } => assert_eq!(name, "Mom"),
            other => panic!("expected Known(Mom), got {other:?}"),
        }
    }

    #[test]
    fn promote_makes_future_utterances_known() {
        let mut id = id_default();
        let label = id.identify(&Embedding(vec![1.0, 0.0, 0.0]));
        let cid = match label { SpeakerLabel::Unknown { cluster_id, .. } => cluster_id, _ => unreachable!() };
        id.promote(cid, "Tom").unwrap();
        match id.identify(&Embedding(vec![0.99, 0.0, 0.0])) {
            SpeakerLabel::Known { name, .. } => assert_eq!(name, "Tom"),
            other => panic!("expected Known(Tom), got {other:?}"),
        }
    }

    fn cluster_of(l: &SpeakerLabel) -> hearable_core::ClusterId {
        match l { SpeakerLabel::Unknown { cluster_id, .. } => *cluster_id, _ => panic!("not unknown") }
    }

    proptest::proptest! {
        #[test]
        fn identify_never_panics(v in proptest::collection::vec(-10f32..10.0, 1..64)) {
            let mut id = id_default();
            let _ = id.identify(&Embedding(v));
        }
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p hearable-speaker` → Expected: FAIL (`LeaderClusterIdentifier` undefined).

- [ ] **Step 4: Implement `src/identify.rs`**

```rust
use hearable_core::{ClusterId, Embedding, Identifier, Profile, Result, SpeakerLabel};

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub threshold: f32,
    pub short_threshold: f32,
    pub short_secs: f32,
    pub ema_alpha: f32,
}
impl Default for ClusterConfig {
    fn default() -> Self {
        Self { threshold: 0.70, short_threshold: 0.62, short_secs: 1.5, ema_alpha: 0.05 }
    }
}

struct Cluster { id: ClusterId, centroid: Embedding, name: Option<String> }

pub struct LeaderClusterIdentifier {
    cfg: ClusterConfig,
    clusters: Vec<Cluster>,
    next_id: u64,
}

impl LeaderClusterIdentifier {
    pub fn new(cfg: ClusterConfig, profiles: Vec<Profile>) -> Self {
        let mut me = Self { cfg, clusters: Vec::new(), next_id: 0 };
        for p in profiles {
            // Seed one named cluster per profile using the mean of its embeddings.
            if let Some(centroid) = mean(&p.embeddings) {
                let id = me.alloc_id();
                me.clusters.push(Cluster { id, centroid, name: Some(p.name) });
            }
        }
        me
    }

    fn alloc_id(&mut self) -> ClusterId { let id = ClusterId(self.next_id); self.next_id += 1; id }

    fn ema_update(centroid: &mut Embedding, e: &Embedding, alpha: f32) {
        if centroid.0.len() != e.0.len() { return; }
        for (c, x) in centroid.0.iter_mut().zip(&e.0) { *c = (1.0 - alpha) * *c + alpha * *x; }
    }
}

impl Identifier for LeaderClusterIdentifier {
    fn identify(&mut self, e: &Embedding) -> SpeakerLabel {
        // (utterance-length-aware threshold is wired in Task 0.8 via Utterance; default path uses `threshold`)
        let thr = self.cfg.threshold;
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.clusters.iter().enumerate() {
            let s = c.centroid.cosine(e);
            if best.map_or(true, |(_, bs)| s > bs) { best = Some((i, s)); }
        }
        match best {
            Some((i, s)) if s >= thr => {
                Self::ema_update(&mut self.clusters[i].centroid, e, self.cfg.ema_alpha);
                let c = &self.clusters[i];
                match &c.name {
                    Some(name) => SpeakerLabel::Known { name: name.clone(), score: s },
                    None => SpeakerLabel::Unknown { cluster_id: c.id, score: s },
                }
            }
            _ => {
                let id = self.alloc_id();
                self.clusters.push(Cluster { id, centroid: e.clone(), name: None });
                SpeakerLabel::Unknown { cluster_id: id, score: best.map_or(0.0, |(_, s)| s) }
            }
        }
    }

    fn promote(&mut self, cluster: ClusterId, name: &str) -> Result<()> {
        for c in &mut self.clusters {
            if c.id == cluster { c.name = Some(name.to_string()); return Ok(()); }
        }
        Err(hearable_core::Error::Store(format!("no cluster {cluster:?}")))
    }
}

fn mean(es: &[Embedding]) -> Option<Embedding> {
    let dim = es.first()?.0.len();
    let mut acc = vec![0.0f32; dim];
    for e in es { for (a, x) in acc.iter_mut().zip(&e.0) { *a += x; } }
    let n = es.len() as f32;
    for a in &mut acc { *a /= n; }
    Some(Embedding(acc))
}
```
```rust
// src/lib.rs
pub mod identify;
pub use identify::{ClusterConfig, LeaderClusterIdentifier};
```

- [ ] **Step 5: Run tests, then commit**

Run: `cargo test -p hearable-speaker` → Expected: PASS.
```bash
git add crates/hearable-speaker
git commit -m "feat(speaker): online leader-cluster identifier with known-profile match + promotion"
```

---

### Task 0.5: `hearable-store` SQLite profile store

**Files:**
- Create: `crates/hearable-store/Cargo.toml`, `src/lib.rs`, `src/schema.rs`, `src/profile_store.rs`
- Test: `#[cfg(test)]` in `src/profile_store.rs`

**Interfaces:**
- Consumes: `Embedding`, `Profile`, `ProfileStore`, `Result` (core).
- Produces: `SqliteProfileStore::open_in_memory() -> Result<Self>` and `::open(path) -> Result<Self>`, implementing `ProfileStore`. Embeddings serialized as little-endian f32 BLOB with a `u32` count-per-embedding header.

- [ ] **Step 1: Create `Cargo.toml`**

```toml
[package]
name = "hearable-store"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
hearable-core.workspace = true
rusqlite.workspace = true
```

- [ ] **Step 2: Write failing roundtrip test in `src/profile_store.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{Embedding, ProfileStore};

    #[test]
    fn upsert_then_load_roundtrips_embeddings() {
        let store = SqliteProfileStore::open_in_memory().unwrap();
        let embs = vec![Embedding(vec![0.1, 0.2, 0.3]), Embedding(vec![-1.0, 0.5, 2.0])];
        store.upsert_profile("Mom", &embs).unwrap();
        let loaded = store.load_profiles().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "Mom");
        assert_eq!(loaded[0].embeddings, embs);
    }

    #[test]
    fn upsert_same_name_replaces() {
        let store = SqliteProfileStore::open_in_memory().unwrap();
        store.upsert_profile("Tom", &[Embedding(vec![1.0])]).unwrap();
        store.upsert_profile("Tom", &[Embedding(vec![2.0]), Embedding(vec![3.0])]).unwrap();
        let loaded = store.load_profiles().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].embeddings.len(), 2);
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p hearable-store` → Expected: FAIL.

- [ ] **Step 4: Implement `src/schema.rs`, `src/profile_store.rs`, `src/lib.rs`**

```rust
// src/schema.rs
pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS profiles (
    name       TEXT PRIMARY KEY,
    embeddings BLOB NOT NULL,
    created_ms INTEGER NOT NULL DEFAULT 0,
    updated_ms INTEGER NOT NULL DEFAULT 0
);
"#;
```
```rust
// src/profile_store.rs
use hearable_core::{Embedding, Error, Profile, ProfileStore, Result};
use rusqlite::Connection;
use std::path::Path;

pub struct SqliteProfileStore { conn: Connection }

impl SqliteProfileStore {
    pub fn open_in_memory() -> Result<Self> { Self::from_conn(Connection::open_in_memory().map_err(se)?) }
    pub fn open(path: impl AsRef<Path>) -> Result<Self> { Self::from_conn(Connection::open(path).map_err(se)?) }
    fn from_conn(conn: Connection) -> Result<Self> {
        conn.execute_batch(crate::schema::SCHEMA).map_err(se)?;
        Ok(Self { conn })
    }
}

impl ProfileStore for SqliteProfileStore {
    fn load_profiles(&self) -> Result<Vec<Profile>> {
        let mut stmt = self.conn.prepare("SELECT name, embeddings FROM profiles").map_err(se)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .map_err(se)?;
        let mut out = Vec::new();
        for row in rows {
            let (name, blob) = row.map_err(se)?;
            out.push(Profile { name, embeddings: decode(&blob)? });
        }
        Ok(out)
    }
    fn upsert_profile(&self, name: &str, embeddings: &[Embedding]) -> Result<()> {
        self.conn.execute(
            "INSERT INTO profiles(name, embeddings) VALUES(?1, ?2)
             ON CONFLICT(name) DO UPDATE SET embeddings = excluded.embeddings",
            rusqlite::params![name, encode(embeddings)],
        ).map_err(se)?;
        Ok(())
    }
}

fn se(e: rusqlite::Error) -> Error { Error::Store(e.to_string()) }

// BLOB layout: repeated [u32 dim][dim * f32 LE].
fn encode(embs: &[Embedding]) -> Vec<u8> {
    let mut buf = Vec::new();
    for e in embs {
        buf.extend_from_slice(&(e.0.len() as u32).to_le_bytes());
        for x in &e.0 { buf.extend_from_slice(&x.to_le_bytes()); }
    }
    buf
}
fn decode(buf: &[u8]) -> Result<Vec<Embedding>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        if i + 4 > buf.len() { return Err(Error::Store("truncated embedding header".into())); }
        let dim = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            if i + 4 > buf.len() { return Err(Error::Store("truncated embedding body".into())); }
            v.push(f32::from_le_bytes(buf[i..i + 4].try_into().unwrap()));
            i += 4;
        }
        out.push(Embedding(v));
    }
    Ok(out)
}
```
```rust
// src/lib.rs
pub mod profile_store;
pub mod schema;
pub use profile_store::SqliteProfileStore;
```

- [ ] **Step 5: Run tests, commit**

Run: `cargo test -p hearable-store` → Expected: PASS.
```bash
git add crates/hearable-store
git commit -m "feat(store): SQLite profile store (embeddings as f32 BLOB, never raw audio)"
```

---

### Task 0.6: `hearable-asr` mock engine

**Files:**
- Create: `crates/hearable-asr/Cargo.toml`, `src/lib.rs`, `src/mock.rs`
- Test: `#[cfg(test)]` in `src/mock.rs`

**Interfaces:**
- Consumes: `AsrEngine`, `AsrCaps`, `Utterance`, `UtteranceId`, `TranscriptResult`, `Result` (core).
- Produces: `MockAsrEngine::new(map: HashMap<u64, &str>) -> Self` (utterance-id → text) implementing `AsrEngine`; `capabilities()` returns `{ streaming: false, multilingual: true, auto_detect: true }`.

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "hearable-asr"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
hearable-core.workspace = true
```

- [ ] **Step 2: Failing test in `src/mock.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::{AsrEngine, Utterance, UtteranceId};
    use std::collections::HashMap;

    fn utt(id: u64) -> Utterance { Utterance { id: UtteranceId(id), pcm16k: vec![], t0_ms: 0, t1_ms: 0 } }

    #[test]
    fn returns_mapped_text() {
        let mut e = MockAsrEngine::new(HashMap::from([(7u64, "hello".to_string())]));
        let r = e.transcribe(&utt(7)).unwrap();
        assert_eq!(r.text, "hello");
        assert!(r.is_final);
    }
    #[test]
    fn unmapped_is_empty() {
        let mut e = MockAsrEngine::new(HashMap::new());
        assert_eq!(e.transcribe(&utt(1)).unwrap().text, "");
    }
}
```

- [ ] **Step 3: Run → FAIL.** `cargo test -p hearable-asr`

- [ ] **Step 4: Implement `src/mock.rs` + `src/lib.rs`**

```rust
// src/mock.rs
use hearable_core::{AsrCaps, AsrEngine, Result, TranscriptResult, Utterance};
use std::collections::HashMap;

pub struct MockAsrEngine { map: HashMap<u64, String> }
impl MockAsrEngine {
    pub fn new<S: Into<String>>(map: HashMap<u64, S>) -> Self {
        Self { map: map.into_iter().map(|(k, v)| (k, v.into())).collect() }
    }
}
impl AsrEngine for MockAsrEngine {
    fn capabilities(&self) -> AsrCaps { AsrCaps { streaming: false, multilingual: true, auto_detect: true } }
    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult> {
        let text = self.map.get(&utt.id.0).cloned().unwrap_or_default();
        Ok(TranscriptResult { text, lang: Some("en".into()), confidence: 1.0, is_final: true })
    }
}
```
```rust
// src/lib.rs
pub mod mock;
pub use mock::MockAsrEngine;
```

- [ ] **Step 5: Run → PASS, commit**

```bash
git add crates/hearable-asr
git commit -m "feat(asr): AsrEngine trait surface + deterministic MockAsrEngine"
```

---

### Task 0.7: `hearable-audio` WAV source + energy segmenter

**Files:**
- Create: `crates/hearable-audio/Cargo.toml`, `src/lib.rs`, `src/wav_source.rs`, `src/segmenter.rs`
- Test: `#[cfg(test)]` in both modules (segmenter uses a generated signal; WAV source uses `tempfile`)

**Interfaces:**
- Consumes: `AudioSource`, `VadSegmenter`, `Utterance`, `UtteranceId`, `Result` (core).
- Produces:
  - `WavAudioSource::open(path) -> Result<Self>` implementing `AudioSource` (assumes 16 kHz mono i16/f32 WAV; emits frames of `frame_len`, default 512).
  - `EnergySegmenter::new(cfg: SegConfig) -> Self` implementing `VadSegmenter` (RMS-threshold speech/silence state machine with min-silence hangover). This is a deterministic stand-in for the Phase 1 sherpa Silero VAD, with the same `VadSegmenter` interface.
  - `SegConfig { sample_rate: u32, rms_threshold: f32, min_silence_ms: u64, min_speech_ms: u64 }`.

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "hearable-audio"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
hearable-core.workspace = true
hound.workspace = true

[dev-dependencies]
tempfile.workspace = true
```

- [ ] **Step 2: Failing segmenter test in `src/segmenter.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use hearable_core::VadSegmenter;

    // 16 kHz: 0.3s speech, 0.3s silence, 0.3s speech => 2 utterances.
    fn signal() -> Vec<f32> {
        let sr = 16_000usize;
        let mut v = Vec::new();
        let push = |v: &mut Vec<f32>, secs: f32, amp: f32| {
            for n in 0..(secs * sr as f32) as usize {
                v.push(amp * (n as f32 * 0.1).sin());
            }
        };
        push(&mut v, 0.3, 0.5);
        push(&mut v, 0.3, 0.0);
        push(&mut v, 0.3, 0.5);
        v
    }

    #[test]
    fn splits_two_speech_spans() {
        let mut seg = EnergySegmenter::new(SegConfig::default());
        let sig = signal();
        let mut utts = Vec::new();
        for chunk in sig.chunks(512) { utts.extend(seg.push(chunk)); }
        utts.extend(seg.flush());
        assert_eq!(utts.len(), 2, "expected 2 utterances, got {}", utts.len());
        assert!(utts[0].pcm16k.len() > 0);
    }
}
```

- [ ] **Step 3: Run → FAIL.** `cargo test -p hearable-audio`

- [ ] **Step 4: Implement `src/segmenter.rs`**

```rust
use hearable_core::{Utterance, UtteranceId, VadSegmenter};

#[derive(Debug, Clone)]
pub struct SegConfig {
    pub sample_rate: u32,
    pub rms_threshold: f32,
    pub min_silence_ms: u64,
    pub min_speech_ms: u64,
}
impl Default for SegConfig {
    fn default() -> Self { Self { sample_rate: 16_000, rms_threshold: 0.05, min_silence_ms: 150, min_speech_ms: 100 } }
}

pub struct EnergySegmenter {
    cfg: SegConfig,
    in_speech: bool,
    buf: Vec<f32>,
    silence_run: usize,
    samples_seen: u64,
    seg_start: u64,
    next_id: u64,
}
impl EnergySegmenter {
    pub fn new(cfg: SegConfig) -> Self {
        Self { cfg, in_speech: false, buf: Vec::new(), silence_run: 0, samples_seen: 0, seg_start: 0, next_id: 0 }
    }
    fn ms(&self, samples: u64) -> u64 { samples * 1000 / self.cfg.sample_rate as u64 }
    fn min_silence_samples(&self) -> usize { (self.cfg.min_silence_ms * self.cfg.sample_rate as u64 / 1000) as usize }
    fn min_speech_samples(&self) -> u64 { self.cfg.min_speech_ms * self.cfg.sample_rate as u64 / 1000 }
    fn emit(&mut self) -> Option<Utterance> {
        if self.buf.is_empty() { return None; }
        let len = self.buf.len() as u64;
        if len < self.min_speech_samples() { self.buf.clear(); return None; }
        let id = UtteranceId(self.next_id); self.next_id += 1;
        let u = Utterance {
            id,
            pcm16k: std::mem::take(&mut self.buf),
            t0_ms: self.ms(self.seg_start),
            t1_ms: self.ms(self.seg_start + len),
        };
        Some(u)
    }
}
impl VadSegmenter for EnergySegmenter {
    fn push(&mut self, frame: &[f32]) -> Vec<Utterance> {
        let mut out = Vec::new();
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len().max(1) as f32).sqrt();
        let voiced = rms >= self.cfg.rms_threshold;
        if voiced {
            if !self.in_speech { self.in_speech = true; self.seg_start = self.samples_seen; }
            self.silence_run = 0;
            self.buf.extend_from_slice(frame);
        } else if self.in_speech {
            self.buf.extend_from_slice(frame); // keep trailing audio until hangover trips
            self.silence_run += frame.len();
            if self.silence_run >= self.min_silence_samples() {
                self.in_speech = false;
                self.silence_run = 0;
                if let Some(u) = self.emit() { out.push(u); }
            }
        }
        self.samples_seen += frame.len() as u64;
        out
    }
    fn flush(&mut self) -> Vec<Utterance> {
        let mut out = Vec::new();
        if self.in_speech { self.in_speech = false; if let Some(u) = self.emit() { out.push(u); } }
        out
    }
}
```

- [ ] **Step 5: Implement `src/wav_source.rs` with its own test**

```rust
use hearable_core::{AudioSource, Error, Result};
use std::path::Path;

pub struct WavAudioSource { samples: Vec<f32>, frame_len: usize, stop: bool }
impl WavAudioSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path).map_err(|e| Error::Audio(e.to_string()))?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
            }
        };
        Ok(Self { samples, frame_len: 512, stop: false })
    }
    pub fn with_frame_len(mut self, n: usize) -> Self { self.frame_len = n; self }
}
impl AudioSource for WavAudioSource {
    fn start(&mut self, on_frame: &mut dyn FnMut(&[f32])) -> Result<()> {
        self.stop = false;
        for chunk in self.samples.chunks(self.frame_len) {
            if self.stop { break; }
            on_frame(chunk);
        }
        Ok(())
    }
    fn stop(&mut self) { self.stop = true; }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_all_samples_as_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for n in 0..1000i16 { w.write_sample(n).unwrap(); }
        w.finalize().unwrap();

        let mut src = WavAudioSource::open(&path).unwrap().with_frame_len(256);
        let mut total = 0usize;
        src.start(&mut |f| total += f.len()).unwrap();
        assert_eq!(total, 1000);
    }
}
```
```rust
// src/lib.rs
pub mod segmenter;
pub mod wav_source;
pub use segmenter::{EnergySegmenter, SegConfig};
pub use wav_source::WavAudioSource;
```

- [ ] **Step 6: Run tests, commit**

Run: `cargo test -p hearable-audio` → Expected: PASS.
```bash
git add crates/hearable-audio
git commit -m "feat(audio): WAV AudioSource + deterministic energy VadSegmenter (Silero stand-in)"
```

---

### Task 0.8: `hearable` binary — mock pipeline + end-to-end integration test

**Files:**
- Create: `crates/hearable/Cargo.toml`, `src/main.rs`, `src/cli.rs`, `src/pipeline.rs`
- Test: `crates/hearable/tests/mock_pipeline.rs`

**Interfaces:**
- Consumes: every trait + `MockAsrEngine` (asr), `EnergySegmenter`/`WavAudioSource` (audio), `LeaderClusterIdentifier` (speaker), `BufferCaptionSink` (core::testutil), `Settings` (core).
- Produces: `run_pipeline<A, V, E, I, S>(source, segmenter, asr, embedder, identifier, sink)` that pulls frames → segments → for each utterance transcribes + embeds + identifies → emits a `CaptionEvent`. A `MockEmbeddingExtractor` test double (maps utterance-id → a fixed embedding) lives in the test file.

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "hearable"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[[bin]]
name = "hearable"
path = "src/main.rs"

[dependencies]
hearable-core.workspace = true
hearable-audio.workspace = true
hearable-asr.workspace = true
hearable-speaker.workspace = true
hearable-store.workspace = true
clap.workspace = true
```

- [ ] **Step 2: Write the failing integration test `tests/mock_pipeline.rs`**

```rust
use hearable::pipeline::run_pipeline;
use hearable_asr::MockAsrEngine;
use hearable_audio::{EnergySegmenter, SegConfig, WavAudioSource};
use hearable_core::testutil::BufferCaptionSink;
use hearable_core::{Embedding, EmbeddingExtractor, Result, SpeakerLabel, Utterance};
use hearable_speaker::{ClusterConfig, LeaderClusterIdentifier};
use std::collections::HashMap;

struct MockEmbed { map: HashMap<u64, Embedding> }
impl EmbeddingExtractor for MockEmbed {
    fn embed(&mut self, utt: &Utterance) -> Result<Embedding> {
        Ok(self.map.get(&utt.id.0).cloned().unwrap_or(Embedding(vec![0.0, 0.0, 1.0])))
    }
}

fn two_speaker_wav(path: &std::path::Path) {
    // 2 voiced spans separated by silence, like Task 0.7's signal.
    let sr = 16_000usize;
    let spec = hound::WavSpec { channels: 1, sample_rate: sr as u32, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    let mut push = |secs: f32, amp: f32| { for n in 0..(secs * sr as f32) as usize { w.write_sample((amp * (n as f32 * 0.1).sin() * 30000.0) as i16).unwrap(); } };
    push(0.3, 1.0); push(0.3, 0.0); push(0.3, 1.0);
    w.finalize().unwrap();
}

#[test]
fn mock_pipeline_produces_two_captions_with_distinct_speakers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.wav");
    two_speaker_wav(&path);

    let source = WavAudioSource::open(&path).unwrap();
    let seg = EnergySegmenter::new(SegConfig::default());
    // utterance 0 -> "hello", utterance 1 -> "world"
    let asr = MockAsrEngine::new(HashMap::from([(0u64, "hello"), (1u64, "world")]));
    // distinct embeddings => two clusters
    let embed = MockEmbed { map: HashMap::from([
        (0u64, Embedding(vec![1.0, 0.0, 0.0])),
        (1u64, Embedding(vec![0.0, 1.0, 0.0])),
    ]) };
    let id = LeaderClusterIdentifier::new(ClusterConfig::default(), vec![]);
    let mut sink = BufferCaptionSink::new();

    run_pipeline(source, seg, asr, embed, id, &mut sink).unwrap();

    let evs = sink.events();
    assert_eq!(evs.len(), 2, "expected 2 caption events");
    assert_eq!(evs[0].text, "hello");
    assert_eq!(evs[1].text, "world");
    let c0 = match &evs[0].speaker { SpeakerLabel::Unknown { cluster_id, .. } => cluster_id.0, _ => panic!() };
    let c1 = match &evs[1].speaker { SpeakerLabel::Unknown { cluster_id, .. } => cluster_id.0, _ => panic!() };
    assert_ne!(c0, c1, "distinct embeddings must yield distinct speaker clusters");
}
```

- [ ] **Step 3: Run → FAIL.** `cargo test -p hearable` (no `run_pipeline`).

- [ ] **Step 4: Implement `src/pipeline.rs`**

```rust
use hearable_core::{
    AsrEngine, AudioSource, CaptionEvent, CaptionSink, EmbeddingExtractor, Identifier, Result,
    Utterance, VadSegmenter,
};

/// Synchronous mock/reference pipeline (Phase 0). Phase 1 replaces this with the
/// threaded coordinator described in spec §3.4; the trait seams stay identical.
pub fn run_pipeline<A, V, E, I, S>(
    mut source: A,
    mut segmenter: V,
    mut asr: impl AsrEngine,
    mut embedder: E,
    mut identifier: I,
    sink: &mut S,
) -> Result<()>
where
    A: AudioSource,
    V: VadSegmenter,
    E: EmbeddingExtractor,
    I: Identifier,
    S: CaptionSink,
{
    let mut utts: Vec<Utterance> = Vec::new();
    source.start(&mut |frame| utts.extend(segmenter.push(frame)))?;
    utts.extend(segmenter.flush());

    for utt in utts {
        let tr = asr.transcribe(&utt)?;
        let emb = embedder.embed(&utt)?;
        let speaker = identifier.identify(&emb);
        sink.emit(CaptionEvent {
            utt_id: utt.id,
            text: tr.text,
            lang: tr.lang,
            speaker,
            t0_ms: utt.t0_ms,
            t1_ms: utt.t1_ms,
            is_final: tr.is_final,
        });
    }
    Ok(())
}
```

- [ ] **Step 5: Implement `src/cli.rs` and `src/main.rs`**

```rust
// src/cli.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hearable", version, about = "Real-time captioning + speaker ID for Deaf/HoH users")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Print resolved settings (Phase 0 placeholder for the overlay run loop).
    Config,
}
```
```rust
// src/main.rs
mod cli;
pub mod pipeline;

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
```

- [ ] **Step 6: Run the integration test + whole workspace, then commit**

Run: `cargo test -p hearable` → Expected: PASS.
Run: `cargo test --workspace` → Expected: PASS (all crates).
Run: `cargo clippy --workspace --all-targets -- -D warnings` → Expected: clean.
Run: `cargo fmt --all` then `cargo fmt --all --check` → Expected: clean.
```bash
git add crates/hearable
git commit -m "feat(bin): mock end-to-end pipeline + CLI skeleton + integration test"
```

---

## Phase 0 self-review (against the spec)

- **Spec §3.1 workspace / §3.2 traits** → Tasks 0.1–0.3 (note: `hearable-ui` deferred to Phase 1; `hearable-core` holds traits to break the cycle — documented above).
- **§4.4 speaker clustering/identification** → Task 0.4 (thresholds 0.70/0.62, EMA α=0.05, promotion).
- **§4.5 store / profiles = embeddings only** → Task 0.5.
- **§4.3 AsrEngine seam** → Task 0.6 (mock; real engines Phase 1).
- **§4.1/§4.2 audio seam** → Task 0.7 (WAV source + segmenter stand-in; real cpal+Silero Phase 1).
- **§8 testing rings 1–2** → unit + proptest (0.4), golden-style signal tests (0.7), integration (0.8). Rings 3–4 (contract macro, `real-asr` latency) land with real engines in Phase 1.
- **§1.6 no mic/GPU/model in default tests** → satisfied; all Phase 0 code is pure-Rust + bundled SQLite.

No placeholders in code (the `Command::Config` printout is a real, working command, explicitly the Phase-0 stand-in for the overlay run loop). Types are consistent across tasks (`Identifier`, `SpeakerLabel`, `Embedding`, `ProfileStore::upsert_profile(name, &[Embedding])`).

---

## Roadmap: Phases 1–5 (separate plans, written after Phase 0 lands)

These are **not** detailed here because they depend on facts that can only be confirmed by adding the real crates (spec §10) — e.g. the official `sherpa-onnx` Rust API method names. Writing "complete code in every step" for them now would mean inventing unverified API calls. Each becomes its own plan once Phase 0 exists:

- **Phase 1** — `hearable-audio` real path (`cpal` capture + `rtrb` + `rubato` resample + sherpa Silero VAD), `hearable-asr` `SenseVoiceEngine`, `hearable-speaker` sherpa `EmbeddingExtractor` (ERes2NetV2), `hearable-ui` egui overlay (macOS/X11), threaded coordinator (§3.4), first-run onboarding + model downloader. Adds testing rings 3–4.
- **Phase 2** — whisper-tiny LID routing → SenseVoice/whisper-small; opt-in Zipformer streaming engine.
- **Phase 3** — Wayland SCTK layer-shell overlay + GNOME fallback; Homebrew formula + `.app`, `cargo-deb`, AppImage.
- **Phase 4** — `CloudAsrEngine` (Deepgram) behind feature flag + opt-in consent; local resolution of cloud speaker indices.
- **Phase 5** — encrypted persistent transcript history (ChaCha20-Poly1305 + keyring), search, export, retention controls.
