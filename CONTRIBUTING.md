# Contributing to hearable

Thanks for your interest! hearable is real-time, on-device captioning and speaker
identification for Deaf and hard-of-hearing users.

## Build & test

Requires Rust ≥ 1.85.

```bash
# Default build: pure-Rust, no microphone / GPU / model downloads. This is what CI runs.
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The default build is deliberately free of native ML/audio/GUI dependencies, so the test
suite is fast and hermetic. The real engines live behind Cargo features:

| Feature | Pulls in | Notes |
|---|---|---|
| `sherpa` | the official `sherpa-onnx` crate | Downloads prebuilt native libs at build time (needs network). VAD + ASR + speaker embeddings. |
| `mic` | `cpal` | Real microphone capture. |
| `ui` | `eframe`/`egui` | The caption overlay window. |
| `download` | `reqwest` + `sha2` | First-run model downloader. |
| `live` | = `sherpa` + `mic` + `ui` | Everything needed for `hearable run`. |

```bash
cargo build --release --features live     # full app (downloads sherpa native libs)
cargo test -p hearable-store --features download
```

## Architecture

A Cargo workspace; `hearable-core` is the dependency leaf (shared types + traits), and the
domain crates implement those traits so they're swappable and testable in isolation.

```
hearable-core      shared types, traits (AsrEngine, VadSegmenter, EmbeddingExtractor,
                   Identifier, ProfileStore, CaptionSink), config, errors
hearable-audio     cpal capture, rubato resampling, Silero VAD, energy segmenter
hearable-asr       SenseVoice / Whisper / streaming engines + a mock
hearable-speaker   online leader-clustering speaker identification
hearable-store     SQLite profiles, model downloader
hearable-ui        caption view-model (pure) + egui overlay
hearable           binary: the threaded coordinator + `run` wiring
```

The full design and phase plans are under [`docs/superpowers/`](docs/superpowers/).

## Conventions

- Keep the **default build native-free** — put anything needing native libs, a mic, a GPU, or
  a model download behind a feature flag, and add logic you can unit-test without them.
- `cargo fmt` + `cargo clippy -D warnings` must pass.
- Speaker profiles store **voice embeddings only — never raw audio** (privacy is core).
- Conventional-commit-style messages (`feat:`, `fix:`, `docs:`, `chore:`).

## Filing issues

Bugs and feature requests welcome via GitHub Issues. For real-run feedback (accuracy, speaker
grouping, latency), include your OS, the engine (`--engine`), and example phrases.
