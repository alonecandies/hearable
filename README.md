# hearable

[![CI](https://github.com/alonecandies/hearable/actions/workflows/ci.yml/badge.svg)](https://github.com/alonecandies/hearable/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Real-time, on-device captioning and speaker identification for Deaf and hard-of-hearing
users. `hearable` continuously listens to your microphone, transcribes speech as it happens,
separates who is speaking, and lets you name voices so recognized people are labelled
automatically — shown in a floating, always-on-top caption overlay.

> **Status: early development.** The architecture, full design spec, and a tested
> pure-logic foundation (Phase 0) are in place. The on-device ML, audio capture, and overlay
> UI (Phase 1+) are in progress. See the roadmap below.

## Why

Existing live-captioning tools are mostly cloud-based, single-language, or don't tell you
*who* is speaking. `hearable` is built around three principles:

- **Privacy-first.** Always-on listening records bystanders who never consented, so audio is
  processed **on-device by default** and never written to disk unless you explicitly choose
  to. A cloud mode exists but is strictly opt-in.
- **Identity, not just transcription.** The same per-utterance voice embedding both separates
  unknown speakers ("Speaker 1/2/3") and recognizes people you've named — and that
  recognition persists across sessions.
- **Multilingual.** Languages are auto-detected; you don't switch manually.

## How it works

```
mic → capture → VAD → utterance → ┬→ speech-to-text ───────────────┐
                                  └→ voice embedding → identify ────┤
                                                                    ↓
                          caption overlay  ←  { text, speaker, language }
```

Speakers are matched by cosine similarity against saved voice profiles (known people) or
online-clustered into anonymous groups (everyone else). Naming an anonymous speaker promotes
their fingerprint into a saved profile.

## Design & planning docs

- Full design specification: [`docs/superpowers/specs/2026-06-26-hearable-design.md`](docs/superpowers/specs/2026-06-26-hearable-design.md)
- Phase 0 implementation plan: [`docs/superpowers/plans/2026-06-26-hearable-phase0-scaffold.md`](docs/superpowers/plans/2026-06-26-hearable-phase0-scaffold.md)

## Tech stack

All-Rust. On-device ML via the official [`sherpa-onnx`](https://crates.io/crates/sherpa-onnx)
crate (VAD, multilingual ASR, speaker embeddings); caption overlay in
[`egui`](https://github.com/emilk/egui); local storage in SQLite. Optional cloud ASR
(Deepgram) lives behind a trait. Targets macOS (Apple Silicon) and Linux (X11 + Wayland);
distributed via Homebrew and apt.

## Building

Requires Rust ≥ 1.85.

```bash
cargo test --workspace     # runs with no microphone, GPU, or model download
cargo run -p hearable -- config
```

## Running the live build (developer preview)

```bash
./scripts/fetch-models.sh ./models           # download the on-device models (~300 MB, one-time)
```

**macOS (recommended — proper mic permission):**
```bash
./scripts/bundle-macos.sh                     # builds + assembles target/hearable.app
open target/hearable.app --args run --models "$PWD/models"
# Whisper (better for accents / Vietnamese): add  --engine whisper --language vi
```
Grant Microphone access to *hearable* when prompted. Use an absolute models path with `open`
(its working directory isn't your shell's). Needs a microphone and a display.

**Bare binary (any platform):**
```bash
cargo build --release --features live         # downloads sherpa-onnx native libs at build time
./target/release/hearable run --models ./models
```

## Roadmap

- [x] **Phase 0** — Workspace, trait surface, online speaker clustering/ID, mock pipeline, CI.
- [x] **Phase 1** — full live pipeline: capture → VAD → SenseVoice/Whisper ASR + language-ID →
  ERes2NetV2 embeddings → online clustering → resizable scrollable overlay with click-to-name.
  Runs on macOS today. (Logic covered by 45 native-free tests.)
- [~] **Phase 3** — packaging in progress: macOS `.app`, `cargo-deb` metadata, Homebrew formula
  done; Wayland layer-shell overlay + `.deb`/AppImage builds need a Linux host.
- [ ] Word-by-word streaming mode (lower latency); cloud hybrid mode; encrypted transcript history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build/test instructions, the feature-flag map, and
architecture. Changelog: [CHANGELOG.md](CHANGELOG.md).

## License

MIT © Long Hoang
