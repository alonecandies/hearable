# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses [SemVer](https://semver.org/).

## [0.1.0] - 2026-06-27

First working preview: real-time, on-device captioning with speaker identification on macOS.

### Added
- **Live pipeline** (`hearable run --models <dir>`): microphone → 16 kHz resample → Silero VAD →
  ASR → speaker embedding → online clustering → translucent always-on-top caption window.
- **ASR engines** (on-device, via the official `sherpa-onnx` crate): **SenseVoice** (multilingual
  utterance-chunk, zh/en/ja/ko/yue) and **Whisper** (broad multilingual incl. Vietnamese), plus a
  Whisper language-ID pass. Select at runtime: `--engine whisper --language vi`.
- **Speaker identification**: online leader-clustering with per-utterance embeddings (ERes2NetV2),
  cross-session persistence, and **interactive click-to-name** in the overlay that promotes a
  cluster to a saved profile.
- **Caption overlay**: resizable, scrollable history (500 lines), colorblind-safe speaker colors,
  committed/partial rendering.
- **Threaded coordinator** with a drop-oldest overload policy (an always-on listener never blocks
  capture); panic-safe thread shutdown; surfaced capture/inference errors.
- **Model downloader** (`download` feature): resumable, SHA-256-verified.
- **Privacy**: ephemeral by default; profiles store voice embeddings only, never raw audio.
- **Packaging**: macOS `.app` bundle (`scripts/bundle-macos.sh`), Homebrew formula, `cargo-deb`
  metadata; models fetched at runtime via `scripts/fetch-models.sh`.

### Known limitations
- Single short words are mis-recognized and can split one speaker into several (on-device model
  limits below ~2 s of audio); full sentences work well.
- Word-by-word streaming, Linux (Wayland) overlay, and `.deb`/AppImage builds are not yet shipped.
- The macOS build is an unsigned local preview (no notarization yet).
