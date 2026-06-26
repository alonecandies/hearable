# hearable Phase 1 (Real Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the Phase 0 stand-ins with the real on-device pipeline — microphone capture, Silero VAD, multilingual SenseVoice ASR, ERes2NetV2 speaker embeddings, and an egui caption overlay — on macOS/X11, wired into a `hearable run` command with first-run onboarding and a model downloader.

**Architecture:** Native-dependent code (sherpa-onnx ML, cpal mic, egui UI) sits behind Cargo features (`sherpa`, `mic`, `ui`) so the **default build and CI stay native-free and green** (no microphone, GPU, or model download). The trait seams from Phase 0 are unchanged; Phase 1 supplies real implementations of the same traits.

**Tech Stack:** `sherpa-onnx` 1.13.3 (static, downloads prebuilt libs), `cpal` 0.18, `rubato` 3.0, `eframe`/`egui` 0.32, `reqwest` 0.13 + `sha2` 0.11, `directories` 6.

## Global Constraints

- Default `cargo test --workspace` must stay green with no native deps (verified: 29 tests).
- All native code behind features: `sherpa` (ML), `mic` (cpal), `ui` (egui), `download` (model fetch).
- Use the official `sherpa-onnx` crate API (see Appendix) — NOT the deprecated `sherpa-rs`.
- On-device default; CPU provider (CoreML not reliably faster — spec §2).

---

## Status (2026-06-26)

**Done and verified this phase:**
- [x] **Threaded coordinator** (`hearable::coordinator::run_threaded`) — capture/VAD + inference threads, bounded **drop-oldest** queue. Unit + threaded integration tests (happy path + overload conservation).
- [x] **`Resampler16k`** (rubato) — arbitrary rate → 16 kHz mono, 16 kHz identity fast-path. Tested.
- [x] **`SenseVoiceEngine`** (`AsrEngine`, behind `sherpa`) — utterance-chunk multilingual ASR. Compile+link verified.
- [x] **`SileroVad`** (`VadSegmenter`, behind `sherpa`) — 512-window segmenter, drop-in for `EnergySegmenter`. Compile+link verified.
- [x] **`SherpaEmbeddingExtractor`** (`EmbeddingExtractor`, behind `sherpa`) — ERes2NetV2 embeddings. Compile+link verified.
- [x] **Feature architecture** — `sherpa`/`mic` flags; default build native-free; binary links with `--features sherpa`.

**Also done since:**
- [x] **Task 1.1 `MicAudioSource`** (cpal, `mic`) — realtime callback → rtrb → resample; compile-verified.
- [x] **Task 1.2 language-ID pass** (`sherpa`) — `LanguageId` (Whisper SLID); SenseVoice fills `lang`; compile+link-verified.
- [x] **Task 1.4 caption overlay** (`hearable-ui`, `ui`) — testable `CaptionView` + egui rendering; compile-verified.
- [x] **Task 1.5 model downloader** (`hearable-store`, `download`) — resumable + SHA-256; tested vs a local server.
- [x] **Task 1.6 `hearable run` wiring** (`live`) — full chain compile+link-verified end-to-end.
- [x] **Interactive click-to-name** — shared `Arc<Mutex<Identifier>>`, `UiCommand` channel, promote + persist to SQLite; compile+link-verified.
- [x] **First-run settings persistence** — `Settings::save_to` + first-run write (tested).

**Genuinely remaining (larger or hardware/asset-dependent):**
- **Streaming word-by-word mode** — needs a streaming-aware *pipeline path* (feed sub-utterance chunks, emit interim `CaptionEvent`s), not just a streaming model behind the utterance trait. Real feature; own design.
- **Auto-download onboarding** — the downloader exists; needs a model registry with **pinned SHA-256s** (compute by downloading the real release assets once) + an interactive retention prompt.
- **Run-on-hardware validation** — the live path is compile+link-verified but unrun (needs mic + display + models on a real machine).
- Wayland layer-shell overlay + packaging are **Phase 3**.

**Original task list (for reference):**

### Task 1.1: `MicAudioSource` (cpal, behind `mic`) — DONE
- Create `crates/hearable-audio/src/mic_source.rs`: implement `AudioSource` over `cpal`. Pick the default input device + config; in the realtime callback, push raw samples to an `rtrb` ring buffer (no alloc/lock/log); a drain method downmixes to mono and feeds `Resampler16k`, calling `on_frame` with 16 kHz frames. Register an error callback that sets an atomic "rebuild" flag for device-change/XRUN.
- Compile-check: `cargo check -p hearable-audio --features mic`. Runtime needs a real mic (manual/local only).
- Note: cpal 0.18 `Stream` is `Send`; `BufferSize::Fixed`; verify `NSMicrophoneUsageDescription` path on macOS (Task 1.6).

### Task 1.2: Spoken-language-ID pass (behind `sherpa`)
- Create `crates/hearable-asr/src/lang_id.rs`: wrap `SpokenLanguageIdentification` (Whisper encoder/decoder) → `lang: String`. `SenseVoiceEngine` optionally runs it (or the coordinator does) to fill `TranscriptResult.lang`, since SenseVoice's result omits language. Route rare languages to a `whisper-small` offline recognizer.

### Task 1.3: Streaming Zipformer engine (opt-in, behind `sherpa`)
- Create `crates/hearable-asr/src/streaming.rs`: `ZipformerStreamingEngine` over `OnlineRecognizer` (encoder/decoder/joiner) for the single-language word-by-word mode. `capabilities().streaming = true`; emit interim non-final + final results with endpointing.

### Task 1.4: Caption overlay (`hearable-ui`, behind `ui`)
- Create crate `hearable-ui`: `eframe`/`egui` (glow) overlay implementing `CaptionSink` via a channel. Translucent, always-on-top, mouse-passthrough `ViewportBuilder` (macOS/X11). Rolling 3-line display, committed+tail-partial opacity, Okabe-Ito speaker colors + 4px accent bar + name tag, `rgba(0,0,0,0.82)`/white, bundled Noto Sans + Noto Sans SC + Noto Emoji. Click a `Speaker N` tag → inline name field → `Identifier::promote` + `ProfileStore::upsert_profile`.
- Compile-check on macOS; headless UI snapshot tests via `egui_kittest` where feasible. (Wayland layer-shell is Phase 3.)

### Task 1.5: Model downloader (`hearable-store`, behind `download`)
- Create `crates/hearable-store/src/model_manager.rs`: `reqwest` 0.13 streaming download with HTTP Range resume + `sha2` SHA-256 verify + atomic `.part`→final rename, into the `directories` 6 data dir. A registry of model {url, sha256, filename, size}. Testable against a local `tiny_http`/`hyper` server serving a known blob (no GitHub).

### Task 1.6: First-run onboarding + `hearable run`
- In the binary: a `run` subcommand that loads `Settings`, ensures models (download if missing), builds the engines (sherpa) + `MicAudioSource` (mic), spawns `run_threaded`, and feeds the egui overlay. First-run flow: privacy/retention choice → mic permission (macOS TCC: embed `Info.plist` `NSMicrophoneUsageDescription`) → model download → overlay. Behind `sherpa`+`mic`+`ui`.

---

## Appendix: verified `sherpa-onnx` 1.13.3 Rust API (primary-source confirmed)

> Constructors are `create(...) -> Option<Self>` (not `new`); recognizer/stream methods take `&self`/`&stream`. The crate links **static by default** and `build.rs` **downloads prebuilt libs from GitHub releases** (no cmake; needs network unless `SHERPA_ONNX_LIB_DIR`/`SHERPA_ONNX_ARCHIVE_DIR` is set). Features: only `static` (default) and `shared`.

- **VAD:** `VoiceActivityDetector::create(&VadModelConfig, buffer_size_secs: f32) -> Option`; `accept_waveform(&[f32])` (no rate arg); `front() -> Option<SpeechSegment>`, `pop()`, `flush()`, `is_empty()`. `SpeechSegment::{start() -> i32, n() -> i32, samples() -> &[f32]}` (no duration getter). `VadModelConfig{ silero_vad, ten_vad, sample_rate, num_threads, provider, debug }`; `SileroVadModelConfig{ model, threshold, min_silence_duration, min_speech_duration, window_size, max_speech_duration }`. Ten VAD selectable via `ten_vad`.
- **Offline ASR (SenseVoice):** `OfflineRecognizer::create(&OfflineRecognizerConfig) -> Option`; `create_stream() -> OfflineStream`; `stream.accept_waveform(rate: i32, &[f32])`; `recognizer.decode(&stream)`; `stream.get_result() -> Option<OfflineRecognizerResult{ text, tokens, timestamps, durations }>` — **no `lang` field**. Config: `config.model_config.sense_voice = OfflineSenseVoiceModelConfig{ model, language ("auto"|...), use_itn }`, `config.model_config.{tokens, provider, num_threads}`.
- **Streaming ASR:** `OnlineRecognizer::create(&OnlineRecognizerConfig)`; `create_stream() -> OnlineStream`; `is_ready(&stream)`, `decode(&stream)`, `get_result(&stream) -> Option<RecognizerResult{ text, tokens, is_final, ... }>`, `is_endpoint(&stream)`, `reset(&stream)`; `stream.accept_waveform(rate, &[f32])`, `stream.input_finished()`. `OnlineTransducerModelConfig{ encoder, decoder, joiner }`.
- **Speaker embedding:** `SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig{ model, num_threads, debug, provider })`; `create_stream() -> Option<OnlineStream>`; `dim() -> i32`; `compute(&stream) -> Option<Vec<f32>>`. Audio via `stream.accept_waveform(rate, &[f32])` + `stream.input_finished()`.
- **Speaker manager:** `SpeakerEmbeddingManager::create(dim: i32)`; `add(name, &[f32]) -> bool`; `search(&[f32], threshold) -> Option<String>` (**name only**); `get_best_matches(&[f32], threshold, n) -> Vec<SpeakerEmbeddingMatch{ name, score }>` (the only score-bearing call); `verify(name, &[f32], threshold) -> bool`.
- **Language ID:** `SpokenLanguageIdentification::create(&config{ whisper{ encoder, decoder, tail_paddings }, num_threads, provider, debug })`; `create_stream() -> OfflineStream`; `compute(&stream) -> Option<SpokenLanguageIdentificationResult{ lang: String }>`.
- **Models (GitHub release assets):** SenseVoice int8 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` (~228 MB onnx); streaming Zipformer `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` (~488 MiB); `silero_vad.onnx`; ERes2NetV2 `3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx` (~68 MiB). URL: `https://github.com/k2-fsa/sherpa-onnx/releases/download/<tag>/<file>`.
- **Cargo:** `sherpa-onnx = "1.13.3"` (static+download default), or `default-features=false` + `SHERPA_ONNX_LIB_DIR` for offline/CI.
