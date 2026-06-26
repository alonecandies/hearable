# hearable — Design Specification

**Status:** Approved design · 2026-06-26
**Author:** Long Hoang (`long.hoang@krystal.app`) with Claude Code
**License:** MIT

---

## 1. Purpose & Vision

`hearable` is an open-source desktop application for Deaf and hard-of-hearing users. It
continuously listens to the microphone, transcribes speech to text in real time, separates
who is speaking, and lets the user attach names to voices so that recognized people are
labelled automatically — all displayed in a floating, always-on-top caption overlay.

It is **privacy-first and on-device by default**: the audio of bystanders who never consented
to being recorded must never leave the machine unless the user explicitly opts into a cloud
mode. It runs on **macOS (Apple Silicon)** and **Linux (X11 + Wayland)**, and is distributed
through **Homebrew** and **apt**.

### Confirmed product decisions

| Axis | Decision |
|---|---|
| Primary scenario | Always-on ambient listening |
| Processing | Hybrid — on-device default, cloud optional behind a trait |
| Interface | Translucent, always-on-top caption overlay |
| Language | Multilingual with auto language detection |
| Speaker identity | Persistent voice profiles + auto-cluster of unknown speakers |
| Retention | User chooses on first run; **default = ephemeral** |
| Caption mode | **Utterance-chunk by default**, opt-in word-by-word streaming for single-language users |
| Stack | All-Rust · official `sherpa-onnx` crate · `egui`/`eframe` |

### Success criteria

1. With the overlay open, spoken phrases appear as captions within **~0.3–1.0 s** of the
   phrase ending (utterance mode), or word-by-word within **~200 ms** (opt-in streaming mode).
2. Distinct speakers are visually separated; a user can tap an unknown "Speaker N" and name
   them, and that person is **recognized automatically in later sessions**.
3. Multiple languages are transcribed without the user manually switching language.
4. In the default privacy mode, **no raw audio and no transcript text are written to disk**;
   only voice fingerprints of named people persist.
5. Installs cleanly via `brew install` and `apt install`, downloading models on first run.
6. The mock/test code path builds and passes in CI **without a microphone, GPU, or model
   download**.

### Non-goals (v1)

- Transcription of recorded audio files (this is a live tool).
- Translation between languages (we transcribe in the spoken language).
- Mobile platforms or Windows (architecture stays portable, but they are out of scope).
- Frame-level overlapping-speech diarization (we do robust *per-utterance* speaker
  assignment, not continuous overlap separation).

---

## 2. Key technical findings that shaped this design

This design was written after an 11-subsystem research pass with adversarial verification.
Four findings materially changed the architecture and are recorded here so future
contributors understand *why* the obvious choices were not taken.

1. **`sherpa-rs` is dead.** The popular `thewh1teagle/sherpa-rs` binding was **archived and
   deprecated on 2026-06-06**; its README redirects to the **official k2-fsa `sherpa-onnx`
   crate** (crates.io/`sherpa-onnx`, docs.rs/`sherpa-onnx`, ~v1.13). We build on the official
   crate. Its Rust API differs from `sherpa-rs`; exact method names for embedding extraction,
   diarization, and VAD must be confirmed against the official docs during implementation.

2. **Streaming and multilingual auto-detect are mutually exclusive in sherpa-onnx.** Streaming
   transducers (Zipformer/Conformer) are mono- or bi-lingual with **no language ID**.
   Multilingual + auto-detect (SenseVoice, Whisper) runs **offline-chunk only**. Because the
   product requires multilingual auto-detect, the **default path is utterance-chunked offline
   ASR**; word-by-word streaming is an opt-in mode restricted to a single fixed language.

3. **CoreML is not a guaranteed speedup.** On Apple Silicon, the ONNX Runtime CoreML execution
   provider has been measured **slower than CPU** for some sherpa-onnx ASR models
   (k2-fsa/sherpa-onnx issue #2910). We therefore **default to INT8 CPU inference** (the small
   INT8 models are fast enough — SenseVoice INT8 is ~15× faster than Whisper-large) and treat
   CoreML as an experimental, benchmark-gated opt-in.

4. **A forced always-on-top overlay is impossible on GNOME Wayland.** `with_always_on_top()` is
   a hint Mutter ignores, and `winit`/`eframe` do not implement `wlr-layer-shell`. The overlay
   uses `eframe`/`egui` on macOS and X11, a **`smithay-client-toolkit` layer-shell path** for
   wlroots compositors (Sway, Hyprland, KWin), and **degrades to a normal floating window with
   an in-app notice on GNOME Wayland**.

A secondary finding: depending on **both** the `ort` crate and `sherpa-onnx` causes a
dual-ONNX-Runtime symbol/threadpool clash. Resolution: **do not depend on `ort`** — route VAD,
ASR, embeddings, and diarization through the single statically-linked `sherpa-onnx` crate.

---

## 3. Architecture

### 3.1 Workspace layout

A Cargo workspace of seven crates. Each crate has one clear responsibility, communicates
through a small trait surface, and is unit-testable in isolation.

```
hearable/
├── crates/
│   ├── hearable-audio/      # capture, resampling, VAD, utterance segmentation
│   ├── hearable-asr/        # AsrEngine trait + sherpa offline/streaming + cloud + mock
│   ├── hearable-speaker/    # embeddings, online clustering, identification
│   ├── hearable-store/      # SQLite profiles, optional encrypted history, model downloader
│   ├── hearable-core/       # pipeline orchestration, event types, config
│   ├── hearable-ui/         # egui overlay + Wayland layer-shell path
│   └── hearable/            # binary: CLI, first-run onboarding, wiring
├── docs/superpowers/specs/
├── packaging/               # Homebrew formula, cargo-deb metadata, AppImage recipe
└── tests/fixtures/          # WAV clips, tiny models (git-lfs or downloaded in CI)
```

Workspace-wide `rust-version` floor: **1.85** (required by `rubato` 3.0 and recent `cpal`).

### 3.2 Core trait surface

These traits are the seams that make the system testable and the cloud/engine choices
swappable. Concrete signatures are finalized in implementation; shapes shown for intent.

```rust
// hearable-audio
trait AudioSource {                 // real mic, or WAV fixture in tests
    fn start(&mut self, sink: SampleSink) -> Result<()>;
    fn stop(&mut self);
}
trait VadSegmenter {                // emits one Utterance per detected speech span
    fn push(&mut self, frame: &[f32]) -> Vec<Utterance>;
}

// hearable-asr
trait AsrEngine: Send {             // offline-chunk, streaming, cloud, or mock
    fn capabilities(&self) -> AsrCaps;          // { streaming, multilingual, auto_detect }
    fn transcribe(&mut self, utt: &Utterance) -> Result<TranscriptResult>;
}

// hearable-speaker
trait EmbeddingExtractor: Send { fn embed(&mut self, utt: &Utterance) -> Result<Embedding>; }
trait Identifier {                  // clustering + known-profile match in one step
    fn identify(&mut self, e: &Embedding) -> SpeakerLabel;
    fn promote(&mut self, cluster: ClusterId, name: &str) -> Result<()>;
}

// hearable-store
trait ProfileStore {
    fn load_profiles(&self) -> Result<Vec<Profile>>;
    fn upsert_profile(&self, p: &Profile) -> Result<()>;
}

// hearable-core
trait CaptionSink: Send { fn emit(&mut self, ev: CaptionEvent); }   // overlay, or test buffer
```

### 3.3 Data flow & event types

```
cpal callback ──rtrb (wait-free SPSC)──▶ coordinator thread
   └ BufferSize::Fixed; no alloc/lock/println in callback
coordinator ──▶ resample to 16 kHz mono f32 (rubato, zero-alloc) ──▶ VadSegmenter
VadSegmenter ──▶ Utterance{ id, pcm16k, t0, t1 }
   ├─ crossbeam bounded(4), try_send/drop-oldest ─▶ ASR worker  ─▶ TranscriptResult{ text, lang, conf, final }
   └─ crossbeam bounded(4), try_send/drop-oldest ─▶ Embed worker ─▶ Embedding ─▶ Identifier ─▶ SpeakerLabel
merge by utterance id ─▶ CaptionEvent{ utt_id, text, lang, speaker, t0, t1, final } ─▶ CaptionSink (overlay)
```

- `SpeakerLabel = Known { name, score } | Unknown { cluster_id, score }`.
- In **utterance mode**, one `final` `TranscriptResult` per utterance. In **streaming mode**,
  the engine emits interim non-`final` results (word-growth) plus a `final` at end-of-utterance.
- The overlay can render a caption before its speaker label arrives (label fills in a beat
  later); merge keys on `utt_id`.

### 3.4 Concurrency model

- **cpal callback**: real-time safe — writes raw samples to an `rtrb` ring buffer (≥2 s
  capacity), sets atomic flags on error, never allocates/locks/logs.
- **Coordinator (std::thread)**: drains the ring, resamples, runs VAD, fans utterances out to
  ML workers over bounded channels with `try_send` + drop-oldest (an always-on stream must
  never block; dropped utterances increment a counter surfaced unobtrusively in the UI).
- **ML workers (std::thread)**: blocking ONNX inference (ASR, embedding) on a small dedicated
  pool — kept off the audio and UI threads.
- **Cloud adapter**: a single off-main **single-threaded tokio runtime** on its own thread
  (async websocket I/O only); used only when cloud mode is enabled.
- **UI**: `egui`/`eframe` owns the OS main thread/event loop; receives `CaptionEvent`s over a
  channel and renders.

### 3.5 Crate dependency map (no `ort`)

```
hearable (bin)
 ├─ hearable-core ── hearable-audio, hearable-asr, hearable-speaker, hearable-store
 ├─ hearable-ui ──── hearable-core (event types)
 └─ hearable-store
sherpa-onnx (official, static-linked)  ← used by hearable-audio (VAD), hearable-asr, hearable-speaker
```

---

## 4. Subsystem specifications

### 4.1 Audio capture (`hearable-audio`)

- `cpal` (pin the current release at implementation time; **verify `Stream: Send`** — it is
  `Send + Sync` in the 0.18 line, so the supervisor thread may own/rebuild streams directly).
- Capture at the device's native rate/format; **downmix + resample to 16 kHz mono f32 on the
  worker thread** (never in the callback) via `rubato` 3.0 (`Async::new_sinc`,
  `FixedAsync::Input`, `process_into_buffer` for zero-alloc; pulls in `audioadapter` +
  `audioadapter-buffers`, both `^3.0`).
- `rtrb` 0.3.4 wait-free SPSC for callback→coordinator handoff; on overflow, count and drop
  (never overwrite/block).
- Set `BufferSize::Fixed` (e.g. 512) — ALSA defaults can yield 250–500 ms buffers.
- **Linux PipeWire**: rely on PipeWire's ALSA/Pulse compatibility layer; do **not** use cpal's
  `pipewire` feature (git-master only, breaks `cargo publish`/packaging). Use the `default`
  device to stay on the compat path.
- **Device hot-plug**: cpal signals revocation via the stream error callback; set an atomic
  flag and rebuild the stream on the supervisor thread. There is no proactive "device added"
  event — refresh device lists on user action.
- **macOS TCC**: requires `NSMicrophoneUsageDescription` in an embedded `Info.plist` **and**
  the `com.apple.security.device.audio-input` entitlement on a signed binary, else silent
  denial. See §7 for how the Homebrew formula provides this.

### 4.2 VAD & utterance segmentation (`hearable-audio`)

- Use **sherpa-onnx's built-in Silero v5 VAD** (avoids a second ONNX Runtime). Config exposes
  `threshold`, `min_speech_duration`, `min_silence_duration`, `max_speech_duration`,
  `window_size`, `neg_threshold` — **but no padding parameter**, so we add a small pre/post
  pad (~150–250 ms) ourselves when slicing the utterance PCM, so ASR and the speaker embedder
  both get a clean segment.
- Tunables tuned for an accessibility false-positive budget; **benchmark TEN-VAD** (also
  bundled by sherpa-onnx) against Silero for fewer false triggers in noisy rooms.
- Output: `Utterance { id, pcm16k, t0, t1 }`. Enforce a `max_speech_duration` so a
  never-ending talker still yields periodic utterances.

### 4.3 ASR (`hearable-asr`)

`AsrEngine` implementations:

1. **`SenseVoiceEngine` (default, multilingual, offline-chunk).** Model
   `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-…-int8` (~250 MB; zh/en/ja/ko/yue strong, broader
   superset usable). Non-autoregressive, ~15× faster than Whisper-large; runs per utterance on
   **INT8 CPU**. Carries built-in language tags.
2. **Language-ID routing.** A tiny **whisper-tiny** language-probability pass (RTF ~0.04)
   detects the spoken language per utterance. If it falls **outside** SenseVoice's strong set,
   route to a **whisper-small** fallback engine (~47 MB) for that utterance.
3. **`ZipformerStreamingEngine` (opt-in, single language).** Streaming transducer (~200 ms
   first word, ~25 MB INT8) for users who fix one language and want word-by-word captions.
   `capabilities().streaming == true`; pipeline emits interim results.
4. **`CloudAsrEngine`** (see §4.6).
5. **`MockAsrEngine`** (tests): deterministic text from a fixture map; no model.

`TranscriptResult { text, lang, confidence, final }`. The active engine is chosen by config;
the pipeline is identical regardless. A small default-model footprint matters for first-run
download time — see §4.5 for the default/upgrade split.

### 4.4 Speaker embeddings, clustering & identification (`hearable-speaker`)

- **Embedding model:** **ERes2NetV2** (3D-Speaker), artifact
  `3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx` (~71 MB) — *not* the similarly-named
  ERes2Net "base" (v1, 39.6 MB). EER ~0.6 % full-duration, ~1.5 % at 2 s. Read the embedding
  dimension from the loaded model rather than hard-coding. Alternatives: WeSpeaker ResNet34-LM
  (256-d, 26.5 MB, English-only) as a lightweight fallback; NeMo TitaNet (large 101 MB / small
  40 MB ONNX) exists if needed.
- **Online clustering (`Identifier`):** cosine-similarity **leader clustering**, O(K) per
  utterance, no batch re-clustering. New cluster when max similarity < threshold. Thresholds:
  **~0.70 normal**, **~0.62 for short (<~1.5 s) utterances**. Update the matched centroid with
  an EMA (α ≈ 0.05) to track within-session vocal drift.
- **Identification vs known profiles:** the same per-utterance embedding is compared (cosine)
  against stored named profiles first. A match above threshold ⇒ `Known { name }`; otherwise it
  feeds the anonymous clusters ⇒ `Unknown { cluster_id }`.
- **Promotion:** naming an `Unknown` cluster persists its centroid (and a few recent member
  embeddings) as a `Profile`, so the person is recognized in future sessions.
- **API note:** sherpa-onnx's `EmbeddingManager::search()` returns the matched name only (no
  score). To drive thresholds and a confidence indicator, **compute cosine similarity
  ourselves** over the loaded profile vectors (a handful of people — brute force is ample, no
  vector DB).
- The offline `Diarize`/pyannote-segmentation path is **not** used in the realtime pipeline;
  it is reserved for an optional post-hoc "re-label this session" feature.

### 4.5 Storage, config & model manager (`hearable-store`)

- **`rusqlite` 0.40** with the `bundled` feature (vendors SQLite 3.53.x — no system lib).
  `rusqlite_migration` for schema versioning at startup.
- **Schema:** `profiles(id, name, embeddings BLOB, created, updated)`;
  `settings(key, value)`; `transcript_history(id, ts, speaker, lang, text_enc BLOB)` (only used
  in the opt-in persistent-history mode).
- **Profiles store voice embeddings, never raw audio.**
- **Encryption (history mode only):** app-level **ChaCha20-Poly1305** column encryption of
  `text_enc`, key stored in the OS keychain via `keyring` 3 (Keychain on macOS, Secret Service
  on Linux) — chosen over SQLCipher to stay pure-Rust and avoid an OpenSSL link.
- **Paths:** `directories` 6 (`~/Library/Application Support/hearable` on macOS,
  `$XDG_DATA_HOME/hearable` on Linux). `None` return ⇒ headless/no-home ⇒ degrade gracefully.
- **Config:** `figment` + TOML (compiled defaults → `config.toml` → `HEARABLE_*` env overrides),
  serde-derived settings struct.
- **Model manager:** hand-rolled `reqwest` 0.12 (`stream`, `rustls-tls`) streaming download with
  HTTP Range resume, `sha2` SHA-256 verification, atomic `.part`→final rename. Models live under
  the data dir; downloaded at **first run** (never bundled).
  - **Default download (multilingual — matches the default caption mode):** SenseVoice INT8
    (~250 MB) + Silero VAD (~2 MB) + ERes2NetV2 (~71 MB) + whisper-tiny LID (~40 MB).
  - **Added when the user enables streaming mode:** streaming Zipformer bilingual (~80 MB).
  - **Added on first need:** whisper-small rare-language fallback (~47 MB).
  - *Open question #6 revisits a "lite first run" that ships bilingual streaming first and
    defers the SenseVoice download — trading initial multilingual coverage for a faster, smaller
    install.* Either way, onboarding explains the size/latency tradeoff.

### 4.6 Cloud ASR adapter (`hearable-asr`, opt-in)

- Behind a feature flag and a runtime opt-in. **Primary: Deepgram Nova-3.**
  On-the-wire selector is **`model=nova-3` + `language=multi`** (there is no `nova-3-multilingual`
  model id), with **`diarize_model`** (do **not** also send `diarize=true` — sending both is
  rejected) and **`mip_opt_out=true`** so audio is not used for model training.
- Transport: `tokio-tungstenite` 0.29 with `rustls-tls-webpki-roots` (pure-Rust TLS, no OpenSSL).
- **Identity stays local even in cloud mode:** Deepgram receives only PCM and returns *ephemeral*
  speaker indices; the local ERes2NetV2 + `Identifier` resolves those to named profiles. Cloud
  never sees or stores names/embeddings.
- **Cost (with privacy opt-out):** `mip_opt_out=true` forfeits Deepgram's ~50 % Model
  Improvement Program discount, so realistic cost is **~$0.70/hr** of audio (Nova-3 multilingual
  streaming + diarization). Surfaced in the cloud-mode UI. **Soniox (~$0.12/hr)** is a documented
  cheaper secondary adapter (hand-rolled, no Rust SDK).
- **Consent UX:** enabling cloud requires an explicit confirmation explaining that audio leaves
  the device; the recording indicator changes to a distinct "cloud" state.

### 4.7 Caption overlay UI (`hearable-ui`)

**Rendering stack:** `eframe`/`egui` with the **glow (OpenGL)** backend.

- **macOS & Linux/X11:** `ViewportBuilder` with `with_transparent(true)`,
  `with_always_on_top(true)`, `with_mouse_passthrough(true)`; `clear_color → TRANSPARENT`. This
  is the full overlay.
- **Linux/Wayland — wlroots (Sway, Hyprland, KWin):** a **`smithay-client-toolkit`
  `zwlr_layer_shell_v1`** path (bypassing winit) gives a true always-on-top overlay layer.
- **Linux/Wayland — GNOME/Mutter:** no protocol for forced always-on-top; **degrade to a normal
  floating window** and show a one-time in-app notice explaining the limitation.
- Compositor/path is **detected at runtime**.

**Caption presentation (accessibility-validated):**

- Rolling **"append-within-stable-line"**, **3 visible lines**, no smooth-scroll animation
  (snap on line break); older lines fade to ~40 % opacity.
- **Committed + tail-partial** split: committed text opacity 1.0, unstable partial tail ~0.65,
  with a 150–250 ms stability window to avoid flicker from ASR re-scoring.
- **Speaker differentiation that does not rely on color alone (WCAG):** a **4 px left-edge accent
  bar** + a name/Speaker-N tag on turn change, colored from the **Okabe-Ito colorblind-safe
  palette** (cap at 5 active speaker slots; avoid red/vermillion; amber instead of yellow in
  light mode).
- Background `rgba(0,0,0,0.82)`, white text (~14.7:1 contrast, WCAG AAA), 8 px corner radius,
  semi-opaque so the user keeps spatial awareness.
- **Fonts (multilingual):** egui ships Latin-only, so **bundle at compile time**: Noto Sans
  (Latin/Greek/Cyrillic/Arabic/Hebrew/Devanagari) + subsetted Noto Sans SC (CJK) + Noto Emoji.
  Default ~26 px, weight 500, +0.01em letter-spacing; line length ~42 chars (EBU R95 / SMPTE).
- **Interaction:** clicking a `Speaker N` tag opens an inline name field (promotes to a profile);
  a tray/menu item and a global hotkey toggle **pause/mute**; a always-visible **recording
  indicator** (distinct local vs cloud states).

### 4.8 First-run onboarding (`hearable` binary)

1. Explain what the app does and the privacy implications of always-on recording.
2. **Retention choice** (default ephemeral): *Ephemeral* (rolling in-memory ~10 min, nothing on
   disk) vs *Persistent local history* (encrypted, searchable).
3. Trigger the **microphone permission** flow (macOS TCC dialog; on Linux, verify device access).
4. **Model download**: default small set, with the explicit upgrade-to-multilingual option.
5. Land in the overlay.

---

## 5. Error handling

The audio and UI threads must never panic. Strategy per failure:

| Failure | Handling |
|---|---|
| Mic permission denied | Clear onboarding message with OS-specific fix steps; app stays open, overlay shows "no microphone access". |
| Model download fails / checksum mismatch | Resume on retry; never use a partially-written or unverified model; show progress + retry button. |
| Audio device unplugged / default changed | Error callback sets atomic flag; supervisor rebuilds the stream; brief "reconnecting" indicator. |
| Ring-buffer overflow | Count + drop oldest; surface a subtle "audio dropping" indicator if sustained. |
| ASR / embedding inference error | Skip that utterance, log, continue; repeated failures surface a non-blocking warning. |
| Worker channel full (overload) | `try_send` drop-oldest; never block the coordinator. |
| Cloud disconnect | Auto-reconnect with backoff; on repeated failure, fall back to the local engine and notify. |
| Wayland/GNOME always-on-top unsupported | Floating-window fallback + one-time notice (not an error). |
| No home dir / headless | `directories` returns `None`; run with in-memory-only state, log a warning. |

---

## 6. Privacy & security

- **Default ephemeral:** no raw audio and no transcript text written to disk; transcript is a
  rolling in-memory window (~10 min). Only **voice fingerprints of named people** persist.
- **Persistent history** is opt-in and **encrypted at rest** (ChaCha20-Poly1305, key in OS
  keychain).
- **Cloud is strictly opt-in**, with `mip_opt_out=true`, an explicit consent step, and a
  distinct recording indicator; identity/embeddings never leave the device.
- **Always-visible recording indicator** and an easy global pause/mute.
- Profiles and history are local files under the user's data dir; documented for easy deletion.
- The README will note the ethical/legal reality that always-on recording captures
  non-consenting bystanders, and that local laws on recording apply.

---

## 7. Packaging & distribution

- **Static-link** ONNX Runtime + sherpa-onnx into the release binary (`sherpa-onnx` `static`
  feature / `SHERPA_ONNX_LIB_DIR` / `ORT_LIB_PATH`) — no system `libonnxruntime` dependency.
- **macOS — Homebrew formula** (source build in a third-party tap): no Apple Developer Program
  cost and no notarization required for a formula build. A **post-install step assembles a
  minimal `.app` bundle** carrying `Info.plist` (`NSMicrophoneUsageDescription`) so the TCC
  microphone dialog appears and captions work. (CLI-only binaries can instead embed the plist
  via the linker `-sectcreate __TEXT __info_plist`, but the `.app` is the reliable path for the
  GUI + TCC.) Note: Homebrew 6.0's Linux Bubblewrap sandbox also covers post-install, so any
  network/model fetch happens at **first app run**, not in post-install.
- **Linux — `cargo-deb`** produces a `.deb` for a self-hosted apt channel; declare runtime deps
  (`libasound2`, `libxcb1`, `libwayland-client0`, etc.). **AppImage** (via `cargo-appimage`) is
  built in the same CI job as a portable artifact for non-Debian distros.
- **Models are downloaded at first run**, never shipped in the package — keeps artifacts small.
- CI builds release binaries per platform, runs the default (no-model) test suite, and produces
  the `.deb` + AppImage and the Homebrew formula bottle inputs.

---

## 8. Testing strategy

Four concentric rings; the default `cargo test` needs **no microphone, GPU, or model download**.

1. **Unit (pure logic, no models):** `Identifier` clustering/ID over precomputed embeddings
   (`proptest` 1.11 for invariants — cosine symmetry, cluster idempotence); config parsing;
   event merge logic; store CRUD on an in-memory SQLite. ML boundaries mocked with `mockall`
   0.14 (`AsrEngine`, `EmbeddingExtractor`, `AudioSource`).
2. **Golden / snapshot:** `insta` 1.48 (+ float `rounded_redaction`) over `hound`-loaded WAV
   fixtures for **VAD segmentation** boundaries and **ASR text**, tolerating ASR nondeterminism.
3. **Contract tests:** a shared `rstest`-parameterized suite that **every `AsrEngine`** (local
   offline, streaming, cloud, mock) must satisfy — same fixtures, same invariants.
4. **Integration / latency (feature-gated `--features real-asr`):** runs tiny cached models
   (Silero ~1.8 MB, whisper-tiny.en ~40 MB) on CPU in CI; `criterion` measures per-stage and
   end-to-end latency against the §1 budgets. Pin both the ONNX Runtime and sherpa-onnx native
   versions in `Cargo.lock` for cross-arch determinism.

- **`AudioSource` trait injection** lets a `WavAudioSource` replay fixtures deterministically;
  all live `cpal` code sits behind a feature flag CI never enables.
- **Headless UI:** `egui_kittest` (AccessKit semantic queries + software rasterizer) snapshots
  the overlay; use one fixed software rasterizer (lavapipe) as the canonical baseline.

---

## 9. Implementation phases

All phases are part of this one spec; built in order, each leaving the test suite green.

- **Phase 0 — Scaffold.** Workspace, seven crates, the trait surface, event types, `MockAsr`
  and `WavAudioSource`, config, CI with the no-model test suite. *Exit: `cargo test` green.*
- **Phase 1 — Core pipeline (macOS + X11).** cpal capture → resample → sherpa Silero VAD →
  default ASR → ERes2NetV2 embedding → online clustering + tap-to-name → SQLite profiles →
  egui overlay (committed+partial, speaker colors, accent bars) → first-run onboarding
  (retention choice, mic permission, model download). *Exit: live captions + persistent naming
  on macOS.*
- **Phase 2 — Multilingual + streaming.** whisper-tiny LID routing → SenseVoice / whisper-small
  fallback; opt-in single-language Zipformer streaming mode. *Exit: auto-detect multilingual
  captions; streaming toggle works.*
- **Phase 3 — Wayland + packaging.** SCTK `wlr-layer-shell` overlay + GNOME fallback; Homebrew
  formula (+`.app`), `cargo-deb`, AppImage; first-run model download hardened. *Exit:
  `brew install` and `apt install` produce a working app on the target platforms.*
- **Phase 4 — Cloud hybrid.** `CloudAsrEngine` (Deepgram) behind the trait + feature flag;
  opt-in consent flow; local identity resolution of cloud speaker indices. *Exit: cloud mode
  transcribes while identity stays local.*
- **Phase 5 — Persistent history.** Encrypted `transcript_history`, search, export, retention
  controls. *Exit: the non-ephemeral retention mode is fully usable.*

---

## 10. Open questions to resolve during implementation

1. Exact official `sherpa-onnx` Rust API names for VAD, ASR (offline + streaming), embedding
   extraction, and language ID (the API differs from the deprecated `sherpa-rs`).
2. Final `cpal` version pin and confirmation of `Stream: Send` on that version.
3. Whether the official `sherpa-onnx` prebuilt static archive is CoreML-enabled, and per-model
   CoreML-vs-CPU benchmarks on Apple Silicon (default CPU until proven faster).
4. Silero vs TEN-VAD false-positive comparison in noisy rooms; final VAD tunables.
5. Clustering thresholds (0.70 / 0.62) validated against real multi-speaker recordings.
6. Whether to ship Zipformer-bilingual or SenseVoice as the first-run default given download
   size vs the multilingual promise (current plan: bilingual default + one-click upgrade).
```

