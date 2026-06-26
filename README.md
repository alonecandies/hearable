# hearable

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

## Roadmap

- [x] **Phase 0** — Workspace, trait surface, online speaker clustering/identification,
  profile store, mock pipeline, CI.
- [~] **Phase 1** (vertical slice wired) — threaded coordinator, resampler, model downloader,
  and the caption-overlay view-model are **built & tested**; Silero VAD, SenseVoice ASR,
  ERes2NetV2 embeddings, the `cpal` mic source, the egui overlay, and the `hearable run`
  wiring are **done and compile+link-verified** behind the `live` feature. Remaining
  refinements: interactive click-to-name, language-ID pass, opt-in streaming mode, and
  first-run onboarding with automatic model download.

### Running the live build (developer preview)

```bash
./scripts/fetch-models.sh ./models          # download the on-device models (~300 MB)
```

**macOS (recommended — proper mic permission):**
```bash
./scripts/bundle-macos.sh                    # builds + assembles a signed target/hearable.app
open target/hearable.app --args run --models ./models
```

**Any platform (bare binary):**
```bash
cargo build --release --features live        # downloads sherpa-onnx native libs at build time
./target/release/hearable run --models ./models
```
On macOS the bare binary makes *Terminal* request the mic; the `.app` bundle requests it as
"hearable" instead. Needs a microphone and a display.
- [ ] **Phase 2** — Language-ID routing + opt-in word-by-word streaming mode.
- [ ] **Phase 3** — Wayland (layer-shell) overlay + Homebrew/apt/AppImage packaging.
- [ ] **Phase 4** — Optional cloud hybrid mode (identity stays local).
- [ ] **Phase 5** — Encrypted persistent transcript history + search.

## License

MIT © Long Hoang
