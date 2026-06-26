# hearable Phase 3 (Packaging & Wayland) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Several tasks here
> can only be built/tested on a real macOS host or a Linux host (noted per task).

**Goal:** Make hearable installable (Homebrew, apt, AppImage) and give it a first-class macOS
app identity, plus the Wayland layer-shell overlay path the design calls for.

## Status

**Done (verifiable / reviewable from the macOS dev box):**
- [x] **macOS `.app` bundle** — `packaging/macos/Info.plist` (`NSMicrophoneUsageDescription`),
  `hearable.entitlements` (`com.apple.security.device.audio-input`), and
  `scripts/bundle-macos.sh` (builds `--features live`, assembles the bundle, ad-hoc signs with
  hardened runtime). Fixes mic-permission attribution. Plists lint clean.
- [x] **cargo-deb metadata** — `[package.metadata.deb]` in `crates/hearable/Cargo.toml`
  (runtime deps: libasound2, libxkbcommon0, libwayland-client0, libxcb1, libgl1; assets).
- [x] **Homebrew formula (draft)** — `packaging/homebrew/hearable.rb` (head-installable;
  stable url/sha256 + sherpa-libs resource flagged as TODO).

**Remaining (needs a real build host):**

### Task 3.1: Wayland layer-shell overlay  *(Linux host required)*
The winit/eframe overlay's `with_always_on_top()` is ignored by GNOME/Mutter Wayland, and
winit doesn't implement `wlr-layer-shell`. Add a Linux-only path using
`smithay-client-toolkit` (`zwlr_layer_shell_v1`) for wlroots/KWin compositors, detected at
runtime, with a graceful normal-window fallback on GNOME. Gate behind `#[cfg(target_os =
"linux")]` + a `wayland` feature. **Cannot be compile-checked on macOS** (SCTK is Linux-only);
develop on a Linux/Wayland machine. Reference: design spec §4.7.

### Task 3.2: Homebrew sherpa-libs resource  *(macOS host)*
Homebrew's build sandbox blocks network, but sherpa-onnx's build.rs downloads native libs.
Declare them as a `resource` in the formula, stage to a dir, and set `SHERPA_ONNX_LIB_DIR`
before `cargo build`. Pin the resource URL + sha256 from the sherpa-onnx release. Then verify
`brew install --build-from-source` end to end. Cut a `v0.1.0` git tag and fill the formula's
stable `url`/`sha256`.

### Task 3.3: Build the `.deb`  *(Linux host)*
`cargo build --release --features live` then `cargo deb --no-build` (the `--features live`
build must happen first so the binary exists). Verify the dependency list installs on a clean
Debian/Ubuntu, and that the overlay launches. Host the `.deb` in an apt channel.

### Task 3.4: AppImage  *(Linux host)*
Produce a portable AppImage (e.g. `cargo-appimage` or linuxdeploy) bundling the binary +
required `.so`s for non-Debian distros. Build in the same CI job as the `.deb`.

### Task 3.5: Release signing/notarization  *(macOS host, paid Apple Developer)*
For distribution (not local preview), replace ad-hoc signing with a Developer ID, hardened
runtime, and `xcrun notarytool` notarization so Gatekeeper accepts the `.app`/bundle.

## How to use what's done now (macOS)

```bash
./scripts/fetch-models.sh ./models
./scripts/bundle-macos.sh                       # -> target/hearable.app (ad-hoc signed)
open target/hearable.app --args run --models ./models
```
The `.app` requests microphone access as "hearable" (not your terminal).
