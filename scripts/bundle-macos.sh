#!/usr/bin/env bash
#
# Build a macOS .app bundle for hearable so the microphone permission prompt attributes to
# the app (not your terminal). Ad-hoc code-signs it with the audio-input entitlement, which
# is enough for a locally-run preview build (a real release needs a Developer ID + notarize).
#
# Usage:  ./scripts/bundle-macos.sh
# Output: target/hearable.app
set -euo pipefail
cd "$(dirname "$0")/.."

APP="target/hearable.app"
PLIST="packaging/macos/Info.plist"
ENTITLEMENTS="packaging/macos/hearable.entitlements"

echo "==> Building release binary (--features live)"
cargo build --release --features live

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/hearable "$APP/Contents/MacOS/hearable"
cp "$PLIST" "$APP/Contents/Info.plist"

echo "==> Ad-hoc code-signing with hardened runtime + entitlements"
codesign --force --options runtime --sign - \
    --entitlements "$ENTITLEMENTS" "$APP"
codesign --verify --verbose "$APP" || true

cat <<EOF

Built $APP

Run it (passes args through to the binary):
    open "$APP" --args run --models ./models

or directly (also gets the app's TCC identity):
    "$APP/Contents/MacOS/hearable" run --models ./models

Grant Microphone access for "hearable" when prompted (System Settings ->
Privacy & Security -> Microphone).
EOF
