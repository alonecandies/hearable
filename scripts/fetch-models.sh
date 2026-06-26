#!/usr/bin/env bash
#
# Download the on-device model files `hearable run` expects into a directory.
# Usage:  ./scripts/fetch-models.sh [target-dir]   (default: ./models)
#
# URLs point at the k2-fsa/sherpa-onnx GitHub releases. If any 404s, check the
# releases page (https://github.com/k2-fsa/sherpa-onnx/releases) for the current
# asset name and adjust below — upstream occasionally re-dates model archives.
set -euo pipefail

DIR="${1:-./models}"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
mkdir -p "$DIR"
cd "$DIR"

echo "==> Silero VAD (~2 MB)"
[ -f silero_vad.onnx ] || curl -fL -o silero_vad.onnx "$BASE/asr-models/silero_vad.onnx"

echo "==> ERes2NetV2 speaker embedding (~68 MB)  [note: upstream tag misspells 'recongition']"
EMB="3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
[ -f "$EMB" ] || curl -fL -o "$EMB" "$BASE/speaker-recongition-models/$EMB"

echo "==> SenseVoice multilingual ASR (~230 MB archive -> model.int8.onnx + tokens.txt)"
SV="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
if [ ! -f model.int8.onnx ] || [ ! -f tokens.txt ]; then
  curl -fL -o "$SV.tar.bz2" "$BASE/asr-models/$SV.tar.bz2"
  tar xjf "$SV.tar.bz2"
  cp "$SV/model.int8.onnx" model.int8.onnx
  cp "$SV/tokens.txt" tokens.txt
  rm -rf "$SV" "$SV.tar.bz2"
fi

echo
echo "Done. Models in: $DIR"
ls -1
echo
echo "Optional (language tags): drop whisper-encoder.onnx + whisper-decoder.onnx here too."
echo "Next:  cargo build --release --features live"
echo "       ./target/release/hearable run --models \"$DIR\""
