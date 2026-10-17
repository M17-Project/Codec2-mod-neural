#!/usr/bin/env bash
# compare_adv.sh - decode one speech file with fargan_adv_30/40/50 (16 kHz each) for A/B listening
#
#   bash compare_adv.sh                 # random LibriSpeech dev-clean file
#   bash compare_adv.sh my_voice.wav    # your own recording (any sample rate)
#
# Models are read from ~/Desktop (override: MODELS=/some/dir). Python venv: VENV=, corpus: CORPUS=
# Output: ~/Desktop/compare/<name>_adv30.wav, _adv40.wav, _adv50.wav (16 kHz),
#         plus <name>_classic.wav (classic Codec2, 8 kHz) for reference.
set -euo pipefail
MODELS=${MODELS:-$HOME/Desktop}
C2F=${C2F:-$(cd "$(dirname "$0")" && pwd)}
OUT=${OUT:-$HOME/Desktop/compare}
EPOCHS=${EPOCHS:-"30 40 50"}

VENV=${VENV:-$HOME/c2nn/venv}
CORPUS=${CORPUS:-$HOME/c2nn/corpora/LibriSpeech/dev-clean}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
IN=${1:-$(find "$CORPUS" -name '*.flac' | shuf -n1)}
[ -f "$IN" ] || { echo "input not found: $IN"; exit 1; }
name=$(basename "${IN%.*}")
mkdir -p "$OUT"
echo "input: $IN"

files=()
for e in $EPOCHS; do
  m="$MODELS/fargan_adv_$e.pth"
  [ -f "$m" ] || { echo "missing $m"; exit 1; }
  o="$OUT/${name}_adv$e.wav"
  python3 "$C2F/synth.py" "$m" "$IN" "$o" >/dev/null
  mv -f "$OUT/${name}_adv${e}_classic.wav" "$OUT/${name}_classic.wav"
  files+=("$o")
  echo "  adv $e -> $o"
done
echo "  classic Codec2 (8 kHz) -> $OUT/${name}_classic.wav"

if command -v aplay >/dev/null; then
  for f in "${files[@]}"; do echo "playing $(basename "$f")"; aplay -q "$f"; sleep 0.5; done
fi
