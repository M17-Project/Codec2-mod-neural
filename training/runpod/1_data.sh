#!/usr/bin/env bash
# 1_data.sh - Opus' FARGAN training corpus -> Codec2-conditioned training set
#   corpus : https://media.xiph.org/lpcnet/speech/tts_speech_negative_16k.sw (as in opus/dnn/torch/fargan/README.md)
#   prep   : prep_opus.py = dump_data -train augmentation, Codec2 features
# Output: $W/data/opus/{features.f32,target.s16}   (persistent, ~15 GB for 20000 sequences)
set -Eeuo pipefail
trap 'echo "ERROR: 1_data.sh line $LINENO: \"$BASH_COMMAND\" failed (exit $?)" >&2' ERR
W=${W:-/workspace}
C2F=$(cd "$(dirname "$0")/.." && pwd)
URL=${URL:-https://media.xiph.org/lpcnet/speech/tts_speech_negative_16k.sw}
SEQUENCES=${SEQUENCES:-20000}       # dump_data.c: maxCount = 20000
NOISE_FILE=${NOISE_FILE:-}          # optional raw 16 kHz s16 noise; mixed into input AND target (Opus-style)
JOBS=${JOBS:-$(nproc)}
TMP=${TMP:-/data/tmp}

C=$W/corpora/xiph; OUT=$W/data/opus
mkdir -p "$C" "$TMP"
f=$C/$(basename "$URL")
if [ ! -f "$f.ok" ]; then
  echo "== downloading $URL"
  aria2c -c -x 16 -s 16 -k 32M --file-allocation=none --max-tries=50 --retry-wait=10 \
         --console-log-level=warn --download-result=hide --summary-interval=0 -d "$C" -o "$(basename "$f")" "$URL"
  touch "$f.ok"
fi
echo "== corpus: $(du -h "$f" | cut -f1)"

if [ -f "$OUT/done" ]; then echo "== $OUT already built"; else
  rm -rf "$OUT"
  args=("$f" "$OUT" --sequences "$SEQUENCES" --jobs "$JOBS" --tmp "$TMP")
  [ -n "$NOISE_FILE" ] && args+=(--noise "$NOISE_FILE")
  python3 "$C2F/prep_opus.py" "${args[@]}"
  touch "$OUT/done"
fi
python3 "$C2F/check_data.py" "$OUT"
echo "data OK -> $OUT"
