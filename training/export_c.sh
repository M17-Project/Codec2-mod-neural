#!/usr/bin/env bash
# export_c.sh - turn a FARGAN checkpoint into the C weight files in ../model
#
#   bash training/export_c.sh fargan_adv_50.pth
#
# Needs the Opus source tree (default: next to this repository, override with OPUS=).
set -euo pipefail
PTH=${1:?usage: export_c.sh model.pth}
HERE=$(cd "$(dirname "$0")" && pwd)
OPUS=${OPUS:-$HERE/../../opus}
OUT=$HERE/../model
python3 "$OPUS/dnn/torch/fargan/dump_fargan_weights.py" "$PTH" "$OUT" | grep -v -e '^dumping' -e '^Ignoring' || true
{
  echo "checkpoint: $(basename "$PTH")"
  echo "sha256:     $(sha256sum "$PTH" | cut -d' ' -f1)"
  echo "opus:       $(git -C "$OPUS" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "exported:   $(date -u +%Y-%m-%d)"
} > "$OUT/MODEL.txt"
cat "$OUT/MODEL.txt"
