#!/usr/bin/env bash
# 2_train.sh - stage 1, Opus' FARGAN pre-training recipe:
#   train_fargan.py --epochs 400 --batch-size 4096 --lr 0.002   (opus/dnn/torch/fargan/README.md)
# Data is copied to the fast local container disk first ($DATA).
# Resume after an interruption:  RESUME=<last finished epoch> bash 2_train.sh
set -Eeuo pipefail
trap 'echo "ERROR: 2_train.sh line $LINENO: \"$BASH_COMMAND\" failed (exit $?)" >&2' ERR
W=${W:-/workspace}
SRC=${SRC:-$W/data/opus}
DATA=${DATA:-/data/opus}
RUN=${RUN:-$W/runs/opus_v1}
EPOCHS=${EPOCHS:-400}
BATCH=${BATCH:-4096}
LR=${LR:-0.002}
RESUME=${RESUME:-}

mkdir -p "$DATA"
for x in features.f32 target.s16; do
  if [ ! -f "$DATA/$x" ] || [ "$(stat -c %s "$DATA/$x")" != "$(stat -c %s "$SRC/$x")" ]; then
    echo "== copying $x to local disk"; cp "$SRC/$x" "$DATA/$x"
  fi
done

cd "$W/opus/dnn/torch/fargan"
args=("$DATA/features.f32" "$DATA/target.s16" "$RUN" --batch-size "$BATCH" --cuda-visible-devices 0)
if [ -n "$RESUME" ]; then
  # optimizer state is not stored in the checkpoint: continue the 1/(1+1e-4*step) LR decay by hand
  rows=$(( $(stat -c %s "$DATA/features.f32") / 144 ))
  steps=$(( (rows / 15 - 4) / BATCH * RESUME ))
  lr=$(awk -v l="$LR" -v s="$steps" 'BEGIN{printf "%.6g", l/(1+1e-4*s)}')
  args+=(--epochs $((EPOCHS - RESUME)) --lr "$lr" --suffix "_from$RESUME"
         --initial-checkpoint "$RUN/checkpoints/fargan_$RESUME.pth")
  echo "== resuming after epoch $RESUME (step $steps): lr $lr, $((EPOCHS - RESUME)) epochs left"
else
  args+=(--epochs "$EPOCHS" --lr "$LR")
fi
echo "train: ${args[*]}"
python3 train_fargan.py "${args[@]}"
