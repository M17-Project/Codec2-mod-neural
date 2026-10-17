#!/usr/bin/env bash
# 3_adv.sh - stage 2, Opus' adversarial fine-tuning recipe:
#   adv_train_fargan.py --lr 0.000002 --reg-weight 5 --batch-size 160 --initial-checkpoint fargan_400.pth
#   (50 epochs, the script default)
# INIT defaults to the newest stage-1 checkpoint in $RUN.
# RESUME=N continues after fargan_adv_N.pth (epochs N+1..EPOCHS; EPOCHS is the total, not "more").
set -Eeuo pipefail
trap 'echo "ERROR: 3_adv.sh line $LINENO: \"$BASH_COMMAND\" failed (exit $?)" >&2' ERR
W=${W:-/workspace}
DATA=${DATA:-/data/opus}
RUN=${RUN:-$W/runs/opus_v1}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-160}
RESUME=${RESUME:-0}
if [ "$RESUME" -gt 0 ]; then
  INIT=$RUN/checkpoints/fargan_adv_$RESUME.pth
  [ -f "$INIT" ] || { echo "$INIT not found"; exit 1; }
else
  INIT=${INIT:-$(ls "$RUN"/checkpoints/fargan_[0-9]*.pth "$RUN"/checkpoints/fargan_from*_[0-9]*.pth 2>/dev/null \
                 | grep -v _adv_ | xargs -r ls -t | head -1)}
fi
[ -n "$INIT" ] || { echo "no stage-1 checkpoint found in $RUN/checkpoints"; exit 1; }
[ -f "$DATA/features.f32" ] || { echo "$DATA missing - copy it: mkdir -p $DATA && cp $W/data/opus/{features.f32,target.s16} $DATA/"; exit 1; }
grep -q -- '--start-epoch' "$W/opus/dnn/torch/fargan/adv_train_fargan.py" || { echo "run 0_setup.sh first (resume patch missing)"; exit 1; }
echo "== adversarial stage from $INIT (epochs $((RESUME+1))..$EPOCHS, batch $BATCH)"
cd "$W/opus/dnn/torch/fargan"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python3 adv_train_fargan.py "$DATA/features.f32" "$DATA/target.s16" "$RUN" \
        --lr 0.000002 --reg-weight 5 --batch-size "$BATCH" --epochs "$EPOCHS" --start-epoch "$RESUME" \
        --initial-checkpoint "$INIT" --cuda-visible-devices 0
