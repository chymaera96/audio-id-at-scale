#!/usr/bin/env bash

CHECKPOINT="/data/home/acw723/audio-id-at-scale/checkpoints/best/tc31-087.ckpt"
OUT_PATH="/data/scratch/acw723/synth/GraFP/tc_27_epoch87_100K/dummy_db.mm"

for NUM_SAMPLES in "$@"; do
  if [ "$NUM_SAMPLES" -lt 1000000 ]; then
    VAL=$(( NUM_SAMPLES / 1000 ))
    SUFFIX="K"
  else
    VAL=$(( NUM_SAMPLES / 1000000 ))
    SUFFIX="M"
  fi

  STR="${VAL}${SUFFIX}"
  OUT_CUR="${OUT_PATH//100K/${STR}}"
  mkdir -p "$(dirname "$OUT_CUR")"

  echo "Output path..."
  echo "$OUT_CUR"

  python generate.py --checkpoint="$CHECKPOINT" --num_samples="$NUM_SAMPLES" --out="$OUT_CUR"
done
