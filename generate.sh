#!/usr/bin/env bash

CHECKPOINT="audio-id-at-scale/tc28/checkpoints/epoch=99-step=361900.ckpt"
OUT_PATH="/data/scratch/acw723/synth/GraFP/tc_29_epoch99_100K/dumm_db.mm"

for NUM_SAMPLES in "$@"; do
  KVAL=$(( NUM_SAMPLES / 1000 ))
  KSTR="${KVAL}K"
  OUT_CUR="${OUT_PATH//100K/${KSTR}}"
  mkdir -p "$(dirname "$OUT_CUR")"

  echo "Output path..."
  echo $CHECKPOINT


#   python generate.py --checkpoint="$CHECKPOINT" --num_samples="$NUM_SAMPLES" --out="$OUT_CUR"
# done
