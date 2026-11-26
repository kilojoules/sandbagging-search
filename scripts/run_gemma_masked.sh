#!/bin/bash
# Steering grid on a Gemma-3 pod with the massive-activation dimension MASKED (sbx.run --mask-massive): the raw directions of a finished dirs step
# are copied into a NEW out dir (cells are keyed by direction name, so raw and masked cells must not share a store) and the grid is run
# for the given steering framings.  Usable as the RUN_CMD of pod_launch.sh (it waits for dirs.npz when SRC is not on this machine yet).
#
#   bash scripts/run_gemma_masked.sh MODEL SRC_OUT DST_OUT MODE FRAMINGS DIR_NAMES STAGES MAXMIN [extra sbx.run args ...]
#   e.g. bash scripts/run_gemma_masked.sh gemma-3-12b-it results/sbx/gemma-3-12b-it/native results/sbx/gemma-3-12b-it/native_m native \
#          pos_bio_hazard B,A,LIE,EA_test,EXPERT behavior,steer 170 --dir-layers 1-23 --layer-sets auto1
set -u
MODEL="$1"; SRC="$2"; DST="$3"; MODE="$4"; FR="$5"; DN="$6"; STAGES="$7"; MAXMIN="$8"; shift 8
EXTRA="$*"
say() { echo "[masked $(date +%H:%M:%S)] $*"; }
mkdir -p "$DST"
if [ "$SRC" != "$DST" ]; then
  for f in dirs.npz dirs_meta.json; do [ -f "$SRC/$f" ] && [ ! -f "$DST/$f" ] && cp "$SRC/$f" "$DST/$f"; done
fi
for i in $(seq 1 480); do [ -f "$DST/dirs.npz" ] && break; [ $i -eq 1 ] && say "waiting for $DST/dirs.npz"; sleep 10; done
[ -f "$DST/dirs.npz" ] || { say "FAILED: no dirs.npz"; exit 1; }
say "model $MODEL mode $MODE steer framings $FR directions $DN stages $STAGES (cap $MAXMIN min)"
python -m sbx.run --model "$MODEL" --out "$DST" --answer-mode "$MODE" --preset steer --stages "$STAGES" \
  --evals wmdp_bio --splits disc --framings "N,$FR" --steer-framings "$FR" --dir-names "$DN" --mask-massive 20 \
  --max-per-split 120 --rand-ctrl 2 --rand-for B --max-minutes "$MAXMIN" $EXTRA
say "run exit $?"
touch "$DST/MASKED_DONE"
say "done"
