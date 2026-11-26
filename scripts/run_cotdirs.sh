#!/bin/bash
# sandbag_cot directions (mean residual over the GENERATED tokens, natural trigger vs neutral) + validity steering under nat_goal_doc.
# Meant as the RUN_CMD of pod_launch.sh:   bash scripts/run_cotdirs.sh MODEL OUTKEY DIR_LAYERS N_PAIRS [extra sbx.run args]
set -u
MODEL="$1"; OUTKEY="$2"; DIRL="$3"; NP="$4"; shift 4
EXTRA="$*"
OUT=results/sbx/$OUTKEY/native
say() { echo "[cotdirs $(date +%H:%M:%S)] $*"; }
say "step 1/2: building sandbag_cot directions from nat_goal_doc vs N ($NP questions)"
python -m sbx.run --model "$MODEL" --out "$OUT" --answer-mode native --stages dirs --recipes sandbag_cot --n-pairs "$NP" \
  --max-per-split 120 --explicit-framing nat_goal_doc --dir-layers "$DIRL" --layer-sets 13-23,auto1 --mask-massive 20 --max-minutes 120 $EXTRA
say "dirs exit $?"
touch "$OUT/COTDIRS_BUILT"
say "step 2/2: steering under nat_goal_doc (validity)"
bash scripts/run_gemma_masked.sh "$MODEL" "$OUT" "$OUT" native nat_goal_doc CF,CM behavior,steer 150 --layer-sets 13-23,auto1 --rand-for CF $EXTRA
say "done"
