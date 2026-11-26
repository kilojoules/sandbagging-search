#!/bin/bash
# Natural-organism instrument: build the sandbag directions from the NATURAL trigger's own pairs (right under N, wrong under
# --explicit-framing nat_goal_doc, no instruction to underperform anywhere), then steer under nat_goal_doc (validity: does the
# direction restore the natural drop?) and under N (does it change the default model?).  Meant to be the RUN_CMD of pod_launch.sh.
#
#   RUN_CMD="bash scripts/run_natdirs.sh MODEL OUTKEY DIR_LAYERS LAYER_SETS [N_PAIRS] [extra sbx.run args ...]" scripts/pod_launch.sh ...
#   e.g. bash scripts/run_natdirs.sh gemma-3-12b-it gemma-3-12b-it-nat 1-23 auto1,13-23 400
set -u
MODEL="$1"; OUTKEY="$2"; DIRL="$3"; LSETS="$4"; NP="${5:-400}"; shift 5 2>/dev/null || shift $#
EXTRA="$*"
OUT=results/sbx/$OUTKEY/native
say() { echo "[natdirs $(date +%H:%M:%S)] $*"; }

say "step 1/2: directions from natural-trigger pairs (explicit framing nat_goal_doc), n_pairs=$NP"
python -m sbx.run --model "$MODEL" --out "$OUT" --answer-mode native --preset dirs --stages dirs --recipes sandbag_bio \
  --n-pairs "$NP" --max-per-split 150 --explicit-framing nat_goal_doc --dir-layers "$DIRL" --layer-sets "$LSETS" \
  --gen-cap 64 --max-minutes 90 $EXTRA
say "dirs exit $?"

say "step 2/2: steering grid under nat_goal_doc and N with the natural-pair directions, then automatic replication"
python -m sbx.run --model "$MODEL" --out "$OUT" --answer-mode native --preset steer \
  --steer-framings nat_goal_doc,N --dir-names B,A,M --explicit-framing nat_goal_doc --dir-layers "$DIRL" --layer-sets "$LSETS" \
  --max-per-split 150 --gen-cap 64 --max-minutes 170 $EXTRA
say "steer exit $?"
touch "$OUT/NATDIRS_DONE"
say "done"
