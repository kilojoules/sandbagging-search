#!/bin/bash
# Full-sample confirmation of one validated CoT-direction instrument (default: CF, cap, layers 13-23) under ONE steering framing:
# both bio and non-bio evals, both splits, 3 random-direction controls.  RUN_CMD of pod_launch.sh (waits for dirs.npz):
#   bash scripts/run_cot_confirm.sh MODEL OUT_DIR FRAMING [extra sbx.run args]
set -u
MODEL="$1"; OUT="$2"; FR="$3"; shift 3
EXTRA="$*"
say() { echo "[cotconfirm $(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT"
for i in $(seq 1 480); do [ -f "$OUT/dirs.npz" ] && break; [ $i -eq 1 ] && say "waiting for $OUT/dirs.npz"; sleep 10; done
[ -f "$OUT/dirs.npz" ] || { say "FAILED: no dirs.npz"; exit 1; }
say "model $MODEL framing $FR instrument ${CONF_DIR:-CF} / ${CONF_INT:-cap} / ${CONF_LAYERS:-13-23}"
python -m sbx.run --model "$MODEL" --out "$OUT" --answer-mode native --preset steer --stages behavior,steer \
  --evals "${CONF_EVALS:-wmdp_bio,mmlu_nonbio}" --splits disc,rep --grid-splits disc,rep --max-per-split 0 \
  --framings "N,$FR" --steer-framings "$FR" --steer-evals "${CONF_EVALS:-wmdp_bio,mmlu_nonbio}" \
  --dir-names "${CONF_DIR:-CF}" --interventions "${CONF_INT:-cap}" --layer-sets "${CONF_LAYERS:-13-23}" --strengths=-0.4 \
  --rand-ctrl "${CONF_RAND:-3}" --rand-for "${CONF_DIR:-CF}" --mask-massive 20 --skip-rerun --gen-cap "${CONF_GENCAP:-64}" --max-minutes "${CONF_MAXMIN:-300}" $EXTRA
say "run exit $?"
touch "$OUT/COTCONFIRM_DONE"
