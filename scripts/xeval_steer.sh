#!/bin/bash
# Env: XEVALS (comma list), MAXPS (max questions per split), XBASE (baseline framings, default N,E), XSTEER (steering framings, default E,N), INSTR_FILE.
# Cross-eval sweep of a few VALIDATED instruments on one model: does any of them raise DEFAULT (N) accuracy on ANY bio eval, and does each still
# restore E on those evals?  One sbx.run invocation per instrument (cells are keyed by eval/split/framing/direction/intervention/layers/strength).
#   [INSTR_FILE=file with lines "dirs interventions layer_sets strengths"] bash scripts/xeval_steer.sh MODEL SRC_OUT DST_OUT MODE MAXMIN [extra sbx.run args]
set -u
MODEL="$1"; SRC="$2"; DST="$3"; MODE="$4"; MAXMIN="$5"; shift 5
EXTRA="$*"
EVALS="${XEVALS:-gpqa_bio,labbench_litqa2,labbench_protocolqa,labbench_seqqa,mmlupro_bio,supergpqa_bio,medxpertqa_text,medqa_usmle,medmcqa_bio,mmlu_bio_all,wmdp_chem,wmdp_cyber}"
say() { echo "[xeval $(date +%H:%M:%S)] $*"; }
mkdir -p "$DST"
for f in dirs.npz dirs_meta.json; do [ -f "$SRC/$f" ] && [ ! -f "$DST/$f" ] && cp "$SRC/$f" "$DST/$f"; done
run() { # dirs interventions layer_sets strengths
  say "instrument: dirs=$1 interventions=$2 layers=$3 strengths=$4"
  python -m sbx.run --model "$MODEL" --out "$DST" --answer-mode "$MODE" --preset steer --stages behavior,steer \
    --evals "$EVALS" --steer-evals "$EVALS" --splits disc,rep --grid-splits disc,rep --max-per-split "${MAXPS:-150}" --framings "${XBASE:-N,E}" --steer-framings "${XSTEER:-E,N}" \
    --dir-names "$1" --interventions "$2" --layer-sets "$3" --strengths="$4" --rand-ctrl 0 --max-minutes "$MAXMIN" $EXTRA
  say "exit $?"
}
if [ -n "${INSTR_FILE:-}" ] && [ -f "$INSTR_FILE" ]; then
  while read -r d i l s; do [ -n "$d" ] && run "$d" "$i" "$l" "$s"; done < "$INSTR_FILE"
else
  run LIE proj0 10-28 -0.2
  run EXPERT add auto1 -0.4
  run EA_test proj0 10-28 -0.2
  run B cap 10-28 -0.2
fi
touch "$DST/XEVAL_DONE"
say "done"
