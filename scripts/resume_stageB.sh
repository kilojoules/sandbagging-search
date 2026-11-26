#!/bin/bash
# Resume chain: redo crashed 9B heldout eval (JSON fix), then Stage B remainder.
set -x
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
R=/workspace/sandbagging
LOG=/workspace/logs

# re-run just the crashed 9B heldout eval with the fixed script
python $R/scripts/sandbag_pipeline.py eval \
  --repo THUDM/glm-4-9b-chat --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/pilot9b_dirs.npz --out $R/results/pilot9b_eval_heldout.jsonl \
  --alpha 0.5 1.0 > $LOG/pilot9b_eval_heldout.log 2>&1
echo HELDOUT9B_DONE

MODELAIR=zai-org/GLM-4.5-Air-FP8

python $R/scripts/sandbag_pipeline.py probs \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --out $R/results/air_probs.jsonl --know-thresh 0.5 > $LOG/air_probs.log 2>&1

python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open('/workspace/sandbagging/results/air_probs.jsonl')]
kept = [r for r in rows if r['top1'] == r['answer'] and r['p_correct'] >= 0.5]
with open('/workspace/sandbagging/results/air_pairs.jsonl', 'w') as f:
    for r in kept:
        f.write(json.dumps(r) + '\n')
print('pairs:', len(kept), 'of', len(rows))
PYEOF

python $R/scripts/sandbag_pipeline.py acts \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_dirs.npz > $LOG/air_acts.log 2>&1

python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_heldout.jsonl \
  --alpha 0.5 1.0 > $LOG/air_eval_heldout.log 2>&1

python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_control.jsonl > $LOG/air_eval_control.log 2>&1

python $R/scripts/sandbag_pipeline.py convince \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_convince.jsonl > $LOG/air_convince.log 2>&1

python $R/scripts/judge.py \
  --pairs $R/results/air_convince.jsonl \
  --repo unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit \
  --out $R/results/air_judge.jsonl > $LOG/air_judge.log 2>&1

echo STAGE_B_DONE > $LOG/stageB.done
echo RESUME_ALL_DONE
