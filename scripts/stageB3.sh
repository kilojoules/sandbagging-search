#!/bin/bash
# Stage B v3: full Air campaign with BOTH bio and math directions (2x2 specificity matrix)
set -x
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
R=/workspace/sandbagging
LOG=/workspace/logs
MODELAIR=zai-org/GLM-4.5-Air

# ---- BIO direction (from bio pilot pairs) ----
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

# ---- MATH direction (from control pairs, same method) ----
python $R/scripts/sandbag_pipeline.py probs \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --out $R/results/air_math_probs.jsonl --know-thresh 0.5 > $LOG/air_math_probs.log 2>&1
python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open('/workspace/sandbagging/results/air_math_probs.jsonl')]
kept = [r for r in rows if r['top1'] == r['answer'] and r['p_correct'] >= 0.5]
with open('/workspace/sandbagging/results/air_math_pairs.jsonl', 'w') as f:
    for r in kept:
        f.write(json.dumps(r) + '\n')
print('math pairs:', len(kept), 'of', len(rows))
PYEOF
python $R/scripts/sandbag_pipeline.py acts \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --pairs $R/results/air_math_pairs.jsonl --out $R/results/air_math_dirs.npz > $LOG/air_math_acts.log 2>&1

# ---- Cross-domain eval matrix (bio & math directions on bio & math evals) ----
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_heldout.jsonl \
  --alpha 0.5 1.0 > $LOG/air_eval_heldout.log 2>&1
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_control.jsonl > $LOG/air_eval_control.log 2>&1
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/air_math_dirs.npz --out $R/results/air_mathdir_on_bio.jsonl > $LOG/air_mathdir_on_bio.log 2>&1
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --directions $R/results/air_math_dirs.npz --out $R/results/air_mathdir_on_math.jsonl > $LOG/air_mathdir_on_math.log 2>&1

# ---- 9B heldout steering redo (fixed Ablator) ----
python $R/scripts/sandbag_pipeline.py eval \
  --repo THUDM/glm-4-9b-chat --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/pilot9b_dirs.npz --out $R/results/pilot9b_eval_heldout.jsonl \
  --alpha 0.5 1.0 > $LOG/pilot9b_eval_heldout.log 2>&1

# ---- Convince + judge (bio pairs) ----
python $R/scripts/sandbag_pipeline.py convince \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_convince.jsonl > $LOG/air_convince.log 2>&1
python $R/scripts/judge.py \
  --pairs $R/results/air_convince.jsonl \
  --repo unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit \
  --out $R/results/air_judge.jsonl > $LOG/air_judge.log 2>&1

echo STAGE_B_V3_DONE > $LOG/stageB.done
echo ALL_DONE_V3
