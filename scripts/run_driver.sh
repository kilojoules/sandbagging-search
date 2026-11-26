#!/bin/bash
# Sandbagging-direction campaign driver. Run under setsid on the pod.
set -x
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
R=/workspace/sandbagging
LOG=/workspace/logs; mkdir -p $LOG results

MODEL9B=THUDM/glm-4-9b-chat
MODELAIR=zai-org/GLM-4.5-Air-FP8

# ---- pre-download model 1 under cover of pip warmup ----
python - <<'PY' 2>&1 | tee $LOG/dl_9b.log
from huggingface_hub import snapshot_download
snapshot_download("THUDM/glm-4-9b-chat")
print("9B done")
PY

# ================= STAGE A: 9B pilot (validates pipeline) =================
python $R/scripts/sandbag_pipeline.py probs \
  --repo $MODEL9B --questions $R/data/questions_pilot.jsonl \
  --out $R/results/pilot9b_probs.jsonl --know-thresh 0.5 2>&1 | tee $LOG/pilot9b_probs.log
python - <<'PY' 2>&1 | tee $LOG/make_pairs.log
import json
rows=[json.loads(l) for l in open('/workspace/sandbagging/results/pilot9b_probs.jsonl')]
kept=[r for r in rows if r['top1']==r['answer'] and r['p_correct']>=0.5]
with open('/workspace/sandbagging/results/pilot9b_pairs.jsonl','w') as f:
    for r in kept: f.write(json.dumps(r)+'\n')
print('pairs:',len(kept),'of',len(rows))
PY
python $R/scripts/sandbag_pipeline.py acts \
  --repo $MODEL9B --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/pilot9b_pairs.jsonl --out $R/results/pilot9b_dirs.npz 2>&1 | tee $LOG/pilot9b_acts.log
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODEL9B --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/pilot9b_dirs.npz --out $R/results/pilot9b_eval_heldout.jsonl \
  --alpha 0.5 1.0 2>&1 | tee $LOG/pilot9b_eval_heldout.log
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODEL9B --questions $R/data/questions_control_mathphys.jsonl \
  --directions $R/results/pilot9b_dirs.npz --out $R/results/pilot9b_eval_control.jsonl 2>&1 | tee $LOG/pilot9b_eval_control.log
echo "STAGE A DONE" > $LOG/stageA.done

# ---- download model 2 ----
python - <<'PY' 2>&1 | tee $LOG/dl_air.log
from huggingface_hub import snapshot_download
snapshot_download("zai-org/GLM-4.5-Air-FP8")
print("Air done")
PY

# ================= STAGE B: GLM-4.5-Air main run =================
python $R/scripts/sandbag_pipeline.py probs \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --out $R/results/air_probs.jsonl --know-thresh 0.5 2>&1 | tee $LOG/air_probs.log
python - <<'PY' 2>&1 | tee $LOG/make_pairs_air.log
import json
rows=[json.loads(l) for l in open('/workspace/sandbagging/results/air_probs.jsonl')]
kept=[r for r in rows if r['top1']==r['answer'] and r['p_correct']>=0.5]
with open('/workspace/sandbagging/results/air_pairs.jsonl','w') as f:
    for r in kept: f.write(json.dumps(r)+'\n')
print('pairs:',len(kept),'of',len(rows))
PY
python $R/scripts/sandbag_pipeline.py acts \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_dirs.npz 2>&1 | tee $LOG/air_acts.log
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_heldout.jsonl \
  --alpha 0.5 1.0 2>&1 | tee $LOG/air_eval_heldout.log
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_control.jsonl 2>&1 | tee $LOG/air_eval_control.log
python $R/scripts/sandbag_pipeline.py convince \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_convince.jsonl 2>&1 | tee $LOG/air_convince.log
# judge with Llama-3.1-70B-4bit (pre-quantized, per runpod-ops)
python - <<'PY' 2>&1 | tee $LOG/dl_judge.log
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit")
print("judge done")
PY
python $R/scripts/judge.py --pairs $R/results/air_convince.jsonl \
  --repo unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --out $R/results/air_judge.jsonl 2>&1 | tee $LOG/air_judge.log
echo "STAGE B DONE" > $LOG/stageB.done
echo ALL_DONE
