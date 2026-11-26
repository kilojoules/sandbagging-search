#!/bin/bash
# LAST chain: bio direction + evals + convince + judge + mathdir_on_bio, all fixes in.
set -x
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
# no expandable_segments (caused VA over-reserve); disable speculative warmup allocations
export PYTORCH_CUDA_ALLOC_CONF=
export TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP=1
R=/workspace/sandbagging
LOG=/workspace/logs
MODELAIR=zai-org/GLM-4.5-Air

# 1. bio acts (pairs exist: 417)
python $R/scripts/sandbag_pipeline.py acts \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_dirs.npz > $LOG/air_acts.log 2>&1

# 2. bio-direction evals (bio heldout w/ steering + math control)
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_heldout.jsonl \
  --alpha 0.5 1.0 > $LOG/air_eval_heldout.log 2>&1
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_control_mathphys.jsonl \
  --directions $R/results/air_dirs.npz --out $R/results/air_eval_control.jsonl > $LOG/air_eval_control.log 2>&1

# 3. math-direction on bio eval (the cross-domain cell)
python $R/scripts/sandbag_pipeline.py eval \
  --repo $MODELAIR --questions $R/data/questions_heldout_eval.jsonl \
  --directions $R/results/air_math_dirs.npz --out $R/results/air_mathdir_on_bio.jsonl \
  > $LOG/air_mathdir_on_bio.log 2>&1

# 4. convince + judge
python $R/scripts/sandbag_pipeline.py convince \
  --repo $MODELAIR --questions $R/data/questions_pilot.jsonl \
  --pairs $R/results/air_pairs.jsonl --out $R/results/air_convince.jsonl > $LOG/air_convince.log 2>&1
python $R/scripts/judge.py \
  --pairs $R/results/air_convince.jsonl \
  --repo unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit \
  --out $R/results/air_judge.jsonl > $LOG/air_judge.log 2>&1

echo LAST_CHAIN_DONE > $LOG/last_chain.done
echo TRULY_ALL_DONE
