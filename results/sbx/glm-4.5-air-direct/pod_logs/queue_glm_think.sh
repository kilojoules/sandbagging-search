#!/bin/bash
cd /workspace/sandbagging
while ps -eo cmd | grep -qE "[x]eval_steer|[q]ueue_glm_xeval|[s]bx.run --model glm-4.5-air-direct"; do sleep 20; done
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/sbx/glm-4.5-air-direct/think; mkdir -p $OUT
echo "[campaign $(date +%H:%M:%S)] PHASE D: thinking-mode screen (hidden reasoning channel): N, E, natural trigger, twin, n=60" >> /workspace/logs/campaign.log
python -m sbx.run --model glm-4.5-air-think --out $OUT --preset screen --answer-mode native --evals wmdp_bio --splits disc --framings N,E,nat_goal_doc,neg_goal_doc_inverted --max-per-split 60 --gen-cap 160 --max-minutes 80 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE D done" >> /workspace/logs/campaign.log
