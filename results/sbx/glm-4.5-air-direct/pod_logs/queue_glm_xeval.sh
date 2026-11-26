#!/bin/bash
cd /workspace/sandbagging
while ps -eo cmd | grep -qE "[s]bx.campaign --model glm-4.5-air-direct|[s]bx.run --model glm-4.5-air-direct|[r]un_glm_phases"; do sleep 20; done
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[campaign $(date +%H:%M:%S)] PHASE C: cross-eval sweep of validated instruments (E, pos_generic, N) on 12 bio evals" >> /workspace/logs/campaign.log
XBASE=N,E,pos_generic XSTEER=E,pos_generic,N INSTR_FILE=/workspace/instr_glm.txt bash scripts/xeval_steer.sh glm-4.5-air-direct results/sbx/glm-4.5-air-direct/full/native results/sbx/glm-4.5-air-direct/full/xeval native 45 --mask-massive 20 --gen-cap 128 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE C done" >> /workspace/logs/campaign.log
