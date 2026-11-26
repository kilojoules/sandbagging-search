#!/bin/bash
# Phase E4: cross-eval sweep of 5 validated Llama-3.3-70B (bf16, letters-mode) instruments on 12 bio evals: restoration under E and change under N.
cd /workspace/sandbagging
while ps -eo cmd | grep -qE "[p]hase_e3.sh|[s]bx.campaign --model llama-3.3-70b-instruct|[s]bx.run --model llama-3.3-70b-instruct"; do sleep 20; done
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[campaign $(date +%H:%M:%S)] PHASE E4: cross-eval sweep of 5 validated Llama-70B bf16 instruments (A add, H proj0, LIE cap, REF add, EA_test proj0) on 12 bio evals, E and N" >> /workspace/logs/campaign.log
XBASE=N,E XSTEER=E,N INSTR_FILE=/workspace/instr_llama.txt bash scripts/xeval_steer.sh llama-3.3-70b-instruct results/sbx/glm-4.5-air-direct/llama70b-bf16/letters results/sbx/glm-4.5-air-direct/llama70b-bf16/xeval letters 45 --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --gen-cap 128 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E4 done" >> /workspace/logs/campaign.log
