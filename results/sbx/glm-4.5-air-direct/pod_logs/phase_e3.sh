#!/bin/bash
# Phase E3 (Llama-3.3-70B bf16): after the campaign's noise step, (1) random-direction controls for the letters-mode instruments (the campaign ran none),
# (2) the native (visible reasoning) screen in bf16 on the same 5 framings as the earlier 4-bit screen.
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "sbx.campaign --model llama-3.3-70b-instruct" > /dev/null; do sleep 5; done
sleep 5
OUT=results/sbx/glm-4.5-air-direct/llama70b-bf16
COMMON="--model llama-3.3-70b-instruct --repo unsloth/Llama-3.3-70B-Instruct --quant bf16"
echo "[campaign $(date +%H:%M:%S)] PHASE E3a: random-direction controls (3 random directions) for the letters-mode instruments, E and N" >> /workspace/logs/campaign.log
python -m sbx.run $COMMON --out $OUT/letters --preset steer --answer-mode letters --max-minutes 40 --max-per-split 0 --gen-cap 128 --rand-ctrl 3 --rand-for B >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E3a done" >> /workspace/logs/campaign.log
echo "[campaign $(date +%H:%M:%S)] PHASE E3b: native (visible reasoning) screen in bf16, n=120, N,E,pos_bio_plain,pos_bio_hazard,nat_goal_doc" >> /workspace/logs/campaign.log
mkdir -p $OUT/native
python -m sbx.run $COMMON --out $OUT/native --preset screen --answer-mode native --evals wmdp_bio --splits disc --framings N,E,pos_bio_plain,pos_bio_hazard,nat_goal_doc --max-per-split 120 --gen-cap 128 --max-minutes 45 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E3b done" >> /workspace/logs/campaign.log
