#!/bin/bash
# Phase E v2: Llama-3.3-70B-Instruct in full bf16 from the ungated sha256-identical mirror, letters campaign then a small native screen.
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[campaign $(date +%H:%M:%S)] PHASE E2: downloading unsloth/Llama-3.3-70B-Instruct (bf16 mirror of the gated official repo)" >> /workspace/logs/campaign.log
df -h / | tail -n 1 >> /workspace/logs/campaign.log
python - >> /workspace/logs/campaign.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download as s
p = s('unsloth/Llama-3.3-70B-Instruct', ignore_patterns=['original/*','metal/*','*.gguf','*.onnx','*.msgpack','*.h5','consolidated*.safetensors'])
print('downloaded', p)
PY
rm -rf results/sbx/glm-4.5-air-direct/llama70b-bf16
echo "[campaign $(date +%H:%M:%S)] PHASE E2: letters-mode campaign (screen, kprobe, behavior, dirs, steer, noise as gated), bf16, budget 2 h" >> /workspace/logs/campaign.log
python -m sbx.campaign --model llama-3.3-70b-instruct --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --out results/sbx/glm-4.5-air-direct/llama70b-bf16 --modes letters --budget-hours 2 --gen-cap 128 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E2: native (visible reasoning) screen on 5 framings, n=120" >> /workspace/logs/campaign.log
mkdir -p results/sbx/glm-4.5-air-direct/llama70b-bf16/native
python -m sbx.run --model llama-3.3-70b-instruct --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --out results/sbx/glm-4.5-air-direct/llama70b-bf16/native --preset screen --answer-mode native --evals wmdp_bio --splits disc --framings N,E,pos_bio_plain,pos_bio_hazard,nat_goal_doc --max-per-split 120 --gen-cap 128 --max-minutes 45 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E2 done" >> /workspace/logs/campaign.log
