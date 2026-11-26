#!/bin/bash
# Phase E: Llama-3.3-70B-Instruct in full bf16 (the earlier screen used 4-bit) on the same 2-GPU pod, after the GLM thinking-mode screen.
cd /workspace/sandbagging
while ps -eo cmd | grep -qE "[q]ueue_glm_think|[s]bx.run --model glm-4.5-air"; do sleep 20; done
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[campaign $(date +%H:%M:%S)] PHASE E: freeing disk, downloading Llama-3.3-70B-Instruct (bf16)" >> /workspace/logs/campaign.log
rm -rf /root/.cache/huggingface/hub/models--zai-org--GLM-4.5-Air /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B /root/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat-hf
df -h / | tail -n 1 >> /workspace/logs/campaign.log
python - >> /workspace/logs/campaign.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download as s
p = s('meta-llama/Llama-3.3-70B-Instruct', ignore_patterns=['original/*','metal/*','*.gguf','*.onnx','*.msgpack','*.h5','consolidated*.safetensors'])
print('downloaded', p)
PY
echo "[campaign $(date +%H:%M:%S)] PHASE E: campaign letters+native, bf16 (--quant bf16), budget 3.5 h" >> /workspace/logs/campaign.log
python -m sbx.campaign --model llama-3.3-70b-instruct --out results/sbx/glm-4.5-air-direct/llama70b-bf16 --modes letters,native --budget-hours 3.5 --quant bf16 --gen-cap 128 --max-per-split 120 --noise-include default >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E done" >> /workspace/logs/campaign.log
