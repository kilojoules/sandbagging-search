#!/bin/bash
# Gate for the GLM-4.5-Air run on a fresh 2-GPU pod (RUN_CMD of pod_launch.sh):
#   1. a small model sharded vs single-GPU on THIS hardware (must give 2 x "[check] ALL OK"),
#   2. the real GLM-4.5-Air load + absolute checks (weights fully on GPU, ~93/106 GiB per device, hooks across the GPU boundary, noise, ...).
# Only after "[preflight] PASSED" is the campaign restarted (scripts/pod_restart.sh).
set -u
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[preflight $(date +%H:%M:%S)] $*"; }
say "step 1: small model, sharded vs single GPU"
bash scripts/run_multigpu_validation.sh qwen3-8b > /workspace/logs/preflight_small.log 2>&1
grep -E '^\[(mgpu|check)\]|^\[mgpu|Traceback|Error|assert' /workspace/logs/preflight_small.log | cut -c1-220
N=$(grep -c '^\[check\] ALL OK' /workspace/logs/preflight_small.log)
if [ "$N" -lt 2 ]; then say "FAILED: small-model validation gave $N of 2 ALL OK"; exit 1; fi
say "step 2: GLM-4.5-Air real model"
python scripts/multigpu_check.py glm-4.5-air-direct > /workspace/logs/preflight_glm.log 2>&1
grep -E '^\[(load|check)\]|Traceback|Error|assert|ffload' /workspace/logs/preflight_glm.log | cut -c1-240
if grep -q '^\[check\] ALL OK' /workspace/logs/preflight_glm.log; then say "PASSED"; else say "FAILED: GLM-4.5-Air real-model check"; exit 2; fi
