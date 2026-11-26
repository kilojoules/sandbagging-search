#!/bin/bash
# Validation of the sharded (device_map=auto) code paths on a cheap 2-GPU pod: each model once on GPU 0 only (reference) and once sharded.
set -u
cd /workspace/sandbagging
for M in "$@"; do
  echo "[mgpu $(date +%H:%M:%S)] $M: single-GPU reference"
  CUDA_VISIBLE_DEVICES=0 python scripts/multigpu_check.py "$M" --out /workspace/ref_$M.npz 2>&1 | grep -v "Warning\|warn\|Loading checkpoint\|Fetching"
  echo "[mgpu $(date +%H:%M:%S)] $M: sharded over both GPUs, compared with the reference"
  python scripts/multigpu_check.py "$M" --compare /workspace/ref_$M.npz 2>&1 | grep -v "Warning\|warn\|Loading checkpoint\|Fetching"
done
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo "[mgpu $(date +%H:%M:%S)] DONE"
