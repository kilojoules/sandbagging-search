#!/bin/bash
# After E6a (running): E5 native steering E/N only (A add L34, H proj0 22-62 + 1 random direction each), then E4 letters-mode cross-eval (3 instruments x 12 evals).
cd /workspace/sandbagging
while ps -eo cmd | grep -qE "[p]hase_e3.sh|[s]bx.campaign --model llama-3.3-70b-instruct|[s]bx.run --model llama-3.3-70b-instruct"; do sleep 20; done
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/sbx/glm-4.5-air-direct/llama70b-bf16
echo "[campaign $(date +%H:%M:%S)] PHASE E6a done" >> /workspace/logs/campaign.log
[ -f /workspace/SKIP_E5 ] || {
echo "[campaign $(date +%H:%M:%S)] PHASE E5: native (visible reasoning) steering, WMDP-bio disc n=120: A add L34, H proj0 22-62 + 1 random direction each, framings E and N" >> /workspace/logs/campaign.log
XEVALS=wmdp_bio MAXPS=120 XBASE=N,E XSTEER=E,N INSTR_FILE=/workspace/instr_llama_native.txt bash scripts/xeval_steer.sh llama-3.3-70b-instruct $OUT/letters $OUT/native native 30 --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --gen-cap 128 --splits disc --grid-splits disc --rand-ctrl 1 --rand-for A,H >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E5 done" >> /workspace/logs/campaign.log
}
echo "[campaign $(date +%H:%M:%S)] PHASE E4: cross-eval sweep of 3 validated Llama-70B bf16 instruments (A add, H proj0, LIE cap) on 12 bio evals, E and N" >> /workspace/logs/campaign.log
MAXPS=100 XBASE=N,E XSTEER=E,N INSTR_FILE=/workspace/instr_llama.txt bash scripts/xeval_steer.sh llama-3.3-70b-instruct $OUT/letters $OUT/xeval letters 40 --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --gen-cap 128 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E4 done" >> /workspace/logs/campaign.log
