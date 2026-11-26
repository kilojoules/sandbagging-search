#!/bin/bash
# After phase_e3.sh (random controls + native screen): E3c small-sigma weight noise, E5 native (visible reasoning) steering of 3 instruments on wmdp_bio, E4 letters-mode cross-eval of 4 instruments on 12 bio evals.
cd /workspace/sandbagging
while ps -eo cmd | grep -qE "[p]hase_e3.sh|[s]bx.campaign --model llama-3.3-70b-instruct|[s]bx.run --model llama-3.3-70b-instruct"; do sleep 20; done
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/sbx/glm-4.5-air-direct/llama70b-bf16
COMMON="--model llama-3.3-70b-instruct --repo unsloth/Llama-3.3-70B-Instruct --quant bf16"
echo "[campaign $(date +%H:%M:%S)] PHASE E3c: weight noise at small sigma (0.0025,0.005,0.01 of per-tensor rms; 0.02 collapsed the model to chance), N and E, wmdp_bio + mmlu_nonbio, 2 seeds" >> /workspace/logs/campaign.log
python -m sbx.run $COMMON --out $OUT/letters --preset noise --answer-mode letters --max-minutes 30 --max-per-split 0 --gen-cap 128 --noise-kinds weight --noise-sigmas-weight 0.0025,0.005,0.01 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E3c done" >> /workspace/logs/campaign.log
echo "[campaign $(date +%H:%M:%S)] PHASE E5: native (visible reasoning) steering, WMDP-bio disc n=120: A add L34, H proj0 22-62, B cap 22-62 + 2 random directions each, E and N" >> /workspace/logs/campaign.log
XEVALS=wmdp_bio MAXPS=120 XBASE=N,E XSTEER=E,N INSTR_FILE=/workspace/instr_llama_native.txt bash scripts/xeval_steer.sh llama-3.3-70b-instruct $OUT/letters $OUT/native native 30 --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --gen-cap 128 --splits disc --grid-splits disc --rand-ctrl 2 --rand-for A,H,B >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E5 done" >> /workspace/logs/campaign.log
echo "[campaign $(date +%H:%M:%S)] PHASE E4: cross-eval sweep of 4 validated Llama-70B bf16 instruments (A add, H proj0, LIE cap, REF add) on 12 bio evals, E and N" >> /workspace/logs/campaign.log
MAXPS=100 XBASE=N,E XSTEER=E,N INSTR_FILE=/workspace/instr_llama.txt bash scripts/xeval_steer.sh llama-3.3-70b-instruct $OUT/letters $OUT/xeval letters 40 --repo unsloth/Llama-3.3-70B-Instruct --quant bf16 --gen-cap 128 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE E4 done" >> /workspace/logs/campaign.log
