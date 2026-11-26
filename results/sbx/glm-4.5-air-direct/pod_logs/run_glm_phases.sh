#!/bin/bash
cd /workspace/sandbagging
export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTB=results/sbx/glm-4.5-air-direct/cot; mkdir -p $OUTB
echo "[campaign $(date +%H:%M:%S)] PHASE B1: reasoning-mode (non-thinking chain of thought) screen: N, E, natural trigger, twin, plain/hazard instructions"
python -m sbx.run --model glm-4.5-air --out $OUTB --preset screen --answer-mode native --evals wmdp_bio --splits disc --framings N,E,nat_goal_doc,neg_goal_doc_inverted,pos_bio_plain,pos_bio_hazard --max-per-split 120 --gen-cap 160 --max-minutes 60 >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE B1 done"
echo "[campaign $(date +%H:%M:%S)] PHASE A: forced full-sample campaign on the direct key (screen, kprobe, behavior, dirs, steer E/pos_generic/N, noise)"
python -m sbx.campaign --model glm-4.5-air-direct --out results/sbx/glm-4.5-air-direct/full --budget-hours 6 --force-steer --max-per-split 0 --noise-include attn --gen-cap 128 --mask-massive 20 --caps steer=240 --n-pairs 400 --steer-framings E,pos_generic,N,auto2 --rand-ctrl 2 --rand-for B,LIE,EXPERT >> /workspace/logs/campaign.log 2>&1
echo "[campaign $(date +%H:%M:%S)] PHASE A done"
