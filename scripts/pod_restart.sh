#!/bin/bash
# Restart the campaign driver of a running model pod with the CURRENT synced code (cells already on the pod are reused):
#   scripts/pod_restart.sh KEY BUDGET_HOURS [campaign args...]
KEY="$1"; HOURS="$2"; shift 2
CAMP_ARGS="$*"
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_ed25519"
read HOST PORT < /tmp/sbx_$KEY.ep
ssh $SSHO -p "$PORT" root@"$HOST" 'ps -eo pid,cmd | awk "/[s]bx.campaign --model/ || /[s]bx.run --model/ {print \$1}" | xargs -r kill; sleep 3; ps -eo pid,cmd | grep -c "[s]bx\."' 2>&1 | grep -v Warning | sed "s/^/[$KEY] remaining sbx procs: /"
(ssh $SSHO -p "$PORT" root@"$HOST" "cd /workspace/sandbagging && cp -f /workspace/logs/campaign.log /workspace/logs/campaign.prev.\$(date +%H%M).log 2>/dev/null; export HF_TOKEN=\$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0 USE_TF=0 USE_TORCH=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; setsid nohup python -m sbx.campaign --model $KEY --out results/sbx/$KEY --budget-hours $HOURS $CAMP_ARGS > /workspace/logs/campaign.log 2>&1 < /dev/null & disown; echo started" > /dev/null 2>&1 &)
sleep 10
ssh $SSHO -p "$PORT" root@"$HOST" 'ps -eo pid,etime,cmd | grep -E "[s]bx\.(campaign|run)" | cut -c1-170; tail -n 1 /workspace/logs/campaign.log | cut -c1-200' 2>&1 | grep -v Warning | sed "s/^/[$KEY] /"
