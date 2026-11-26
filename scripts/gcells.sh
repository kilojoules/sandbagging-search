#!/bin/bash
# Compact view of the steered cells of a running pod:  scripts/gcells.sh KEY [N_LINES]
KEY="$1"; N="${2:-30}"
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_ed25519"
read HOST PORT < /tmp/sbx_$KEY.ep
ssh $SSHO -p "$PORT" root@"$HOST" 'grep -a "^\[cell\]" /workspace/logs/campaign.log' 2>&1 | grep -v Warning \
 | sed -E 's/\[cell\] //;s/\|0\.0\|0 acc=/ acc=/;s/\|0 acc=/ acc=/;s/ ans=1\.00 ref=0\.00 acc\|ans=[0-9.]+//;s/ \([0-9.]+m, total [0-9]+m\)//' | cut -c1-150 | tail -n "$N"
