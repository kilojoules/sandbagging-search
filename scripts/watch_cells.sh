#!/bin/bash
# Stream only the NOTABLE steered cells of running pods:  scripts/watch_cells.sh KEY [KEY ...]
# notable = a steered cell (direction != none) whose paired change vs its unsteered baseline is >= 8 points in either direction,
# or whose answered fraction collapsed (< 0.9: the intervention broke generation).  Plain baselines and small effects are not streamed.
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_ed25519"
watch_one() {
  KEY="$1"
  while [ ! -s /tmp/sbx_$KEY.ep ]; do sleep 10; done
  read HOST PORT < /tmp/sbx_$KEY.ep
  while true; do
    ssh $SSHO -p "$PORT" root@"$HOST" 'tail -n 0 -F /workspace/logs/campaign.log 2>/dev/null' 2>/dev/null \
      | grep --line-buffered -E '^\[cell\] ' \
      | grep --line-buffered -v -E '\|none\|none\|-\|0.0\|0 ' \
      | awk -v k="$KEY" '{
          d=0; a=1; hasd=0
          if (match($0, /Δvs-base=[+-][0-9.]+/)) { s=substr($0, RSTART+11, RLENGTH-11); d=s+0; hasd=1 }
          if (match($0, / ans=[0-9.]+/)) { a=substr($0, RSTART+5, RLENGTH-5)+0 }
          if ((hasd && (d>=0.08 || d<=-0.08)) || a<0.9) { gsub(/\|0.0\|0 acc=/, " acc="); print "[" k "] " substr($0, 1, 230); fflush() }
        }'
    sleep 30
  done
}
for k in "$@"; do watch_one "$k" & done
wait
