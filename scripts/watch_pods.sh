#!/bin/bash
# Stream the milestone lines of several campaign logs, prefixed with the model key.  Waits for /tmp/sbx_<key>.ep (written by pod_launch.sh).
#   scripts/watch_pods.sh KEY [KEY ...]
# Filter: campaign driver lines, step/stage milestones, warnings, errors, tracebacks, gate results.  [cell] lines are NOT streamed.
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_ed25519"
watch_one() {
  KEY="$1"
  while [ ! -s /tmp/sbx_$KEY.ep ]; do sleep 10; done
  read HOST PORT < /tmp/sbx_$KEY.ep
  while true; do
    ssh $SSHO -p "$PORT" root@"$HOST" 'tail -n 0 -F /workspace/logs/campaign.log 2>/dev/null' 2>/dev/null \
      | grep --line-buffered -E '^\[campaign|^\[natdirs|^\[masked|^\[calib\] --mask|^\[oom\]|^\[(warn|error|replicate|stop|done)|\[kprobe\] decision|\[dirs\] recipe|\[refusal-validation|\[behavior\] stage B|Traceback|Error:|OutOfMemory' \
      | grep --line-buffered -v -E 'validation grid|hidden states |generation flags are not valid|layers \[' \
      | sed -u "s/^/[$KEY] /"
    sleep 30     # ssh dropped: reconnect (the log is append-only, tail -n 0 only misses what happened while disconnected)
  done
}
for k in "$@"; do watch_one "$k" & done
wait
