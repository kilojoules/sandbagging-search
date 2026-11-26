#!/bin/bash
# Compact table of the baseline cells a running campaign has written so far:  scripts/peek_pod.sh KEY [mode]
KEY="$1"; MODE="${2:-native}"
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_ed25519"
read HOST PORT < /tmp/sbx_$KEY.ep
ssh $SSHO -p "$PORT" root@"$HOST" "python3 - <<'PY'
import json, os
p='/workspace/sandbagging/results/sbx/$KEY/$MODE/cells.jsonl'
if not os.path.exists(p): print('no cells yet'); raise SystemExit
rows=[json.loads(l) for l in open(p) if l.strip()]
base={(r['eval'],r['split'],r['framing']):r for r in rows if r['direction']=='none' and r['intervention']=='none'}
print('%-14s %-5s %-24s %6s %5s %5s %8s %4s %7s' % ('eval','split','framing','acc','ans','ref','acc|ans','n','d_vs_N'))
for (ev,sp,fr),r in sorted(base.items()):
    n=base.get((ev,sp,'N'))
    d='' if (n is None or fr=='N') else '%+.3f'%(r['acc']-n['acc'])
    aa=r.get('acc_answered')
    print('%-14s %-5s %-24s %6.3f %5.2f %5.2f %8s %4d %7s' % (ev,sp,fr,r['acc'],r['parse_rate'],r.get('refusal_rate',0),'nan' if aa is None else '%.3f'%aa,r['n'],d))
print('cells total:', len(rows))
PY" 2>&1 | grep -v Warning
