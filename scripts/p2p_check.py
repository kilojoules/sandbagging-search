#!/usr/bin/env python3
"""Pre-flight for multi-GPU pods: are GPU-to-GPU copies intact?  A host with broken peer-to-peer access (torch still reports
can_device_access_peer == True) silently corrupts every tensor a sharded model moves between its GPUs: hidden states turn into NaN / garbage.
Exit code 0 = all copies intact, 3 = corrupted (destroy the pod and take another host / GPU type), 2 = fewer than 2 GPUs.
  python3 scripts/p2p_check.py"""
import sys
import torch

n = torch.cuda.device_count()
if n < 2:
    print(f"[p2p] only {n} GPU(s) visible"); sys.exit(2)
print(f"[p2p] {n} GPUs: {[torch.cuda.get_device_name(i) for i in range(n)]}; peer access reported: "
      f"{[(i, j, torch.cuda.can_device_access_peer(i, j)) for i in range(n) for j in range(n) if i != j]}", flush=True)
torch.manual_seed(0)
bad = 0
trials = 0
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        for _ in range(20):
            x = torch.randn(4, 4096, 211, dtype=torch.bfloat16, device=f"cuda:{i}")
            y = x.to(f"cuda:{j}")
            z = y.to(f"cuda:{i}")
            torch.cuda.synchronize()
            trials += 1
            if not torch.equal(x, z) or bool(torch.isnan(y).any()):
                bad += 1
print(f"[p2p] roundtrip copies corrupted in {bad} of {trials} trials", flush=True)
sys.exit(0 if bad == 0 else 3)
