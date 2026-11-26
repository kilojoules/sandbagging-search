# Cross-eval table (results/sbx/gptoss-120b/xeval)

Per eval (disc + rep pooled): baseline accuracy N / E; for each instrument the change under E (restoration) and under N (default), in points, with 95% CI for N.


## instrument: EXPERT / add / layers 15 / strength -0.4

| eval | n | N acc | E acc | restore under E | change under N [95% CI] |
|---|---|---|---|---|---|
| gpqa_bio | 59 | 0.644 | 0.390 | +10.2 | -6.8 [-18.6,+5.1] |
| labbench_litqa2 | 144 | 0.410 | 0.229 | +19.4 | +4.2 [-4.2,+12.5] |
| labbench_protocolqa | 73 | 0.438 | 0.329 | +5.5 | +4.1 [-6.8,+15.1] |
| labbench_seqqa | 74 | 0.595 | 0.338 | +0.7 | -12.2 [-24.3,+1.4] |
| mmlupro_bio | 147 | - | 0.381 | +37.4 | - |

  -> restores E by >= +10 pts on 2/4 evals; on those, the largest change under N is +4.2 [-4.2,+12.5] (labbench_litqa2, n=144); evals with N-change CI above 0: 0

## instrument: LIE / proj0 / layers 10-28 / strength 0.0

| eval | n | N acc | E acc | restore under E | change under N [95% CI] |
|---|---|---|---|---|---|
| gpqa_bio | 59 | 0.644 | 0.390 | +30.5 | +0.0 [-8.5,+10.2] |
| labbench_litqa2 | 144 | 0.410 | 0.229 | +14.6 | +0.7 [-5.6,+6.9] |
| labbench_protocolqa | 73 | 0.438 | 0.329 | +13.7 | +8.2 [+0.0,+16.4] |
| labbench_seqqa | 142 | 0.549 | 0.338 | +16.2 | -4.2 [-12.0,+2.8] |
| mmlupro_bio | 70 | 0.857 | 0.381 | +42.2 | +0.0 [-5.7,+5.7] |

  -> restores E by >= +10 pts on 5/5 evals; on those, the largest change under N is +8.2 [+0.0,+16.4] (labbench_protocolqa, n=73); evals with N-change CI above 0: 0
