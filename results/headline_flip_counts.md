| Model (answer mode) | Questions | Validated instruments run on the default prompt | **Wrong → right** | **Right → wrong** | Net (95% upper bound, in questions) | Instrument (largest net) | Same instrument when the model IS sandbagging: wrong → right / right → wrong |
|---|---|---|---|---|---|---|---|
| gpt-oss-120B | 150 | 50 | **2** | **0** | +2 (≤ ~5) | B cap L10-28 | 40 / 12 |
| gpt-oss-20B | 120 | 17 | **3** | **2** | +1 (≤ ~5) | REF proj0 L7-19 | 25 / 12 |
| GLM-4.5-Air 107B (generic instruction as validity framing) | 291 | 21 (16 informative) | **5** | **3** | +2 (≤ ~7) | A add L19 -0.4 | 58 / 9 |
| GLM-4.5-Air 107B (bio-specific instruction E as validity framing) | 291 | 1 | **5** | **5** | 0 (≤ ~6) | LIE add L19 -0.4 | 40 / 9 |
| Llama-3.3-70B bf16, direct answers | 291 | 39 (35 informative) | **5** | **2** | +3 (≤ ~8) | H proj0 L22-62 | 151 / 9 |
| Llama-3.3-70B bf16, visible reasoning | 120 | 2 | **3** | **3** | 0 (≤ ~5) | H proj0 L22-62 | 37 / 8 |
| Gemma-3-12B, direct answers | 291 | 26 | **7** | **2** | +5 (≤ ~11) | EXPERT add L20 -0.2 | 78 / 8 |
| Gemma-3-12B, visible reasoning † | 593 | 1 | **27** | **27** | 0 (≤ ~14) | CF cap L13-23 | 140 / 51 † |
| Gemma-3-27B, visible reasoning † | 593 | 1 | **29** | **17** | +12 (≤ ~25) | CF cap L17-30 | 283 / 47 † |
| GLM-4-9B | 120 | 6 | **10** | **7** | +3 (≤ ~11) | B add L17 -0.2 | 25 / 8 |
| Llama-3.1-8B | 120 | 3 | **9** | **6** | +3 (≤ ~11) | B add L13 -0.2 | 25 / 10 |

| Model (answer mode) | Instruments | Wrong → right, min–max (median) | Right → wrong, min–max (median) | Random directions on the default prompt: wrong → right / right → wrong |
|---|---|---|---|---|
| gpt-oss-120B | 50 | 0–5 (2) | 0–13 (5) | – |
| gpt-oss-20B | 17 | 3–9 (5) | 2–15 (8) | – |
| GLM-4.5-Air 107B (generic instruction as validity framing) | 16 | 0–9 (4.5) | 0–13 (4) | 1.4 / 0.6 (mean of 12 cells) |
| GLM-4.5-Air 107B (bio-specific instruction E as validity framing) | 1 | 5–5 (5) | 5–5 (5) | 1.4 / 0.6 (mean of 12 cells) |
| Llama-3.3-70B bf16, direct answers | 35 | 0–16 (2) | 0–28 (2) | 0.2 / 0.5 (mean of 18 cells) |
| Llama-3.3-70B bf16, visible reasoning | 2 | 3–3 (3) | 3–9 (6) | 1.5 / 8.0 (mean of 2 cells) |
| Gemma-3-12B, direct answers | 26 | 0–12 (5) | 0–14 (4) | 3.8 / 3.7 (mean of 12 cells) |
| Gemma-3-12B, visible reasoning † | 1 | 27 | 27 | 16 / 14, 10 / 19, 15 / 17 (3 random caps, same layers) |
| Gemma-3-27B, visible reasoning † | 1 | 29 | 17 | 22 / 8, 21 / 12, 14 / 9 (3 random caps, same layers) |
| GLM-4-9B | 6 | 4–12 (8.5) | 6–10 (7) | – |
| Llama-3.1-8B | 3 | 6–9 (6) | 4–9 (6) | – |
