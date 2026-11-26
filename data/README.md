# data/

The benchmark question files that the runs read are **not redistributed in this repository**.

* **GPQA** (`data/hard/gpqa_biology.jsonl`, `gpqa_control.jsonl`): the dataset is gated and its authors ask that its examples are not posted in
  plain text online, to keep them out of training corpora.
* **Everything else** is a public benchmark with its own licence (below). It is fetched from its source rather than copied here, and raw
  questions of hazard-proxy benchmarks (WMDP) do not belong in a results repository.

`sbx/evals.py` expects these paths, and its registry reads every registered file on first use, so rebuild or fetch all of them before running an
eval (`python3 -m sbx.evals check` verifies the layout).

| path | what it is | source and licence |
|---|---|---|
| `data/wmdp_train.jsonl`, `data/questions_heldout_eval.jsonl` | WMDP-bio (1273 questions = 1073 + 200) and 200 MMLU bio / medical questions | `cais/wmdp` `wmdp-bio` (MIT); `cais/mmlu` (MIT) |
| `data/questions_control_mathphys.jsonl` | 400 MMLU physics, maths, astronomy and ML test questions (non-bio control) | `cais/mmlu` (MIT) |
| `data/questions_pilot.jsonl` | 250 WMDP + 250 MMLU questions used for early exploration (not a registered eval) | `cais/wmdp`, `cais/mmlu` (MIT) |
| `data/hard/mmlupro_biology.jsonl` | 200 of the 717 MMLU-Pro biology questions | `TIGER-Lab/MMLU-Pro` (MIT) |
| `data/hard/gpqa_biology.jsonl`, `gpqa_control.jsonl` | GPQA main: 78 biology, 76 chemistry, 74 physics questions | `Idavidrein/gpqa` (CC BY 4.0, gated, see above) |
| `data/hard/labbench_*.jsonl` | LAB-Bench SeqQA, LitQA2, ProtocolQA, CloningScenarios | `futurehouse/lab-bench` (CC BY-SA 4.0) |
| `data/evals/*.jsonl` | MMLU bio / medical subjects, MedQA, MedMCQA, PubMedQA, SuperGPQA bio, MedXpertQA, WMDP-chem / cyber, MMLU and MMLU-Pro non-bio controls | rebuilt by `python3 -m sbx.evals build` (deterministic: fixed seed, pinned dataset revisions; needs network, the `datasets` package, and an HF token for gated sets) |
| `data/dirs/*.jsonl` | prompt sets for the refusal and honesty directions | built by the dataset builders in `sbx/dirs/refusal.py` and `sbx/dirs/honesty.py`; see `docs/dirs_refusal_evalaware.md`, `docs/dirs_honesty_persona.md` |

Source, licence and selection rule of every registered eval: the `origin` and `licence` fields in `sbx/evals.py`, and the table in `docs/evals.md`.
The per-question results in `results/sbx/` refer to questions only by id (`qids/`), so the analysis scripts run without these files.
