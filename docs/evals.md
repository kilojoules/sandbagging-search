# Evals: registry, splits, pools (`sbx/evals.py`)

One module answers three questions: *which questions may a run evaluate on* (`get_eval`), *which questions may be used to build
directions* (`get_pool`), and *which split is a given question in* (`split_of`). Data live in the existing files plus 18 new,
small, deterministic files under `data/evals/` (3.8 MB in total, none larger than 650 KB).

```python
from sbx.evals import EVALS, get_eval, get_pool, split_of, split_sizes, list_evals

qs = get_eval("wmdp_bio", "disc")        # evaluation questions, fixed pseudo-random order (a prefix is an unbiased subsample)
qs = get_eval("mmlu_virology", "rep")    # splits: "disc" | "rep" | "all"      ("all" = disc + rep, NEVER dir)
pool = get_pool()                        # {"wmdp": [...], "hard": [...], "nonbio": [...]}  direction building only (dir bucket)
split_of(q)                              # "dir" | "disc" | "rep"
cluster_of(q)                            # id of the group of questions that share context (one protocol / plasmid); see section 1
EVALS["wmdp_cyber"].category, .hazard, .pool, .n, .files, .licence, .origin, .notes, .clustered
list_evals("bio")                        # names, registry order; union views only with include_unions=True
```

Everything returned is a fresh copy of `{"source", "qid", "question", "choices", "answer"}` (answer = 0-based int, 2..10 unique
non-empty options). `get_eval(name, "dir")` raises on purpose: dir questions are only reachable through `get_pool()`.
`python3 -m sbx.evals check | report | docs | counts | build` are the diagnostics / rebuild commands (section 6). The registry is
global (de-duplication and clustering need every file): the first call reads all registered files and raises if one is missing
(about 1.5 s once per process: most of it is the cluster rule on the three LAB-Bench sets), so sync all of `data/` (including
`data/evals/`) to a pod. Two guards apply to everything the module returns: the **cluster rule** (section 1: questions that share
a protocol / plasmid are split together) and the **prompt-length cap** (section 3: questions over 7500 characters are not served).

## 1. Splits

`b = zlib.crc32(f"{source}|{qid}".encode()) % 100`; `b < 25` is **dir** (direction pool), `b < 63` is **disc** (discovery),
otherwise **rep** (replication). It depends only on `(source, qid)`: no tokenizer, no RNG, no process state.

**WMDP-bio special case** (the pool already used for the gpt-oss-120b directions): `dir` = the first 680 rows of
`data/wmdp_train.jsonl` (qids `t0000`..`t0679`), regardless of their hash. Every other WMDP-bio question (the remaining 393 train rows and
the 200 `wmdp/bio` rows of `data/questions_heldout_eval.jsonl`) is split disc/rep by the same bucket: **disc if b < 50 else rep**
(not the 25/63 cut, which would give 63/37; this keeps the halves balanced: 291 disc / 302 rep). `wmdp_bio` has no other dir questions.

Achieved proportions over all file-based sets except `wmdp_bio`: dir 25.6% / disc 37.7% / rep 36.7% (n = 6616) against the 25/38/37
design. Individual small sets fluctuate a lot (binomial noise): e.g. `mmlu_virology` is 32 / 72 / 43, `mmlu_med_genetics` 19 / 29 / 40.

**Clusters (`EvalSpec.clustered`: `labbench_seqqa`, `labbench_protocolqa`, `labbench_cloning`).** The literal rule treats every
question as an independent draw, which is false for LAB-Bench: ProtocolQA asks up to 16 questions about **one** protocol text (108
questions = 25 protocols), CloningScenarios reuses the same plasmid sequences (33 questions = 5 scenario families). With the literal
rule 40 of the 49 ProtocolQA rep questions were about a protocol that also has a disc question, and 66 of the 85 served questions
about a protocol of a dir question (CloningScenarios: 16 of 17 rep questions shared a scenario with a disc question). Consequences:
a discovery fluke specific to one protocol "replicates" on rep (the replication step is no longer independent evidence), the McNemar
test sees far fewer independent units than questions, and a direction built from dir questions is evaluated on questions that quote
the very text it was built from. For a clustered eval:

* two stems are **linked** when they share at least 300 distinct character 12-grams (whitespace collapsed, case kept) and at least
  half of the shorter stem's 12-grams. A short and an extended version of one protocol link (containment, not similarity: Jaccard
  would call them different); two questions built from one template about different genes or sequences do not (they share too little
  text): task-type similarity is not shared context, and SeqQA's "clone gene X into pUC19" instances are separate problems;
* the clusters are the connected components, and every member takes the bucket of the cluster's **representative**, its member with
  the lowest qid: `b = crc32(f"{source}|{representative qid}") % 100` and the usual 25 / 63 cuts. So the rule is unchanged except for
  the id that is hashed; singletons, and every eval that is not clustered, follow the literal rule exactly (a test checks both);
* the clusters are computed over all valid rows of the eval *before* de-duplication and before the length cap, so neither changes a
  split. `cluster_of(q)` returns the id (`source|representative qid`) for cluster-level inference.

The parameters sit on a plateau, not on an edge: ProtocolQA / CloningScenarios / SeqQA give the same 25 / 5 / 196 clusters for every
n-gram length from 8 to 24, every minimum-shared count from 200 to 500 and every overlap from 0.4 to 0.65 (the pair overlaps have an
empty gap there: ProtocolQA nothing between 0.12 and 0.75, CloningScenarios nothing between 0.40 and 0.60; SeqQA's 4 linked pairs are
all above 0.9; a test pins the stability of the partition). An overlap threshold of 0.8 misses three questions about one short H&E protocol (overlap 0.78 to
0.81, the question sentence is a large part of a short stem). ProtocolQA's clusters equal its protocol titles (24 titles; the generic
title "Section:" covers two different protocols), which a test uses as an independent oracle.

What this fixes and what it does not. It makes disc, rep and dir **cluster-disjoint** (verified against a naive pairwise check on
every row of the three sets) and it deviates from the literal SPEC rule only for the questions of multi-member clusters (the cluster rule
gives a different split than the literal rule to 2 of the 200 SeqQA questions, 49 of the 108 ProtocolQA and 12 of the 33 CloningScenarios questions). It does **not** make the questions *inside* one split independent: after
the cap ProtocolQA's disc split is 24 questions in 7 protocols and its rep split 49 questions in 12, so the effective sample size is
about the number of protocols and a question-level McNemar is anticonservative there. Treat the clustered evals as exploratory, or
test at the cluster level (cluster bootstrap / permutation over `cluster_of`); the module cannot do that for the analysis. Cluster
sizes make the split sizes lumpy (ProtocolQA 12 / 24 / 49 instead of 25 / 38 / 37 per cent). SeqQA stays templated, deliberately: the 71
disc / rep pairs with a character 8-gram Jaccard of at least 0.8 are all short stems (about 200 characters) of one template ("I want to
clone the X gene from E. coli into the plasmid pUC19 ...") about different genes, none identical. Those are separate problems from one task
type, not one text, so an effect on that task type replicating from disc to rep is a real finding about the task type; the effective
diversity of SeqQA is its number of templates, not 193. Other evals are not clustered
(`wmdp_bio` is fixed by file order; elsewhere a handful of passages is shared by 2 to 4 questions, see section 7).

**Order.** `get_eval` serves each split in a fixed pseudo-random order (`crc32("order|source|qid")`), so `--max-per-split` truncation is
an unbiased subsample even for pooled sets (e.g. the first 25 `mmlu_nonbio` disc questions already span at least 5 of its 8 subjects). `get_pool()["wmdp"]` is
the 680 rows in **file order** (identical content); `"hard"` and `"nonbio"` are de-duplicated and pseudo-randomly ordered.

## 2. De-duplication and leakage control

Identity of a question = its normalised question text (`question_key`: NFKC, casefold, punctuation / whitespace collapsed; the
options are *not* part of the key, so a question re-published with regenerated options, e.g. MMLU -> MMLU-Pro, is the same
question). Groups of equal key are resolved across **all** registered evals, in registry order then file order:

* if some member is a **dir** question by the split rule: all dir members are kept (as dir) and every other member is dropped
  (`duplicate_of_dir`) -- this is what guarantees that a dir-pool question never reappears in a disc / rep eval of any set;
* otherwise only the first member is kept, in the split its own hash gives (`duplicate`) -- this guarantees that no question is in
  two splits and that disc / rep evals do not overlap each other.

Dir members are never dropped (a duplicate inside the dir pool is harmless), which keeps the WMDP pool *exactly* the first 680 rows.
Rows that violate the schema (here: duplicated options) are dropped and listed. `split_of(q)` is registry-aware: a question whose
key is registered gets its group's split, so a copy living in some other file (e.g. the pilot file) is also dir if its twin is dir.

What the registry did on the current data (`python3 -m sbx.evals report`; a test pins these numbers):

* Rows read: 7964 (non-union evals). Invalid rows dropped: **5** (all "duplicated options": `labbench_litqa2_0011`,
  `labbench_litqa2_0033`, `gpqa_control_0044`, `gpqa_control_0123`, `c0223` of `mathphys`; in `labbench_litqa2_0011`,
  `labbench_litqa2_0033` and `gpqa_control_0123` only distractors are repeated, in `gpqa_control_0044` and `c0223` the correct option has an
  identical twin, so those two are ill-posed).
* Cross-set duplicate groups: **7** -- all between `mmlu_bio` and `mmlupro_bio` (MMLU-Pro re-uses MMLU questions with extra
  options; about 30% of MMLU-Pro biology originates from MMLU).
* Within-set duplicate groups: **9** -- 6 in `mmlupro_bio`, 1 in `mmlu_bio` (`h0363` repeats `h0295`), 1 in `mathphys`
  (meteorite question), 1 in `labbench_seqqa`.
* Duplicate questions dropped: **20** (11 `duplicate_of_dir`, 9 `duplicate`): `mmlupro_bio` 10, `labbench_seqqa` 7, `mmlu_bio` 3.
* Questions removed by the prompt-length cap: **50** (`labbench_protocolqa` 23, `labbench_cloning` 27; section 3). They are applied
  *after* the de-duplication and the split assignment, so removing one never changes another question's split and never lets a copy of
  a removed dir question into disc / rep (`split_of` still answers for it; a test builds that situation). Every row read is either kept,
  invalid, a dropped duplicate or too long (`rows = n + invalid + duplicate + too long` per eval, checked).
* The new files are built to be disjoint from every pre-existing registered file *and* from `data/questions_pilot.jsonl`, so they
  contribute no duplicates (a test checks this); the builder also removes within-set repeated stems (`duplicate_within_set` in the
  build log: e.g. 155 in the MMLU-Pro non-bio candidates, 8 in the MedMCQA candidates).

Side effect of stem-only identity: `labbench_seqqa` has 10 questions with one boilerplate stem that differ only in the option
sequences; because other members of that group are dir questions, 7 were dropped from disc / rep (193 of 200 SeqQA questions remain).
This is deliberate (literal "no repeated question text"), but they are not true duplicates.

**Near-duplicates (not removed).** `near_duplicate_report(threshold)` compares every served question with the direction pools by Jaccard
similarity of word 3-gram sets of the stems (stems under 6 words skipped). At 0.5, 157 of 5515 served questions are similar to a
pool question: `labbench_seqqa` 142 / 142 (templated: the stems come from a few templates but are different problems, section 1),
`wmdp_bio` 8 / 593, `mmlu_nonbio` 4 / 294, `mmlu_bio`, `mmlupro_bio` and `mathphys` 1 each. At 0.8: 39 (SeqQA 36, `mmlu_nonbio` 3).
Before the cluster rule this was 226 of 5550 at 0.5, of which `labbench_protocolqa` 66 / 85 (57 / 85 at 0.8, the protocol text of a dir
question reappeared in disc / rep) and `labbench_cloning` 3 / 29: both are 0 now. Topic-level overlap (WMDP clusters generated from the
same source papers) is *not* detected by this measure and certainly exists.

**Pilot overlap (pre-existing sets, not fixed).** `data/questions_pilot.jsonl` (250 WMDP + 250 MMLU, used for exploration) is not a
registered eval, but 77 questions of `questions_heldout_eval.jsonl` (44 WMDP, 33 MMLU) and 206 of `wmdp_train.jsonl` (126 in the
first 680 = dir pool, 80 in the rest) also occur in it. So 124 of the 593 `wmdp_bio` disc / rep questions and 21 of `mmlu_bio`'s served
questions were seen during exploration. The new sets exclude all pilot questions.

## 3. Direction pools

| pool | built from (dir bucket of) | size |
|---|---|---|
| `wmdp` | `wmdp_bio` (exactly the first 680 rows of `wmdp_train.jsonl`, file order) | 680 |
| `hard` | `labbench_seqqa`, `labbench_litqa2`, `labbench_protocolqa`, `labbench_cloning`, `mmlupro_bio`, `gpqa_bio`, `supergpqa_bio` | 281 (supergpqa 107, LitQA2 53, SeqQA 49, MMLU-Pro 41, GPQA 19, ProtocolQA 12, Cloning 0) |
| `nonbio` | `mmlu_nonbio`, `mathphys` | 213 |

The dir buckets of all other sets (`mmlu_*`, `medqa_usmle`, `medmcqa_bio`, `pubmedqa`, `medxpertqa_text`, `gpqa_control`,
`wmdp_cyber`, `wmdp_chem`, `mmlupro_nonbio`) are in no pool: those questions are simply never evaluated or used (change `pool=` in
`EVALS` if a recipe needs them; evaluation sets of other evals are unaffected because the split rule does not depend on pools).
Before the two guards of this module the `hard` pool had 296 questions, among them `labbench_cloningscenarios_0026` (51k characters,
26k gpt-oss tokens) and 8 more prompts of over 3000 tokens; `sandbag_hard` runs `Ctx.run` and `Ctx.hidden_last` on the whole pool.

### Prompt-length cap (`MAX_PROMPT_CHARS = 7500`)

Every question with `prompt_chars(q) = len(question) + sum(len(choices)) > 7500` is removed from every split and from the pools and
listed in `registry_report()["too_long"]` (50 questions: ProtocolQA 23, CloningScenarios 27; no other eval has one; the served
`labbench_protocolqa` goes from 85 to 73 questions and `labbench_cloning` from 29 to 6, together with the cluster rule of section 1).

*Why.* The pre-existing LAB-Bench files contain prompts of up to 51k characters, and `labbench_cloning` served 23 of 29 prompts of over
3000 tokens (up to 10k). With eager attention and 64 heads one 26k-token prompt needs about 88 GB of attention scores per layer
(64 x 26259^2 x 2 B), and `Ctx.hidden_last` also keeps every layer's hidden states and all-position logits: `Ctx._gen_chunk` raises
"OOM on a single prompt" (shortest prompts first, so after the rest of the cell's work) and `hidden_last` has no OOM handling, so one
prompt loses a whole recipe or cell. The legacy pipeline skipped prompts of more than 3000 tokens (`scripts/gptoss_dir.py`: `MAX_TOK`,
`short_enough`, applied to every pool and eval set); the registry had dropped that filter and nothing downstream re-adds it (`run.py` and
the recipes take `get_eval` / `get_pool` as given), so the cap lives here, in the one place every caller goes through.

*Why characters, and why 7500.* The cap must not depend on the tokenizer: every model has to see the same questions or the cross-model
comparison and the pre-registered question sets break. 7500 characters is the stand-in for 3000 tokens: on the registered data the two
filters keep **exactly the same questions** in the gpt-oss tokenizer (the longest kept question has 7431 characters, the shortest removed
one 7522; a test checks the equivalence when the tokenizer is cached), and every kept question builds a framing-E prompt of at most
3000 tokens in the gpt-oss, Llama-3 and Gemma-3 tokenizers (the longest: 2939, 2892 and 2474 tokens; also tested). Characters per token on the long questions range from
1.8 (DNA) to 5 (prose), so a pure-DNA prompt of 7500 characters would exceed 3000 tokens; the data has none, and the test fails if one
appears. The builder's own `MAX_CHARS = 6000` for new files is stricter and unchanged (a test keeps it at or below the registry cap).

*Consequences.* The ProtocolQA results cover the shorter protocols only (median served prompt 4.4k characters against 4.6k in the file).
`labbench_cloning` keeps 6 questions, all in rep, so it has **no dir and no disc question and cannot be used for discovery**; it stays
registered (SPEC name) so that raising the cap brings it back (lumpy even then: its 5 scenario families fall into rep and disc only).
*To relax the cap* on a bigger GPU edit `MAX_PROMPT_CHARS`, re-pin `_COUNTS` (`python3 -m sbx.evals counts`) and regenerate the tables
below (`python3 -m sbx.evals docs`); `hidden_last` memory grows with the longest prompt, so check it for your model first (the model cards'
`direction_building_memory.max_single_prompt_tokens_estimate` gives the limit per model: about 20k tokens for the 32B models on one 80 GB
card, and eager attention needs more on top).

## 4. The sets

Columns: `clustered` = questions that share a protocol / plasmid are split together (section 1); `n` = all questions of the eval after validation, de-duplication and the length cap (dir + disc + rep, pinned in `EvalSpec.n` and
tested; the evaluable ones are disc + rep); `rows read` = rows in the source file(s) (= `n` + invalid + duplicate + too long); `too long` =
questions over `MAX_PROMPT_CHARS`, removed (section 3); `options` = number of options per question; `chance` = mean 1/#options over the served
(disc + rep) questions; `top letter` = most frequent correct position and its share (a model that collapses onto one letter after
steering scores about that share, which can fake or mask effects on position-skewed sets).

**Flags.** `category` bio / control. `hazard` is True only for sets built as a proxy for hazardous knowledge (WMDP bio / cyber / chem);
all other sets are False, including borderline ones: `mmlu_virology` (pathogen topics at textbook level), `labbench_protocolqa` and
`labbench_cloning` (wet-lab protocols and cloning design, used in dangerous-capability evaluations but built to measure practical
research skill). Flip `EvalSpec.hazard` if you want a broader definition. `mmlu_bio_all` is a **union view**: the same questions as its ten members
(no new data, no extra dedupe), for statistical power; never analyse a view together with its members.

<!-- BEGIN GENERATED (python3 -m sbx.evals docs) -->
| eval | category | hazard | pool | clustered | n | dir | disc | rep |
|---|---|---|---|---|---|---|---|---|
| `wmdp_bio` | bio | yes | wmdp | - | 1273 | 680 | 291 | 302 |
| `mmlu_bio` | bio | no | - | - | 197 | 54 | 84 | 59 |
| `mmlupro_bio` | bio | no | hard | - | 190 | 43 | 70 | 77 |
| `gpqa_bio` | bio | no | hard | - | 78 | 19 | 29 | 30 |
| `labbench_seqqa` | bio | no | hard | yes | 193 | 51 | 74 | 68 |
| `labbench_litqa2` | bio | no | hard | - | 197 | 53 | 63 | 81 |
| `labbench_protocolqa` | bio | no | hard | yes | 85 | 12 | 24 | 49 |
| `labbench_cloning` | bio | no | hard | yes | 6 | 0 | 0 | 6 |
| `gpqa_control` | control | no | - | - | 148 | 47 | 48 | 53 |
| `mathphys` | control | no | nonbio | - | 399 | 108 | 151 | 140 |
| `mmlu_college_bio` | bio | no | - | - | 110 | 26 | 44 | 40 |
| `mmlu_hs_bio` | bio | no | - | - | 238 | 53 | 108 | 77 |
| `mmlu_virology` | bio | no | - | - | 147 | 32 | 72 | 43 |
| `mmlu_med_genetics` | bio | no | - | - | 88 | 19 | 29 | 40 |
| `mmlu_anatomy` | bio | no | - | - | 104 | 29 | 42 | 33 |
| `mmlu_clinical` | bio | no | - | - | 208 | 58 | 84 | 66 |
| `mmlu_college_med` | bio | no | - | - | 70 | 20 | 26 | 24 |
| `mmlu_prof_med` | bio | no | - | - | 303 | 75 | 118 | 110 |
| `mmlu_nutrition` | bio | no | - | - | 248 | 62 | 96 | 90 |
| `mmlu_bio_all` | bio | no | - | - | 1713 | 428 | 703 | 582 |
| `medqa_usmle` | bio | no | - | - | 400 | 103 | 141 | 156 |
| `medmcqa_bio` | bio | no | - | - | 400 | 98 | 165 | 137 |
| `pubmedqa` | bio | no | - | - | 400 | 113 | 135 | 152 |
| `supergpqa_bio` | bio | no | hard | - | 400 | 107 | 133 | 160 |
| `medxpertqa_text` | bio | no | - | - | 400 | 101 | 144 | 155 |
| `wmdp_cyber` | control | yes | - | - | 400 | 90 | 156 | 154 |
| `wmdp_chem` | control | yes | - | - | 407 | 110 | 164 | 133 |
| `mmlu_nonbio` | control | no | nonbio | - | 400 | 106 | 154 | 140 |
| `mmlupro_nonbio` | control | no | - | - | 400 | 105 | 138 | 157 |

| eval | rows read | invalid | duplicate | too long | options | chance | top letter |
|---|---|---|---|---|---|---|---|
| `wmdp_bio` | 1273 | 0 | 0 | 0 | 4 | 25.0% | A 26% |
| `mmlu_bio` | 200 | 0 | 3 | 0 | 4 | 25.0% | D 29% |
| `mmlupro_bio` | 200 | 0 | 10 | 0 | 3-10 | 10.9% | B 14% |
| `gpqa_bio` | 78 | 0 | 0 | 0 | 4 | 25.0% | B 29% |
| `labbench_seqqa` | 200 | 0 | 7 | 0 | 4-5 | 24.6% | D 29% |
| `labbench_litqa2` | 199 | 2 | 0 | 0 | 2-8 | 24.8% | B 27% |
| `labbench_protocolqa` | 108 | 0 | 0 | 23 | 4-7 | 23.3% | D 27% |
| `labbench_cloning` | 33 | 0 | 0 | 27 | 4-7 | 23.2% | D 50% |
| `gpqa_control` | 150 | 2 | 0 | 0 | 4 | 25.0% | C 28% |
| `mathphys` | 400 | 1 | 0 | 0 | 4 | 25.0% | D 33% |
| `mmlu_college_bio` | 110 | 0 | 0 | 0 | 4 | 25.0% | D 31% |
| `mmlu_hs_bio` | 238 | 0 | 0 | 0 | 4 | 25.0% | D 36% |
| `mmlu_virology` | 147 | 0 | 0 | 0 | 4 | 25.0% | A 32% |
| `mmlu_med_genetics` | 88 | 0 | 0 | 0 | 4 | 25.0% | B 33% |
| `mmlu_anatomy` | 104 | 0 | 0 | 0 | 4 | 25.0% | C 33% |
| `mmlu_clinical` | 208 | 0 | 0 | 0 | 4 | 25.0% | D 33% |
| `mmlu_college_med` | 70 | 0 | 0 | 0 | 4 | 25.0% | C 30% |
| `mmlu_prof_med` | 303 | 0 | 0 | 0 | 4 | 25.0% | D 43% |
| `mmlu_nutrition` | 248 | 0 | 0 | 0 | 4 | 25.0% | D 28% |
| `mmlu_bio_all` | 1716 | 0 | 3 | 0 | 4 | 25.0% | D 31% |
| `medqa_usmle` | 400 | 0 | 0 | 0 | 4 | 25.0% | A 30% |
| `medmcqa_bio` | 400 | 0 | 0 | 0 | 4 | 25.0% | A 34% |
| `pubmedqa` | 400 | 0 | 0 | 0 | 3 | 33.3% | A 52% |
| `supergpqa_bio` | 400 | 0 | 0 | 0 | 4-10 | 10.3% | A 13% |
| `medxpertqa_text` | 400 | 0 | 0 | 0 | 10 | 10.0% | A 12% |
| `wmdp_cyber` | 400 | 0 | 0 | 0 | 4 | 25.0% | D 28% |
| `wmdp_chem` | 407 | 0 | 0 | 0 | 4 | 25.0% | B 26% |
| `mmlu_nonbio` | 400 | 0 | 0 | 0 | 4 | 25.0% | D 29% |
| `mmlupro_nonbio` | 400 | 0 | 0 | 0 | 4-10 | 11.9% | C 14% |

| eval | files | origin | licence |
|---|---|---|---|
| `wmdp_bio` | `data/wmdp_train.jsonl`, `data/questions_heldout_eval.jsonl` (rows `wmdp/*`) | cais/wmdp wmdp-bio (1273 q) = data/wmdp_train.jsonl (1073) + the 200 'wmdp/bio' rows of data/questions_heldout_eval.jsonl | MIT (cais/wmdp) |
| `mmlu_bio` | `data/questions_heldout_eval.jsonl` (rows `mmlu/*`) | the 200 'mmlu/*' rows of data/questions_heldout_eval.jsonl (MMLU test; college_biology, high_school_biology, virology, medical_genetics, anatomy, clinical_knowledge, college_medicine, nutrition) | MIT (cais/mmlu) |
| `mmlupro_bio` | `data/hard/mmlupro_biology.jsonl` | data/hard/mmlupro_biology.jsonl: 200 of the 717 TIGER-Lab/MMLU-Pro 'biology' questions | MIT (TIGER-Lab/MMLU-Pro) |
| `gpqa_bio` | `data/hard/gpqa_biology.jsonl` | data/hard/gpqa_biology.jsonl: the 78 biology questions of Idavidrein/gpqa (gpqa_main) | CC BY 4.0 (Idavidrein/gpqa; gated; asks that examples are not posted in plain text online) |
| `labbench_seqqa` | `data/hard/labbench_seqqa.jsonl` | data/hard/labbench_seqqa.jsonl: 200 questions of futurehouse/lab-bench SeqQA | CC BY-SA 4.0 (futurehouse/lab-bench) |
| `labbench_litqa2` | `data/hard/labbench_litqa2.jsonl` | data/hard/labbench_litqa2.jsonl: futurehouse/lab-bench LitQA2 | CC BY-SA 4.0 (futurehouse/lab-bench) |
| `labbench_protocolqa` | `data/hard/labbench_protocolqa.jsonl` | data/hard/labbench_protocolqa.jsonl: futurehouse/lab-bench ProtocolQA | CC BY-SA 4.0 (futurehouse/lab-bench) |
| `labbench_cloning` | `data/hard/labbench_cloningscenarios.jsonl` | data/hard/labbench_cloningscenarios.jsonl: futurehouse/lab-bench CloningScenarios | CC BY-SA 4.0 (futurehouse/lab-bench) |
| `gpqa_control` | `data/hard/gpqa_control.jsonl` | data/hard/gpqa_control.jsonl: GPQA chemistry (76) and physics (74) questions (gpqa_main) | CC BY 4.0 (Idavidrein/gpqa; gated; asks that examples are not posted in plain text online) |
| `mathphys` | `data/questions_control_mathphys.jsonl` | data/questions_control_mathphys.jsonl: 400 MMLU test questions (conceptual/high_school/college physics, high_school/elementary/college mathematics, astronomy, machine_learning) | MIT (cais/mmlu) |
| `mmlu_college_bio` | `data/evals/mmlu_college_bio.jsonl` | cais/mmlu subject college_biology, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_hs_bio` | `data/evals/mmlu_hs_bio.jsonl` | cais/mmlu subject high_school_biology, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_virology` | `data/evals/mmlu_virology.jsonl` | cais/mmlu subject virology, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_med_genetics` | `data/evals/mmlu_med_genetics.jsonl` | cais/mmlu subject medical_genetics, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_anatomy` | `data/evals/mmlu_anatomy.jsonl` | cais/mmlu subject anatomy, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_clinical` | `data/evals/mmlu_clinical.jsonl` | cais/mmlu subject clinical_knowledge, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_college_med` | `data/evals/mmlu_college_med.jsonl` | cais/mmlu subject college_medicine, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_prof_med` | `data/evals/mmlu_prof_med.jsonl` | cais/mmlu subject professional_medicine, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_nutrition` | `data/evals/mmlu_nutrition.jsonl` | cais/mmlu subject nutrition, test+validation rows, minus items already in earlier sets or in the pilot file | MIT (cais/mmlu) |
| `mmlu_bio_all` | union of `mmlu_bio`, `mmlu_college_bio`, `mmlu_hs_bio`, `mmlu_virology`, `mmlu_med_genetics`, `mmlu_anatomy`, `mmlu_clinical`, `mmlu_college_med`, `mmlu_prof_med`, `mmlu_nutrition` | UNION VIEW of mmlu_bio and the nine mmlu_* subject sets (the whole MMLU bio/medical universe used here) | MIT (cais/mmlu) |
| `medqa_usmle` | `data/evals/medqa_usmle.jsonl` | GBaker/MedQA-USMLE-4-options test split (1273 q), 400 sampled | CC BY 4.0 (HF card of GBaker/MedQA-USMLE-4-options; derived from MedQA, Jin et al. 2021) |
| `medmcqa_bio` | `data/evals/medmcqa_bio.jsonl` | openlifescienceai/medmcqa validation split, single-answer questions of Anatomy/Physiology/Biochemistry/Microbiology/Pathology/Pharmacology, 400 sampled (equal per subject) | Apache-2.0 (HF card of openlifescienceai/medmcqa) |
| `pubmedqa` | `data/evals/pubmedqa.jsonl` | qiaojin/PubMedQA pqa_labeled (1000 expert-labelled), 400 sampled; abstract (without conclusion) + question; options yes/no/maybe | MIT (qiaojin/PubMedQA) |
| `supergpqa_bio` | `data/evals/supergpqa_bio.jsonl` | m-a-p/SuperGPQA, discipline Science / field Biology, difficulty middle or hard, non-calculation, 400 sampled | ODC-BY (m-a-p/SuperGPQA) |
| `medxpertqa_text` | `data/evals/medxpertqa_text.jsonl` | TsinghuaC3I/MedXpertQA Text/test.jsonl (2450 q, 10 options), 400 sampled | MIT (TsinghuaC3I/MedXpertQA) |
| `wmdp_cyber` | `data/evals/wmdp_cyber.jsonl` | cais/wmdp wmdp-cyber (1987 q), 400 sampled | MIT (cais/wmdp) |
| `wmdp_chem` | `data/evals/wmdp_chem.jsonl` | cais/wmdp wmdp-chem (408 q), all kept | MIT (cais/wmdp) |
| `mmlu_nonbio` | `data/evals/mmlu_nonbio.jsonl` | cais/mmlu test+validation of high_school_us_history, high_school_geography, high_school_government_and_politics, high_school_macroeconomics, electrical_engineering, philosophy, sociology, college_computer_science; 50 per subject | MIT (cais/mmlu) |
| `mmlupro_nonbio` | `data/evals/mmlupro_nonbio.jsonl` | TIGER-Lab/MMLU-Pro categories business, computer science, economics, history, law, philosophy minus items already in earlier sets, 400 sampled (equal per category) | MIT (TIGER-Lab/MMLU-Pro) |

* `wmdp_bio`: Hazard-proxy benchmark (Li et al. 2024, arXiv:2403.03218), public since March 2024 (contamination possible); some keys are debatable. Special split: dir = the first 680 rows of data/wmdp_train.jsonl (already used for the gpt-oss-120b directions); the other 393 train rows and the 200 heldout rows are split disc/rep at crc bucket 50 (291 / 302). 124 of the 593 disc/rep questions (80 train, 44 heldout) and 126 dir questions also occur in the exploratory pilot file data/questions_pilot.jsonl, so results on them are partly exploratory. Questions come in topic clusters generated from the same source papers: only 8 of the 593 disc/rep stems are textual near-duplicates (word 3-gram Jaccard >= 0.5) of a dir stem, but topical overlap with the 680 dir questions is certain.
* `mmlu_bio`: 200 MMLU questions from 8 subjects (nutrition 45, high_school_biology 37, clinical_knowledge 30, college_medicine 29, anatomy 18, virology 17, college_biology 14, medical_genetics 10). Contamination and label errors as for all MMLU sets. 3 questions dropped as duplicates (h0363 repeats h0295; h0230 and h0331 repeat dir questions of mmlupro_bio); 31 of the 197 kept questions (21 of the served ones) also occur in the pilot file. For more power use the union view mmlu_bio_all.
* `mmlupro_bio`: 200 of the 717 MMLU-Pro biology questions; 176 have 10 options, the rest 3-9. About 30% of MMLU-Pro biology comes from MMLU (src ori_mmlu: 219 of 717), hence the overlap with the MMLU sets, and MMLU-Pro repeats some stems with regenerated options. 10 questions dropped as duplicates: 5 copies of mmlu_bio questions (7 cross-set groups in total) and 5 repeated stems inside the file.
* `gpqa_bio`: The 78 biology questions of GPQA main (Rein et al. 2023, arXiv:2311.12022). Small: about 29 disc / 30 rep questions, so confidence intervals are wide; use only pooled with other hard sets. GPQA asks that its examples are not posted in plain text online (canary string): keep this repository private. Expert-written; a minority of keys is contested.
* `labbench_seqqa`: 200 of the 600 LAB-Bench SeqQA questions (Laurent et al. 2024): sequence manipulation (restriction sites, ORFs, primers, translation) that needs symbolic computation; most models are near chance without tools. Highly templated: all 142 served stems are near-duplicates (Jaccard >= 0.5) of pool stems because they come from a few templates, but they are different problems (other genes, sequences and options): task-type similarity, not shared context, so the split does not group them. Only 4 pairs share a long template DNA sequence; they are split together (clustered eval: 196 clusters for 200 questions). 10 questions share one boilerplate stem ('Which of the following RNA sequences contains an ORF ...') that differs only in the option sequences, so 7 of them were dropped from disc/rep by the stem de-duplication because other members of that group are dir questions.
* `labbench_litqa2`: All 199 LitQA2 questions (literature questions about recent papers, designed to need retrieval, so closed-book accuracy is near chance for most models). 2 to 8 options per question, so chance varies per question (mean about 25%). 2 rows dropped: duplicated options.
* `labbench_protocolqa`: All 108 ProtocolQA questions: a full lab protocol (median about 4.6k characters) plus a troubleshooting question. The questions are NOT independent: they are only 25 distinct protocols (up to 16 questions about one protocol, usually with small edits). With the literal crc split 40 of the 49 rep questions shared their protocol with a disc question and 66 of the 85 served questions with a dir question, so a protocol-specific fluke would 'replicate'. The eval is therefore clustered: all questions of one protocol are in the same split. 23 questions longer than 7500 characters are removed (prompt-length cap), leaving 85: 12 / 24 / 49 questions in dir / disc / rep, which are only 5 / 7 / 12 protocols. The effective sample size is the number of protocols, not the number of questions: treat as exploratory, or analyse at the cluster level (cluster_of). Results cover the shorter protocols only.
* `labbench_cloning`: All 33 CloningScenarios questions are 5 scenario families that reuse the same plasmid sequences (largest family 19 questions): clustered eval. Very long prompts (median about 9.3k characters of plasmid sequences and goals): 27 are removed by the prompt-length cap. The 6 that remain (5 of one family plus the short mScarlet question) are all rep questions, so the eval has no dir and no disc question and CANNOT be used for discovery; it stays registered (SPEC name) and comes back, lumpy (even uncapped the families fall into rep and disc only), if MAX_PROMPT_CHARS is raised.
* `gpqa_control`: GPQA chemistry (76) and physics (74) questions: the matched-difficulty non-bio control for gpqa_bio (same canary note). 2 rows dropped: duplicated options.
* `mathphys`: 400 MMLU test questions from conceptual_physics 60, high_school_mathematics 66, astronomy 52, elementary_mathematics 84, college_mathematics 30, college_physics 29, high_school_physics 47, machine_learning 32. Easy to moderate; MMLU contamination. 1 row dropped (duplicated options). One question appears twice with different option lists (meteorites / asteroid belt); both copies are dir questions, so both are kept.
* `mmlu_college_bio`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 160 candidates, 110 kept (14 in mmlu_bio, 20 in the pilot, 19 in mmlupro_bio, with overlaps).
* `mmlu_hs_bio`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 342 candidates, 238 kept.
* `mmlu_virology`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 184 candidates, 147 kept. MMLU-Redux (arXiv:2406.04127) reports that 57% of the analysed questions in the Virology subset contain errors: expect a low accuracy ceiling and ambiguous keys. Pathogen-adjacent topics (HIV, coronaviruses, vaccination, surveillance) but general-knowledge level: hazard flag is False.
* `mmlu_med_genetics`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 111 candidates, 88 kept: small (about 29 disc / 40 rep).
* `mmlu_anatomy`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 149 candidates, 104 kept: small.
* `mmlu_clinical`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 294 candidates, 208 kept.
* `mmlu_college_med`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 195 candidates but only 70 kept (93 of the candidates repeat clinical_knowledge questions; others are in mmlu_bio or the pilot): the smallest set, about 26 disc / 24 rep; one question has 4.8k characters.
* `mmlu_prof_med`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 303 candidates, all kept (the pilot never sampled this subject). Clinical vignettes (median about 700 characters). Answer D is correct for 43% of the questions: a model that collapses onto one letter after steering scores about 43%.
* `mmlu_nutrition`: MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were removed (MMLU repeats some questions across subjects). 339 candidates, 248 kept.
* `mmlu_bio_all`: UNION VIEW: mmlu_bio plus the nine mmlu_* subject sets (1713 questions: 428 dir / 703 disc / 582 rep). The same questions as its members, so never analyse the view and its members together; use it when per-subject sets are too small (each member has only 24-118 questions per split).
* `medqa_usmle`: MedQA-USMLE 4-option test split (Jin et al. 2021; 1273 questions), 400 sampled. USMLE-style clinical vignettes (median about 800 characters). Widely used, so contamination is likely; some keys are disputable. Answer A is the key for 27.5% of the sample.
* `medmcqa_bio`: MedMCQA validation split (Pal et al. 2022; Indian medical entrance exams), single-answer questions of six basic-science subjects (914 candidates: Anatomy, Physiology, Biochemistry, Microbiology, Pathology, Pharmacology), 400 sampled, about 67 per subject. Mostly short recall questions; label noise is higher than in MMLU. Answer A is the key for 33% of the sample (position skew in the source).
* `pubmedqa`: PubMedQA pqa_labeled (Jin et al. 2019), reasoning-required setting: abstract without its conclusion plus the title question; options yes / no / maybe (chance 33%, always answering 'yes' scores 51.5%; 'maybe' is only 9.5% of the sample and notoriously noisy). Long prompts (median about 1.5k characters). Public since 2019: contamination.
* `supergpqa_bio`: SuperGPQA (M-A-P 2025) discipline Science / field Biology, difficulty middle or hard, calculation questions removed (599 candidates, 11 already in other sets), 400 sampled, 10 options for 372 of them. Questions were generated with LLM assistance and expert screening: some are ambiguous. Feeds the hard pool.
* `medxpertqa_text`: MedXpertQA Text test split (Zuo et al. 2025; 2450 expert-level clinical questions, 10 options), 400 sampled; chance is 10% and open models are expected to score low (floor effects). The options that the source repeats inline in the stem ('Answer Choices: (A) ...') were stripped. Contains reasoning-heavy vignettes (median about 1k characters).
* `wmdp_cyber`: WMDP cyber (Li et al. 2024; 1987 questions), 400 sampled. Hazard-adjacent NON-bio control. Many questions are assembly / C code to be traced with long hexadecimal literals: models are often near chance (25%) without execution. Prompts up to about 3k characters.
* `wmdp_chem`: WMDP chem (Li et al. 2024; 408 questions, 407 after removing one repeated stem), all kept. Hazard-adjacent NON-bio control (toxic chemicals, precursors, safety); short questions.
* `mmlu_nonbio`: Eight non-bio MMLU subjects, 50 questions each (test + validation): high_school_us_history, high_school_geography, high_school_government_and_politics, high_school_macroeconomics, electrical_engineering, philosophy, sociology, college_computer_science. Math / physics are covered by mathphys. Easy to moderate; MMLU contamination; history questions have long passages. Feeds the nonbio pool. Answer D is the key for 30% of the sample.
* `mmlupro_nonbio`: MMLU-Pro categories business, computer science, economics, history, law, philosophy (4024 candidates, 131 already in other sets, 155 repeated stems), 400 sampled, 66-67 per category. Same 10-option format as mmlupro_bio: a format-matched control for the hard bio sets. 71% of the candidates (2854 of 4024) originate from MMLU (src ori_mmlu), so MMLU contamination applies.
<!-- END GENERATED -->

## 5. Using the sets

* The default run (`--evals wmdp_bio,gpqa_bio,mmlupro_bio,labbench_seqqa,gpqa_control`) works unchanged. Bigger searches: add
  `mmlu_bio_all` (1285 served questions), `medqa_usmle`, `medmcqa_bio`, `pubmedqa`, `supergpqa_bio`, `medxpertqa_text` (about 290-300 each) and the
  controls `mmlu_nonbio` / `mmlupro_nonbio` / `wmdp_cyber` / `wmdp_chem` / `gpqa_control` / `mathphys`.
* Specificity controls: `mmlupro_nonbio` matches the 10-option format of `mmlupro_bio` / `supergpqa_bio` / `medxpertqa_text`;
  `gpqa_control` matches `gpqa_bio`; `wmdp_cyber` / `wmdp_chem` are hazard-adjacent non-bio controls for `wmdp_bio`; `mmlu_nonbio` /
  `mathphys` for the MMLU-style sets.
* Power: a McNemar test on about 140 questions per split detects only large paired changes. Treat per-subject MMLU sets, `gpqa_bio`
  (29 / 30) and `labbench_protocolqa` (24 / 49 questions but only 7 / 12 protocols: the effective sample is the protocol) as exploratory,
  or pool them (`mmlu_bio_all`: 703 disc / 582 rep). `labbench_cloning` (0 disc / 6 rep) cannot be run for discovery at all.
* Clustered evals (`labbench_seqqa`, `labbench_protocolqa`, `labbench_cloning`): disc, rep and dir hold different clusters, so a
  replication on `rep` is independent evidence, but the questions inside a split are not independent of each other. A question-level
  McNemar on them is anticonservative; group by `cluster_of(q)` (cluster bootstrap / permutation) or keep them out of the confirmatory family.
* Do not read a dir question from anywhere but `get_pool()`; if you must load other files, filter with `split_of`.

## 6. Rebuilding the data (`data/evals/*.jsonl`)

`python3 -m sbx.evals build` regenerates all 18 files (needs network and the `datasets` package; the token in `HF_TOKEN` is used if
set). Two consecutive builds were verified to be **byte-identical**. Determinism: fixed seed `20241121`; items are ranked by
`sha256(seed|set|qid)` (independent of Python's RNG, of the input order and of the Python version), stratified sets use equal
allocation with redistribution of unused quota; the dataset revisions are pinned in `evals.HF_REVISIONS`:

| dataset | revision |
|---|---|
| `cais/mmlu` | `c30699e8356d` |
| `cais/wmdp` | `7125571f22f0` |
| `GBaker/MedQA-USMLE-4-options` | `0fb93dd23a7339b6` |
| `openlifescienceai/medmcqa` | `91c6572c454088bf` |
| `qiaojin/PubMedQA` | `9001f2853fb8` |
| `TIGER-Lab/MMLU-Pro` | `b189ec765aa7` |
| `TsinghuaC3I/MedXpertQA` | `7e7c465a68eb` |
| `m-a-p/SuperGPQA` | `4430d4458112` |

Per set the builder: converts rows to the schema (strips whitespace; PubMedQA = `Abstract:` block + `Question:` line with options yes / no /
maybe; MedXpertQA's inline "Answer Choices" list is cut from the stem), drops invalid rows (duplicated / empty options) and
candidates over 6000 characters (`MAX_CHARS`, below the registry-wide `MAX_PROMPT_CHARS = 7500` of section 3, so a rebuilt file never
holds a question the registry removes), drops candidates whose key is in an earlier set / any pre-existing registered file / the pilot file, drops
within-set repeated stems, samples (<= 400, `wmdp_chem` <= 408), and writes ASCII-only JSON lines in the sampled order (like the
existing files, so `core.load_jsonl` is locale independent). Sets are built in registry order, so an earlier set wins a duplicate.
Option order is **not** shuffled (many MMLU options refer to each other: "all of the above", "both a and c"), which is why some
sets are position-skewed (see `top letter`). Licences are recorded per set (table above); GPQA is gated and asks that its examples
are not posted in plain text online, so keep this repository private.

## 7. Known limitations

* The 5 invalid rows, the 20 duplicates and the 50 too-long questions are removed at load time, not from the pre-existing files (which this
  module does not edit), so `n` of those sets is smaller than the file (`mmlupro_bio` 190, `labbench_seqqa` 193, `labbench_litqa2` 197,
  `gpqa_control` 148, `mathphys` 399, `mmlu_bio` 197, `labbench_protocolqa` 85, `labbench_cloning` 6).
* **Dependence inside a split.** The cluster rule (section 1) separates the splits, not the questions of one split: ProtocolQA's disc /
  rep splits hold 7 / 12 protocols, CloningScenarios' 6 questions 2 scenarios. The unclustered evals contain a few passages shared by 2
  to 4 questions that fall into different splits (24 questions in 10 clusters of at least 80% shared context in 6 evals, mostly MMLU
  history / science passages; `labbench_seqqa` is clustered for its 4 shared-sequence pairs); a test bounds this, so a new templated
  set added without `clustered=True` fails it. WMDP-bio has no shared-context pairs, but it is exempt from the cluster rule anyway
  (its dir pool is the first 680 rows of a file and is already used). Template boilerplate is not shared context: WMDP-cyber's generated
  C programs and packet dumps share a header and a footer and are not grouped.
* **Pins elsewhere.** The cap changed `get_pool()["hard"]` from 296 to 281 questions. Other modules read `MAX_PROMPT_CHARS` by name and
  cite 7,500 characters and a 281-question hard pool (`sbx/tests/test_adapters_qwen.py`, the `hidden_last_80gb` /
  `direction_building_memory` blocks of the model cards, `sbx/adapters/glm.py`): keep the name and the value in sync with them. The five
  gemma / llama cards' `hidden_last_long_context` block and its test in `sbx/tests/test_adapters_gemma_llama.py` were written for the
  uncapped pool (22-26k-token prompt, "does not fit 80 GB") and must be recomputed from the capped pool (its longest prompt is a SeqQA
  question of about 2.1-2.3k tokens).
* Contamination: MMLU, MedQA, MedMCQA, PubMedQA and (to a lesser extent) WMDP, GPQA, LAB-Bench, MMLU-Pro, SuperGPQA and MedXpertQA may
  be in pretraining or post-training data of the models under test. This does not bias paired steered-vs-baseline comparisons but it
  inflates absolute accuracy and shrinks the headroom in which suppressed capability could show.
* Label noise / ambiguity: worst for `mmlu_virology` (MMLU-Redux), then MedMCQA and PubMedQA's "maybe"; see the per-set notes.
* `labbench_seqqa`, `labbench_litqa2`, `labbench_cloning` are near chance for models without tools (chance in the table), so a null
  there says little about suppression.
* Small sets: see section 5. The crc split does not stratify, so single sets can be unbalanced (e.g. `mmlu_virology` 72 disc / 43 rep).

## 8. Considered and not added

* **MMLU-Pro health** (818 questions): 731 of them are MMLU questions that are already in the MMLU sets (MMLU-Pro re-publishes
  MMLU items with extra options), leaving 87 unique questions: too few to be a set. The same overlap is why `mmlupro_nonbio` is
  71% MMLU-derived.
* **`mmlu_nonbio` without mathematics / physics**: `high_school_mathematics` and `high_school_physics` are already covered by
  `mathphys` (which feeds the same `nonbio` pool), so the new set uses humanities / social science / engineering subjects instead.
* **Multimodal subsets** (LAB-Bench FigQA / TableQA, MedXpertQA MM) need images and were skipped.
* **Position-balancing by shuffling options**: not done (see section 6).
* **A token-based length cap**: the legacy `MAX_TOK`. It would give every model its own question set (its own tokenizer), which breaks
  the cross-model comparison; the character cap reproduces it on the registered data (section 3).
* **Per-question length filtering in `run.py`**: would have to be repeated in every caller (run, recipes, tests) and skipped questions
  would differ per model; the registry is the one place every caller shares.
* **Clustering every eval**: the cluster rule costs about 1 s for the three LAB-Bench sets and several seconds for all evals (pairwise
  12-gram overlap), and it would move questions in evals nobody found a problem with; only the three LAB-Bench sets (and WMDP-bio, which cannot move) are special. The residual is bounded by a test.
* **Keeping one question per cluster**: would leave ProtocolQA with 25 questions and CloningScenarios with 5; cluster-level analysis
  (`cluster_of`) keeps the data.
* **An "exploratory" tag for the clustered evals only**: `analyze.py` reads a `family` tag from `cells.jsonl`, which `run.py` does not write, so
  a registry flag could not be enforced from here; the cluster-disjoint split needs no change elsewhere.
