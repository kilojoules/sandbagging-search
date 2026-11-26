"""sbx evals: eval registry, deterministic dir/disc/rep splits and direction-building pools (SPEC section 4).

Public API (everything else is internal)
  EVALS                    {name: EvalSpec}: category "bio"|"control", hazard flag, source files, n, pool, licence, notes
  get_eval(name, split)    questions for EVALUATION. split in {"all", "disc", "rep"}; "all" = disc + rep and NEVER
                           contains a dir question (dir questions are only reachable through get_pool()).
  get_pool()               {"wmdp": [...], "hard": [...], "nonbio": [...]}: questions for BUILDING directions (dir only)
  split_of(q)              "dir" | "disc" | "rep"
  cluster_of(q)            id of the group of questions that share context (see "Clusters"); the question's own id for most
  prompt_chars(q)          len(question) + sum(len(choices)): the quantity MAX_PROMPT_CHARS bounds
  split_sizes(name)        {"dir": n, "disc": n, "rep": n} for one eval
  list_evals(...)          eval names filtered by category / hazard / pool (union views only with include_unions=True)
  registry_report()        what was dropped (invalid rows, duplicates, too-long prompts) when the registry was built
  near_duplicate_report()  how many served questions have a near-duplicate stem in the direction pools
  eval_profile(name)       options / chance level / answer-position skew of the questions served for evaluation
The registry is global: the first call reads EVERY registered file (de-duplication and clustering need all of them; about
1-2 s, once per process) and raises if one is missing or malformed. A union view (EvalSpec.union, e.g. mmlu_bio_all) has the
same questions as its members.

Split rule (SPEC section 4; deterministic, tokenizer- and process-independent)
  b = zlib.crc32(f"{source}|{qid}".encode()) % 100;  b < 25 -> "dir";  b < 63 -> "disc";  else "rep".
  WMDP-bio special case (the dir pool already used for the gpt-oss-120b directions): dir = the FIRST 680 rows of
  data/wmdp_train.jsonl; every other WMDP-bio question (the remaining 393 train rows and the 200 'wmdp/bio' rows of
  data/questions_heldout_eval.jsonl) is split disc/rep by the same crc bucket (disc if b < 50 else rep), so WMDP-bio has no
  other dir questions and its disc/rep halves are balanced.

Clusters (EvalSpec.clustered; LAB-Bench SeqQA / ProtocolQA / CloningScenarios)
  Some questions are not independent draws: ProtocolQA asks up to 16 questions about ONE protocol text, CloningScenarios asks
  about the same plasmid sequences, so with the literal rule a protocol-specific fluke found on disc "replicates" on rep
  and a direction built from dir questions is evaluated on questions that quote the same protocol. For a clustered eval two
  stems are LINKED when they share at least CLUSTER_MIN_SHARED (300) distinct character CLUSTER_K-grams (whitespace collapsed)
  and at least CLUSTER_MIN_OVERLAP (half) of the shorter stem's CLUSTER_K-grams (a short and a long version of one protocol
  link; two questions built from one template but about different genes / sequences do not). The connected components are the clusters;
  every member takes the bucket of the cluster's REPRESENTATIVE, the member with the lowest qid, i.e. the rule above applied
  to the representative's (source, qid). Singleton clusters, and every eval that is not clustered, follow the rule unchanged.
  Clusters are computed over all valid rows of the eval, before de-duplication and before the length cap, so neither changes
  a split. This guarantees that disc, rep and dir contain different clusters; it does NOT remove the dependence between the
  questions inside one split (use cluster_of for cluster-level inference).

Prompt length (MAX_PROMPT_CHARS)
  A prompt of a few thousand tokens already needs gigabytes of attention memory per layer (eager attention, 64 heads), and
  Ctx.hidden_last keeps every layer's hidden states and all-position logits; the legacy pipeline therefore skipped prompts of
  more than 3000 tokens (scripts/gptoss_dir.py: MAX_TOK). The registry applies the tokenizer-independent equivalent: every
  question with prompt_chars(q) > MAX_PROMPT_CHARS is removed from every split and pool and listed in
  registry_report()["too_long"]. It happens AFTER de-duplication, clustering and the split assignment, so it can neither move
  another question to a different split nor let a copy of a removed dir question into disc / rep (split_of still answers "dir").

De-duplication (no question may be in two splits, and no dir question may leak into any disc/rep eval)
  Questions are identified by their normalised question text (`question_key`: NFKC, casefold, punctuation/whitespace
  collapsed). Within a group of equal keys (across ALL registered evals, in EVALS order, then file order):
    * if some member is a dir question by the rule above: every dir member is kept (as dir) and all other members are
      dropped from their evals ("duplicate_of_dir");
    * otherwise only the first member is kept (in the split its own hash gives) and the rest are dropped ("duplicate").
  Rows that violate the question schema (e.g. duplicated or empty options) are dropped and listed in registry_report().
  `split_of(q)` is registry-aware: a question registered under its (source, qid), or whose key is registered, gets the
  bucket of its duplicate group (so a copy of a dir question that lives in some other file, e.g. questions_pilot.jsonl, is
  also "dir"); any other question gets the pure rule. It needs only "source" and "qid" ("question" is used if present).

Ordering: get_eval returns each split in a fixed pseudo-random order (crc32 of "order|source|qid"), so a prefix
(`--max-per-split`) is an unbiased subsample. get_pool()["wmdp"] is the first 680 rows of wmdp_train.jsonl in FILE order;
"hard" and "nonbio" are de-duplicated and in pseudo-random order. All returned question dicts are fresh copies.

Question schema: {"source": str, "qid": str, "question": str, "choices": [str, ...2..10 unique non-empty], "answer": int}.

Rebuilding the data/evals/*.jsonl files (needs network + `datasets`; HF_TOKEN from the environment is used if present):
  python3 -m sbx.evals build [--only NAME ...]      deterministic (fixed seed, pinned dataset revisions)
  python3 -m sbx.evals docs                         print the generated block of docs/evals.md
  python3 -m sbx.evals report | check | counts      registry diagnostics
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import chain
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (the parent of sbx/)

SPLITS = ("dir", "disc", "rep")
EVAL_SPLITS = ("all", "disc", "rep")
CATEGORIES = ("bio", "control")
POOLS = ("wmdp", "hard", "nonbio")
KEYS = ("source", "qid", "question", "choices", "answer")

DIR_BELOW, DISC_BELOW = 25, 63          # crc bucket b: b < 25 -> dir, b < 63 -> disc, else rep
WMDP_SOURCE = "wmdp/bio"
WMDP_TRAIN = "data/wmdp_train.jsonl"
WMDP_DIR_N = 680                        # dir = first 680 rows of WMDP_TRAIN
WMDP_REST_DISC_BELOW = 50               # non-dir WMDP-bio: disc if b < 50 else rep
PILOT_FILE = "data/questions_pilot.jsonl"  # exploratory pilot (not registered); the builder keeps new sets disjoint from it

# Prompt-length guard (see the module docstring). 7500 characters of stem + options is the tokenizer-independent equivalent of
# the legacy 3000-token filter (scripts/gptoss_dir.py MAX_TOK, gpt-oss tokenizer, framing E): on the registered data the two
# keep exactly the same questions, and every kept question builds a framing-E prompt of at most 3000 tokens in the gpt-oss
# (longest 2939), Llama-3 and Gemma-3 tokenizers (a test measures this when the tokenizers are cached). Raising it (bigger
# GPUs) means re-pinning _COUNTS.
MAX_PROMPT_CHARS = 7500
# Cluster rule for EvalSpec.clustered evals (see the module docstring). The three parameters sit on a plateau of the
# registered data: ProtocolQA / CloningScenarios / SeqQA give the SAME 25 / 5 / 196 clusters for every K in 8..24, every
# MIN_SHARED in 200..500 and every overlap in 0.4..0.65. The overlap of the pairs that share >= MIN_SHARED k-grams has an empty
# gap there (ProtocolQA: nothing between 0.12 and 0.75, CloningScenarios: nothing between 0.40 and 0.60; SeqQA's 4 linked
# pairs are all >= 0.9), and 0.5 is in the middle of it. A test re-checks the stability. (A threshold of 0.8 would miss
# three questions about one short protocol whose overlap is 0.78-0.81 because the question sentence is a large part of it.)
CLUSTER_K = 12
CLUSTER_MIN_SHARED = 300
CLUSTER_MIN_OVERLAP = 0.5


# ================================================================================================ registry definition
@dataclass(frozen=True)
class Src:
    """One source file of an eval: a repo-relative .jsonl path, optionally restricted to rows whose `source` starts with
    `source_prefix` (used to carve the WMDP and MMLU rows out of data/questions_heldout_eval.jsonl)."""
    path: str
    source_prefix: str = ""


@dataclass(frozen=True)
class EvalSpec:
    """name, category ("bio"|"control"), hazard (True iff built as a proxy for hazardous knowledge: the WMDP sets), files.

    n      total number of questions of the eval (dir + disc + rep) after validation, de-duplication and the prompt-length
           cap, pinned by a test; the evaluable questions are disc + rep (see split_sizes)
    pool   which direction pool the dir bucket feeds: "wmdp" | "hard" | "nonbio" | None (dir bucket then unused)
    origin / licence / notes   provenance, licence and known caveats (rendered into docs/evals.md)
    union  a UNION VIEW: files == () and union == names of earlier (non-union) evals with the same category and hazard.
           Its splits are the union of the members' splits (no new questions, no new dedupe, pool must be None): a bigger
           eval for statistical power. A view and its members contain the same questions; never run both in one analysis.
    clustered  True: questions that share a long stretch of context (the same protocol / plasmid / template sequence) are
           split together, so disc, rep and dir hold different clusters (see "Clusters" in the module docstring). Not
           allowed for union views or for sets containing WMDP-bio rows (their dir pool is fixed by file order).
    """
    name: str
    category: str
    hazard: bool
    files: Tuple[Src, ...]
    n: int = 0
    pool: Optional[str] = None
    origin: str = field(default="", repr=False)
    licence: str = field(default="", repr=False)
    notes: str = field(default="", repr=False)
    union: Tuple[str, ...] = ()
    clustered: bool = False

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.name}: category must be one of {CATEGORIES}, got {self.category!r}")
        if not isinstance(self.hazard, bool):
            raise ValueError(f"{self.name}: hazard must be a bool")
        if not isinstance(self.clustered, bool):
            raise ValueError(f"{self.name}: clustered must be a bool")
        if self.pool is not None and self.pool not in POOLS:
            raise ValueError(f"{self.name}: pool must be None or one of {POOLS}, got {self.pool!r}")
        files = tuple(f if isinstance(f, Src) else Src(f) for f in self.files)
        union = tuple(self.union)
        if bool(files) == bool(union):
            raise ValueError(f"{self.name}: give either source files or a union of evals, not both/neither")
        if union and self.pool is not None:
            raise ValueError(f"{self.name}: a union view cannot feed a direction pool (its members do)")
        if union and self.clustered:
            raise ValueError(f"{self.name}: a union view cannot be clustered (its members are)")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "union", union)

    @property
    def generated(self) -> bool:
        """True for sets built by `python -m sbx.evals build` (their files live in data/evals/)."""
        return bool(self.files) and all(f.path.startswith("data/evals/") for f in self.files)

    @property
    def is_union(self) -> bool:
        return bool(self.union)

    def __getitem__(self, key):  # allow spec["category"] as well as spec.category
        return getattr(self, key)


# per-set total counts (dir + disc + rep, after validation, de-duplication and the MAX_PROMPT_CHARS cap) pinned by tests;
# regenerate with `python3 -m sbx.evals counts` after changing data or the cap
_COUNTS: Dict[str, int] = {
    'wmdp_bio': 1273,
    'mmlu_bio': 197,
    'mmlupro_bio': 190,
    'gpqa_bio': 78,
    'labbench_seqqa': 193,
    'labbench_litqa2': 197,
    'labbench_protocolqa': 85,
    'labbench_cloning': 6,
    'gpqa_control': 148,
    'mathphys': 399,
    'mmlu_college_bio': 110,
    'mmlu_hs_bio': 238,
    'mmlu_virology': 147,
    'mmlu_med_genetics': 88,
    'mmlu_anatomy': 104,
    'mmlu_clinical': 208,
    'mmlu_college_med': 70,
    'mmlu_prof_med': 303,
    'mmlu_nutrition': 248,
    'mmlu_bio_all': 1713,
    'medqa_usmle': 400,
    'medmcqa_bio': 400,
    'pubmedqa': 400,
    'supergpqa_bio': 400,
    'medxpertqa_text': 400,
    'wmdp_cyber': 400,
    'wmdp_chem': 407,
    'mmlu_nonbio': 400,
    'mmlupro_nonbio': 400,
}

_MIT = "MIT"


def _gen(name):
    return (Src(f"data/evals/{name}.jsonl"),)


def _mmlu_set(name, subject):
    return EvalSpec(
        name, "bio", False, _gen(name), _COUNTS.get(name, 0), None,
        origin=f"cais/mmlu subject {subject}, test+validation rows, minus items already in earlier sets or in the pilot file",
        licence=f"{_MIT} (cais/mmlu)")


_SPECS: List[EvalSpec] = [
    # ---- existing data, canonical names (SPEC section 4) -------------------------------------------------------
    EvalSpec("wmdp_bio", "bio", True, (Src(WMDP_TRAIN), Src("data/questions_heldout_eval.jsonl", "wmdp/")),
             _COUNTS.get("wmdp_bio", 0), "wmdp",
             origin="cais/wmdp wmdp-bio (1273 q) = data/wmdp_train.jsonl (1073) + the 200 'wmdp/bio' rows of data/questions_heldout_eval.jsonl",
             licence=f"{_MIT} (cais/wmdp)"),
    EvalSpec("mmlu_bio", "bio", False, (Src("data/questions_heldout_eval.jsonl", "mmlu/"),),
             _COUNTS.get("mmlu_bio", 0), None,
             origin="the 200 'mmlu/*' rows of data/questions_heldout_eval.jsonl (MMLU test; college_biology, high_school_biology, virology, medical_genetics, anatomy, clinical_knowledge, college_medicine, nutrition)",
             licence=f"{_MIT} (cais/mmlu)"),
    EvalSpec("mmlupro_bio", "bio", False, (Src("data/hard/mmlupro_biology.jsonl"),),
             _COUNTS.get("mmlupro_bio", 0), "hard",
             origin="data/hard/mmlupro_biology.jsonl: 200 of the 717 TIGER-Lab/MMLU-Pro 'biology' questions",
             licence=f"{_MIT} (TIGER-Lab/MMLU-Pro)"),
    EvalSpec("gpqa_bio", "bio", False, (Src("data/hard/gpqa_biology.jsonl"),),
             _COUNTS.get("gpqa_bio", 0), "hard",
             origin="data/hard/gpqa_biology.jsonl: the 78 biology questions of Idavidrein/gpqa (gpqa_main)",
             licence="CC BY 4.0 (Idavidrein/gpqa; gated; asks that examples are not posted in plain text online)"),
    EvalSpec("labbench_seqqa", "bio", False, (Src("data/hard/labbench_seqqa.jsonl"),),
             _COUNTS.get("labbench_seqqa", 0), "hard",
             origin="data/hard/labbench_seqqa.jsonl: 200 questions of futurehouse/lab-bench SeqQA",
             licence="CC BY-SA 4.0 (futurehouse/lab-bench)", clustered=True),
    EvalSpec("labbench_litqa2", "bio", False, (Src("data/hard/labbench_litqa2.jsonl"),),
             _COUNTS.get("labbench_litqa2", 0), "hard",
             origin="data/hard/labbench_litqa2.jsonl: futurehouse/lab-bench LitQA2",
             licence="CC BY-SA 4.0 (futurehouse/lab-bench)"),
    EvalSpec("labbench_protocolqa", "bio", False, (Src("data/hard/labbench_protocolqa.jsonl"),),
             _COUNTS.get("labbench_protocolqa", 0), "hard",
             origin="data/hard/labbench_protocolqa.jsonl: futurehouse/lab-bench ProtocolQA",
             licence="CC BY-SA 4.0 (futurehouse/lab-bench)", clustered=True),
    EvalSpec("labbench_cloning", "bio", False, (Src("data/hard/labbench_cloningscenarios.jsonl"),),
             _COUNTS.get("labbench_cloning", 0), "hard",
             origin="data/hard/labbench_cloningscenarios.jsonl: futurehouse/lab-bench CloningScenarios",
             licence="CC BY-SA 4.0 (futurehouse/lab-bench)", clustered=True),
    EvalSpec("gpqa_control", "control", False, (Src("data/hard/gpqa_control.jsonl"),),
             _COUNTS.get("gpqa_control", 0), None,
             origin="data/hard/gpqa_control.jsonl: GPQA chemistry (76) and physics (74) questions (gpqa_main)",
             licence="CC BY 4.0 (Idavidrein/gpqa; gated; asks that examples are not posted in plain text online)"),
    EvalSpec("mathphys", "control", False, (Src("data/questions_control_mathphys.jsonl"),),
             _COUNTS.get("mathphys", 0), "nonbio",
             origin="data/questions_control_mathphys.jsonl: 400 MMLU test questions (conceptual/high_school/college physics, high_school/elementary/college mathematics, astronomy, machine_learning)",
             licence=f"{_MIT} (cais/mmlu)"),
    # ---- new bio / medical sets (data/evals) ----------------------------------------------------------------------
    _mmlu_set("mmlu_college_bio", "college_biology"),
    _mmlu_set("mmlu_hs_bio", "high_school_biology"),
    _mmlu_set("mmlu_virology", "virology"),
    _mmlu_set("mmlu_med_genetics", "medical_genetics"),
    _mmlu_set("mmlu_anatomy", "anatomy"),
    _mmlu_set("mmlu_clinical", "clinical_knowledge"),
    _mmlu_set("mmlu_college_med", "college_medicine"),
    _mmlu_set("mmlu_prof_med", "professional_medicine"),
    _mmlu_set("mmlu_nutrition", "nutrition"),
    EvalSpec("mmlu_bio_all", "bio", False, (), _COUNTS.get("mmlu_bio_all", 0), None,
             origin="UNION VIEW of mmlu_bio and the nine mmlu_* subject sets (the whole MMLU bio/medical universe used here)",
             licence=f"{_MIT} (cais/mmlu)",
             union=("mmlu_bio", "mmlu_college_bio", "mmlu_hs_bio", "mmlu_virology", "mmlu_med_genetics", "mmlu_anatomy",
                    "mmlu_clinical", "mmlu_college_med", "mmlu_prof_med", "mmlu_nutrition")),
    EvalSpec("medqa_usmle", "bio", False, _gen("medqa_usmle"), _COUNTS.get("medqa_usmle", 0), None,
             origin="GBaker/MedQA-USMLE-4-options test split (1273 q), 400 sampled",
             licence="CC BY 4.0 (HF card of GBaker/MedQA-USMLE-4-options; derived from MedQA, Jin et al. 2021)"),
    EvalSpec("medmcqa_bio", "bio", False, _gen("medmcqa_bio"), _COUNTS.get("medmcqa_bio", 0), None,
             origin="openlifescienceai/medmcqa validation split, single-answer questions of Anatomy/Physiology/Biochemistry/Microbiology/Pathology/Pharmacology, 400 sampled (equal per subject)",
             licence="Apache-2.0 (HF card of openlifescienceai/medmcqa)"),
    EvalSpec("pubmedqa", "bio", False, _gen("pubmedqa"), _COUNTS.get("pubmedqa", 0), None,
             origin="qiaojin/PubMedQA pqa_labeled (1000 expert-labelled), 400 sampled; abstract (without conclusion) + question; options yes/no/maybe",
             licence=f"{_MIT} (qiaojin/PubMedQA)"),
    EvalSpec("supergpqa_bio", "bio", False, _gen("supergpqa_bio"), _COUNTS.get("supergpqa_bio", 0), "hard",
             origin="m-a-p/SuperGPQA, discipline Science / field Biology, difficulty middle or hard, non-calculation, 400 sampled",
             licence="ODC-BY (m-a-p/SuperGPQA)"),
    EvalSpec("medxpertqa_text", "bio", False, _gen("medxpertqa_text"), _COUNTS.get("medxpertqa_text", 0), None,
             origin="TsinghuaC3I/MedXpertQA Text/test.jsonl (2450 q, 10 options), 400 sampled",
             licence=f"{_MIT} (TsinghuaC3I/MedXpertQA)"),
    # ---- new controls (data/evals) --------------------------------------------------------------------------------
    EvalSpec("wmdp_cyber", "control", True, _gen("wmdp_cyber"), _COUNTS.get("wmdp_cyber", 0), None,
             origin="cais/wmdp wmdp-cyber (1987 q), 400 sampled",
             licence=f"{_MIT} (cais/wmdp)"),
    EvalSpec("wmdp_chem", "control", True, _gen("wmdp_chem"), _COUNTS.get("wmdp_chem", 0), None,
             origin="cais/wmdp wmdp-chem (408 q), all kept",
             licence=f"{_MIT} (cais/wmdp)"),
    EvalSpec("mmlu_nonbio", "control", False, _gen("mmlu_nonbio"), _COUNTS.get("mmlu_nonbio", 0), "nonbio",
             origin="cais/mmlu test+validation of high_school_us_history, high_school_geography, high_school_government_and_politics, high_school_macroeconomics, electrical_engineering, philosophy, sociology, college_computer_science; 50 per subject",
             licence=f"{_MIT} (cais/mmlu)"),
    EvalSpec("mmlupro_nonbio", "control", False, _gen("mmlupro_nonbio"), _COUNTS.get("mmlupro_nonbio", 0), None,
             origin="TIGER-Lab/MMLU-Pro categories business, computer science, economics, history, law, philosophy minus items already in earlier sets, 400 sampled (equal per category)",
             licence=f"{_MIT} (TIGER-Lab/MMLU-Pro)"),
]

# Known caveats per set (rendered into docs/evals.md). Numbers refer to the data as built (see `python3 -m sbx.evals report`).
_MMLU_COMMON = ("MMLU is in most pretraining corpora (contamination) and has label errors (MMLU-Redux, Gema et al. 2024). "
                "Test + validation rows; questions already in mmlu_bio, the pilot file, mmlupro_bio or an earlier mmlu_* set were "
                "removed (MMLU repeats some questions across subjects). ")
_NOTES: Dict[str, str] = {
    "wmdp_bio": (
        "Hazard-proxy benchmark (Li et al. 2024, arXiv:2403.03218), public since March 2024 (contamination possible); some keys "
        "are debatable. Special split: dir = the first 680 rows of data/wmdp_train.jsonl (already used for the gpt-oss-120b "
        "directions); the other 393 train rows and the 200 heldout rows are split disc/rep at crc bucket 50 (291 / 302). 124 of "
        "the 593 disc/rep questions (80 train, 44 heldout) and 126 dir questions also occur in the exploratory pilot file "
        "data/questions_pilot.jsonl, so results on them are partly exploratory. Questions come in topic clusters generated from "
        "the same source papers: only 8 of the 593 disc/rep stems are textual near-duplicates (word 3-gram Jaccard >= 0.5) of "
        "a dir stem, but topical overlap with the 680 dir questions is certain."),
    "mmlu_bio": (
        "200 MMLU questions from 8 subjects (nutrition 45, high_school_biology 37, clinical_knowledge 30, college_medicine 29, "
        "anatomy 18, virology 17, college_biology 14, medical_genetics 10). Contamination and label errors as for all MMLU "
        "sets. 3 questions dropped as duplicates (h0363 repeats h0295; h0230 and h0331 repeat dir questions of mmlupro_bio); "
        "31 of the 197 kept questions (21 of the served ones) also occur in the pilot file. For more power use the union view "
        "mmlu_bio_all."),
    "mmlupro_bio": (
        "200 of the 717 MMLU-Pro biology questions; 176 have 10 options, the rest 3-9. About 30% of MMLU-Pro biology comes from "
        "MMLU (src ori_mmlu: 219 of 717), hence the overlap with the MMLU sets, and MMLU-Pro repeats some stems with "
        "regenerated options. 10 questions dropped as duplicates: 5 copies of mmlu_bio questions (7 cross-set groups in total) "
        "and 5 repeated stems inside the file."),
    "gpqa_bio": (
        "The 78 biology questions of GPQA main (Rein et al. 2023, arXiv:2311.12022). Small: about 29 disc / 30 rep questions, "
        "so confidence intervals are wide; use only pooled with other hard sets. GPQA asks that its examples are not posted in "
        "plain text online (canary string): keep this repository private. Expert-written; a minority of keys is contested."),
    "labbench_seqqa": (
        "200 of the 600 LAB-Bench SeqQA questions (Laurent et al. 2024): sequence manipulation (restriction sites, ORFs, "
        "primers, translation) that needs symbolic computation; most models are near chance without tools. Highly templated: "
        "all 142 served stems are near-duplicates (Jaccard >= 0.5) of pool stems because they come from a few templates, but they "
        "are different problems (other genes, sequences and options): task-type similarity, not shared context, so the split "
        "does not group them. Only 4 pairs share a long template DNA sequence; they are split together (clustered eval: 196 "
        "clusters for 200 questions). 10 questions share one boilerplate stem ('Which of the following RNA sequences contains an "
        "ORF ...') that differs only in the option sequences, so 7 of them were dropped from disc/rep by the stem de-duplication "
        "because other members of that group are dir questions."),
    "labbench_litqa2": (
        "All 199 LitQA2 questions (literature questions about recent papers, designed to need retrieval, so closed-book "
        "accuracy is near chance for most models). 2 to 8 options per question, so chance varies per question (mean about "
        "25%). 2 rows dropped: duplicated options."),
    "labbench_protocolqa": (
        "All 108 ProtocolQA questions: a full lab protocol (median about 4.6k characters) plus a troubleshooting question. The "
        "questions are NOT independent: they are only 25 distinct protocols (up to 16 questions about one protocol, usually "
        "with small edits). With the literal crc split 40 of the 49 rep questions shared their protocol with a disc question and "
        "66 of the 85 served questions with a dir question, so a protocol-specific fluke would 'replicate'. The eval is therefore "
        "clustered: all questions of one protocol are in the same split. 23 questions longer than 7500 characters are removed "
        "(prompt-length cap), leaving 85: 12 / 24 / 49 questions in dir / disc / rep, which are only 5 / 7 / 12 protocols. The "
        "effective sample size is the number of protocols, not the number of questions: treat as exploratory, or analyse at the "
        "cluster level (cluster_of). Results cover the shorter protocols only."),
    "labbench_cloning": (
        "All 33 CloningScenarios questions are 5 scenario families that reuse the same plasmid sequences (largest family 19 "
        "questions): clustered eval. Very long prompts (median about 9.3k characters of plasmid sequences and goals): 27 are "
        "removed by the prompt-length cap. The 6 that remain (5 of one family plus the short mScarlet question) are all rep "
        "questions, so the eval has no dir and no disc question and CANNOT be used for discovery; it stays registered (SPEC "
        "name) and comes back, lumpy (even uncapped the families fall into rep and disc only), if MAX_PROMPT_CHARS is raised."),
    "gpqa_control": (
        "GPQA chemistry (76) and physics (74) questions: the matched-difficulty non-bio control for gpqa_bio (same canary note). "
        "2 rows dropped: duplicated options."),
    "mathphys": (
        "400 MMLU test questions from conceptual_physics 60, high_school_mathematics 66, astronomy 52, elementary_mathematics 84, "
        "college_mathematics 30, college_physics 29, high_school_physics 47, machine_learning 32. Easy to moderate; MMLU "
        "contamination. 1 row dropped (duplicated options). One question appears twice with different option lists "
        "(meteorites / asteroid belt); both copies are dir questions, so both are kept."),
    "mmlu_college_bio": (_MMLU_COMMON + "160 candidates, 110 kept (14 in mmlu_bio, 20 in the pilot, 19 in mmlupro_bio, "
                         "with overlaps)."),
    "mmlu_hs_bio": (_MMLU_COMMON + "342 candidates, 238 kept."),
    "mmlu_virology": (_MMLU_COMMON + "184 candidates, 147 kept. MMLU-Redux (arXiv:2406.04127) reports that 57% of the analysed "
                      "questions in the Virology subset contain errors: expect a low accuracy ceiling and ambiguous keys. "
                      "Pathogen-adjacent topics (HIV, coronaviruses, vaccination, surveillance) but general-knowledge level: "
                      "hazard flag is False."),
    "mmlu_med_genetics": (_MMLU_COMMON + "111 candidates, 88 kept: small (about 29 disc / 40 rep)."),
    "mmlu_anatomy": (_MMLU_COMMON + "149 candidates, 104 kept: small."),
    "mmlu_clinical": (_MMLU_COMMON + "294 candidates, 208 kept."),
    "mmlu_college_med": (_MMLU_COMMON + "195 candidates but only 70 kept (93 of the candidates repeat clinical_knowledge "
                         "questions; others are in mmlu_bio or the pilot): the smallest set, about 26 disc / 24 rep; one question has 4.8k characters."),
    "mmlu_prof_med": (_MMLU_COMMON + "303 candidates, all kept (the pilot never sampled this subject). Clinical vignettes "
                      "(median about 700 characters). Answer D is correct for 43% of the questions: a model that collapses onto "
                      "one letter after steering scores about 43%."),
    "mmlu_nutrition": (_MMLU_COMMON + "339 candidates, 248 kept."),
    "mmlu_bio_all": (
        "UNION VIEW: mmlu_bio plus the nine mmlu_* subject sets (1713 questions: 428 dir / 703 disc / 582 rep). The same "
        "questions as its members, so never analyse the view and its members together; use it when per-subject sets are too "
        "small (each member has only 24-118 questions per split)."),
    "medqa_usmle": (
        "MedQA-USMLE 4-option test split (Jin et al. 2021; 1273 questions), 400 sampled. USMLE-style clinical vignettes (median "
        "about 800 characters). Widely used, so contamination is likely; some keys are disputable. Answer A is the key for 27.5% "
        "of the sample."),
    "medmcqa_bio": (
        "MedMCQA validation split (Pal et al. 2022; Indian medical entrance exams), single-answer questions of six basic-science "
        "subjects (914 candidates: Anatomy, Physiology, Biochemistry, Microbiology, Pathology, Pharmacology), 400 sampled, about "
        "67 per subject. Mostly short recall questions; label noise is higher than in MMLU. Answer A is the key for 33% of the "
        "sample (position skew in the source)."),
    "pubmedqa": (
        "PubMedQA pqa_labeled (Jin et al. 2019), reasoning-required setting: abstract without its conclusion plus the title "
        "question; options yes / no / maybe (chance 33%, always answering 'yes' scores 51.5%; 'maybe' is only 9.5% of the sample "
        "and notoriously noisy). Long prompts (median about 1.5k characters). Public since 2019: contamination."),
    "supergpqa_bio": (
        "SuperGPQA (M-A-P 2025) discipline Science / field Biology, difficulty middle or hard, calculation questions removed "
        "(599 candidates, 11 already in other sets), 400 sampled, 10 options for 372 of them. Questions were generated with "
        "LLM assistance and expert screening: some are ambiguous. Feeds the hard pool."),
    "medxpertqa_text": (
        "MedXpertQA Text test split (Zuo et al. 2025; 2450 expert-level clinical questions, 10 options), 400 sampled; chance is "
        "10% and open models are expected to score low (floor effects). The options that the source repeats inline in the stem "
        "('Answer Choices: (A) ...') were stripped. Contains reasoning-heavy vignettes (median about 1k characters)."),
    "wmdp_cyber": (
        "WMDP cyber (Li et al. 2024; 1987 questions), 400 sampled. Hazard-adjacent NON-bio control. Many questions are "
        "assembly / C code to be traced with long hexadecimal literals: models are often near chance (25%) without execution. "
        "Prompts up to about 3k characters."),
    "wmdp_chem": (
        "WMDP chem (Li et al. 2024; 408 questions, 407 after removing one repeated stem), all kept. Hazard-adjacent NON-bio "
        "control (toxic chemicals, precursors, safety); short questions."),
    "mmlu_nonbio": (
        "Eight non-bio MMLU subjects, 50 questions each (test + validation): high_school_us_history, high_school_geography, "
        "high_school_government_and_politics, high_school_macroeconomics, electrical_engineering, philosophy, sociology, "
        "college_computer_science. Math / physics are covered by mathphys. Easy to moderate; MMLU contamination; history "
        "questions have long passages. Feeds the nonbio pool. Answer D is the key for 30% of the sample."),
    "mmlupro_nonbio": (
        "MMLU-Pro categories business, computer science, economics, history, law, philosophy (4024 candidates, 131 already in "
        "other sets, 155 repeated stems), 400 sampled, 66-67 per category. Same 10-option format as mmlupro_bio: a "
        "format-matched control for the hard bio sets. 71% of the candidates (2854 of 4024) originate from MMLU (src ori_mmlu), "
        "so MMLU contamination applies."),
}
if set(_NOTES) != {s.name for s in _SPECS}:
    raise ValueError(f"notes missing/unknown for: {sorted(set(_NOTES) ^ {s.name for s in _SPECS})}")
_SPECS = [replace(s, notes=_NOTES[s.name]) for s in _SPECS]

EVALS: Dict[str, EvalSpec] = {}
for _s in _SPECS:
    if _s.name in EVALS:
        raise ValueError(f"duplicate eval name {_s.name!r}")
    for _m in _s.union:
        if _m not in EVALS or EVALS[_m].is_union:
            raise ValueError(f"{_s.name}: union member {_m!r} must be an earlier, non-union eval")
        if (EVALS[_m].category, EVALS[_m].hazard) != (_s.category, _s.hazard):
            raise ValueError(f"{_s.name}: union member {_m!r} has a different category/hazard")
    EVALS[_s.name] = _s
del _s


def list_evals(category: Optional[str] = None, hazard: Optional[bool] = None, pool: Optional[str] = None,
               include_unions: bool = False) -> List[str]:
    """Eval names (registry order) with the given category / hazard flag / pool (None = do not filter on that field).
    Union views (e.g. mmlu_bio_all) are left out unless include_unions=True: their questions are already in their members,
    so a loop over "all bio evals" must not count them twice."""
    return [n for n, s in EVALS.items()
            if (category is None or s.category == category) and (hazard is None or s.hazard == hazard)
            and (pool is None or s.pool == pool) and (include_unions or not s.is_union)]


# ================================================================================================ text, validation, splits
def norm_text(s: str) -> str:
    """Normalised text used to identify a question: NFKC, casefold, every run of non-word characters -> one space."""
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()


def question_key(q: dict) -> str:
    """Identity of a question for de-duplication: its normalised question text (choices are NOT part of the key, so a
    question re-published with regenerated options, e.g. MMLU -> MMLU-Pro, is still recognised)."""
    return norm_text(q["question"])


def validate_question(q) -> Optional[str]:
    """None if `q` satisfies the question schema, else the first problem found. Reasons starting with "structure" mean the
    row is malformed (a data error); "choices: ..." means the options are unusable (duplicated / empty options)."""
    if not isinstance(q, dict):
        return "structure: row is not an object"
    if set(q) != set(KEYS):
        return f"structure: keys {sorted(q)} != {sorted(KEYS)}"
    for k in ("source", "qid", "question"):
        if not isinstance(q[k], str) or not q[k].strip():
            return f"structure: {k} must be a non-empty string"
    ch = q["choices"]
    if not isinstance(ch, list) or not all(isinstance(c, str) for c in ch):
        return "structure: choices must be a list of strings"
    if not 2 <= len(ch) <= 10:
        return f"structure: {len(ch)} choices (need 2..10)"
    a = q["answer"]
    if isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < len(ch):
        return "structure: answer must be a 0-based int index into choices"
    if not norm_text(q["question"]):
        return "structure: question has no alphanumeric text"
    stripped = [c.strip() for c in ch]
    if any(not c for c in stripped):
        return "choices: empty option"
    if len(set(stripped)) != len(stripped):
        return "choices: duplicated options"
    return None


def prompt_chars(q: dict) -> int:
    """Length of a question as the registry bounds it: characters of the stem plus all options (the ASK line, the option
    letters and the framing add under ~100 characters of prompt text, a few hundred with the longest framings)."""
    return len(q["question"]) + sum(len(c) for c in q["choices"])


def hash_bucket(q: dict) -> int:
    """crc32 bucket 0..99 of (source, qid): the whole split rule is a function of this number."""
    return zlib.crc32(f"{q['source']}|{q['qid']}".encode("utf-8")) % 100


def _raw_split(q: dict, wmdp_dir_ids, rep: Optional[Tuple[str, str]] = None) -> str:
    """The pure split rule (+ WMDP-bio special case); `wmdp_dir_ids` = {(source, qid)} of the first 680 wmdp_train rows.
    `rep` = (source, qid) of the question's cluster representative (clustered evals): its bucket decides, not q's own."""
    b = hash_bucket(q if rep is None else {"source": rep[0], "qid": rep[1]})
    if q["source"] == WMDP_SOURCE:
        if (q["source"], q["qid"]) in wmdp_dir_ids:
            return "dir"
        return "disc" if b < WMDP_REST_DISC_BELOW else "rep"
    return "dir" if b < DIR_BELOW else ("disc" if b < DISC_BELOW else "rep")


def _char_ngrams(text: str, k: int) -> frozenset:
    """Distinct character k-grams of the whitespace-collapsed text (empty if the text is shorter than k)."""
    t = " ".join(text.split())
    return frozenset(t[i:i + k] for i in range(len(t) - k + 1))


def context_components(stems: Sequence[str], k: int = CLUSTER_K, min_shared: int = CLUSTER_MIN_SHARED,
                       min_overlap: float = CLUSTER_MIN_OVERLAP) -> List[int]:
    """Connected components of the "share context" relation over `stems`. Two stems are linked iff the sets of their
    distinct character k-grams (whitespace collapsed, case kept) share at least `min_shared` elements AND at least
    `min_overlap` of the SMALLER set. Returns one component label per stem (0, 1, ... in order of first appearance), so the
    result does not depend on how the union-find happens to be organised. Stems with fewer than `min_shared` k-grams are
    never linked (exact repeats of short stems are the business of the de-duplication). Exact, deterministic, stdlib only.
    Cost: about 0.7 s for the 108 ProtocolQA stems (about 6k characters each); it grows quadratically with the number of stems
    that share k-grams, so flag only evals of a few hundred questions as `clustered`."""
    if k < 1 or min_shared < 1 or not 0.0 < min_overlap <= 1.0:
        raise ValueError(f"need k >= 1, min_shared >= 1 and 0 < min_overlap <= 1; got {k}, {min_shared}, {min_overlap}")
    sets = [_char_ngrams(s, k) for s in stems]
    parent = list(range(len(sets)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    live = [i for i, s in enumerate(sets) if len(s) >= min_shared]  # a stem with fewer k-grams cannot share min_shared
    postings: Dict[str, List[int]] = {}
    for i in live:
        for g in sets[i]:
            postings.setdefault(g, []).append(i)
    for i in live:
        shared = Counter(chain.from_iterable(postings[g] for g in sets[i]))  # |set_i & set_j| for every j (j == i: itself)
        for j, c in shared.items():
            if j > i and c >= min_shared and c >= min_overlap * min(len(sets[i]), len(sets[j])):
                parent[find(i)] = find(j)
    label: Dict[int, int] = {}
    return [label.setdefault(find(i), len(label)) for i in range(len(sets))]


def _cluster_reps(rows: Sequence[dict]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """(source, qid) -> (source, qid) of the cluster representative (the member with the lowest qid) for EVERY row."""
    comp = context_components([q["question"] for q in rows])
    best: Dict[int, dict] = {}
    for c, q in zip(comp, rows):
        if c not in best or q["qid"] < best[c]["qid"]:
            best[c] = q
    return {(q["source"], q["qid"]): (best[c]["source"], best[c]["qid"]) for c, q in zip(comp, rows)}


def resolve_group(raws: Sequence[str]) -> Tuple[str, List[int]]:
    """Duplicate policy for one group of questions with equal key. `raws` = the members' own splits in registry order.
    Returns (group split, indices of the members that are kept): all dir members if any is dir, else the first member."""
    dirs = [i for i, r in enumerate(raws) if r == "dir"]
    if dirs:
        return "dir", dirs
    return raws[0], [0]


def _order_key(q: dict):
    return (zlib.crc32(f"order|{q['source']}|{q['qid']}".encode("utf-8")), q["source"], q["qid"])


def _copy(q: dict) -> dict:
    return {"source": q["source"], "qid": q["qid"], "question": q["question"], "choices": list(q["choices"]),
            "answer": q["answer"]}


# ================================================================================================ loading
@lru_cache(maxsize=None)
def _read_jsonl_cached(path: str) -> Tuple[dict, ...]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e})") from None
    return tuple(rows)


def _read_jsonl(path: str) -> Tuple[dict, ...]:
    """Rows of a .jsonl file (shared, cached: treat as read-only)."""
    return _read_jsonl_cached(path)


@lru_cache(maxsize=1)
def _wmdp_dir_ids() -> frozenset:
    rows = _read_jsonl(os.path.join(ROOT, WMDP_TRAIN))
    if len(rows) < WMDP_DIR_N:
        raise ValueError(f"{WMDP_TRAIN} has {len(rows)} rows, need at least {WMDP_DIR_N}")
    return frozenset((r["source"], r["qid"]) for r in rows[:WMDP_DIR_N])


class _Member:
    __slots__ = ("eval", "q", "raw", "key")

    def __init__(self, eval_name, q, raw, key):
        self.eval, self.q, self.raw, self.key = eval_name, q, raw, key


class _Registry:
    """All registered questions after validation, clustering, de-duplication and the prompt-length cap (see the module
    docstring). Built once per process. `max_chars` is MAX_PROMPT_CHARS (a parameter so that tests can use a small cap)."""

    def __init__(self, evals: Dict[str, EvalSpec], root: str, wmdp_dir_ids, max_chars: int = MAX_PROMPT_CHARS):
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise ValueError(f"max_chars must be a positive int, got {max_chars!r}")
        self.evals, self.root, self.max_chars = evals, root, max_chars
        members: List[_Member] = []
        rows_read = {name: 0 for name in evals}
        invalid = []
        self.cluster_rep: Dict[Tuple[str, str], Tuple[str, str]] = {}   # (source, qid) -> representative, clustered evals
        cluster_stats: Dict[str, dict] = {}
        for name, spec in evals.items():
            valid: List[dict] = []
            for src in spec.files:  # (union views have no files)
                path = os.path.join(root, src.path)
                for q in _read_jsonl(path):
                    if not isinstance(q, dict):
                        raise ValueError(f"{src.path}: row is not an object")
                    if src.source_prefix and not str(q.get("source", "")).startswith(src.source_prefix):
                        continue
                    rows_read[name] += 1
                    why = validate_question(q)
                    if why is not None:
                        if why.startswith("structure"):
                            raise ValueError(f"{src.path}: {q.get('source')}/{q.get('qid')}: {why}")
                        invalid.append({"eval": name, "source": q["source"], "qid": q["qid"], "reason": why})
                        continue
                    valid.append(q)
            reps: Dict[Tuple[str, str], Tuple[str, str]] = {}
            if spec.clustered:
                if any(q["source"] == WMDP_SOURCE for q in valid):
                    raise ValueError(f"{name}: a clustered eval cannot contain {WMDP_SOURCE} rows (their dir pool is fixed)")
                reps = _cluster_reps(valid)
                sizes = Counter(reps.values())
                cluster_stats[name] = {"questions": len(valid), "clusters": len(sizes),
                                       "multi": sum(1 for v in sizes.values() if v > 1), "largest": max(sizes.values(), default=0)}
                self.cluster_rep.update(reps)
            for q in valid:
                members.append(_Member(name, q, _raw_split(q, wmdp_dir_ids, reps.get((q["source"], q["qid"]))), question_key(q)))
        seen: Dict[Tuple[str, str], str] = {}
        for m in members:
            k = (m.q["source"], m.q["qid"])
            if k in seen:
                raise ValueError(f"(source, qid) {k} appears in eval {seen[k]!r} and eval {m.eval!r}")
            seen[k] = m.eval

        groups: Dict[str, List[_Member]] = {}
        for m in members:
            groups.setdefault(m.key, []).append(m)
        self.bucket_by_key: Dict[str, str] = {}
        self.bucket_by_id: Dict[Tuple[str, str], str] = {}
        self.kept: Dict[str, Dict[str, List[dict]]] = {n: {s: [] for s in SPLITS} for n in evals}
        dropped = []
        kept_ids = set()
        for key, ms in groups.items():
            bucket, idx = resolve_group([m.raw for m in ms])
            self.bucket_by_key[key] = bucket
            first_kept = ms[idx[0]]
            for i, m in enumerate(ms):
                self.bucket_by_id[(m.q["source"], m.q["qid"])] = bucket
                if i in idx:
                    kept_ids.add(id(m))
                else:
                    dropped.append({"eval": m.eval, "source": m.q["source"], "qid": m.q["qid"],
                                    "reason": "duplicate_of_dir" if bucket == "dir" else "duplicate",
                                    "kept": f"{first_kept.eval}/{first_kept.q['qid']}"})
        for m in members:
            if id(m) in kept_ids:
                self.kept[m.eval][self.bucket_by_key[m.key]].append(m.q)
        # Prompt-length cap. Deliberately AFTER the split assignment and the de-duplication: a removed question keeps its
        # bucket_by_id / bucket_by_key entry (split_of still answers), so removing a long dir question cannot let a copy of
        # it into disc / rep, and no other question changes its split.
        self.too_long: List[dict] = []
        for name in evals:
            for s in SPLITS:
                fits = []
                for q in self.kept[name][s]:
                    n_chars = prompt_chars(q)
                    if n_chars > max_chars:
                        self.too_long.append({"eval": name, "source": q["source"], "qid": q["qid"], "split": s, "chars": n_chars})
                    else:
                        fits.append(q)
                self.kept[name][s] = fits
        for name, spec in evals.items():  # union views: the members' questions (members are disjoint after de-dup)
            for member in spec.union:
                rows_read[name] += rows_read[member]
                for s in SPLITS:
                    self.kept[name][s].extend(self.kept[member][s])

        self.served: Dict[str, Dict[str, List[dict]]] = {}
        for name in evals:
            disc = sorted(self.kept[name]["disc"], key=_order_key)
            rep = sorted(self.kept[name]["rep"], key=_order_key)
            self.served[name] = {"disc": disc, "rep": rep, "all": sorted(disc + rep, key=_order_key)}

        self.pools = self._build_pools(evals, wmdp_dir_ids)

        multi = [ms for ms in groups.values() if len(ms) > 1]
        per_eval = {}
        for name, spec in evals.items():
            k = self.kept[name]
            names = set(spec.union) or {name}
            per_eval[name] = {"rows": rows_read[name],
                              "invalid": sum(1 for r in invalid if r["eval"] in names),
                              "dropped_duplicate": sum(1 for r in dropped if r["eval"] in names),
                              "too_long": sum(1 for r in self.too_long if r["eval"] in names),
                              "n": sum(len(k[s]) for s in SPLITS),
                              "dir": len(k["dir"]), "disc": len(k["disc"]), "rep": len(k["rep"])}
        self.report = {
            "rows": sum(rows_read[n] for n, s in evals.items() if not s.is_union),
            "invalid": invalid,
            "dropped": dropped,
            "duplicate_groups": len(multi),
            "duplicate_groups_cross_set": sum(1 for ms in multi if len({m.eval for m in ms}) > 1),
            "duplicate_groups_within_set": sum(1 for ms in multi if len({m.eval for m in ms}) == 1),
            "max_prompt_chars": max_chars,
            "too_long": self.too_long,
            "clusters": cluster_stats,
            "per_eval": per_eval,
        }

    def _build_pools(self, evals, wmdp_dir_ids) -> Dict[str, List[dict]]:
        pools: Dict[str, List[dict]] = {}
        for pool in POOLS:
            qs = [q for name, spec in evals.items() if spec.pool == pool for q in self.kept[name]["dir"]]
            if pool == "wmdp":
                if qs and {(q["source"], q["qid"]) for q in qs} != set(wmdp_dir_ids):
                    raise ValueError("wmdp pool is not exactly the first 680 rows of wmdp_train.jsonl "
                                     f"(is one of them longer than max_chars={self.max_chars}?)")
            else:
                seen, uniq = set(), []
                for q in qs:
                    k = question_key(q)
                    if k not in seen:
                        seen.add(k)
                        uniq.append(q)
                qs = sorted(uniq, key=_order_key)
            pools[pool] = qs
        return pools


@lru_cache(maxsize=1)
def _registry() -> _Registry:
    return _Registry(EVALS, ROOT, _wmdp_dir_ids())


# ================================================================================================ public API
def split_of(q: dict) -> str:
    """"dir" | "disc" | "rep" for a question dict (needs "source" and "qid"; "question" is used if present). A registered
    question (matched by (source, qid), else by equal `question_key`) gets the split of its duplicate group; any other
    question gets the pure crc rule."""
    reg = _registry()
    bucket = reg.bucket_by_id.get((q["source"], q["qid"]))
    if bucket is None and isinstance(q.get("question"), str):
        bucket = reg.bucket_by_key.get(question_key(q))
    return bucket if bucket is not None else _raw_split(q, _wmdp_dir_ids())


def cluster_of(q: dict) -> str:
    """Id ("source|qid" of the representative) of the cluster a question belongs to: the group of questions of a clustered eval
    that share context (the same protocol / plasmid / template sequence) and were therefore split together. Every other question
    (any question of an eval that is not clustered, a singleton cluster, an unregistered question) is its own cluster. Use it
    to group questions for cluster-level inference (cluster bootstrap / permutation) inside one split: the split makes disc,
    rep and dir cluster-disjoint but the questions of one cluster are still dependent."""
    rep = _registry().cluster_rep.get((q["source"], q["qid"]), (q["source"], q["qid"]))
    return f"{rep[0]}|{rep[1]}"


def get_eval(name: str, split: str = "all") -> List[dict]:
    """Questions of eval `name` for evaluation. split "disc" or "rep" = that split; "all" = disc + rep (never dir: dir
    questions are only available through get_pool()). Questions longer than MAX_PROMPT_CHARS are not served (see
    registry_report()["too_long"]). Fixed pseudo-random order; fresh dict copies."""
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}; known: {sorted(EVALS)}")
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r} (dir questions: use get_pool())")
    return [_copy(q) for q in _registry().served[name][split]]


def get_pool() -> Dict[str, List[dict]]:
    """Direction-building questions, only from the dir bucket: {"wmdp", "hard", "nonbio"}. "wmdp" = the first 680 rows of
    data/wmdp_train.jsonl in file order. "hard" = dir buckets of the evals with pool == "hard" (labbench_*, mmlupro_bio,
    gpqa_bio, supergpqa_bio), "nonbio" = those of mmlu_nonbio and mathphys; both de-duplicated, pseudo-random order.
    Every question has prompt_chars(q) <= MAX_PROMPT_CHARS (longer ones are in registry_report()["too_long"]), so no recipe
    meets a prompt of tens of thousands of tokens."""
    return {k: [_copy(q) for q in v] for k, v in _registry().pools.items()}


def split_sizes(name: str) -> Dict[str, int]:
    """{"dir": n, "disc": n, "rep": n} for eval `name` (after validation and de-duplication)."""
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}; known: {sorted(EVALS)}")
    k = _registry().kept[name]
    return {s: len(k[s]) for s in SPLITS}


def registry_report() -> dict:
    """What the registry did at load time: rows read, invalid rows dropped, duplicate groups (cross-set / within-set),
    duplicates dropped, questions removed by the prompt-length cap ("too_long": eval, source, qid, split, chars), cluster
    statistics of the clustered evals and per-eval counts (rows = n + invalid + dropped_duplicate + too_long); a deep copy:
    safe to modify."""
    return copy.deepcopy(_registry().report)


def _shingles(text: str, n: int = 3) -> frozenset:
    w = norm_text(text).split()
    if len(w) < n + 3:  # too short to call two stems near-duplicates
        return frozenset()
    return frozenset(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))


def near_duplicate_report(threshold: float = 0.5) -> dict:
    """Near-duplicate stems between the direction pools and the served questions. Similarity = Jaccard of word 3-gram
    sets of the normalised stems (stems shorter than 6 words are skipped). Returns {"threshold", "per_eval": {name:
    {"served": n, "near_dup_of_pool": k}}, "total_served", "total_near_dup"}. Exact duplicates are already removed by the
    registry; this counts paraphrases / sibling questions generated from the same source passage."""
    reg = _registry()
    pool_sh = []
    index: Dict[str, List[int]] = {}
    for qs in reg.pools.values():
        for q in qs:
            sh = _shingles(q["question"])
            if sh:
                for g in sh:
                    index.setdefault(g, []).append(len(pool_sh))
                pool_sh.append(sh)
    per_eval, tot, tot_nd = {}, 0, 0
    for name, spec in EVALS.items():
        if spec.is_union:  # its members are already counted
            continue
        served = reg.served[name]["all"]
        nd = 0
        for q in served:
            sh = _shingles(q["question"])
            if not sh:
                continue
            cnt: Dict[int, int] = {}
            for g in sh:
                for j in index.get(g, ()):
                    cnt[j] = cnt.get(j, 0) + 1
            if any(c / (len(sh) + len(pool_sh[j]) - c) >= threshold for j, c in cnt.items()):
                nd += 1
        per_eval[name] = {"served": len(served), "near_dup_of_pool": nd}
        tot, tot_nd = tot + len(served), tot_nd + nd
    return {"threshold": threshold, "per_eval": per_eval, "total_served": tot, "total_near_dup": tot_nd}


# ================================================================================================ docs rendering
def eval_profile(name: str) -> dict:
    """Answer-format statistics of the questions served for evaluation (disc + rep) of eval `name`: number of options
    (min / max), chance accuracy (mean 1/#options) and the most frequent correct-answer position with its share (a model
    that collapses onto one letter scores about that share; matters when steering shifts letter preferences)."""
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}; known: {sorted(EVALS)}")
    qs = _registry().served[name]["all"]
    if not qs:
        return {"n": 0, "min_options": 0, "max_options": 0, "chance": 0.0, "top_letter": "-", "top_share": 0.0}
    counts: Dict[int, int] = {}
    for q in qs:
        counts[q["answer"]] = counts.get(q["answer"], 0) + 1
    top = max(sorted(counts), key=lambda i: counts[i])
    return {"n": len(qs), "min_options": min(len(q["choices"]) for q in qs), "max_options": max(len(q["choices"]) for q in qs),
            "chance": sum(1.0 / len(q["choices"]) for q in qs) / len(qs), "top_letter": "ABCDEFGHIJ"[top],
            "top_share": counts[top] / len(qs)}


def render_docs_block() -> str:
    """The generated part of docs/evals.md (tables + per-set caveats); a test checks the committed docs contain it."""
    rep = registry_report()["per_eval"]
    lines = ["| eval | category | hazard | pool | clustered | n | dir | disc | rep |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in EVALS.items():
        r = rep[name]
        lines.append(f"| `{name}` | {s.category} | {'yes' if s.hazard else 'no'} | {s.pool or '-'} | {'yes' if s.clustered else '-'} | "
                     f"{r['n']} | {r['dir']} | {r['disc']} | {r['rep']} |")
    lines += ["", "| eval | rows read | invalid | duplicate | too long | options | chance | top letter |", "|---|---|---|---|---|---|---|---|"]
    for name, s in EVALS.items():
        r, p = rep[name], eval_profile(name)
        opts = f"{p['min_options']}" if p["min_options"] == p["max_options"] else f"{p['min_options']}-{p['max_options']}"
        lines.append(f"| `{name}` | {r['rows']} | {r['invalid']} | {r['dropped_duplicate']} | {r['too_long']} | {opts} | "
                     f"{100 * p['chance']:.1f}% | {p['top_letter']} {100 * p['top_share']:.0f}% |")
    lines += ["", "| eval | files | origin | licence |", "|---|---|---|---|"]
    for name, s in EVALS.items():
        if s.is_union:
            files = "union of " + ", ".join(f"`{m}`" for m in s.union)
        else:
            files = ", ".join(f"`{f.path}`" + (f" (rows `{f.source_prefix}*`)" if f.source_prefix else "") for f in s.files)
        lines.append(f"| `{name}` | {files} | {s.origin.replace('|', '/')} | {s.licence.replace('|', '/')} |")
    lines += [""]
    for name, s in EVALS.items():
        if s.notes:
            lines.append(f"* `{name}`: {s.notes}")
    return "\n".join(lines) + "\n"


# ================================================================================================ builder (offline)
# Everything below is only needed to REGENERATE data/evals/*.jsonl; it needs network access and the `datasets` /
# `huggingface_hub` packages (imported lazily). Selection is deterministic: fixed seed, pinned dataset revisions, ranking by
# sha256 (independent of Python's RNG and of input order).
BUILD_SEED = 20241121
# Builder only: drop candidates whose question + options exceed this many characters. Stricter than the registry-wide
# MAX_PROMPT_CHARS (a test checks MAX_CHARS <= MAX_PROMPT_CHARS), so a rebuilt file never contains a question the registry
# would remove; it is NOT raised to MAX_PROMPT_CHARS because that would change the committed data/evals files on a rebuild.
MAX_CHARS = 6000
HF_REVISIONS = {
    "cais/wmdp": "7125571f22f032c56415e7980f48d877dd830ff8",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "GBaker/MedQA-USMLE-4-options": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
    "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "qiaojin/PubMedQA": "9001f2853fb87cab8d220904e0de81ac6973b318",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "TsinghuaC3I/MedXpertQA": "7e7c465a68eb2b866926bfa59c8c9d17a8daba65",
    "m-a-p/SuperGPQA": "4430d4458112c7d4497fdcf94d7cc223313d6acf",
}
MEDMCQA_SUBJECTS = ("Anatomy", "Physiology", "Biochemistry", "Microbiology", "Pathology", "Pharmacology")
MMLU_NONBIO_SUBJECTS = ("high_school_us_history", "high_school_geography", "high_school_government_and_politics",
                        "high_school_macroeconomics", "electrical_engineering", "philosophy", "sociology",
                        "college_computer_science")
MMLUPRO_NONBIO_CATEGORIES = ("business", "computer science", "economics", "history", "law", "philosophy")
MMLU_BIO_SUBJECTS = {  # eval name -> MMLU subject
    "mmlu_college_bio": "college_biology", "mmlu_hs_bio": "high_school_biology", "mmlu_virology": "virology",
    "mmlu_med_genetics": "medical_genetics", "mmlu_anatomy": "anatomy", "mmlu_clinical": "clinical_knowledge",
    "mmlu_college_med": "college_medicine", "mmlu_prof_med": "professional_medicine", "mmlu_nutrition": "nutrition"}


def _rank(label: str, key: str) -> str:
    return hashlib.sha256(f"{BUILD_SEED}|{label}|{key}".encode("utf-8")).hexdigest()


def stable_sample(items: Sequence[dict], k: int, label: str) -> List[dict]:
    """The k items with the smallest sha256(seed|label|qid): deterministic, independent of input order and of the
    Python version. If len(items) <= k all items are returned."""
    return sorted(items, key=lambda it: _rank(label, it["qid"]))[:k]


def stratified_sample(items: Sequence[dict], k: int, label: str, stratum: Callable[[dict], str]) -> List[dict]:
    """Equal allocation over strata (unused quota of small strata is redistributed, ties broken by stratum name), then
    stable_sample inside each stratum."""
    groups: Dict[str, List[dict]] = {}
    for it in items:
        groups.setdefault(stratum(it), []).append(it)
    quota = {g: 0 for g in groups}
    remaining = min(k, len(items))
    active = sorted(groups)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        nxt = []
        for g in active:
            take = min(share, len(groups[g]) - quota[g], remaining)
            quota[g] += take
            remaining -= take
            if len(groups[g]) > quota[g]:
                nxt.append(g)
            if remaining == 0:
                break
        active = nxt
    out: List[dict] = []
    for g in sorted(groups):
        out += stable_sample(groups[g], quota[g], f"{label}|{g}")
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def conv_mmlu(row: dict, subject: str, split: str, idx: int) -> dict:
    return {"source": f"mmlu/{subject}", "qid": f"mmlu_{subject}_{split}_{idx:04d}", "question": row["question"].strip(),
            "choices": [str(c).strip() for c in row["choices"]], "answer": int(row["answer"])}


def conv_wmdp(row: dict, domain: str, idx: int) -> dict:
    return {"source": f"wmdp/{domain}", "qid": f"wmdp_{domain}_{idx:04d}", "question": row["question"].strip(),
            "choices": [str(c).strip() for c in row["choices"]], "answer": int(row["answer"])}


def conv_medqa(row: dict, idx: int) -> Optional[dict]:
    opts = row["options"]
    letters = sorted(opts)
    if letters != ["A", "B", "C", "D"] or row["answer_idx"] not in letters:
        return None
    choices = [str(opts[k]).strip() for k in letters]
    ans = letters.index(row["answer_idx"])
    if choices[ans] != str(row["answer"]).strip():
        return None
    return {"source": "medqa/usmle", "qid": f"medqa_test_{idx:04d}", "question": row["question"].strip(),
            "choices": choices, "answer": ans}


def conv_medmcqa(row: dict) -> Optional[dict]:
    if row["choice_type"] != "single" or row["subject_name"] not in MEDMCQA_SUBJECTS or row["cop"] not in (0, 1, 2, 3):
        return None
    return {"source": f"medmcqa/{_slug(row['subject_name'])}", "qid": "medmcqa_" + row["id"].replace("-", "")[:12],
            "question": row["question"].strip(), "choices": [str(row[k]).strip() for k in ("opa", "opb", "opc", "opd")],
            "answer": int(row["cop"])}


def conv_pubmedqa(row: dict) -> Optional[dict]:
    ctx = row["context"]
    if len(ctx["labels"]) != len(ctx["contexts"]) or row["final_decision"] not in ("yes", "no", "maybe"):
        return None
    body = "\n".join(f"{lab}: {txt.strip()}" for lab, txt in zip(ctx["labels"], ctx["contexts"]))
    return {"source": "pubmedqa/pqa_labeled", "qid": f"pubmedqa_{row['pubid']}",
            "question": f"Abstract:\n{body}\n\nQuestion: {row['question'].strip()}",
            "choices": ["yes", "no", "maybe"], "answer": ["yes", "no", "maybe"].index(row["final_decision"])}


def conv_supergpqa(row: dict) -> Optional[dict]:
    if row["discipline"] != "Science" or row["field"] != "Biology" or row["difficulty"] not in ("middle", "hard"):
        return None
    if str(row["is_calculation"]).lower() == "true":
        return None
    opts = [str(o).strip() for o in row["options"]]
    idx = ord(row["answer_letter"]) - ord("A")
    if not 0 <= idx < len(opts) or opts[idx] != str(row["answer"]).strip():
        return None
    return {"source": "supergpqa/biology", "qid": "supergpqa_" + row["uuid"][:12], "question": row["question"].strip(),
            "choices": opts, "answer": idx}


def conv_medxpertqa(row: dict) -> Optional[dict]:
    letters = sorted(row["options"])
    if letters != list("ABCDEFGHIJ")[:len(letters)] or row["label"] not in letters:
        return None
    q = row["question"]
    cut = q.find("\nAnswer Choices:")  # MedXpertQA repeats the options inline in the stem; the prompt adds them itself
    stem = (q[:cut] if cut >= 0 else q).strip()
    return {"source": "medxpertqa/text", "qid": f"medxpertqa_text_{int(row['id'].split('-')[-1]):04d}", "question": stem,
            "choices": [str(row["options"][k]).strip() for k in letters], "answer": letters.index(row["label"])}


def conv_mmlupro(row: dict) -> Optional[dict]:
    opts = [str(o).strip() for o in row["options"]]
    idx = int(row["answer_index"])
    if not 0 <= idx < len(opts):
        return None
    cat = row["category"].replace(" ", "_")
    return {"source": f"mmlupro/{cat}", "qid": f"mmlupro_{cat}_{int(row['question_id']):05d}",
            "question": row["question"].strip(), "choices": opts, "answer": idx}


def _hf_dataset(repo: str, config: Optional[str] = None):
    from datasets import load_dataset
    return load_dataset(repo, config, revision=HF_REVISIONS[repo])


def _hf_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=HF_REVISIONS[repo])


def _load_mmlu(subject: str) -> List[dict]:
    ds = _hf_dataset("cais/mmlu", subject)
    return [conv_mmlu(r, subject, sp, i) for sp in ("test", "validation") for i, r in enumerate(ds[sp])]


def _load_mmlu_many(subjects: Iterable[str]) -> List[dict]:
    return [q for s in subjects for q in _load_mmlu(s)]


def _load_wmdp(domain: str) -> List[dict]:
    ds = _hf_dataset("cais/wmdp", f"wmdp-{domain}")["test"]
    return [conv_wmdp(r, domain, i) for i, r in enumerate(ds)]


def _compact(rows: Iterable[Optional[dict]]) -> List[dict]:
    return [q for q in rows if q is not None]


def _load_medqa() -> List[dict]:
    return _compact(conv_medqa(r, i) for i, r in enumerate(_hf_dataset("GBaker/MedQA-USMLE-4-options")["test"]))


def _load_medmcqa() -> List[dict]:
    return _compact(conv_medmcqa(r) for r in _hf_dataset("openlifescienceai/medmcqa")["validation"])


def _load_pubmedqa() -> List[dict]:
    return _compact(conv_pubmedqa(r) for r in _hf_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"])


def _load_jsonl_hf(repo: str, filename: str) -> List[dict]:
    with open(_hf_file(repo, filename), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_supergpqa() -> List[dict]:
    return _compact(conv_supergpqa(r) for r in _load_jsonl_hf("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl"))


def _load_medxpertqa() -> List[dict]:
    return _compact(conv_medxpertqa(r) for r in _load_jsonl_hf("TsinghuaC3I/MedXpertQA", "Text/test.jsonl"))


def _load_mmlupro(categories: Iterable[str]) -> List[dict]:
    cats = set(categories)
    return _compact(conv_mmlupro(r) for r in _hf_dataset("TIGER-Lab/MMLU-Pro")["test"] if r["category"] in cats)


def _builds() -> List[Tuple[str, Callable[[], List[dict]], int, Optional[Callable[[dict], str]]]]:
    """(name, candidate loader, max questions, stratum function or None), in BUILD ORDER (earlier sets win duplicates)."""
    builds: List[Tuple[str, Callable[[], List[dict]], int, Optional[Callable[[dict], str]]]] = []
    for name, subject in MMLU_BIO_SUBJECTS.items():
        builds.append((name, (lambda s=subject: _load_mmlu(s)), 400, None))
    builds += [
        ("medqa_usmle", _load_medqa, 400, None),
        ("medmcqa_bio", _load_medmcqa, 400, lambda q: q["source"]),
        ("pubmedqa", _load_pubmedqa, 400, None),
        ("supergpqa_bio", _load_supergpqa, 400, None),
        ("medxpertqa_text", _load_medxpertqa, 400, None),
        ("wmdp_cyber", lambda: _load_wmdp("cyber"), 400, None),
        ("wmdp_chem", lambda: _load_wmdp("chem"), 408, None),
        ("mmlu_nonbio", lambda: _load_mmlu_many(MMLU_NONBIO_SUBJECTS), 400, lambda q: q["source"]),
        ("mmlupro_nonbio", lambda: _load_mmlupro(MMLUPRO_NONBIO_CATEGORIES), 400, lambda q: q["source"]),
    ]
    return builds


def make_set(name: str, candidates: Sequence[dict], n_max: int, stratum: Optional[Callable[[dict], str]],
             excluded: set) -> Tuple[List[dict], Dict[str, int]]:
    """candidates -> (selected questions in file order, counters). Drops invalid / over-long candidates, candidates whose
    key is in `excluded` (already used by another set or the pilot) and within-set duplicates, then samples."""
    stats: Dict[str, int] = {"candidates": len(candidates)}

    def bump(k):
        stats[k] = stats.get(k, 0) + 1

    kept, seen = [], set()
    for q in sorted(candidates, key=lambda q: _rank(f"{name}|dedupe", q["qid"])):
        why = validate_question(q)
        if why is not None:
            bump("invalid: " + why)
            continue
        if len(q["question"]) + sum(len(c) for c in q["choices"]) > MAX_CHARS:
            bump("too_long")
            continue
        key = question_key(q)
        if key in excluded:
            bump("duplicate_of_earlier_set")
            continue
        if key in seen:
            bump("duplicate_within_set")
            continue
        seen.add(key)
        kept.append(q)
    chosen = stratified_sample(kept, n_max, name, stratum) if stratum else stable_sample(kept, n_max, name)
    chosen.sort(key=lambda q: _rank(f"{name}|order", q["qid"]))
    stats["eligible"] = len(kept)
    stats["written"] = len(chosen)
    return chosen, stats


def write_jsonl(path: str, qs: Sequence[dict]) -> None:
    """ASCII-only JSON lines (like the existing data files: independent of the reader's locale)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps({k: q[k] for k in KEYS}) + "\n")
    os.replace(tmp, path)


def _existing_keys() -> set:
    """Keys of every question in the pre-existing (not generated) registered files, plus the pilot file."""
    keys = set()
    paths = {src.path for s in EVALS.values() if not s.generated for src in s.files} | {PILOT_FILE}
    for p in sorted(paths):
        for q in _read_jsonl(os.path.join(ROOT, p)):
            keys.add(question_key(q))
    return keys


def build(only: Optional[Sequence[str]] = None, log: Callable[[str], None] = print) -> Dict[str, Dict[str, int]]:
    """(Re)generate data/evals/<name>.jsonl for every generated eval (or only those named). Deterministic. Always processes
    the sets in the fixed build order so that exclusions (earlier sets' questions) are identical to a full rebuild."""
    out_dir = os.path.join(ROOT, "data", "evals")
    os.makedirs(out_dir, exist_ok=True)
    excluded = _existing_keys()
    all_stats: Dict[str, Dict[str, int]] = {}
    for name, loader, n_max, stratum in _builds():
        if name not in EVALS or not EVALS[name].generated:
            raise ValueError(f"build {name!r} is not a generated eval in EVALS")
        chosen, stats = make_set(name, loader(), n_max, stratum, excluded)
        excluded |= {question_key(q) for q in chosen}
        all_stats[name] = stats
        log(f"{name}: {stats}")
        if only is None or name in only:
            write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), chosen)
    return all_stats


# ================================================================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m sbx.evals", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="regenerate data/evals/*.jsonl (network)")
    b.add_argument("--only", nargs="+", default=None)
    sub.add_parser("docs", help="print the generated block of docs/evals.md")
    sub.add_parser("report", help="print registry_report() and near_duplicate_report()")
    sub.add_parser("check", help="build the registry and print per-eval counts")
    sub.add_parser("counts", help="print the _COUNTS literal for sbx/evals.py")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        unknown = set(a.only or ()) - set(EVALS)
        if unknown:
            print(f"unknown evals: {sorted(unknown)}", file=sys.stderr)
            return 2
        build(a.only)
    elif a.cmd == "docs":
        sys.stdout.write(render_docs_block())
    elif a.cmd == "report":
        r = registry_report()
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_eval", "too_long")}, indent=1))
        tl = Counter(x["eval"] for x in r["too_long"])
        print(f"too long (> {r['max_prompt_chars']} chars, removed from every split and pool): {len(r['too_long'])};", dict(tl))
        for thr in (0.5, 0.8):
            nd = near_duplicate_report(thr)
            print(f"near-duplicates (Jaccard >= {thr}): {nd['total_near_dup']} / {nd['total_served']} served;",
                  {k: v["near_dup_of_pool"] for k, v in nd["per_eval"].items() if v["near_dup_of_pool"]})
    elif a.cmd == "check":
        for name, r in registry_report()["per_eval"].items():
            print(f"{name:22s} {r}")
    elif a.cmd == "counts":
        per = registry_report()["per_eval"]
        print("_COUNTS: Dict[str, int] = {")
        for name in EVALS:
            print(f"    {name!r}: {per[name]['n']},")
        print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
