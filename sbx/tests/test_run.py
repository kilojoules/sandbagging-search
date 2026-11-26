"""sbx.run: helpers, store, dry-run and a tiny end-to-end pass (random tiny model: only mechanics are checked)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sbx import run as R


def test_parse_layers_validates_range_and_forms():
    assert R.parse_layers("auto1", 36) == [15]
    assert R.parse_layers("autoN", 36)[0] == 10 and R.parse_layers("autoN", 36)[-1] == 28
    assert R.parse_layers("10-12", 36) == [10, 11, 12]
    assert R.parse_layers("12,10,10", 36) == [10, 12]
    for bad in ("0", "36", "1-99", "x", "", "5,40"):
        with pytest.raises(ValueError):
            R.parse_layers(bad, 36)


def test_layer_tag_roundtrip_and_no_collisions():
    sets = [[15], [10, 11, 12], [10, 12, 15], [10, 11, 12, 13, 14, 15]]
    tags = [R.layer_tag(s) for s in sets]
    assert len(set(tags)) == len(tags)
    assert R.layer_tag([10, 12, 15]) != R.layer_tag(list(range(10, 16)))
    for s, t in zip(sets, tags):
        assert R.layers_from_tag(t) == s
    assert R.layers_from_tag("-") == []


def test_expand_framings_groups_globs_and_order():
    from sbx.framings import FRAMINGS, GROUPS
    fr = dict(FRAMINGS)
    fr["N2"] = FRAMINGS["N"]
    allf = R.expand_framings("all", fr, GROUPS)
    assert allf[0] == "N" and allf[1] == "E" and "N2" not in allf and set(allf) == set(FRAMINGS)
    assert R.expand_framings("nat_*,E", fr, GROUPS)[:2] == ["E", "nat_doc_user"] or R.expand_framings("nat_*,E", fr, GROUPS)[0] == "E"
    nat = R.expand_framings("natural_trigger", fr, GROUPS)
    assert set(nat) == set(GROUPS["natural_trigger"])
    with pytest.raises(ValueError):
        R.expand_framings("no_such_framing", fr, GROUPS)
    with pytest.raises(ValueError):
        R.expand_framings("zzz*", fr, GROUPS)


def test_one_sided_mcnemar():
    assert R.p_one_sided(0, 0) == 1.0
    assert abs(R.p_one_sided(5, 0) - 1 / 32) < 1e-12
    assert R.p_one_sided(0, 5) == 1.0


def test_store_roundtrip_and_consolidate(tmp_path):
    st = R.Store(str(tmp_path))
    rec = dict(key="wmdp_bio|disc|N|none|none|-|0.0|0", eval="wmdp_bio", split="disc")
    st.put_qids("wmdp_bio", "disc", ["a", "b", "c"])
    st.put(rec, [1, 0, 1], [1, 1, 1], [0, 0, 0])
    st.consolidate()
    z = np.load(os.path.join(str(tmp_path), "arrays.npz"))
    assert f"corr::{rec['key']}" in z.files and f"parsed::{rec['key']}" in z.files and f"refused::{rec['key']}" in z.files
    assert list(z["qids::wmdp_bio|disc"]) == ["a", "b", "c"]
    st2 = R.Store(str(tmp_path))                       # resume: the cell is known
    assert st2.has(rec["key"]) and st2.corr(rec["key"]).tolist() == [1.0, 0.0, 1.0]


def test_dry_run_loads_no_weights_and_plans(tmp_path, capsys):
    R.main(["--model", "qwen3-8b", "--out", str(tmp_path), "--dry-run", "--stages", "behavior,steer", "--framings", "all",
            "--dir-names", "B,A", "--steer-framings", "E,N,auto2"])
    out = capsys.readouterr().out
    assert "[plan] behavior cells:" in out and "cells planned" in out


def test_dry_run_rejects_unknown_eval(tmp_path):
    with pytest.raises(SystemExit):
        R.main(["--model", "qwen3-8b", "--out", str(tmp_path), "--dry-run", "--evals", "no_such_eval"])


@pytest.mark.parametrize("mode", ["native", "letters"])
def test_tiny_end_to_end_all_stages(tmp_path, mode):
    out = str(tmp_path / mode)
    R.main(["--model", "tiny", "--out", out, "--answer-mode", mode, "--stages", "behavior,dirs,steer,replicate,probe,noise",
            "--evals", "wmdp_bio,gpqa_control", "--splits", "disc,rep", "--max-per-split", "8", "--framings", "N,E,C",
            "--recipes", "sandbag_bio,refusal", "--dir-names", "B,REF", "--interventions", "cap,proj0,add", "--strengths=-0.2",
            "--layer-sets", "auto1,autoN", "--steer-evals", "wmdp_bio,gpqa_control", "--n-pairs", "12", "--max-new", "12",
            "--noise-kinds", "act_fixed,weight", "--noise-sigmas-act", "0.1", "--noise-sigmas-weight", "0.02", "--noise-seeds", "1",
            "--skip-refusal-validation"])
    cells = [json.loads(l) for l in open(os.path.join(out, "cells.jsonl"))]
    assert cells and all(c["n"] == 8 or c["n"] <= 8 for c in cells)
    for c in cells:
        for k in ("acc", "parse_rate", "refusal_rate", "nofinal_rate", "acc_answered", "key", "layers", "strength"):
            assert k in c
    assert {c["framing"] for c in cells} >= {"N", "E", "C", "N2"}
    assert any(c["intervention"] == "noise_act" for c in cells) and any(c["intervention"] == "noise_w" for c in cells)
    z = np.load(os.path.join(out, "arrays.npz"))
    assert any(k.startswith("refused::") for k in z.files)
    st = json.load(open(os.path.join(out, "status.json")))
    assert st["cells"] == len(cells)
    if mode == "letters":
        assert all(c["parse_rate"] == 1.0 and c["refusal_rate"] == 0.0 for c in cells)
    # resume: nothing new is executed
    n0 = len(cells)
    R.main(["--model", "tiny", "--out", out, "--answer-mode", mode, "--stages", "behavior", "--evals", "wmdp_bio,gpqa_control",
            "--splits", "disc,rep", "--max-per-split", "8", "--framings", "N,E,C", "--max-new", "12"])
    assert len([1 for _ in open(os.path.join(out, "cells.jsonl"))]) == n0


def test_missing_direction_warns_and_strict_raises(tmp_path):
    base = ["--model", "tiny", "--out", str(tmp_path / "a"), "--stages", "dirs", "--recipes", "refusal", "--dir-names", "B,REF",
            "--n-pairs", "8", "--max-new", "8", "--skip-refusal-validation", "--evals", "wmdp_bio", "--max-per-split", "4"]
    R.main(base)
    st = json.load(open(str(tmp_path / "a" / "status.json")))
    assert any("not built" in w for w in st["warnings"])
    with pytest.raises(RuntimeError):
        R.main(["--model", "tiny", "--out", str(tmp_path / "b"), "--stages", "dirs", "--recipes", "refusal", "--dir-names", "B,REF",
                "--n-pairs", "8", "--max-new", "8", "--skip-refusal-validation", "--strict-dirs", "--evals", "wmdp_bio",
                "--max-per-split", "4"])
