"""sbx.run: the checkpoint record (what was really loaded) written once per run directory, and the refusal to mix checkpoints."""
from __future__ import annotations

import json
import types

import pytest

from sbx import run as R


class _Model:
    def __init__(self, quant=None, rev="abc123", n=3):
        self.config = types.SimpleNamespace(_commit_hash=rev, torch_dtype="bfloat16",
                                            quantization_config=({"quant_method": "bnb_4bit"} if quant else None), num_hidden_layers=n)
        self.n = n

    def parameters(self):
        return [types.SimpleNamespace(numel=lambda: 1_000_000_000)]


class _Ad:
    repo = "org/default-repo"
    key = "somekey"

    def num_layers(self, model):
        return model.n


def test_record_holds_the_repo_actually_loaded_and_both_quantisations():
    r = R.checkpoint_record("cedar", _Ad(), None, _Model(), repo=None, quant=None)
    assert r["model"] == "cedar" and r["repo"] == "org/default-repo" and r["repo_is_override"] is False
    assert r["quant"] is None and r["checkpoint_quant"] is None and r["effective_precision"] == "bfloat16"
    assert r["revision"] == "abc123" and r["num_layers"] == 3 and r["params_b"] == 1.0
    r2 = R.checkpoint_record("llama-3.3-70b-instruct", _Ad(), None, _Model(), repo="other/mirror", quant="4bit")
    assert r2["repo"] == "other/mirror" and r2["repo_is_override"] and r2["effective_precision"] == "4bit"


def test_a_prequantised_checkpoint_cannot_pass_for_bf16():
    """quant=None but the checkpoint declares 4-bit: effective precision must say so."""
    r = R.checkpoint_record("k", _Ad(), None, _Model(quant=True), quant=None)
    assert r["quant"] is None and r["checkpoint_quant"] == "bnb_4bit" and r["effective_precision"] == "bnb_4bit"


def test_write_appends_and_refuses_a_second_checkpoint(tmp_path):
    a = dict(model="cedar", repo="sandbagging-games/cedar", quant=None, checkpoint_quant=None, revision="r1")
    R.write_checkpoint(str(tmp_path), dict(a, params_b=70.5))
    R.write_checkpoint(str(tmp_path), dict(a, params_b=70.5))                   # a resumed step of the same run: fine
    lines = [json.loads(l) for l in open(tmp_path / "checkpoint.jsonl", encoding="utf-8")]
    assert len(lines) == 2 and all(l["model"] == "cedar" for l in lines)
    other = dict(model="llama-3.3-70b-instruct", repo="meta-llama/Llama-3.3-70B-Instruct", quant=None, checkpoint_quant=None, revision="r2")
    with pytest.raises(SystemExit, match="refusing to add cells"):
        R.write_checkpoint(str(tmp_path), dict(other, params_b=70.5))
    with pytest.raises(SystemExit, match="refusing to add cells"):
        R.write_checkpoint(str(tmp_path), dict(a, quant="4bit"))                # same repo, other precision: also a different checkpoint
    assert len(open(tmp_path / "checkpoint.jsonl", encoding="utf-8").read().splitlines()) == 2


def test_record_has_no_secret_and_no_home_path(tmp_path):
    r = R.checkpoint_record("cedar", _Ad(), None, _Model())
    R.write_checkpoint(str(tmp_path), r)
    t = (tmp_path / "checkpoint.jsonl").read_text()
    assert "/Users/" not in t and "rpa_" not in t and "hf_" not in t          # a repo id, a revision hash and sizes: nothing private
