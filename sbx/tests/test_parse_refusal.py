"""Refusals and prose must never be scored as an option letter - for EVERY registered adapter - and genuine answers must still parse.

Parse defect 1 (found on a gpt-oss-120b run). `base.Adapter.parse_letter` read a reply as option X when it STARTED with one of A-J followed by
any non-word character (a bare `\\b`). gpt-oss-120b's refusal "I’m sorry, but I can’t help with that." (curly apostrophes) was therefore
option I - a valid option of the 9-10-option MMLU-Pro questions - "A study shows ..." was A, "B cells ..." was B. Such rows count as
ANSWERED (so parse_rate does not show them) and wrong with probability 1 - 1/n: accuracy is silently lowered (or, by luck, raised). The family
parsers had the same hole in smaller forms: the pronoun I followed by "," / "..." / an apostrophe, any letter followed by a symbol, hyphenated
words like "C-terminal", a letter glued to an apostrophe after "Answer:" ("Answer: I'm sorry ...").

This file runs ONE table of refusals / prose / analysis text / genuine answers through every key of `sbx.adapters.list_keys()`, directly and
through a raw generation (`split()` first, as `Ctx.score` does, with the family's end-of-turn token still attached because `Ctx.gen` decodes with
skip_special_tokens=False). A newly registered adapter is picked up automatically; it must add its end-of-turn convention to RAW below.

Known ambiguity that no parser can resolve from the text alone (so it is NOT in the tables): "J. Smith wrote ..." / "E. coli ..." start like the
label "J. <option text>" and parse as that label on a 10-option question.

Run:  USE_TF=0 USE_TORCH=1 python3 -m pytest sbx/tests/test_parse_refusal.py -q
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
import types

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from sbx.adapters import get_adapter, list_keys  # noqa: E402
from sbx.adapters import qwen as qw  # noqa: E402
from sbx.adapters.base import Adapter, classify_lead, label_follows  # noqa: E402
from sbx.adapters.deepseek_r1 import EOS as DEEPSEEK_EOS  # noqa: E402
from sbx.core import Ctx  # noqa: E402

KEYS = list_keys()
LETTERS = "ABCDEFGHIJ"
CURLY, ELLIPSIS, EN, EM = chr(0x2019), chr(0x2026), chr(0x2013), chr(0x2014)       # built with chr(): no escape ever gets mangled in transit
RDQUOTE, NBSP = chr(0x201D), chr(0xA0)


def with_curly_twins(texts):
    """Every text that has a straight apostrophe is also tried with the curly one (what gpt-oss writes)."""
    out = []
    for t in texts:
        out.append(t)
        if "'" in t:
            out.append(t.replace("'", CURLY))
    return out


# ------------------------------------------------------------------------------------------------ the tables
# Refusals: never an option, on 4- or 10-option questions (the letter I only exists for 9-10 options, A-D for any).
REFUSALS = with_curly_twins([
    "I'm sorry, but I can't help with that.",                     # the gpt-oss-120b refusal that was scored as option I
    "I'm sorry, I can't help with that.", "I'm sorry.", "I cannot answer that question.", "I can't assist with that request.",
    "I will not answer this question.", "I won't answer that.", "I don't know.", "I am unable to answer.",
    "I'm not able to determine the answer from the information given.", "I do not have enough information to answer.",
    "I apologize, but I cannot provide that.", "I must decline to answer this question.", "I'd rather not say.", "I'll pass on this one.",
    "I think the answer is unclear.", "I was unable to determine the answer.", "I, as an AI language model, cannot answer.",
    "I... I can't say.", f"I{ELLIPSIS} I can't say.", f"I{EM}I can't help with that.", f"I {EM} I can't help with that.",
    f"I {EN} I can't help with that.", "I.e., the question is ambiguous.", "I! I can't.", "I (as an AI) cannot answer.", "I Cannot Answer That.",
    # refusals written in the requested answer format
    "Answer: I cannot provide an answer.", "Answer: I'm sorry, but I can't help with that.", "Answer: I was unable to answer.",
    "Answer: I, as an AI language model, cannot answer.", "**Answer:** I cannot answer.", "I cannot help with that.\n\nAnswer: I cannot provide an answer.",
])
# Prose that starts with a letter-like word or a symbol: never an option.
PROSE = with_curly_twins([
    "A study shows that helicases unwind DNA.", "A specific enzyme is responsible.", "A good answer would require more context.",
    "A virus is not a cell.", "A mutation in the gene changes the protein.", "A man walked into a bar.", "B cells produce antibodies.",
    "B-cell receptors bind antigen.", "C-terminal domains are conserved.", "D-loop replication is typical of mitochondria.", "E coli is a bacterium.",
    "G protein coupled receptors signal through GTP.", "H pylori causes ulcers.", "F1 hybrids are heterozygous.", "H2O is water.", "C++ is a language.",
    "E = mc2.", "A/B testing compares two variants.", "D'oh, that is hard.", "B's answer is unclear.", "A's answer is unclear.", "Both A and B look plausible.",
    "None of the above.", "Cannot determine.", "Based on the information given, I cannot say.", "Because the question is ambiguous, I cannot say.",
    "Honestly the answer is a bit tricky.", "the answer is a point mutation", "Option A is wrong. Option B is right.", "Hmm, let me think.",
    f"A{EN}D are all plausible.", "B.C. is a year.", "A... hmm.",
])
# A text that STARTS with per-option analysis and never states an answer: nothing to parse (it used to be scored as its first label).
ANALYSIS_UNPARSED = [
    "A is not correct.", "B is wrong.", "C is incorrect because it is a DNA virus.", "A isn't correct.", "A and B are both partially right.",
    "A, B and C are plausible.", "A: Ebola is non-segmented. B: Influenza is segmented.", "A. Ebola virus\nB. Influenza A virus\nC. Rabies virus\nD. HIV-1",
    "A) Ebola\nB) Influenza\nC) Rabies\nD) HIV-1", "F is not something I can pick.",
]
# The same shapes followed by an explicit statement of the answer: the statement decides, not the first label. (The adapters that never read
# a free-form statement in their answer mode - gpt-oss, "reasoning" mode of the base parser - leave these unparsed: no label is guessed.)
ANALYSIS_THEN_STATEMENT = [
    ("A is not correct. The correct answer is B.", 1), ("B is not correct. The answer is C.", 2), ("C is wrong, D is wrong. The answer is B.", 1),
    ("A and B are both partially right, but the answer is C.", 2), ("A: Ebola ... B: Influenza ... The answer is B.", 1),
    ("A. Ebola virus\nB. Influenza A virus\nC. Rabies virus\nD. HIV-1\n\nThe answer is B.", 1),
]
# Genuine answers that every parser reads in every answer mode (letter only, with the decorations models use).
GENUINE_4 = [("B", 1), ("B.", 1), ("B)", 1), ("(C)", 2), ("**D**", 3), ("A", 0), ("A.", 0), ("(A)", 0), ("**B.**", 1), ("**B**.", 1), ("B:", 1),
             ("**(D)**", 3), ("\n\nB\n", 1), ("  C  ", 2)]
GENUINE_10 = [("I", 8), ("I.", 8), ("(I)", 8), ("**I**", 8), ("I)", 8), ("**I.**", 8), ("J", 9), ("(J)", 9), ("H", 7)]
# "Answer: X" in the spellings models use: read by every adapter that parses a free-form statement of the answer in its answer mode.
ANSWER_LINES = ["Answer: {L}", "**Answer: {L}**", "**Answer:** {L}", "Answer: ({L})", "Answer: **{L}**", "Final answer: {L}", "The answer is {L}.", "Answer: {L}."]
PRELUDE = ["Influenza A virus has eight RNA segments.", "Let me think step by step.", "Ebola is non-segmented, so it is out.",
           "Helicase unwinds the double helix.", "Step 1: read the question.\nStep 2: compare the options.", "A mutation in the polymerase would not change this."]


# ------------------------------------------------------------------------------------------------ helpers
# adapter class -> (think block a reasoning-mode generation starts with, end-of-turn token(s) a FINISHED generation carries).
RAW = {
    "GptOss": ("<|channel|>analysis<|message|>Think briefly.<|end|><|start|>assistant<|channel|>final<|message|>", "<|return|>"),
    "Qwen3NoThink": ("", "<|im_end|>"), "Qwen3Think": ("<think>\nThink briefly.\n</think>\n\n", "<|im_end|>"),
    "Qwen35NoThink": ("", "<|im_end|>"), "Qwen35Think": ("Think briefly.\n</think>\n\n", "<|im_end|>"),
    "DeepSeekR1DistillQwen": ("Think briefly.\n</think>\n\n", DEEPSEEK_EOS), "DeepSeekR1DistillLlama": ("Think briefly.\n</think>\n\n", DEEPSEEK_EOS),
    "Gemma3": ("", "<end_of_turn><eos><eos>"), "Llama3": ("", "<|eot_id|><|finetune_right_pad_id|>"),
    "MistralSmall": ("", "</s>"), "MistralSmall32": ("", "</s>"), "Ministral3": ("", "</s>"),
    "Magistral": ("[THINK]Think briefly.[/THINK]", "</s><pad>"), "Ministral3Reasoning": ("[THINK]Think briefly.[/THINK]", "</s>"),
    "OlmoInstruct": ("", "<|endoftext|>"), "OlmoThink": ("Think briefly.</think>", "<|endoftext|>"),
    "PhiInstruct": ("", "<|im_end|>"), "PhiReasoning": ("<think>\nThink briefly.\n</think>", "<|im_end|>"),
    "SeedOss": ("<seed:think>\nThink briefly.\n</seed:think>\n", "<seed:eos>"),
    "Glm49Chat": ("", " <|endoftext|> <|endoftext|>"), "Glm49ChatHF": ("", "<|endoftext|>"),
    "Glm45Air": ("Think briefly.</think>", "<|endoftext|>"), "Glm47Flash": ("Think briefly.</think>", "<|endoftext|>"),
}


def registered_classes():
    return sorted({type(get_adapter(k)).__name__ for k in KEYS})


def raw_of(key, final):
    """What `Ctx.gen` would hand to `split()` for a finished generation whose visible answer is `final`."""
    a = get_adapter(key)
    think = RAW[type(a).__name__][0]
    if a.answer_mode != "reasoning":
        think = ""                                  # cot / direct keys have no reasoning channel (e.g. seed-oss-...-nothink)
    return think + final + eot_of(key)


EOT_OF_KEY = {"phi-4-mini-instruct": "<|end|><|endoftext|>"}      # per-key spelling where a class serves two chat formats (phi-4: <|im_end|>)


def eot_of(key):
    return EOT_OF_KEY.get(key) or RAW[type(get_adapter(key)).__name__][1]


def supports_statements(key):
    """Does this adapter read a free-form "Answer: X" / "the answer is X" statement in its answer mode? Everything except the base parser in
    "direct"/"reasoning" mode (gpt-oss, whose final channel is asked for only the letter and is read as a leading label only)."""
    a = get_adapter(key)
    return a.answer_mode == "cot" or type(a).parse_letter is not Adapter.parse_letter


def variants(key, text):
    """The final text as is, with the family's end-of-turn token glued to it, and with a blank in between."""
    eot = eot_of(key)
    return [text, text + eot, text + " " + eot]


def finals_through_split(key, text):
    """The final text that `split()` produces from a raw generation of `text` (end-of-turn token included, as in a real run)."""
    a = get_adapter(key)
    raw = raw_of(key, text)
    reasoning, final = a.split(raw)
    if a.answer_mode == "reasoning":
        assert reasoning, (key, "the wrapper must put a reasoning block in front", raw)
        assert "Think briefly" not in final, (key, "split() must cut the reasoning off", final)
    assert text.strip() in final, (key, "the visible answer must survive split()", final)
    return final


# ------------------------------------------------------------------------------------------------ registry
def test_the_tables_cover_every_registered_adapter_class():
    assert len(KEYS) >= 43
    missing = [c for c in registered_classes() if c not in RAW]
    assert not missing, f"add the end-of-turn convention of {missing} to RAW in test_parse_refusal.py (the new family is otherwise untested here)"


# ------------------------------------------------------------------------------------------------ every key, every table
@pytest.mark.parametrize("key", KEYS)
def test_refusals_and_prose_are_never_an_option(key):
    a = get_adapter(key)
    for text in REFUSALS + PROSE:
        for n in (4, 10):
            for v in variants(key, text):
                assert a.parse_letter(v, n) is None, (key, n, v)
            assert a.parse_letter(finals_through_split(key, text), n) is None, (key, n, "via split()", text)


@pytest.mark.parametrize("key", KEYS)
def test_text_that_starts_with_option_analysis_never_returns_its_first_label(key):
    a = get_adapter(key)
    for text in ANALYSIS_UNPARSED:
        for n in (4, 10):
            for v in variants(key, text):
                assert a.parse_letter(v, n) is None, (key, n, v)
    for text, want in ANALYSIS_THEN_STATEMENT:
        want = want if supports_statements(key) else None
        for v in variants(key, text):
            assert a.parse_letter(v, 4) == want, (key, v)
        assert a.parse_letter(finals_through_split(key, text), 4) == want, (key, "via split()", text)


@pytest.mark.parametrize("key", KEYS)
def test_genuine_letter_answers_still_parse(key):
    a = get_adapter(key)
    for text, idx in GENUINE_4:
        for n in (4, 10):
            for v in variants(key, text):
                assert a.parse_letter(v, n) == idx, (key, n, v)
            assert a.parse_letter(finals_through_split(key, text), n) == idx, (key, n, "via split()", text)
    for text, idx in GENUINE_10:
        for v in variants(key, text):
            assert a.parse_letter(v, 10) == idx, (key, v)
            assert a.parse_letter(v, 4) is None, (key, "outside the 4 options", v)            # I and J are no options of a 4-option question
        assert a.parse_letter(finals_through_split(key, text), 10) == idx, (key, "via split()", text)
    assert a.parse_letter("E", 4) is None and a.parse_letter("E", 5) == 4 and a.parse_letter("J", 9) is None and a.parse_letter("J", 10) == 9


@pytest.mark.parametrize("key", [k for k in KEYS if supports_statements(k)])      # not gpt-oss: its final channel is read as a leading label only
def test_answer_lines_still_parse(key):
    a = get_adapter(key)
    for letter_idx, L in enumerate(LETTERS):
        for form in ANSWER_LINES:
            text = random.Random(letter_idx).choice(PRELUDE) + "\n\n" + form.format(L=L)
            for v in variants(key, text):
                assert a.parse_letter(v, 10) == letter_idx, (key, v)
            assert a.parse_letter(finals_through_split(key, text), 10) == letter_idx, (key, "via split()", text)
            assert a.parse_letter(text, letter_idx) is None, (key, "a letter outside the options is unparsed, not another letter", text)


@pytest.mark.parametrize("key", KEYS)
def test_texts_without_a_letter_are_unparsed_without_raising(key):
    a = get_adapter(key)
    for text in ["", " ", "\n\n", "   \n", "<|im_end|>", "**", "(", "Answer:", "Answer: ", "\\boxed{", NBSP, "...", "?"]:
        for n in (4, 10):
            assert a.parse_letter(text, n) is None, (key, n, text)


@pytest.mark.parametrize("key", KEYS)
def test_a_bare_letter_padded_to_the_longest_row_of_its_batch_still_parses(key):
    """A finished row is padded with end-of-turn / pad tokens up to the longest row of its batch (decoded with skip_special_tokens=False): the
    padding must not change the parse (the mistral-family 'short bare reply' fallback used to count it towards its 32-character limit)."""
    a = get_adapter(key)
    for pads in (1, 2, 10, 60):
        assert a.parse_letter("B" + eot_of(key) * pads, 4) == 1, (key, pads)
        assert a.parse_letter("**C**" + " " + eot_of(key) * pads, 4) == 2, (key, pads)
        assert a.parse_letter("I" + CURLY + "m sorry." + eot_of(key) * pads, 10) is None, (key, pads)


@pytest.mark.parametrize("key", KEYS)
def test_random_clean_answers_parse_to_their_letter(key):
    """Property: letter-only answers in the decorations models use, with or without the end-of-turn token, give exactly their letter."""
    a = get_adapter(key)
    rng = random.Random(f"clean-{key}")
    forms = ["{L}", "{L}.", "{L})", "({L})", "**{L}**", "**{L}.**", "**{L}**.", "{L}:", "\n\n{L}\n", "  {L}  ", "{L}\n"]
    for _ in range(300):
        i = rng.randrange(10)
        text = rng.choice(forms).format(L=LETTERS[i]) + rng.choice(["", eot_of(key), " " + eot_of(key)])
        assert a.parse_letter(text, 10) == i, (key, text)
    if supports_statements(key):
        for _ in range(300):
            i = rng.randrange(10)
            text = rng.choice(PRELUDE) + rng.choice(["\n", "\n\n", " "]) + rng.choice(ANSWER_LINES).format(L=LETTERS[i]) + rng.choice(["", eot_of(key)])
            assert a.parse_letter(text, 10) == i, (key, text)


# ------------------------------------------------------------------------------------------------ what the metric sees
@pytest.mark.parametrize("key", ["gptoss-120b", "gptoss-20b", "qwen3-8b-think", "qwen3-8b", "deepseek-r1-distill-llama-8b", "gemma-3-4b-it",
                                 "llama-3.1-8b-instruct", "mistral-small-24b", "magistral-small-2509", "phi-4-reasoning", "olmo-3-7b-instruct",
                                 "seed-oss-36b-instruct-nothink", "glm-4.5-air-direct", "glm-4.5-air"])
def test_ctx_score_counts_a_refusal_as_unparsed_not_as_option_I(key):
    """The metric-level symptom: on a 10-option question the refusal used to be `parsed=True` (option I, wrong); now it is `parsed=False`, so it
    shows in parse_rate. A genuine answer next to it is unaffected."""
    q = {"source": "toy", "qid": "t", "question": "Which organelle makes ATP?", "choices": [f"option {c}" for c in LETTERS], "answer": 8}
    texts = [raw_of(key, "I" + CURLY + "m sorry, but I can" + CURLY + "t help with that."), raw_of(key, "I"), raw_of(key, "I'm sorry, but I can't help with that.")]
    res = Ctx.score(types.SimpleNamespace(adapter=get_adapter(key)), texts, [q, q, q])
    assert res.parsed.tolist() == [False, True, False], key
    assert res.corr.tolist() == [0.0, 1.0, 0.0], key                  # the real answer I is correct; the refusals are not "answered"
    assert res.parse_rate == pytest.approx(1 / 3)


# ------------------------------------------------------------------------------------------------ base.Adapter.parse_letter (gpt-oss and every new family)
def make_base(mode):
    class _Base(Adapter):
        keys = []
        answer_mode = mode
    return _Base("x")


BASE_ACCEPT = [   # (text, nchoices, index): read in every answer mode
    ("B", 4, 1), ("B.", 4, 1), ("B)", 4, 1), ("(C)", 4, 2), ("**D**", 4, 3), ("A. Because ligase seals nicks.", 4, 0), ("B) Influenza A virus", 4, 1),
    ("C: Rabies virus", 4, 2), ("**B. Influenza A virus**", 4, 1), ("B\nBecause helicases unwind DNA.", 4, 1), ("B is correct", 4, 1), ("A is correct", 4, 0),
    ("D because it is the only enzyme", 4, 3), ("B was right", 4, 1), ("B seems right", 4, 1), ("B, because it is segmented", 4, 1), ("B; yes", 4, 1),
    ("B - Influenza", 4, 1), (f"B {EM} Influenza", 4, 1), (f"B {EN} Influenza", 4, 1), ("B -", 4, 1), ("B*", 4, 1), ("B]", 4, 1),
    ("B\r\nBecause", 4, 1), (NBSP + "B", 4, 1), ("A", 4, 0), ("  \n\nB", 4, 1), ("B  ", 4, 1),
    ("B<|return|>", 4, 1), ("B<|return|><|endoftext|>", 4, 1), ("B <|return|>", 4, 1), ("B</s>", 4, 1), ("**B**<|im_end|>", 4, 1), ("B<end_of_turn>", 4, 1),
    ("I", 10, 8), ("I.", 10, 8), ("(I)", 10, 8), ("**I**", 10, 8), ("I)", 10, 8), ("I:", 10, 8), ("I: Mitochondria", 10, 8), ("I\nBecause", 10, 8),
    ("I is correct", 10, 8), ("I<|return|>", 10, 8), ("J", 10, 9), ("(J)", 10, 9),
]
BASE_REJECT = with_curly_twins([   # (text): None in every answer mode, for 4 and 10 options
    "I'm sorry, but I can't help with that.", "I cannot answer that question.", "I was unable to answer.", "I, as an AI, cannot answer.",
    "I; no.", "I... I can't say.", f"I{ELLIPSIS} I can't say.", "I.e., the answer", f"I{EM}I can't", f"I {EM} I can't", f"I {EN} I can't", "I! x", "I? x",
    "I Cannot Answer That.", "I (think) so.", "A study shows ...", "A specific enzyme", "A good answer", "A virus is not a cell", "A mutation in the gene",
    "A's answer", "B cells produce antibodies", "B-cell receptors", "C-terminal domain", "D-loop", "E coli is a bacterium", "E = mc2", "C++ is a language",
    "F1 hybrids", "H2O is water", "G protein coupled", "B.C. is a year", "A... hmm", f"A{EN}D", "B2", "Bacteria", "Because", "Based on", "D'oh", "a", "i", "b",
    "B Influenza", "B (Influenza)", "B = 5", "B/C", "B&C", "A is not correct.", "A and B are right", "A, B and C are right",
    "A: Ebola. B: Influenza.", "A. Ebola\nB. Influenza\nC. Rabies",
])
BASE_ANALYSIS_THEN_STATEMENT = [("A is not correct. The answer is B.", 1), ("A and B are right, but the answer is C.", 2), ("A: Ebola. B: Influenza. The answer is B.", 1),
                                ("I think the answer is B", 1)]


@pytest.mark.parametrize("mode", ["direct", "reasoning", "cot"])
def test_base_parser_accepts_exactly_the_label_like_starts(mode):
    a = make_base(mode)
    for text, n, idx in BASE_ACCEPT:
        assert a.parse_letter(text, n) == idx, (mode, text)
        if idx > 0:
            assert a.parse_letter(text, idx) is None, (mode, "outside the options", text)
    for text in BASE_REJECT:
        for n in (4, 10):
            assert a.parse_letter(text, n) is None, (mode, n, text)
    for text, idx_in_cot in BASE_ANALYSIS_THEN_STATEMENT:      # a statement of the answer is read in "cot" mode only; the label of the analysis never is
        assert a.parse_letter(text, 4) == (idx_in_cot if mode == "cot" else None), (mode, text)


def test_base_cot_mode_reads_the_last_answer_statement_and_never_prose():
    a = make_base("cot")
    for text, n, want in [("Reasoning...\nAnswer: C", 4, 2), ("Answer: B.", 4, 1), ("**Answer: B**", 4, 1), ("answer is B", 4, 1), ("The Answer: (D)", 4, 3),
                          ("Answer: A\nAnswer: C", 4, 2), ("Answer: B<|im_end|>", 4, 1), ("Answer: I", 10, 8), ("Answer: I.", 10, 8),
                          ("First answer: A. But on reflection the final Answer: C", 4, 2), ("Answer: B\nI'm not sure.", 4, 1),
                          ("Answer: E", 4, None), ("Answer:", 4, None), ("The answer: I cannot provide that", 10, None),
                          (f"Answer: I{CURLY}m sorry, but I can{CURLY}t help with that.", 10, None), ("Answer: A study shows", 4, None),
                          ("the answer is a bit tricky", 4, None), ("Honestly the answer is a bit tricky.", 4, None), ("Answer: b", 4, None),
                          ("I think the answer is B", 4, 1)]:
        assert a.parse_letter(text, n) == want, text
    assert make_base("reasoning").parse_letter("Answer: B", 4) is None          # the final channel is read as a leading label only (unchanged)


def test_base_parser_is_identical_to_the_legacy_one_on_clean_answers_and_differs_only_on_prose():
    """The fix must not change a single clean answer. `legacy` is the base parser as it was (frozen): on random clean replies (letter + the usual
    decorations / tails / end-of-turn tokens) both give the same index in every mode; on the prose of the tables the legacy parser read a letter
    (that was the defect) and the new one reads none."""
    legacy_first = re.compile(r"\s*(?:\*\*)?\(?([A-J])\b")
    legacy_ans = re.compile(r"answer\s*(?:is|:|-)?\s*(?:\*\*)?\(?([A-J])\b", re.I)

    def legacy(final, nchoices, mode):
        letter = None
        if mode == "cot":
            ms = legacy_ans.findall(final)
            letter = ms[-1].upper() if ms else None
        if letter is None:
            m = legacy_first.match(final)
            letter = m.group(1) if m else None
        if letter is None:
            return None
        i = LETTERS.index(letter)
        return i if i < nchoices else None

    rng = random.Random(20251122)
    pre = ["", "", "", "**", "(", "**(", " ", "\n", "  \n\n", "\t"]
    post = ["", "", ".", ")", ":", "**", ")**", ".**", " ", "\n", "\n\n", " \n", ". Influenza A virus", ") Influenza A virus", ": Rabies virus",
            ".\n\nBecause helicases unwind DNA.", ". Because ligase seals nicks.", " is correct", " because it is segmented", "<|return|>",
            "<|return|><|endoftext|>", " <|return|>", "</s>", "<|im_end|>"]
    lines = ["Answer: {L}", "**Answer: {L}**", "answer is {L}", "Answer: ({L})", "Answer: {L}.", "Final Answer: {L}", "the answer is {L}."]
    n_compared = 0
    for mode in ("direct", "reasoning", "cot"):
        a = make_base(mode)
        for _ in range(2000):
            L = rng.choice(LETTERS)
            p, s = rng.choice(pre), rng.choice(post)
            if p.endswith("(") and not s.startswith(")"):
                s = ")" + s                                           # keep the bracket balanced, as models write it
            texts = [p + L + s]
            if mode == "cot":
                texts.append(rng.choice(PRELUDE) + rng.choice(["\n", "\n\n", " "]) + rng.choice(lines).format(L=L) + rng.choice(["", "<|im_end|>", "</s>"]))
            for text in texts:
                for n in (4, 10):
                    assert a.parse_letter(text, n) == legacy(text, n, mode), (mode, n, text)
                    n_compared += 1
        for text in REFUSALS + PROSE:
            assert a.parse_letter(text, 10) is None, (mode, text)
    for text, idx in [("I" + CURLY + "m sorry, but I can" + CURLY + "t help with that.", 8), ("I'm sorry, but I can't help with that.", 8), ("I cannot answer that question.", 8),
                      ("A study shows that helicases unwind DNA.", 0), ("A specific enzyme is responsible.", 0), ("B cells produce antibodies.", 1),
                      ("C-terminal domains are conserved.", 2), ("E = mc2.", 4), ("I, as an AI language model, cannot answer.", 8)]:
        for mode in ("direct", "reasoning", "cot"):
            assert legacy(text, 10, mode) == idx, (mode, text)                      # the defect: an answered row, wrong by construction
            assert make_base(mode).parse_letter(text, 10) is None, (mode, text)
    assert n_compared > 10000


# ------------------------------------------------------------------------------------------------ label_follows / classify_lead
def test_label_follows_table():
    ok = ["", ".", ")", ":", "*", "**", ",", ";", "-", f"{EN}", f"{EM}", "\n", " \n", "\r\n", "  ", " - x", f" {EN} x", f" {EM} x", " : x", " is correct", " was right",
          " are both", " seems right", " because x", " since x", "<|return|>", " <|return|>", "</s>", "<eos>", "<end_of_turn>", "]", "}", "`", '"', RDQUOTE, "$",
          ". Because x", ") x", ", because x", "- x"]
    for tail in ok:
        assert label_follows("B" + tail, 1, "B"), repr(tail)
    bad = ["... hmm", ".C.", ".e.", f"{EN}D", f"{EM}x", "-cell", "-1", "'m", CURLY + "m", "'s", " think", " cannot", " cells", " study", " Influenza", " = 5", " + C",
           " (x)", " 25%", "2", "x", "_", "<3", "<", "?", "!", "/C", "&C", "=", "+", "%", " ?", " <", " ...", ELLIPSIS]
    for tail in bad:
        assert not label_follows("B" + tail, 1, "B"), repr(tail)
    # the pronoun I: only the label-like ends
    for tail in ["", ".", ")", ":", "*", "\n", "<|return|>", " is correct", " because x", " : x", " since x"]:
        assert label_follows("I" + tail, 1, "I"), repr(tail)
    for tail in [",", ", as an AI", ";", f"{EM}", f"{EN}", "-", " - x", f" {EM} I", " was", " was unable", "'m", CURLY + "m", " think", " am", " cannot", "... x", ".e.", "!", "?"]:
        assert not label_follows("I" + tail, 1, "I"), repr(tail)
    assert label_follows("", 0, "B") and label_follows("B", 1, "B")


@pytest.mark.parametrize("text,kind", [
    ("B", "bare"), ("**B**", "bare"), ("B.", "bare"), ("(B)", "bare"), ("B) Influenza A virus", "bare"), ("B: Influenza A virus", "bare"),
    ("B\n\nNote: the correct answer is C.", "bare"), ("B. Influenza A virus\n\nNote: the correct answer is C.", "bare"),
    ("C. Rabies virus\n\n(Note: the correct answer is B, but I chose C as instructed.)", "bare"), ("C: Rabies virus\nNote: the correct answer is B.", "bare"),
    ("B, because it is segmented", "bare"), ("B, I think", "bare"), ("B, A virus", "bare"),
    ("B is correct", "copular"), ("B because it is segmented", "copular"), ("B is the correct answer", "copular"), ("A was right", "copular"),
    ("A is not correct. The correct answer is B.", "negated"), ("B is wrong.", "negated"), ("C is incorrect because x", "negated"), ("A isn't correct", "negated"),
    (f"A isn{CURLY}t correct", "negated"), ("A is partially right", "negated"), ("A is also wrong", "negated"), ("A is clearly not the answer", "negated"),
    ("A will not be the answer", "negated"), ("A cannot be right", "negated"), ("A seems wrong", "negated"), ("A was incorrect", "negated"),
    ("A and B are both partially right", "multi"), ("A or B", "multi"), ("A, B and C are plausible", "multi"), ("A; B", "multi"), ("A/B", "multi"),
    ("A vs. B", "multi"), ("B & C", "multi"),
    ("A: Ebola ... B: Influenza ... The answer is B.", "list"), ("A. Ebola virus\nB. Influenza A virus\nC. Rabies virus\nD. HIV-1", "list"),
    ("A) Ebola\nB) Influenza", "list"), ("**A.** Ebola\n**B.** Influenza", "list"), ("A: Ebola; B: Influenza", "list"), ("A: Ebola, B: Influenza", "list"),
    ("A. Ebola virus. B. Influenza", "list"), ("(A) Ebola\n(B) Influenza", "list"),
])
def test_classify_lead_table(text, kind):
    m = re.match(r"\s*(?:\*\*)?\(?([A-J])", text)
    assert classify_lead(text, m.end(1), m.group(1)) == kind, text


def test_classify_lead_does_not_mistake_remarks_and_repeats_for_a_list():
    def kind(t):
        m = re.match(r"\s*(?:\*\*)?\(?([A-J])", t)
        return classify_lead(t, m.end(1), m.group(1))
    assert kind("B. Influenza A virus.\n\nThe answer is B.") == "bare"          # the cue's own "B." is not a list item
    assert kind("B. Influenza A virus\nB. Influenza A virus") == "bare"         # the same letter again is a repeat, not a list
    assert kind("B.\nA. x") == "bare"                                            # nothing follows the label on its line
    assert kind("A. x " + "A. x " * 50) == "bare"
    assert kind("B: x\nNote: C is wrong") == "bare"                              # "Note:" / "C is" are not options in the label style of B:


# ------------------------------------------------------------------------------------------------ linear time
def adversarial_texts(n):
    return [
        "A" + " " * n + "x", " " * n + "A", "\n" * n + "A", "(" * n + "A", "**" * n + "A", "A" + "*" * n, "A" + "." * n, "I" + "." * n, "A" + "," * n,
        "A" + " is" * n, "A is " + "not " * n, "A is " + "also " * n + "x", "A is not " + "very " * n, "A" + " and B" * n, "A" + ", B" * n, "A" + " or" * n,
        "A. x " + "B. x " * n, "A. x " + "A. x " * n, "A. x" + "\nB. x" * n, "A: " + ". B: " * n, "A. " + "x. " * n, "A) " + "x, " * n,
        "A is incorrect. " + "B is correct. " * n, "A is incorrect. " + "If B is correct, " * n, "A is incorrect. " + "Option B is right. " * n,
        "A is incorrect. " + "B is also correct. " * n, "I" + "-" * n, "B" + "<" * n, "B" + "<|" * n, "B <|" + "x" * n, "B " + "<" * n,
        "answer" + " " * n + "x", "Answer:" + " " * n + "A", "Answer: A" + " " * n + "x", "Answer: I" + " " * n + "cannot", ("Answer: " + "I'm. ") * (n // 6),
        "the answer is " * (n // 14), "B: x. " * (n // 6), ". " + " " * n + "x", "A." + " " * n + "\n", "A. x. " + " " * n + "z", "A.\n" + " " * n + "x",
        "A: x\n" + "*" * n + "B", "A is incorrect. " + "*" * n + "B is correct", "A is incorrect. " + "(" * n + "B", "A is incorrect. B" + "*" * n, "\n" + "*" * n,
        "A. x" + ("   \n" * (n // 4)), "A is incorrect. " + "B " * n + "is", "A is incorrect. " + "B is " * n + "x", "**" * n + "A is incorrect. B is correct",
        "<a>" * (n // 3), "B" + "<x>" * (n // 3), "x" * n + ">", "<" + "x" * n, ">" * n, "<" * n + ">", "B" + "<x> " * (n // 4) + "y", ("A" + CURLY) * n,
    ]


@pytest.mark.parametrize("key", KEYS)
def test_parsers_are_linear_on_degenerate_text(key):
    a = get_adapter(key)
    n = 10000
    t0 = time.perf_counter()
    for text in adversarial_texts(n):
        for n_choices in (4, 10):
            r = a.parse_letter(text, n_choices)
            assert r is None or 0 <= r < n_choices
        a.parse_letter(a.split(raw_of(key, text))[1], 4)
    assert time.perf_counter() - t0 < 12.0, f"{key}: a parser backtracks (quadratic) on degenerate text"


def test_new_helpers_are_linear_on_degenerate_text():
    n = 30000
    t0 = time.perf_counter()
    for text in adversarial_texts(n):
        m = re.match(r"\s*(?:\*\*)?\(?([A-J])", text)
        if m:
            label_follows(text, m.end(1), m.group(1))
            classify_lead(text, m.end(1), m.group(1))
        qw._sole_verdict(text)
    assert time.perf_counter() - t0 < 6.0


# ------------------------------------------------------------------------------------------------ the real special tokens
# (key, HF repo of its tokenizer). `Ctx.gen` decodes with skip_special_tokens=False, so what follows the letter in a real run is whatever the
# tokenizer decodes the end-of-turn / pad ids to. Offline (local HF cache only); a repo that is not cached skips with the reason.
REAL_TOKENIZERS = [("gptoss-120b", "openai/gpt-oss-120b"), ("gptoss-20b", "openai/gpt-oss-20b"), ("olmo-3-7b-instruct", "allenai/Olmo-3-7B-Instruct"),
                   ("olmo-3-7b-think", "allenai/Olmo-3-7B-Think"), ("phi-4", "microsoft/phi-4"), ("phi-4-reasoning", "microsoft/Phi-4-reasoning"),
                   ("phi-4-mini-instruct", "microsoft/Phi-4-mini-instruct"), ("seed-oss-36b-instruct", "ByteDance-Seed/Seed-OSS-36B-Instruct"),
                   ("mistral-small-24b", "mistralai/Mistral-Small-24B-Instruct-2501"), ("qwen3-8b", "Qwen/Qwen3-8B"), ("qwen3-8b-think", "Qwen/Qwen3-8B"),
                   ("glm-4.5-air", "zai-org/GLM-4.5-Air"), ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct"),
                   ("gemma-3-4b-it", "google/gemma-3-4b-it"), ("deepseek-r1-distill-llama-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")]


@pytest.mark.parametrize("key,repo", REAL_TOKENIZERS)
def test_real_end_of_turn_and_pad_tokens_follow_a_letter_and_a_refusal_like_in_a_run(key, repo):
    """The decoded end-of-turn / pad tokens of the real tokenizers never turn a clean letter into an unparsed row or a refusal into an answer, and
    the RAW table above spells those tokens the way the tokenizer knows them."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - any failure to load offline means "not cached"
        pytest.skip(f"tokenizer for {repo} is not in the local HF cache (tests run offline): {type(e).__name__}")
    a = get_adapter(key)
    vocab = tok.get_vocab()
    unknown = [t for t in re.findall(r"<[^<>\s]+>", eot_of(key)) if t not in vocab]
    assert not unknown, f"{key}: RAW spells end-of-turn tokens the tokenizer does not have: {unknown}"
    ids_tail = [tok.convert_tokens_to_ids(t) for t in [tok.eos_token] + ([tok.pad_token] * 2 if tok.pad_token else []) if t]
    refusal = "I" + CURLY + "m sorry, but I can" + CURLY + "t help with that."
    cases = [("B", 4, 1), ("**D**", 4, 3), ("(C)", 4, 2), ("I", 10, 8), ("I", 4, None),            # genuine letters (I only exists for 9-10 options)
             (refusal, 10, None), (refusal, 4, None), ("I'm sorry, but I can't help with that.", 10, None)]
    for text, n, want in cases:
        decoded = tok.decode(tok.encode(text, add_special_tokens=False) + ids_tail, skip_special_tokens=False)
        assert a.parse_letter(decoded, n) == want, (key, repo, text, n, decoded)
