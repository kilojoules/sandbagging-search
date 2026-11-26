"""Qwen3 (dense, hybrid-thinking) and Qwen3.5 (hybrid linear-attention, hybrid-thinking) adapters.

Keys (one registry key == one fixed thinking mode; the HF checkpoint is the same for the two modes of a size):

  qwen3-32b   qwen3-14b   qwen3-8b          Qwen/Qwen3-{32B,14B,8B}      thinking OFF  answer_mode "cot"       max_new 400
  qwen3-32b-think  qwen3-14b-think  qwen3-8b-think                       thinking ON   answer_mode "reasoning" max_new 1800
  qwen3.5-9b  qwen3.5-0.8b                  Qwen/Qwen3.5-{9B,0.8B}       thinking OFF  answer_mode "cot"       max_new 400
  qwen3.5-9b-think  qwen3.5-0.8b-think                                   thinking ON   answer_mode "reasoning" max_new 1800

Thinking OFF  (`enable_thinking=False`). The chat template appends an EMPTY think block to the generation prompt:
    ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n            (the model answers directly; no think tag is generated)
  The model writes a short visible chain of thought ending in "Answer: X" (`ASK["cot"]`); `split()` returns ("", visible text).
  The last prompt token (where activations are read) is the "\\n\\n" after the empty `</think>`.
Thinking ON   (`enable_thinking=True`). Qwen3:   prompt ends `<|im_start|>assistant\\n`; the model itself generates `<think>\\n...`.
                                       Qwen3.5: prompt ends `<|im_start|>assistant\\n<think>\\n` (opening tag is IN the prompt, so
                                       the generated text starts directly with reasoning). `split()` copes with both: it cuts at the
                                       first `</think>`; a MISSING close tag means the reasoning was truncated => final == "".
  `enable_thinking` is always passed explicitly, so behaviour does not depend on the template default (which differs between
  Qwen3.5-0.8B [default OFF] and Qwen3.5-9B [default ON]).

Practical notes (details in sbx/models/<key>.json):
  * Qwen's model cards say "DO NOT use greedy decoding" in thinking mode (endless repetition). sbx is greedy-only, so expect some
    runaway/repetitive reasoning that hits `max_new`; those rows are unparsed (counted wrong) - always read parse_rate.
  * `max_new=1800` truncates a sizeable minority of questions on hard sets (long reasoning tails); raise it (e.g. 4096) for
    confirmatory runs and budget the time: HF `generate` throughput is ~1-2 orders of magnitude below vLLM.
  * Answer parsing (`parse_choice`): the LAST line that starts with `Answer:` wins over every other statement (a remark after it
    such as "(Note: the correct answer is B)" never overrides it); free-form statements ("the correct answer is B", `\\boxed{B}`) must
    sit on one line and are ignored when hypothetical ("if the answer is C, ...") or a remark about an option ("Answer choice D is
    ..."). In the reasoning keys the final text is first read as a LEADING label ("B", "**B**", "B. text"): a bare label beats every
    later remark, but a text that STARTS with per-option analysis ("A is not correct. The correct answer is B.", "A and B are both ...
    but the answer is C.", "A: Ebola ... B: Influenza ... The answer is B.") has no leading label and is read from its explicit
    statements (or, as a last resort, from the single option it declares correct). A reply that starts with the pronoun I or an
    ordinary word is never a label ("I'm sorry ...", "I, as an AI ...", "B cells ..."). thinking-OFF (`cot`) turns that were cut at
    `max_new` (no end-of-turn token) get `CUT_MARK` appended to their final text
    by `split()` and are parsed from an `Answer:` line only, so truncation shows up as parse-rate loss instead of a letter quoted
    from the middle of unfinished reasoning; the visible text itself is kept (mention regexes, `sbx.dirs.refusal.final_text`).
  * Direction building (`Ctx.hidden_last` / `neutral_stats`) keeps all hidden states AND all-position logits of one prompt:
    ~0.97 GB per 1k prompt tokens on the 32B keys, ~20 GB free after the weights => at most ~20k tokens per prompt there (see
    `direction_building_memory` in the cards; `sbx.evals` keeps the pools below that with MAX_PROMPT_CHARS).
  * Framing text goes into the system turn (`sys_role="system"`); `effort` has no meaning for this family (warns if set).
  * Qwen3.5 needs transformers>=5.2.0 (python>=3.10, torch>=2.4): the local py3.9/torch2.2/transformers-4.57 stack cannot
    instantiate it (tokenizer + chat template work everywhere, and are what the tests exercise offline). `load()` fails early with
    a clear message on older versions, and adds `<|im_end|>` to the generation EOS ids (the repo ships no generation_config.json,
    so transformers would otherwise stop only on `<|endoftext|>`).
"""
from __future__ import annotations

import re
import warnings

from .base import Adapter, classify_lead

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
LETTERS = "ABCDEFGHIJ"


# ----------------------------------------------------------------------------------------------- shared output handling
def cut_at_eos(text, eos_texts):
    """Drop the first end-of-turn/pad special token (decode uses skip_special_tokens=False) and everything after it."""
    cut = len(text)
    for s in eos_texts:
        i = text.find(s)
        if 0 <= i < cut:
            cut = i
    return text[:cut]


def split_think(text, eos_texts=()):
    """Generated text -> (reasoning, final) for a model whose reasoning is closed by `</think>`.

    The opening `<think>` may or may not be in the generated text (it is when the model emits it, it is not when the chat template
    already put it in the prompt): a leading one is dropped. A missing `</think>` means the reasoning was truncated (or the model
    never closed it): the whole text is returned as reasoning and final == "". The FIRST `</think>` closes the reasoning.
    """
    t = cut_at_eos(text, eos_texts)
    i = t.find(THINK_CLOSE)
    reasoning, final = (t, "") if i < 0 else (t[:i], t[i + len(THINK_CLOSE):])
    r = reasoning.lstrip()
    if r.startswith(THINK_OPEN):
        r = r[len(THINK_OPEN):]
    return r.strip(), final.strip()


# Letter parsing -------------------------------------------------------------------------------------------------------------
# Regex design rule: every loop is over ONE alternation of mutually exclusive alternatives and no two loops are adjacent, so
# matching is linear even on adversarial text ("*****...", "answer" followed by 50k spaces, "\boxed{" + spaces, ...).
# Line rule: a free-form cue never reaches across a newline ([^\S\n] = whitespace EXCEPT the newline): "to determine the answer:\n\n
# A. Ebola virus ..." is the start of an option list, not the statement "answer: A".
# "Opener" = one decoration in front of the letter: **B**, *B*, __B__, `B`, (B), [B], {B}, $B$, \boxed{B}, \text{B}.
CUT_MARK = "<|sbx_cut|>"   # appended by ThinkTagMixin.split() to the visible text of a cot row that was cut off (no end-of-turn token)
_OPEN = r"(?:[*_`$(\[{]|\\boxed\{|\\text(?:bf)?\{|\\mathrm\{)"
_SEP = r"(?:[^\S\n]|[:=\-\u2013\u2014]|" + _OPEN + r")"
_COLON = r"(?:[^\S\n]|[*_`])*[:=\-\u2013\u2014]"            # "Answer:" / "**Answer**:" / "Answer -" (the separator itself is required)
_TAIL = _SEP + r"*(?i:(?:option|choice|letter)[^\S\n]+)?([A-J])(?![A-Za-z0-9])"
# Explicit answer statements inside a line ("the correct answer is (B)", "answer: B", "Final Answer: \boxed{B}"). The head must be
# followed by a CONNECTOR ("is" / "would be" / "should be" / ":" / "=" / dash): "Answer choice D is ..." and "Answer B: ..." are
# remarks about an option, not statements of the answer. Cue words are case-insensitive, the LETTER is upper-case only (so "the
# answer is a point mutation" never parses as A).
_CUE = re.compile(
    r"\b(?i:answer|(?:correct|right|best)[^\S\n]+(?:option|choice))"
    r"(?i:[^\S\n]+to[^\S\n]+(?:this|the)[^\S\n]+(?:question|problem))?"
    r"(?:(?i:[^\S\n]+(?:is|would[^\S\n]+be|should[^\S\n]+be)\b)|" + _COLON + r")"
    + _TAIL)
# The requested format as a LINE of its own ("Answer: B", "**Answer:** B", "- Final answer: (B)"); `match` is applied to the line
# after _LINE_LEAD markdown (bold, heading, quote, bullet) has been stripped. A line that is only "Answer:" takes the letter from
# the next non-empty line when that line starts with a label ("B", "**B**", "Option B.", "B. Influenza A virus").
_ANSLINE = re.compile(r"(?i:(?:final[^\S\n]+)?answer)" + _COLON + _TAIL)
_ANS_ONLY = re.compile(r"(?i:(?:final[^\S\n]+)?answer)" + _COLON + r"(?:[^\S\n]|[*_`:=\-\u2013\u2014])*")
_LINE_LEAD = " \t\r\f\v*_#>`-\u2022\u00b7"
_BOXED = re.compile(r"\\boxed\{(?:[\s(]|\\(?:text|textbf|mathrm|mathbf)\{)*([A-J])(?![A-Za-z0-9])")
_LEAD = re.compile(r"\s*(?:" + _OPEN + r"\s*)*([A-J])(?![A-Za-z0-9])")
_WHOLE = re.compile(r"(?:" + _OPEN + r"|\s)*(?:(?i:option|choice|letter)\s(?:" + _OPEN + r"|\s)*)?([A-J])[\s.)\]}*_`$:!]*")
# "If the answer is A, ...", "suppose the correct answer is C", "whether the answer is B": a hypothesis, not a statement.
_COND = re.compile(r"\b(?:if|suppose|supposing|assume|assuming|unless|whether)"
                   r"(?:[^\S\n]+(?:the|that|this|it|your|my|our|correct|right|best|real|true|final)){0,3}[^\S\n]+\Z", re.I)
_SPACES = re.compile(r"[ \t]*")
_ATTACHED_OK = frozenset(".)]}:,;!*_`$\\\"\u201d")
_COPULA_OR_EXPLAIN = re.compile(r"(?:is|was|would|should|seems|looks|appears|must|has|can|may|might|will|could"
                                r"|because|since|and|or|but|as|due|given|based)\b")
_SPACED_OPEN = frozenset("([{*_`$\\\"\u201c'")                # a label may be followed by blanks and one of these: "B (Influenza)", "B **Influenza**"
# "A is wrong because ... B is correct because ...": a single option declared correct, in a text made of per-option verdicts.
_VERDICT = re.compile(r"(?<![A-Za-z0-9*_`(\[])[*_`(\[]*([A-J])[*_`)\]]*[^\S\n]+(?:is|was|would[^\S\n]+be)[^\S\n]+"
                      r"(?:(?:definitely|clearly|obviously|certainly|actually|indeed|also|the)[^\S\n]+)*(?:correct|right|true)\b")
_OPTION_WORD_BEFORE = re.compile(r"(?i:option|choice|answer|letter)[^\S\n]*[:\-]?[^\S\n]*[*_`(\[]*\Z")   # "Option B is right": a remark about an option
_FIRST_PERSON = re.compile(r"(?<![A-Za-z0-9])(?:I|I['\u2019](?:m|ll|d|ve)|[Mm]y|[Mm]e|[Mm]yself)(?![A-Za-z0-9])")


def _label_follows(t, pos, letter=""):
    """Does the text of `t` from `pos` on (right after the candidate `letter`) read as an option LABEL rather than the rest of an
    English word/phrase?

    Accepts: end of text; a closing/punctuation character glued to the letter ("B.", "B)", "B:", "B**", "B,"); a newline; " - text";
    whitespace + an upper-case word, digit or opening bracket/quote ("B. Influenza", "C (because ...)", "D 25%"); whitespace + a
    copula/connector ("B is correct", "B because ..."). Rejects "A mutation ...", "I think ...", "B cells are ...", "I'll ...",
    "C-terminal ..." (lowercase word, apostrophe or hyphen glued to the letter), "E = mc2" (symbol after blanks), and what is the
    start of an English sentence when the letter is the pronoun I: "I... I can't", "I.e.", "I, as an AI ...", "I! ...", "I was
    ...", "I \u2014 I can't ...", "I Cannot ..." (after the letter I only an end, a newline or ":" count).
    Reads the text in place (no slice of the rest: linear on adversarial input).
    """
    n = len(t)
    if pos >= n:
        return True
    c = t[pos]
    if not c.isspace():
        if c == ".":
            nxt = t[pos + 1:pos + 2]
            return not (nxt == "." or nxt.isalnum())         # "I... hmm", "I.e.", "B.C." are not labels
        if c in ",;!":
            return letter != "I"                             # "I, as an AI language model, ..."
        return c in _ATTACHED_OK
    j = _SPACES.match(t, pos).end()
    if j >= n or t[j] in "\r\n":
        return True
    c = t[j]
    if letter == "I":
        return c == ":"
    if c in "-\u2013\u2014":
        return True
    if not c.islower():
        return c.isupper() or c.isdigit() or c == ":" or c in _SPACED_OPEN
    return _COPULA_OR_EXPLAIN.match(t, j) is not None


def _hypothetical(t, start):
    """Is the cue that starts at `start` introduced by if/suppose/assuming/unless/whether (+ at most 3 filler words)?"""
    return _COND.search(t, max(0, start - 64), start) is not None


def _answer_line(t):
    """Letter of the LAST line that starts with the requested format ("Answer: B", "**Final answer:** (B)", ...), else None.

    The format the prompt asks for wins over every free-form remark, wherever it sits: a note AFTER it ("Answer: C\\n\\n(Note: the
    correct answer is B, but ...)") or an earlier draft before it must not override it. "Answer: A mutation ..." (article, not a
    label) is no candidate."""
    below = None                                        # closest non-empty line after the current one
    for line in reversed(t.split("\n")):
        s = line.lstrip(_LINE_LEAD)
        if not s:
            continue
        m = _ANSLINE.match(s)
        if m:
            if m.group(1) not in "AI" or _label_follows(s, m.end(), m.group(1)):
                return m.group(1)
        elif below is not None and _ANS_ONLY.fullmatch(s):
            nxt = below.strip()                         # "Answer:" on a line of its own: the letter is on the next line
            ml = _LEAD.match(nxt)
            if ml and _label_follows(nxt, ml.end(), ml.group(1)):
                return ml.group(1)
            mw = _WHOLE.fullmatch(nxt)
            if mw:
                return mw.group(1)
        below = line
    return None


def _explicit_cue(t):
    """Letter of the last (by position) explicit answer statement inside a line (_CUE, \\boxed{X}), else None. Statements that are
    hypotheses (_hypothetical) and "answer: A mutation ..." (A/I that are not labels) are skipped."""
    found = [(m.start(), m, True) for m in _CUE.finditer(t)] + [(m.start(), m, False) for m in _BOXED.finditer(t)]
    found.sort(key=lambda f: f[0], reverse=True)
    for start, m, is_cue in found:
        if is_cue and (_hypothetical(t, start) or (m.group(1) in "AI" and not _label_follows(t, m.end(), m.group(1)))):
            continue
        return m.group(1)
    return None


def _sole_verdict(t):
    """Letter that a text made of per-option verdicts declares correct ("A is incorrect because ... B is correct because ..."), else None.

    Only used when the text STARTED with a rejected analysis label (parse_choice rule 2 found "A is not correct" / "A and B ..." /
    "A: ... B: ..."), after every explicit answer statement came up empty. Deliberately narrow, because "X is correct" is also how a
    model describes the TRUE answer while choosing another one ("I'll pick C. Although B is correct, ..."):
      * no first-person pronoun anywhere in the text (a choice made by "I" is not read off the verdicts);
      * exactly one distinct letter is declared correct ("B is correct", "B is the correct answer", "B is right");
      * not a hypothesis ("if B is correct, ...") and not a remark about an option ("Option B is right", "Answer choice B is correct").
    """
    if _FIRST_PERSON.search(t):
        return None
    found = set()
    for m in _VERDICT.finditer(t):
        if _hypothetical(t, m.start()) or _OPTION_WORD_BEFORE.search(t, max(0, m.start() - 24), m.start()):
            continue
        found.add(m.group(1))
    return found.pop() if len(found) == 1 else None


def parse_choice(text, nchoices=4, visible_cot=False, eos_texts=()):
    """Option index (0-based) from the model's answer text, or None.

    Priority (the first rule that finds a letter decides; a letter outside the option range gives None, it does not fall through):
      1. the LAST line that starts with `Answer:` / `Final answer:` (markdown stripped; see _answer_line) - the format the prompt asks for;
      2. visible_cot=False ("reasoning"/"direct": `text` is the final answer after the hidden reasoning) only: a leading label
         ("B", "**B**", "(B)", "B. text", "B) text", "\\boxed{B}"), sorted by `base.classify_lead`: a bare label is the answer and
         beats every later remark ("B\\n\\nNote: the correct answer is C." is B); "B is correct" / "B because ..." is the answer unless
         rule 3 finds a later explicit statement; "A is not correct. The answer is B.", "A and B are both ... but the answer is C." and
         "A: Ebola ... B: Influenza ... The answer is B." start with analysis: the leading label is skipped (rules 3-5 decide);
      3. the LAST explicit answer statement in a line ("The correct answer is **B**", "answer: B", "\\boxed{B}"), never one that
         reaches across a newline and never a hypothesis ("If the answer is C, ...") or a remark about an option
         ("Answer choice D is ...");
      4. a last line that is only a label ("**B**", "Option B.");
      5. only when rule 2 skipped an analysis label: the single option a first-person-free text declares correct ("A is incorrect
         because ... B is correct because ..." is B; see _sole_verdict).
    visible_cot=True ("cot": `text` is a whole step-by-step answer): no leading-label rule; the first token of free text is never
      the answer ("A virus ..." / "I think ..." would otherwise parse as A / I).
    A text that ends with CUT_MARK (ThinkTagMixin.split: the turn was cut at max_new, no end-of-turn token) is unfinished: only
      rule 1 applies, anything else gives None, so truncation shows up as parse-rate loss and not as a guessed letter.
    A/I after an explicit cue are additionally checked against the English article/pronoun ("answer: A mutation ...").
    Letters >= nchoices give None. No match gives None.
    """
    t = cut_at_eos(text, eos_texts).strip()
    cut = t.endswith(CUT_MARK)
    if cut:
        t = t[:-len(CUT_MARK)].strip()
    if not t:
        return None
    letter = _answer_line(t)
    if letter is None and cut:
        return None
    analysis = False                                # the text STARTS with per-option analysis: its leading label is not the answer
    if letter is None and not visible_cot:
        m = _LEAD.match(t)
        if m and _label_follows(t, m.end(), m.group(1)):
            kind = classify_lead(t, m.end(), m.group(1))
            if kind == "bare":
                letter = m.group(1)
            elif kind == "copular":                 # "B is correct": the answer, unless a later explicit statement says otherwise
                letter = _explicit_cue(t) or m.group(1)
            else:                                   # "A is not correct." / "A and B ..." / "A: ... B: ...": analysis, not an answer
                analysis = True
    if letter is None:
        letter = _explicit_cue(t)
    if letter is None:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        m = _WHOLE.fullmatch(lines[-1].strip()) if lines else None      # last line is only a label: "**B**", "(B)", "Option B."
        letter = m.group(1) if m else None
    if letter is None and analysis:
        letter = _sole_verdict(t)
    if letter is None:
        return None
    i = LETTERS.index(letter)
    return i if i < nchoices else None


def warn_effort(effort, default, family):
    """`effort` is the gpt-oss reasoning-effort knob; this family has none. Warn (visibly) instead of silently ignoring it."""
    if effort is not None and effort != default:
        warnings.warn(f"{family}: no reasoning-effort knob; effort={effort!r} is ignored", stacklevel=3)


def require_transformers(minimum, have=None):
    """Raise a clear RuntimeError if the installed transformers is older than `minimum`."""
    from packaging.version import Version
    if have is None:
        import transformers
        have = transformers.__version__
    if Version(have) < Version(minimum):
        raise RuntimeError(f"transformers>={minimum} is required for this model (installed {have}); "
                           f"see the model card in sbx/models/ and the module docstring")


def add_eos_ids(model, tok, token_strs):
    """Make `model.generate` also stop on the given special tokens (those that exist in the tokenizer); keeps existing ids."""
    cur = model.generation_config.eos_token_id
    ids = [] if cur is None else ([cur] if isinstance(cur, int) else list(cur))
    for s in token_strs:
        i = tok.convert_tokens_to_ids(s)
        if i is not None and i != tok.unk_token_id and i not in ids:
            ids.append(i)
    model.generation_config.eos_token_id = ids
    return ids


class ThinkTagMixin:
    """`split` / `parse_letter` shared by the Qwen and DeepSeek-R1 adapters (not an Adapter itself => never registered).

    Class attributes the host sets: `eos_texts` (end-of-turn / pad strings in decoded text) and `reasoning_in_output`
    (True when the model's output contains a reasoning section closed by `</think>`).

    Truncation. `Ctx.gen` decodes with skip_special_tokens=False, so a turn that ended on its own carries an end-of-turn token and a
    turn that was cut at max_new does not. For `answer_mode == "cot"` (the visible chain of thought IS the answer) `split()` appends
    CUT_MARK to the final text of a cut turn and `parse_choice` then accepts only an explicit `Answer:` line from it: an unfinished
    chain of thought is unparsed (visible as parse-rate loss) instead of yielding a letter quoted from the middle of the reasoning.
    The final text itself stays the visible text, so callers that read generations as chat text (sbx.dirs.refusal.final_text strips
    every `<|...|>` token and keeps a non-empty answer) are unaffected. Reasoning modes need no marker: a missing `</think>` already
    means final == "".
    """
    eos_texts = ("<|im_end|>", "<|endoftext|>")
    reasoning_in_output = False

    def split(self, text):
        t = cut_at_eos(text, self.eos_texts)
        if self.reasoning_in_output or THINK_CLOSE in t:
            # (thinking OFF models normally never emit a think tag; if one does, still separate the hidden part.)
            reasoning, final = split_think(t)
        else:
            reasoning, final = "", t.strip()
        if final and self.answer_mode == "cot" and len(t) == len(text):       # no end-of-turn token: cut at max_new
            final += CUT_MARK
        return reasoning, final

    def parse_letter(self, final, nchoices=4):
        return parse_choice(final, nchoices, visible_cot=(self.answer_mode == "cot"), eos_texts=self.eos_texts)


# ----------------------------------------------------------------------------------------------- adapters
class _QwenChat(ThinkTagMixin, Adapter):
    """Common implementation (keys=[] => not registered). Subclasses set keys/repos/thinking/answer_mode/max_new/gen_suffix."""
    keys = []
    sys_role = "system"
    thinking = False          # value passed as enable_thinking to the chat template
    gen_suffix = ""           # exact end of every rendered prompt (where the assistant's generated text begins)
    min_transformers = "4.51.0"

    def render(self, tok, msgs, effort=None):
        warn_effort(effort, self.effort, type(self).__name__)
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
        if not p.endswith(self.gen_suffix):
            raise ValueError(f"{self.key}: chat template does not end with the expected generation prompt "
                             f"{self.gen_suffix!r} (got ...{p[-80:]!r}); template/tokenizer revision mismatch?")
        return p


class Qwen3NoThink(_QwenChat):
    """Qwen3 dense hybrid models, thinking OFF: empty think block in the prompt, short visible CoT ending 'Answer: X'."""
    keys = ["qwen3-32b", "qwen3-14b", "qwen3-8b"]
    repos = {"qwen3-32b": "Qwen/Qwen3-32B", "qwen3-14b": "Qwen/Qwen3-14B", "qwen3-8b": "Qwen/Qwen3-8B"}
    answer_mode = "cot"
    max_new = 400
    thinking = False
    reasoning_in_output = False
    gen_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    min_transformers = "4.51.0"


class Qwen3Think(_QwenChat):
    """Qwen3 dense hybrid models, thinking ON: the model emits '<think>\\n...</think>\\n\\n<final>'; reasoning split off."""
    keys = ["qwen3-32b-think", "qwen3-14b-think", "qwen3-8b-think"]
    repos = {"qwen3-32b-think": "Qwen/Qwen3-32B", "qwen3-14b-think": "Qwen/Qwen3-14B", "qwen3-8b-think": "Qwen/Qwen3-8B"}
    answer_mode = "reasoning"
    max_new = 1800
    thinking = True
    reasoning_in_output = True
    gen_suffix = "<|im_start|>assistant\n"
    min_transformers = "4.51.0"


class _Qwen35(_QwenChat):
    """Qwen3.5 (qwen3_5: Gated-DeltaNet/full-attention hybrid + vision tower + MTP head; text-only use here)."""
    keys = []
    min_transformers = "5.2.0"

    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        require_transformers(self.min_transformers)
        tok, model = super().load(repo=repo, dtype=dtype, device_map=device_map, quant=quant, **kw)
        add_eos_ids(model, tok, ["<|im_end|>", "<|endoftext|>"])
        return tok, model

    def layers(self, model):
        # AutoModelForCausalLM -> Qwen3_5ForCausalLM: model.model.layers (text-only; vision tower and MTP head are not instantiated).
        # A Qwen3_5ForConditionalGeneration (AutoModelForImageTextToText) keeps them in model.model.language_model.layers.
        inner = model.model
        return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


class Qwen35NoThink(_Qwen35):
    """Qwen3.5 with thinking OFF (empty think block in the prompt), visible short CoT ending 'Answer: X'."""
    keys = ["qwen3.5-9b", "qwen3.5-0.8b"]
    repos = {"qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B"}
    answer_mode = "cot"
    max_new = 400
    thinking = False
    reasoning_in_output = False
    gen_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen35Think(_Qwen35):
    """Qwen3.5 with thinking ON: '<think>\\n' is already in the prompt, so the generated text starts with the reasoning itself."""
    keys = ["qwen3.5-9b-think", "qwen3.5-0.8b-think"]
    repos = {"qwen3.5-9b-think": "Qwen/Qwen3.5-9B", "qwen3.5-0.8b-think": "Qwen/Qwen3.5-0.8B"}
    answer_mode = "reasoning"
    max_new = 1800
    thinking = True
    reasoning_in_output = True
    gen_suffix = "<|im_start|>assistant\n<think>\n"
