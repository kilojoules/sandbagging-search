"""Mistral-family adapters, plus the helpers shared by the "misc" adapter modules (phi.py, olmo.py, seed_oss.py import them).

Keys (see sbx/models/<key>.json for the machine-readable cards)
  mistral-small-24b      mistralai/Mistral-Small-24B-Instruct-2501    text-only MistralForCausalLM, HF tokenizer + template
  mistral-small-3.2-24b  mistralai/Mistral-Small-3.2-24B-Instruct-2506  Mistral3ForConditionalGeneration (vision wrapper)
  ministral-3-14b        mistralai/Ministral-3-14B-Instruct-2512-BF16   Mistral3ForConditionalGeneration + ministral3 text
                                                                         model: needs transformers>=5.0 (NOT loadable locally)
  magistral-small-2509   mistralai/Magistral-Small-2509               reasoning ([THINK]..[/THINK]) variant of the 3.x VLM wrapper
  ministral-3-14b-reasoning  mistralai/Ministral-3-14B-Reasoning-2512  reasoning variant of ministral-3-14b (transformers>=5.0)

Gotchas (all verified against the real tokenizers/templates in sbx/tests/test_adapters_misc.py)
  * Template = tekken "V7": ``<s>[SYSTEM_PROMPT]..[/SYSTEM_PROMPT][INST]..[/INST]``. The prompt ends right after ``[/INST]``;
    ``add_generation_prompt`` is a no-op (the template ignores it). BOS ``<s>`` is already inside the rendered string, so
    encode() must use add_special_tokens=False (base.Adapter does); tok(text) with the default flag gives ``<s><s>...``.
  * The 2501 template inserts a DEFAULT system prompt (with ``strftime_now('%Y-%m-%d')``, i.e. TODAY's date) whenever no
    system turn is given, and a given system turn REPLACES it. To keep prompts deterministic across days/runs and to keep
    N-vs-framing contrasts clean we ALWAYS render an explicit system turn = default prompt (date pinned) and, when a framing
    has sys_text, append it after a blank line ("default + framing"). N therefore equals the template's own default render
    with the date pinned (tested) and E = N with the framing paragraph inserted before ``[/SYSTEM_PROMPT]``. Set the class
    attribute ``keep_default_system = False`` for the vendor's native semantics (framing REPLACES the default prompt).
    The date is pinned by REWRITING the date sentence of the natively rendered default text (``MistralSmall.default_system``),
    never by passing ``strftime_now=<callable>`` to ``apply_chat_template``: recent transformers 5.x (5.24.0 checked) sanitise template
    kwargs and turn a callable into a string (``TypeError: 'str' object is not callable`` inside the template; 4.57.6 and 5.2.0 are fine).
  * Mistral tokenizers need ``fix_mistral_regex=True`` in transformers 4.57 (otherwise a warning about an incorrect
    pre-tokenizer regex). On 906 real texts the unfixed tokenizer was identical to the official tekken tokenization, so the
    fix is precautionary; load() applies it and verifies/patches the regex explicitly (no silent fallback).
  * Mistral-Small-3.1/3.2 and Ministral-3 are Mistral3ForConditionalGeneration checkpoints: AutoModelForCausalLM rejects
    Mistral3Config, so load() uses AutoModelForImageTextToText and the layers live at ``model.model.language_model.layers``
    (transformers>=4.52; older versions had ``model.language_model.model.layers``). Text-only input works unchanged.
  * The 3.2 repo ships only tekken.json (+params.json); there is NO HF tokenizer.json/chat template in it. We use the HF
    tokenizer of Mistral-Small-3.1-24B-Instruct-2503 (vocab verified identical to 3.2's tekken.json for all 130 072
    non-special ranks, and tokenization identical on 906 real texts) and build the V7 string by hand with the vendor's
    SYSTEM_PROMPT.txt (date pinned) as the default system prompt. mistral-common is NOT needed.
  * Ministral-3 checkpoints are FP8; we point at the -BF16 sibling repo (same template). model_type ``ministral3`` exists only
    in transformers>=5.0; its HF template leaves ``{today}``/``{yesterday}`` placeholders unfilled (we fill them, pinned).
  * Reasoning variants (Magistral, Ministral-3-Reasoning): the default system prompt is the vendor's "HOW YOU SHOULD THINK AND
    ANSWER" prompt (it teaches the ``[THINK]..[/THINK]`` format; framing text is appended to it, never replacing it). [THINK] and
    [/THINK] are single special tokens (ids 34/35; checked in Magistral's tekken.json and in the Ministral-3-Reasoning HF
    tokenizer, whose vocab is identical to Magistral's: Magistral has no HF tokenizer files, so it borrows that one). The model
    emits [THINK] itself; split() cuts at [/THINK]. The vendors recommend sampling (T=0.7, top_p=0.95); sbx is greedy-only.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .base import L10, Adapter, classify_lead, label_follows

# ================================================================================================ shared helpers
# ---------------------------------------------------------------------------------------- answer-letter parsing
# base.Adapter.parse_letter had three weaknesses for chatty models: "**Answer:** B" (bold after the colon) is missed, the
# case-insensitive letter class turned "the answer is a bit ..." into option A, and a leading "A man ..." / "I think ..."
# was read as option A / I. parse_mcq_letter fixes these and also understands \boxed{B}.
# ONE character class for everything between the keyword and the letter (separators, markdown, openers): adjacent quantified
# classes that overlap would backtrack quadratically/cubically on degenerate outputs such as "Answer:" + 100k newlines.
_J = r"[\s*_`:=\u2013\u2014\-(\[{$]"
_ANS = re.compile(
    r"\b(?:(?i:answer)(?:\s+(?i:to|for)\s+(?i:this|the)\s+(?i:question|problem))?|(?i:correct\s+(?:option|choice|letter)))" + _J + r"*"
    r"(?:(?i:is|would\s+be|should\s+be|will\s+be)\b" + _J + r"*)?"
    r"(?:(?i:option|choice|letter)\b" + _J + r"*)?"
    r"(?:\\boxed\s*\{\s*(?:\\text(?:bf)?\s*\{\s*)?)?"
    r"([A-J])")
_BOXED = re.compile(r"\\boxed\s*\{\s*(?:\\text(?:bf)?\s*\{\s*)?\(?\s*([A-J])")
_FIRST = re.compile(r"[\s*_`(\[{$]*(?:\\boxed\s*\{\s*)?([A-J])")
# After an explicit cue ("Answer: A ...") the letters A / I are also English words: demand a clear delimiter. The pronoun I gets the
# narrower set ("Answer: I, as an AI ...", "Answer: I... hmm", "Answer: I (an AI) ..." are refusals, not option I).
_A_DELIM = re.compile(r"[ \t]*(?:$|[\n,;:)\]}*`!?<(]|\.(?![.A-Za-z0-9])|[-\u2013\u2014])")
_I_DELIM = re.compile(r"[ \t]*(?:$|[\n:)\]}*`<]|\.(?![.A-Za-z0-9]))")
# A LEADING label (no cue in front of it) must read as a label (base.label_follows: "B", "B.", "B)", "**B**", "B is correct", "B<|im_end|>"
# are; "B cells ...", "E = mc2", "C++", "I'm ...", "I, as an AI ...", "I... hmm" are not) or, for the letters that are not also English
# words (not A, I), be followed by blanks and an upper-case word / digit / bracket / one more connector ("B Influenza", "C (Golgi)").
# The article A and the pronoun I only count before the delimiter sets above (`_standalone`: "A is correct" / "I because ..." stay unparsed here).
_SPACED_LABEL = re.compile(r"[ \t]+(?:[A-Z0-9(\[{*_`$\"\u201c']|(?:were|would|should|could|must|has|can|may|might|will|and|or|but|as|due|given"
                           r"|based|looks|appears)\b)")


_SPECIAL_SPAN = re.compile(r"<[^<>\s]{1,40}>")


def _strip_trailing_special(text: str) -> str:
    """Drop the end-of-turn / pad tokens a finished row ends with when it is decoded with skip_special_tokens=False ("B<|im_end|><|dummy_85|>
    <|dummy_85|>", "B</s><pad><pad>"): a short reply padded to the longest row of its batch is still a short reply. Linear (no slicing in
    the loop)."""
    end = len(text.rstrip())
    while end > 0 and text[end - 1] == ">":
        i = text.rfind("<", 0, end)
        if i < 0 or _SPECIAL_SPAN.fullmatch(text, i, end) is None:
            break
        end = i
        while end > 0 and text[end - 1].isspace():
            end -= 1
    return text[:end]


def _standalone(text: str, end: int, letter: str) -> bool:
    """True if the letter ending at text[end] (found after an explicit cue) is a stand-alone option label, not the start of a word."""
    nxt = text[end:end + 1]
    if nxt.isalnum() or nxt == "_":
        return False                       # "Because", "B2", "A2"
    if nxt and nxt in "-\u2013\u2014" and text[end + 1:end + 2].isalnum():
        return False                       # "B-cell", "D-glucose", "A\u2013D" (a dash that joins what follows)
    if letter == "I":                      # pronoun: demand a clear delimiter ("Answer: I." yes, "Answer: I, as an AI" no)
        return _I_DELIM.match(text, end) is not None
    if letter == "A":                      # article: demand a clear delimiter ("Answer: A." yes, "A man" no)
        return _A_DELIM.match(text, end) is not None
    return True


def _lead(text: str):
    """(letter, kind) of the option label the text STARTS with, or (None, None). `kind` is base.classify_lead's verdict: "bare" and
    "copular" labels are answers, "negated" ("B is not correct."), "multi" ("A and B ...") and "list" ("A: ... B: ...") are analysis."""
    m = _FIRST.match(text)
    if m is None:
        return None, None
    letter, end = m.group(1), m.end(1)
    if letter in "AI":                                  # the article and the pronoun keep the narrow delimiter sets of the cue path
        ok = _standalone(text, end, letter)
    else:
        ok = label_follows(text, end, letter) or _SPACED_LABEL.match(text, end) is not None
    return (letter, classify_lead(text, end, letter)) if ok else (None, None)


def parse_mcq_letter(text: str, nchoices: int = 4, mode: str = "cot") -> Optional[int]:
    """Model's final text -> 0-based option index, or None (unparsed). Never guesses.

    mode "cot": the LAST "Answer: X" / "answer is X" / "correct option is X" / \\boxed{X} wins; a reply that is itself just a
    short option label ("B", "**B**", "(C) Golgi"; at most 32 characters, not counting the end-of-turn / pad tokens a finished row
    carries) is accepted as a fallback. Truncated reasoning without a final answer line -> None.
    mode "direct"/"reasoning" (asked for "only the letter"): a leading label wins (a bare one beats every later statement; "B is
    correct" yields to a later explicit statement; a text that STARTS with analysis - "B is not correct. The answer is C.",
    "A and B ...", "A: ... B: ..." - has no leading label), then the last answer pattern.
    A letter outside range(nchoices) -> None (not a different candidate).
    Letters must be UPPERCASE; A and I additionally need a delimiter ("Answer: A." / "Answer: I think" -> not I).
    """
    if not text:
        return None
    cands = []
    for m in _ANS.finditer(text):
        if _standalone(text, m.end(1), m.group(1)):
            cands.append((m.start(1), m.group(1)))
    for m in _BOXED.finditer(text):
        if _standalone(text, m.end(1), m.group(1)):
            cands.append((m.start(1), m.group(1)))
    letter = None
    if mode == "cot":
        if cands:
            letter = max(cands)[1]
        else:
            short = _strip_trailing_special(text).strip()          # the 32 characters are the visible reply, not its end-of-turn / pad tokens
            lead, kind = _lead(short) if len(short) <= 32 else (None, None)
            letter = lead if kind in ("bare", "copular") else None
    else:
        lead, kind = _lead(text)
        if kind == "bare" or (kind == "copular" and not cands):
            letter = lead
        elif cands:
            letter = max(cands)[1]
    if letter is None:
        return None
    i = L10.index(letter)
    return i if i < nchoices else None


class MCQParseMixin:
    """Adapter mixin (put it BEFORE Adapter in the bases): robust parse_letter (see parse_mcq_letter) and an encode() that never
    returns token_type_ids. The latter matters for Seed-OSS: its PreTrainedTokenizerFast lists token_type_ids in
    model_input_names, and core.Ctx passes the whole encoding to model.generate(), which rejects the unused key."""

    def parse_letter(self, final, nchoices=4):
        return parse_mcq_letter(final, nchoices, self.answer_mode)

    def encode(self, tok, texts):
        return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False)


# ---------------------------------------------------------------------------------------- thinking-tag splitting
def split_think(text: str, open_tag: str, close_tag: str, open_in_prompt: bool):
    """Generated text -> (reasoning, final) for a model that wraps its reasoning in open_tag..close_tag.

    * closing tag present            -> (text before it with the opening tag removed, text after it), both stripped.
    * opening tag but no closing tag -> truncated reasoning: (text after the opening tag, "")  [final is empty => unparsed].
    * neither tag:  open_in_prompt (the template already opened the think block, so the model is inside it) -> (text, "")
                    (truncated); otherwise the model answered without thinking -> ("", text).
    """
    if close_tag in text:
        r, f = text.split(close_tag, 1)
        if open_tag and open_tag in r:
            r = r.split(open_tag, 1)[1]
        return r.strip(), f.strip()
    if open_tag and open_tag in text:
        return text.split(open_tag, 1)[1].strip(), ""
    return (text.strip(), "") if open_in_prompt else ("", text.strip())


class ThinkSplitMixin:
    """Adapter mixin: split() on think_open/think_close. Put it BEFORE Adapter in the bases."""
    think_open = "<think>"
    think_close = "</think>"
    think_open_in_prompt = False      # True when the chat template's generation prompt already ends with the opening tag

    def split(self, text):
        return split_think(text, self.think_open, self.think_close, self.think_open_in_prompt)


# ---------------------------------------------------------------------------------------- prompt helpers
def check_gen_prompt(adapter, prompt: str) -> str:
    """Explicit guard: the rendered prompt must end exactly where the assistant starts (adapter.gen_prompt_suffix)."""
    suffix = getattr(adapter, "gen_prompt_suffix", None)
    if suffix is not None and not prompt.endswith(suffix):
        raise ValueError(f"{adapter.key}: rendered prompt does not end with the generation prompt {suffix!r}: "
                         f"...{prompt[-80:]!r} (chat template changed?)")
    return prompt


def template_default_system(tok, start: str, end: str, **template_kw) -> str:
    """The chat template's own default system text: render a user-only chat and cut between the system-turn markers."""
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, **template_kw)
    i = s.find(start)
    if i < 0:
        raise ValueError(f"template has no default system turn (marker {start!r} not found in {s[:120]!r})")
    i += len(start)
    j = s.find(end, i)
    if j < 0:
        raise ValueError(f"unterminated default system turn (marker {end!r} not found)")
    return s[i:j]


def with_default_system(msgs, default: str, always: bool, merge: bool = True):
    """Return msgs whose system turn is ``default + blank line + framing`` (framing = the given system message, if any).

    always=True  : a system turn is always present (= default when there is no framing).
    always=False : without a system message msgs is returned unchanged (the template inserts its own default itself).
    merge=False  : template-native semantics, the framing REPLACES the default system prompt (adapter.keep_default_system = False).
    """
    msgs = [dict(m) for m in msgs]
    if msgs and msgs[0]["role"] == "system":
        framing = (msgs[0]["content"] or "").strip()
        if merge:
            msgs[0]["content"] = default.rstrip() + ("\n\n" + framing if framing else "")
    elif always:
        msgs.insert(0, {"role": "system", "content": default})
    for m in msgs[1:]:
        if m["role"] == "system":
            raise ValueError("only a leading system message is supported")
    return msgs


def _model_kwargs(dtype, device_map, quant, kw):
    import torch
    kwargs = dict(torch_dtype=dtype if dtype != "bf16" else torch.bfloat16, device_map=device_map)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    kwargs.update(kw)
    return kwargs


# ================================================================================================ Mistral tokenizer
TEKKEN_PATTERN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
                  r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|"
                  r"\s*[\r\n]+|\s+(?!\S)|\s+")


def _backend(tok):
    return getattr(tok, "backend_tokenizer", None) or tok._tokenizer      # transformers 4.x / 5.x


def _pre_tokenizer_regex(tok) -> Optional[str]:
    pt = json.loads(_backend(tok).to_str()).get("pre_tokenizer") or {}
    for p in pt.get("pretokenizers", [pt]):
        if p.get("type") == "Split":
            return (p.get("pattern") or {}).get("Regex")
    return None


def load_mistral_tokenizer(repo: str, **kw):
    """HF tokenizer for a tekken-based Mistral repo: left padding, a pad token, and the corrected tekken pre-tokenizer regex
    (transformers 4.x: ``fix_mistral_regex=True``, which v5 no longer accepts; if the regex is still not the tekken one we
    patch it ourselves, else raise)."""
    import transformers
    from transformers import AutoTokenizer
    if int(transformers.__version__.split(".")[0]) < 5:
        kw = dict(kw, fix_mistral_regex=True)
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    if _pre_tokenizer_regex(tok) != TEKKEN_PATTERN:
        import tokenizers
        _backend(tok).pre_tokenizer[0] = tokenizers.pre_tokenizers.Split(
            pattern=tokenizers.Regex(TEKKEN_PATTERN), behavior="isolated")
        if _pre_tokenizer_regex(tok) != TEKKEN_PATTERN:
            raise RuntimeError(f"could not install the tekken pre-tokenizer regex on {repo}")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<pad>" if "<pad>" in tok.get_vocab() else tok.eos_token
    return tok


# ================================================================================================ mistral-small-24b
_DATE_SENTENCE = re.compile(r"(The current date is )\d{4}-\d{2}-\d{2}")      # the one date in the 2501 template's default system prompt


class MistralSmall(MCQParseMixin, Adapter):
    """Mistral-Small-24B-Instruct-2501 (text-only, 40 layers, H=5120). answer_mode "cot": the model is a non-reasoning
    instruct model, so it gets a brief visible chain of thought ending in 'Answer: X'."""
    keys = ["mistral-small-24b"]
    repos = {"mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501"}
    answer_mode = "cot"
    max_new = 512
    sys_role = "system"
    gen_prompt_suffix = "[/INST]"
    pinned_today = "2025-01-30"          # the date the template's strftime_now('%Y-%m-%d') wrote is replaced by the release date
    keep_default_system = True           # framing text is APPENDED to the default system prompt (False: it replaces it, template-native)

    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        from transformers import AutoModelForCausalLM
        repo = repo or self.repo
        tok = load_mistral_tokenizer(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, **_model_kwargs(dtype, device_map, quant, kw)).eval()
        return tok, model

    def default_system(self, tok) -> str:
        """The template's default system prompt with its date pinned to ``pinned_today``.

        The template is rendered natively (it calls ``strftime_now`` itself, so the text carries the real date of the render) and
        the date sentence is then rewritten INSIDE this default text only. We must not hand the template a ``strftime_now``
        callable instead: recent transformers 5.x (5.24.0) rebuild every template kwarg from plain JSON types and turn a callable into a
        placeholder string, so the template then fails with ``TypeError: 'str' object is not callable`` (checked: 5.24.0 fails,
        4.57.6 and 5.2.0 accept the callable; the failing step is chat_template_utils._plain_template_data). Exactly one date
        sentence is required: a template that words it differently raises here rather than leaking today's date into the prompts."""
        raw = template_default_system(tok, "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]")
        default, n = _DATE_SENTENCE.subn(r"\g<1>" + self.pinned_today, raw)
        if n != 1:
            raise ValueError(f"{self.key}: expected exactly one 'The current date is YYYY-MM-DD' sentence in the template's default "
                             f"system prompt, found {n} (chat template changed?): {raw[:200]!r}")
        return default

    def render(self, tok, msgs, effort=None):
        # an explicit system turn is ALWAYS supplied, so the template's own (dated) default branch never reaches the output
        msgs = with_default_system(msgs, self.default_system(tok), always=True, merge=self.keep_default_system)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return check_gen_prompt(self, s)


# ================================================================================================ Mistral3 (VLM wrapper)
SYSTEM_PROMPT_32 = (
    "You are {name}, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\n"
    "You power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\n"
    "The current date is {today}.\n\n"
    "When you're not sure about some information or when the user's request requires up-to-date or specific data, you must use "
    "the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or "
    "complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid "
    "making up anything.\n"
    "If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the "
    "question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \"What are "
    "some good restaurants around me?\" => \"Where are you?\" or \"When is the next flight to Tokyo\" => \"Where do you travel "
    "from?\").\n"
    "You are always very attentive to dates, in particular you try to resolve dates (e.g. \"yesterday\" is {yesterday}) and when "
    "asked about information at specific dates, you discard information that is at another date.\n"
    "You follow these instructions in all languages, and always respond to the user in the language they use or request.\n"
    "Next sections describe the capabilities that you have.\n\n"
    "# WEB BROWSING INSTRUCTIONS\n\n"
    "You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to "
    "do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n"
    "# MULTI-MODAL INSTRUCTIONS\n\n"
    "You have the ability to read images, but you cannot generate images. You cannot read nor transcribe audio files or videos.\n\n"
    "# TOOL CALLING INSTRUCTIONS\n\n"
    "You may have access to tools that you can use to fetch information or perform actions. You must use these tools in the "
    "following situations:\n\n"
    "1. When the request requires up-to-date information.\n"
    "2. When the request requires specific data that you do not have in your knowledge base.\n"
    "3. When the request involves actions that you cannot perform without tools.\n\n"
    "Always prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the "
    "user that you cannot perform the requested action at the moment.")


def render_v7(msgs, system: str) -> str:
    """Tekken-V7 chat string for single- or multi-turn text chats: ``<s>[SYSTEM_PROMPT]..[/SYSTEM_PROMPT][INST]..[/INST]``
    (assistant turns are ``content</s>``). ``system`` is the full system text (default already merged with any framing)."""
    out = ["<s>[SYSTEM_PROMPT]", system, "[/SYSTEM_PROMPT]"]
    for m in msgs:
        if m["role"] == "user":
            out += ["[INST]", m["content"], "[/INST]"]
        elif m["role"] == "assistant":
            out += [m["content"], "</s>"]
        elif m["role"] != "system":
            raise ValueError(f"unsupported role {m['role']!r}")
    return "".join(out)


class MistralSmall32(MCQParseMixin, Adapter):
    """Mistral-Small-3.2-24B-Instruct-2506: Mistral3ForConditionalGeneration (24.0B incl. a 0.44B pixtral tower; text path
    used only). Same V7 prompt format as 2501; default system prompt = vendor SYSTEM_PROMPT.txt with the date pinned."""
    keys = ["mistral-small-3.2-24b"]
    repos = {"mistral-small-3.2-24b": "mistralai/Mistral-Small-3.2-24B-Instruct-2506"}
    # the 3.2 repo has no HF tokenizer files; this sibling repo's tokenizer.json has the identical vocabulary (tested)
    tokenizer_repos = {"mistral-small-3.2-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"}
    answer_mode = "cot"
    max_new = 512
    sys_role = "system"
    gen_prompt_suffix = "[/INST]"
    system_name = "Mistral-Small-3.2-24B-Instruct-2506"
    pinned_today, pinned_yesterday = "2025-06-20", "2025-06-19"
    min_transformers = (4, 52, 0)
    keep_default_system = True           # framing text is APPENDED to the default system prompt (False: it replaces it)

    def _check_transformers(self):
        import transformers
        have = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:3])
        if have < self.min_transformers:
            need = ".".join(map(str, self.min_transformers))
            raise RuntimeError(f"{self.key} needs transformers>={need}, found {transformers.__version__}")

    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        from transformers import AutoModelForImageTextToText
        self._check_transformers()
        repo = repo or self.repo
        tok = load_mistral_tokenizer(self.tokenizer_repos.get(self.key, repo))
        model = AutoModelForImageTextToText.from_pretrained(repo, **_model_kwargs(dtype, device_map, quant, kw)).eval()
        if len(tok) != model.config.text_config.vocab_size:
            raise RuntimeError(f"tokenizer ({len(tok)}) / model ({model.config.text_config.vocab_size}) vocab size mismatch")
        return tok, model

    def layers(self, model):
        return model.model.language_model.layers

    def default_system(self, tok=None) -> str:
        return (SYSTEM_PROMPT_32.replace("{name}", self.system_name).replace("{today}", self.pinned_today)
                .replace("{yesterday}", self.pinned_yesterday))

    def render(self, tok, msgs, effort=None):
        msgs = with_default_system(msgs, self.default_system(tok), always=True, merge=self.keep_default_system)
        return check_gen_prompt(self, render_v7(msgs[1:], msgs[0]["content"]))


class Ministral3(MistralSmall32):
    """Ministral-3-14B-Instruct-2512 (BF16 sibling repo). 40 layers, H=5120, 13.9B. Needs transformers>=5.0 (model_type
    ``ministral3``): load() raises a clear error on older versions. Uses the repo's own HF tokenizer + chat template; the
    template's default system prompt has unfilled ``{today}``/``{yesterday}`` placeholders, which we fill (pinned)."""
    keys = ["ministral-3-14b"]
    repos = {"ministral-3-14b": "mistralai/Ministral-3-14B-Instruct-2512-BF16"}
    tokenizer_repos = {}
    min_transformers = (5, 0, 0)
    pinned_today, pinned_yesterday = "2025-12-02", "2025-12-01"

    def default_system(self, tok=None) -> str:
        raw = template_default_system(tok, "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]")
        return raw.replace("{today}", self.pinned_today).replace("{yesterday}", self.pinned_yesterday)

    def render(self, tok, msgs, effort=None):
        msgs = with_default_system(msgs, self.default_system(tok), always=True, merge=self.keep_default_system)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return check_gen_prompt(self, s)


# ================================================================================================ reasoning variants
REASONING_SYSTEM_PROMPT = (
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use "
    "LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\n"
    "Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch "
    "paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a "
    "self-contained response.\n")      # == Magistral-Small-2509/SYSTEM_PROMPT.txt (incl. trailing newline, as in the vendor's examples)


class Magistral(ThinkSplitMixin, MistralSmall32):
    """Magistral-Small-2509: Mistral-Small-3.2 architecture (Mistral3ForConditionalGeneration, 24.0B) trained to reason inside
    [THINK]..[/THINK]. answer_mode "reasoning". No HF tokenizer in the repo -> the Ministral-3-14B-Reasoning HF tokenizer (identical
    vocab, real [THINK]/[/THINK] special tokens) is used; the prompt string is built by hand (V7 + the reasoning system prompt)."""
    keys = ["magistral-small-2509"]
    repos = {"magistral-small-2509": "mistralai/Magistral-Small-2509"}
    tokenizer_repos = {"magistral-small-2509": "mistralai/Ministral-3-14B-Reasoning-2512"}
    answer_mode = "reasoning"
    max_new = 4096
    think_open, think_close = "[THINK]", "[/THINK]"
    think_open_in_prompt = False

    def default_system(self, tok=None) -> str:
        return REASONING_SYSTEM_PROMPT


class Ministral3Reasoning(ThinkSplitMixin, Ministral3):
    """Ministral-3-14B-Reasoning-2512 (bf16, transformers>=5.0). Same as ministral-3-14b but the template's default system prompt is the
    reasoning prompt and the model answers as [THINK]..[/THINK] + final text. answer_mode "reasoning"."""
    keys = ["ministral-3-14b-reasoning"]
    repos = {"ministral-3-14b-reasoning": "mistralai/Ministral-3-14B-Reasoning-2512"}
    answer_mode = "reasoning"
    max_new = 4096
    think_open, think_close = "[THINK]", "[/THINK]"
    think_open_in_prompt = False
