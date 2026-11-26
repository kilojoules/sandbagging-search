"""Adapter base class: everything model-family specific (loading, chat template, thinking mode, parsing, layer path)."""
import re

L10 = list("ABCDEFGHIJ")

# One instruction line per answer mode (kept identical across families so framings are comparable).
ASK = {
    "direct": "Answer this multiple-choice question. Reply with only the letter of the correct option.",
    "reasoning": "Answer this multiple-choice question. Reply with only the letter of the correct option.",
    "cot": ("Answer this multiple-choice question. Think step by step briefly, then finish with a final line "
            "of the form 'Answer: <letter>'."),
}

# ------------------------------------------------------------------------------------------------ answer-letter parsing
# A letter at the START of a reply is an option label only if what follows it says so. The earlier rule (`\b` after the letter)
# read "I’m sorry, but I can’t help with that." (a gpt-oss-120b refusal) as option I, "A study shows ..." as A, "B cells ..." as B:
# such rows were scored as ANSWERED (and wrong with probability 1 - 1/n), so they never showed up as parse-rate loss.
# `label_follows` is the one definition of "reads as a label" used by this parser.
_BLANKS = re.compile(r"[^\S\n]*")                                    # whitespace except the newline
_LABEL_WORD = re.compile(r"(?:is|was|are|seems|because|since)\b")      # blanks + one of these: "B is correct", "B because ..."
_SPECIAL_TOKEN = re.compile(r"<(?:\||/|[A-Za-z_:\-]+>)")              # "<|return|>", "</s>", "<eos>", "<end_of_turn>": decoded with skip_special_tokens=False
_FIRST = re.compile(r"\s*(?:\*\*)?\(?([A-J])")
# One `\s*` before the optional connector and none after it that could meet another: linear on "answer" + 50k blanks.
_ANS = re.compile(r"(?i:answer)\s*(?:(?:(?i:is)|:|-)\s*)?(?:\*\*)?\(?([A-J])")


def label_follows(text, pos, letter):
    """True iff `letter`, which ends at text[pos], reads as an option LABEL and not as the first word of a sentence.

    Accepted continuations (nothing else is):
      * end of text, a newline (after optional blanks) or an end-of-turn special token ("B", "B\\n...", "B<|return|>", "B </s>");
      * glued punctuation  )  .  :  *  ,  ;  -  \u2013  \u2014   ("B)", "B.", "B:", "**B**", "B, because ...", "B- ...") or a closing
        bracket / quote  ]  }  `  "  \u201d  $  ("[B]", "`B`", "\\boxed{B}"), except
          - "." that starts an ellipsis or an abbreviation ("I... I can't", "I.e.", "B.C.");
          - a dash directly followed by a letter or digit ("B-cell", "C-terminal", "A\u2013D"), and any glued dash after I;
          - "," and ";" after the pronoun I ("I, as an AI language model, cannot ...");
      * blanks + "-" / "\u2013" / "\u2014" / ":"  ("B - Influenza", "B: Influenza"); after I only ":" ("I \u2014 I can't ..." is a refusal);
      * blanks + is / was / are / seems / because / since  ("B is correct", "B because ..."); not "I was ...".
    Rejected: "I'm", "I\u2019m", "I think", "I cannot", "A study", "A good ...", "B cells", "E = mc2", "C++", "F1", "H2O".
    """
    n = len(text)
    if pos >= n:
        return True
    c = text[pos]
    if not c.isspace():
        if c.isalnum() or c == "_":
            return False
        if c == "<":
            return _SPECIAL_TOKEN.match(text, pos) is not None
        if c in ")]}*:`\"\u201d$":
            return True
        if c in ",;":
            return letter != "I"
        if c == ".":
            nxt = text[pos + 1:pos + 2]
            return not (nxt == "." or nxt.isalnum())
        if c in "-\u2013\u2014":
            return letter != "I" and not text[pos + 1:pos + 2].isalnum()        # not "I\u2014I can't ..."
        return False
    j = _BLANKS.match(text, pos).end()
    if j >= n or text[j] == "\n" or (text[j] == "<" and _SPECIAL_TOKEN.match(text, j) is not None):
        return True
    if letter == "I":
        return text[j] == ":" or (_LABEL_WORD.match(text, j) is not None and not text.startswith("was", j))
    if text[j] in "-\u2013\u2014:":
        return True
    return _LABEL_WORD.match(text, j) is not None


# ------------------------------------------------------------------------------------------------ is a leading label an ANSWER?
# The families that answer after a hidden / long reasoning (qwen, mistral, glm) read a LEADING label before any "the answer is X"
# statement, because the prompt asks for "only the letter". That is wrong when the text starts with per-option ANALYSIS: "A is not
# correct. The correct answer is B.", "A and B are both partially right, but the answer is C.", "A: Ebola ... B: Influenza ... The
# answer is B." were all read as A. `classify_lead` sorts a leading label (one the caller's own plausibility check already accepted
# as label-like) into
#   "bare"     the label alone or with punctuation / option text: "B", "**B**", "B.", "B) Influenza A virus", "B\n\nNote: ..." -> the answer
#   "copular"  the label is the subject of a sentence that does not reject it: "B is correct", "B because ..." -> the answer, unless a
#              later explicit statement ("the answer is C") says otherwise
#   "negated"  "A is not correct", "A is wrong", "A isn't the answer", "A is partially right"      -> NOT the answer
#   "multi"    "A and B ...", "A or B", "A, B and C ..."                                           -> not a single choice
#   "list"     "A: Ebola ... B: Influenza ..." (the same label style repeats for another letter)   -> first item of a list, NOT the answer
# Every pattern is anchored (`match(text, pos)`) or a single left-to-right scan, with no two adjacent quantified groups that can
# match the same character: linear time on degenerate generations.
_B = r"[^\S\n]"                                   # a blank: whitespace except the newline
_NEXT_LABEL = r"(?:[B-HJ]|[AI](?!" + _B + r"+[a-z'\u2019]))(?![A-Za-z0-9'\u2019])"       # a second option letter ("A and B"), not "I think" / "A study"
_MULTI = re.compile(r"(?:" + _B + r"+(?i:and|or|nor|vs\.?|versus)" + _B + r"+|" + _B + r"*[,;/&]" + _B + r"*)[*_`(\[]*" + _NEXT_LABEL)
_ADV = r"(?:(?:also|just|simply|clearly|obviously|definitely|certainly|actually|really|totally|completely|entirely|probably|likely|only|merely|quite|very|rather|still)" + _B + r"+)*"
_BE = r"(?:is|was|are|were|seems?|looks?|appears?|(?:would|should|could|can|may|might|must|will)" + _B + r"+be)"
_BAD = (r"(?:not|never|no|incorrect|wrong|false|untrue|invalid|inaccurate|mistaken|irrelevant|unlikely|impossible|eliminated|excluded"
        r"|ruled" + _B + r"+out|(?:a" + _B + r"+)?distractor|partially|partly|somewhat|mostly|half)")
_NEGATED = re.compile(_B + r"+(?:(?:isn|wasn|aren|weren|doesn|didn|don|won|couldn|wouldn|shouldn|can)['\u2019]t\b|cannot\b"
                      r"|(?:can|could|will|would|should|may|might|must|does|did|do|has|had)" + _B + r"+not\b"
                      r"|" + _BE + _B + r"+" + _ADV + _BAD + r"\b)", re.I)
_COPULAR = re.compile(_B + r"+(?:is|was|were|are|and|or|but|as|due|given|based|because|since|would|should|could|seems|looks|appears"
                      r"|must|has|can|may|might|will)\b")
_LIST_PUNCT = re.compile(r"[*_`]*([.):])")                       # the glued punctuation after a label: "A.", "A)", "A:", "**A.**"
_TEXT_AFTER = re.compile(_B + r"*(?:[*_`]+" + _B + r"*)?\S")      # something follows the label on the same line
_CLAUSE = r"(?:\n" + _B + r"*|[.!?;,]" + _B + r"+)"             # a clause start: a new line, or after sentence / clause punctuation
_MARKER = {".": re.compile(_CLAUSE + r"[*_`(\[{]*([A-J])[*_`]*\.(?![.A-Za-z0-9])"),
           ")": re.compile(_CLAUSE + r"[*_`(\[{]*([A-J])[*_`]*\)"),
           ":": re.compile(_CLAUSE + r"[*_`(\[{]*([A-J])[*_`]*:")}


def _list_item(text, pos, letter):
    """Is the label ending at text[pos] followed by option text and then by ANOTHER letter in the same label style ("A: ...  B: ...")?"""
    pm = _LIST_PUNCT.match(text, pos)
    if pm is None or _TEXT_AFTER.match(text, pm.end()) is None:
        return False
    return any(m.group(1) != letter for m in _MARKER[pm.group(1)].finditer(text, pm.end()))


def classify_lead(text, pos, letter):
    """Kind of the leading option label `letter` that ends at text[pos]: "bare" | "copular" | "negated" | "multi" | "list" (see above)."""
    if _MULTI.match(text, pos):
        return "multi"
    if _NEGATED.match(text, pos):
        return "negated"
    if _COPULAR.match(text, pos):
        return "copular"
    return "list" if _list_item(text, pos, letter) else "bare"


def question_text(q):
    return q["question"].strip() + "\n" + "\n".join(f"{L10[i]}. {c}" for i, c in enumerate(q["choices"]))


class Adapter:
    """Subclass per model family. Set class attrs, implement build() (and usually split()/load()).

    Conventions (do not change):
      * layer index L is 1-based: hooking `layers(model)[L-1]` and reading its output == hidden_states[L].
      * `render()` / `build()` return the FULL prompt string ending exactly where the assistant's generation starts
        (generation prompt included; for thinking models, include the opening think tag if the template does).
      * special tokens are already in the string -> encode() uses add_special_tokens=False.
      * tokenizer padding side is LEFT (the last position of every row is the last prompt token).
    """
    keys = []              # registry keys (lowercase, unique across adapters), e.g. ["qwen3-32b", "qwen3-8b"]
    repos = {}             # key -> HF repo id
    answer_mode = "direct"  # "direct" (letter only) | "reasoning" (hidden CoT channel; use split()) | "cot" (visible CoT)
    max_new = 8            # default max_new_tokens for this family/mode
    sys_role = "system"    # role name for framing text; None => framing text is prepended to the user turn
    effort = None          # default reasoning-effort knob if the family has one

    def __init__(self, key=None):
        self.key = key
        self.repo = self.repos.get(key)

    # ---- loading -------------------------------------------------------------------------------------------
    def load(self, repo=None, dtype="auto", device_map="auto", quant=None, **kw):
        """Return (tok, model). Override if the family needs trust_remote_code, a different Auto class, etc."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo = repo or self.repo
        tok = AutoTokenizer.from_pretrained(repo)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = dict(torch_dtype=dtype if dtype != "bf16" else torch.bfloat16, device_map=device_map)
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs.update(kw)
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs).eval()
        return tok, model

    # ---- structure -----------------------------------------------------------------------------------------
    def layers(self, model):
        return model.model.layers

    def num_layers(self, model):
        return len(self.layers(model))

    def hidden_size(self, model):
        cfg = model.config
        cfg = getattr(cfg, "text_config", cfg)
        return cfg.hidden_size

    # ---- prompts -------------------------------------------------------------------------------------------
    def user_text(self, q, user_note=None):
        head = (user_note.strip() + "\n\n") if user_note else ""
        return head + ASK[self.answer_mode] + "\n\n" + question_text(q)

    def chat_messages(self, user_text, sys_text=None):
        """Generic (non-MCQ) chat messages: used by recipes that contrast arbitrary prompts."""
        if sys_text and self.sys_role:
            return [{"role": self.sys_role, "content": sys_text}, {"role": "user", "content": user_text}]
        if sys_text:
            return [{"role": "user", "content": sys_text.strip() + "\n\n" + user_text}]
        return [{"role": "user", "content": user_text}]

    def messages(self, q, sys_text=None, user_note=None):
        return self.chat_messages(self.user_text(q, user_note), sys_text)

    def render(self, tok, msgs, effort=None):
        """messages -> full prompt string ending where the assistant starts (generation prompt included; for thinking
        models make sure the thinking mode is ON/consistent here). THE ONE METHOD EVERY FAMILY MUST IMPLEMENT."""
        raise NotImplementedError

    def build(self, tok, q, sys_text=None, user_note=None, effort=None):
        return self.render(tok, self.messages(q, sys_text, user_note), effort)

    def build_chat(self, tok, user_text, sys_text=None, effort=None):
        return self.render(tok, self.chat_messages(user_text, sys_text), effort)

    def encode(self, tok, texts):
        return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)

    # ---- outputs -------------------------------------------------------------------------------------------
    def split(self, text):
        """generated text -> (reasoning, final). Default: no separate reasoning."""
        return "", text

    def parse_letter(self, final, nchoices=4):
        """final answer text -> 0-based option index or None (unparsed).

        "cot": the LAST "Answer: X" / "answer is X" statement whose letter reads as a label (see `label_follows`), else the
        leading label. "direct"/"reasoning": the leading label only. A leading/answer letter that is the start of a sentence
        ("I'm sorry ...", "A study shows ...", "I cannot answer") is never an option, nor is the leading label of a text that
        starts with analysis ("A is not correct. ...", "A and B ...", "A: Ebola ... B: Influenza ..."; see `classify_lead`).
        Letters >= nchoices give None."""
        letter = None
        if self.answer_mode == "cot":
            for m in _ANS.finditer(final):
                if label_follows(final, m.end(1), m.group(1)):
                    letter = m.group(1)
        if letter is None:
            m = _FIRST.match(final)
            if m and label_follows(final, m.end(1), m.group(1)) and classify_lead(final, m.end(1), m.group(1)) in ("bare", "copular"):
                letter = m.group(1)             # not "A is not correct. ...", "A and B ...", "A: Ebola ... B: Influenza ..." (analysis)
        if letter is None:
            return None
        i = L10.index(letter)
        return i if i < nchoices else None
