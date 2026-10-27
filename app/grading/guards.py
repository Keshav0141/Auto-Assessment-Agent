"""Cheap, deterministic pre-checks run on every student answer before any LLM call.

They catch the obvious cases (blank answers, instruction-like text aimed at the
grader, a restated question, keyboard-mash gibberish, OCR [illegible] markers, a
different script from the question) without spending a model call, and they feed
the same `integrity_flags` the graders can report. A flag never changes marks by
itself: it forces human review and caps the combined confidence at CONFIDENCE_CAP.
The only text change is that instruction-like sentences are removed before grading,
so they cannot influence the mark.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional

INTEGRITY_FLAGS = (
    "prompt_injection",
    "blank",
    "off_topic",
    "restates_question",
    "gibberish",
    "language_mismatch",
    "illegible",
)

MAX_ANSWER_CHARS = 6000
CONFIDENCE_CAP = 0.5

FLAG_REASONS = {
    "prompt_injection": "The answer contains text that tries to instruct the grader.",
    "blank": "No answer was found for this question.",
    "off_topic": "The answer does not address the question.",
    "restates_question": "The answer mostly repeats the question instead of answering it.",
    "gibberish": "The answer is mostly random or unreadable text.",
    "language_mismatch": "The answer is not in the language the rubric requires.",
    "illegible": "Parts of the answer were marked illegible during OCR.",
}

# --- prompt injection -------------------------------------------------------

_INJECTION_PATTERNS = [
    # "ignore previous instructions", "disregard all prior rules", ...
    r"(?i)\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
    r"\b(?:previous|prior|above|earlier|preceding|all|any|your|the|these|those)\b[^.\n]{0,25}?"
    r"\b(?:instructions?|prompts?|rules?|rubric|directions?|guidelines?)\b",
    # "award full marks", "give this answer maximum marks", "grant 100% credit"
    r"(?i)\b(?:award|give|grant|assign|mark)\b[^.\n]{0,30}?"
    r"\b(?:full|maximum|max|perfect|all(?:\s+the)?|100\s*%?|top)\s+(?:marks?|points?|score|credit|grade)\b",
    # "give me 5/5", "award this answer 10 out of 10"
    r"(?i)\b(?:give|award|grant|score)\s+(?:me|this(?:\s+answer)?|the\s+student|it)\s+"
    r"\d+(?:\.\d+)?\s*(?:/|out\s+of)\s*\d+",
    r"(?i)\byou\s+are\s+now\b",
    r"(?i)\b(?:pretend|act)\s+(?:to\s+be|as)\s+(?:an?\s+|the\s+)?(?:\w+\s+)?(?:grader|examiner|teacher|marker|assistant)\b",
    r"(?i)\bnew\s+instructions?\s*:",
    # role labels: "system:" at a line start / after a bracket, "SYSTEM:" anywhere,
    # chat-template tokens
    r"(?im)(?:^|[\]>}|])\s*(?:system|assistant|developer)\s*:",
    r"\bSYSTEM\s*:",
    r"(?i)<\|?\s*(?:system|im_start|im_end)\s*\|?>",
    # imitated delimiters / prompt markup: lines starting with <<< or >>> or END_,
    # </answer>-style tags, ```system fences, [INST] / <<SYS>>, "### Instruction"
    # headings, "user:" / "Grader note:" role lines
    r"(?m)^[ \t]*(?:<{3,}|>{3,})",
    r"(?m)^[ \t]*END_\w*",
    r"(?i)</?\s*(?:answer|student_answer|response|system|user|assistant|instructions?|prompt)\s*>",
    r"(?im)^[ \t]*`{3,}[ \t]*(?:system|assistant|user|instructions?|prompt)\b",
    r"(?i)\[\s*/?\s*INST\s*\]|<<\s*/?\s*SYS\s*>>",
    r"(?im)^[ \t]*#{2,}[ \t]*(?:instructions?|system(?:\s+prompt)?|response|new\s+task|grader|examiner)\b",
    r"(?im)^[ \t]*(?:user|human|grader|examiner|marker)(?:'s)?[ \t]*(?:note|notes|instructions?|message)?[ \t]*:",
    # Hindi: "ignore the instructions", "give full marks", "you are now (a lenient examiner)"
    r"निर्देशों?\s*को\s*(?:अनदेखा|नज़रअंदाज़|नजरअंदाज|नज़रअंदाज|अनसुना)\s*(?:करें|करो|कीजिए|कीजिये|कर\s*दें)",
    r"(?:पूरे|पूर्ण|सभी|पूरा)\s*(?:के\s*पूरे\s*)?(?:अंक|नंबर|मार्क्स)\s*(?:दें|दो|दीजिए|दीजिये|दे\s*दें)",
    r"अब\s*(?:आप|तुम)\s*(?:एक\s*)?[^।\n]{0,25}(?:परीक्षक|ग्रेडर|शिक्षक|सहायक)\s*(?:हैं|हो)",
]
_INJECTION_RES = [re.compile(p) for p in _INJECTION_PATTERNS]
_SENTENCE_END = ".!?।\n"
REDACTION = "[instruction-like text removed]"


def find_injections(text: str):
    """Return [(start, end, matched_text)] for every instruction-like phrase."""
    hits = []
    for rx in _INJECTION_RES:
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), m.group(0).strip()))
    return sorted(hits)


def redact_injections(text: str):
    """Replace each sentence that contains an injection phrase with REDACTION.
    Returns (redacted_text, matched_phrases)."""
    hits = find_injections(text)
    if not hits:
        return text, []
    spans = []
    for start, end, _ in hits:
        s = start
        while s > 0 and text[s - 1] not in _SENTENCE_END:
            s -= 1
        while s < start and text[s].isspace():
            s += 1  # keep the whitespace before the removed sentence
        e = end
        while e < len(text) and text[e] not in _SENTENCE_END:
            e += 1
        if e < len(text) and text[e] != "\n":
            e += 1  # swallow the sentence terminator
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    out, pos = [], 0
    for s, e in spans:
        out.append(text[pos:s])
        out.append(REDACTION)
        pos = e
    out.append(text[pos:])
    return "".join(out), [h[2] for h in hits]


# --- blank / restated / gibberish / script ---------------------------------

_WORD_RE = re.compile(r"[\wऀ-ॿ]+")
_PLACEHOLDERS = {
    "n/a", "na", "no answer", "not attempted", "not answered", "skip", "skipped",
    "idk", "i don't know", "i dont know", "i do not know", "don't know", "dont know",
}
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are",
    "was", "were", "be", "it", "its", "this", "that", "with", "by", "as", "from",
    "what", "which", "how", "why", "give", "state", "find", "your", "own", "words",
    "explain", "describe", "define", "calculate", "has", "have", "if", "we", "question",
    "asks", "asked", "about", "want", "need", "here",
}
_KEYBOARD_RUNS = ("qwer", "wert", "asdf", "sdfg", "dfgh", "fghj", "ghjk", "hjkl",
                  "zxcv", "xcvb", "cvbn", "vbnm", "uiop", "yuio", "poiu", "lkjh", "mnbv")
_ILLEGIBLE_RE = re.compile(r"(?i)\[\s*illegible\s*\]")


def is_blank(text: str) -> bool:
    t = _ILLEGIBLE_RE.sub(" ", text or "").strip().lower()
    if not _WORD_RE.search(t):
        return True
    return t.strip(" .!-") in _PLACEHOLDERS


def _content_tokens(text: str):
    return [t for t in _WORD_RE.findall((text or "").lower()) if len(t) > 1 and t not in _STOPWORDS]


def restatement_score(answer: str, question: str) -> float:
    """0-1 similarity of the answer to the question text. High when the answer adds
    almost nothing new AND covers most of the question, or is near-verbatim."""
    a = _content_tokens(answer)
    q = set(_content_tokens(question))
    if len(a) < 4 or not q:
        return 0.0
    novel_frac = sum(1 for t in a if t not in q) / len(a)
    q_coverage = len(q & set(a)) / len(q)
    norm = lambda s: " ".join(_WORD_RE.findall(s.lower()))  # noqa: E731
    verbatim = SequenceMatcher(None, norm(answer), norm(question)).ratio()
    overlap = min(1.0 - novel_frac, q_coverage / 0.75)
    return float(max(verbatim, min(1.0, overlap)))


def _gibberish_word(w: str) -> bool:
    w = w.lower()
    vowels = sum(c in "aeiouy" for c in w)
    if len(w) >= 4 and vowels == 0:
        return True
    if len(w) >= 6 and vowels / len(w) < 0.2:
        return True
    if re.search(r"(.)\1{3,}", w) or re.search(r"[^aeiouy]{6,}", w):
        return True
    return any(run in w for run in _KEYBOARD_RUNS)


def gibberish_ratio(text: str) -> float:
    """Share of the answer that looks like keyboard mash / random symbols (0-1).
    Formulas, numbers and Devanagari words are treated as meaningful."""
    chars = re.sub(r"\s", "", text or "")
    if not chars:
        return 0.0
    meaningful = sum(1 for c in chars if c.isalnum() or "ऀ" <= c <= "ॿ" or c in "=+-*/^().,;:%'\"?!<>[]{}|_")
    if len(chars) >= 10 and meaningful / len(chars) < 0.5:
        return 1.0 - meaningful / len(chars)
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 3]
    if len(words) < 3:
        return 0.0
    return sum(_gibberish_word(w) for w in words) / len(words)


# --- answer language (informational) ----------------------------------------
# Detected from the script; Devanagari is reported as Hindi. Answering in another
# language is legitimate, so this is never a flag unless the rubric requires one.

_SCRIPTS = (
    ("English", "A-Za-z"),
    ("Hindi", "ऀ-ॿ"),
    ("Bengali", "ঀ-৿"),
    ("Punjabi", "਀-੿"),
    ("Gujarati", "઀-૿"),
    ("Odia", "଀-୿"),
    ("Tamil", "஀-௿"),
    ("Telugu", "ఀ-౿"),
    ("Kannada", "ಀ-೿"),
    ("Malayalam", "ഀ-ൿ"),
    ("Urdu", "؀-ۿ"),
)
_LANG_ALIASES = {"bangla": "Bengali", "marathi": "Hindi", "sanskrit": "Hindi", "nepali": "Hindi"}
_HINGLISH_WORDS = {
    "hai", "hain", "hota", "hoti", "hote", "ka", "ki", "ke", "aur", "nahi", "nahin", "mein",
    "se", "ko", "kya", "jo", "toh", "bhi", "ek", "yeh", "ye", "woh", "wo", "kaise", "liye",
    "matlab", "jab", "tab", "par", "agar", "lekin", "sabse", "hum", "tum", "aap",
}
_LANG_NAMES = [name.lower() for name, _ in _SCRIPTS] + ["hinglish"] + list(_LANG_ALIASES)
_REQUIRED_LANG_RE = re.compile(
    r"(?i)\b(?:in|written\s+in|answer(?:ed|ing|s)?\s+in|using|use)\s+(?:the\s+)?(" + "|".join(_LANG_NAMES) + r")\b"
)


def detect_language(text: str):
    """Best-effort answer language: 'English', 'Hindi', 'Hinglish', 'Tamil', ...,
    or 'Hindi + English' when a second script makes up 20%+ of the letters.
    None when there is too little text to tell."""
    counts = [(len(re.findall(f"[{rng}]", text or "")), name) for name, rng in _SCRIPTS]
    counts.sort(reverse=True)
    total = sum(c for c, _ in counts)
    if total < 10:
        return None
    (n1, lang), (n2, second) = counts[0], counts[1]
    if lang == "English":
        words = re.findall(r"[a-z]+", text.lower())
        hits = sum(1 for w in words if w in _HINGLISH_WORDS)
        if words and hits >= 2 and hits / len(words) >= 0.15:
            lang = "Hinglish"
    if n2 >= 0.2 * total:
        return f"{lang} + {second}"
    return lang


def required_language(rubric_text: str):
    """The language the rubric explicitly requires ('in English', 'answer in Hindi'),
    or None."""
    m = _REQUIRED_LANG_RE.search(rubric_text or "")
    if not m:
        return None
    name = m.group(1).lower()
    return _LANG_ALIASES.get(name, "Hinglish" if name == "hinglish" else name.capitalize())


# --- entry point ------------------------------------------------------------

@dataclass
class GuardReport:
    text: str                       # text to grade (injections redacted, truncated)
    blank: bool = False
    flags: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    language: Optional[str] = None  # informational, never a flag by itself

    def add(self, flag, reason):
        if flag not in self.flags:
            self.flags.append(flag)
            self.reasons.append(reason)


def check_answer(answer: str, question: str = "", rubric_text: str = "") -> GuardReport:
    answer = answer or ""
    report = GuardReport(text=answer)
    illegible = len(_ILLEGIBLE_RE.findall(answer))
    if is_blank(answer):
        report.blank = True
        report.text = ""
        report.add("blank", FLAG_REASONS["blank"] + " It was given 0 marks without calling the grader.")
        if illegible:
            report.add("illegible", "The answer only contains [illegible] OCR markers.")
        return report

    text, phrases = redact_injections(answer)
    if phrases:
        quoted = "; ".join(f"'{p[:60]}'" for p in list(dict.fromkeys(phrases))[:3])
        report.add("prompt_injection",
                   f"Instruction-like text aimed at the grader was found and removed before grading: {quoted}.")
        report.text = text
    if illegible:
        report.add("illegible", f"{illegible} part(s) of the answer were marked [illegible] by OCR.")
    score = restatement_score(answer, question)
    if score >= 0.8:
        report.add("restates_question",
                   f"The answer is {score:.0%} similar to the question text and adds little of its own.")
    ratio = gibberish_ratio(answer)
    if ratio >= 0.5:
        report.add("gibberish", f"{ratio:.0%} of the answer looks like random characters.")
    report.language = detect_language(answer)
    required = required_language(rubric_text)
    if required and report.language:
        primary = report.language.split(" + ")[0]
        if primary != required:
            report.add("language_mismatch",
                       f"The rubric requires an answer in {required}, but it is written in {report.language}.")

    if len(report.text) > MAX_ANSWER_CHARS:
        report.warnings.append(
            f"the answer is {len(report.text)} characters long; only the first "
            f"{MAX_ANSWER_CHARS} were graded."
        )
        report.text = report.text[:MAX_ANSWER_CHARS] + "\n[... answer truncated ...]"
    return report


def parse_flags(value) -> List[str]:
    """Normalise a model-reported integrity_flags value to whitelisted flags."""
    if isinstance(value, str):
        value = re.split(r"[,;]", value)
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for v in value:
        f = re.sub(r"[\s-]+", "_", str(v).strip().lower())
        if f in INTEGRITY_FLAGS and f not in out:
            out.append(f)
    return out


def merge_flags(report: GuardReport, gradings) -> tuple:
    """Combine guard flags with flags the graders reported.

    Guard flags always count. A model-reported flag counts when at least half of the
    gradings (primary samples + second grader + referee, whichever ran) report it,
    so one noisy sample at temperature > 0 cannot force review on its own.
    Returns (flags, reasons).
    """
    flags, reasons = list(report.flags), list(report.reasons)
    gradings = [g for g in gradings if g is not None]
    if not gradings:
        return flags, reasons
    votes = {}
    for g in gradings:
        for f in set(g.integrity_flags or []):
            votes[f] = votes.get(f, 0) + 1
    for f in INTEGRITY_FLAGS:
        n = votes.get(f, 0)
        if f in flags or not n or 2 * n < len(gradings):
            continue
        if f in ("blank", "language_mismatch"):
            # The guard decides these: a non-empty answer is graded, and another
            # language is only a problem when the rubric requires one.
            continue
        flags.append(f)
        reasons.append(f"{FLAG_REASONS[f]} (reported by {n} of {len(gradings)} gradings)")
    return flags, reasons
