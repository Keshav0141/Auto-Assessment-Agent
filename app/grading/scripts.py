"""Writing-script detection for model feedback.

Used to catch feedback that drifts into an unrelated script, such as a Japanese
fragment inside Telugu feedback. Only letters and combining marks are classified;
digits, punctuation and symbols (including the danda shared by several Indic
scripts) are ignored.
"""
import unicodedata
from typing import Iterable, List, Optional, Set

# (script, code point ranges). Latin covers ASCII letters and the Latin-1 / Extended
# blocks; Greek is listed so that symbols like pi or Delta can be allowed.
SCRIPT_RANGES = (
    ("Latin", ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF))),
    ("Greek", ((0x370, 0x3FF),)),
    ("Cyrillic", ((0x400, 0x4FF),)),
    ("Hebrew", ((0x590, 0x5FF),)),
    ("Arabic", ((0x600, 0x6FF), (0x750, 0x77F))),
    ("Devanagari", ((0x900, 0x97F), (0xA8E0, 0xA8FF))),
    ("Bengali", ((0x980, 0x9FF),)),
    ("Gurmukhi", ((0xA00, 0xA7F),)),
    ("Gujarati", ((0xA80, 0xAFF),)),
    ("Odia", ((0xB00, 0xB7F),)),
    ("Tamil", ((0xB80, 0xBFF),)),
    ("Telugu", ((0xC00, 0xC7F),)),
    ("Kannada", ((0xC80, 0xCFF),)),
    ("Malayalam", ((0xD00, 0xD7F),)),
    ("Sinhala", ((0xD80, 0xDFF),)),
    ("Thai", ((0xE00, 0xE7F),)),
    ("Hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("Japanese kana", ((0x3040, 0x30FF),)),
    ("Han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF))),
)

# Script each feedback language is written in.
LANGUAGE_SCRIPTS = {
    "English": "Latin", "Hinglish": "Latin", "Hindi": "Devanagari", "Marathi": "Devanagari",
    "Tamil": "Tamil", "Telugu": "Telugu", "Bengali": "Bengali", "Assamese": "Bengali",
    "Kannada": "Kannada", "Malayalam": "Malayalam", "Gujarati": "Gujarati", "Punjabi": "Gurmukhi",
    "Odia": "Odia", "Urdu": "Arabic",
}

# Always acceptable in feedback: English terms and Greek symbols in formulas.
ALWAYS_ALLOWED = frozenset({"Latin", "Greek"})


def _script_of(char: str) -> Optional[str]:
    cp = ord(char)
    for name, ranges in SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def scripts_in(text: str) -> Set[str]:
    """Return the scripts of the letters and combining marks in `text`.

    Args:
        text: Any text.

    Returns:
        Script names from SCRIPT_RANGES; characters outside them are ignored.
    """
    found = set()
    for char in text or "":
        if ord(char) < 0x41 or not unicodedata.category(char).startswith(("L", "M")):
            continue
        script = _script_of(char)
        if script:
            found.add(script)
    return found


def foreign_scripts(text: str, language: Optional[str], context: Iterable[str] = ()) -> List[str]:
    """Find scripts in feedback that belong neither to Latin nor to the feedback language.

    Args:
        text: Feedback and justifications.
        language: The feedback language ("Telugu", "Hinglish", ...).
        context: Texts whose scripts are also acceptable (question, rubric, model
            answer and the student's answer, which feedback may quote).

    Returns:
        The foreign script names, sorted; empty when the language's script is unknown.
    """
    target = LANGUAGE_SCRIPTS.get(language or "English")
    if target is None:
        return []
    allowed = set(ALWAYS_ALLOWED) | {target}
    for extra in context:
        allowed |= scripts_in(extra)
    return sorted(scripts_in(text) - allowed)
