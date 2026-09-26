"""Rule-based romanisation of Indic scripts, built only from the Unicode character
database that ships with Python (unicodedata). No external data or libraries.

Source 2/3 write about a quarter of Indian business names (and many state names)
in Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada or
Malayalam. All nine scripts share one abugida layout, and their Unicode names
follow the same pattern ("<SCRIPT> LETTER KA", "<SCRIPT> VOWEL SIGN AA",
"<SCRIPT> SIGN VIRAMA"), so one table-free decoder romanises all of them:

    consonant  -> its sound plus an inherent 'a'
    vowel sign -> replaces the inherent 'a'
    virama     -> removes the inherent 'a'
    anusvara / candrabindu / tippi / bindi -> 'n',  visarga -> 'h'

The output is a loose, lower-case romanisation ("अर्बन डेवलपर्स" -> "arban
devalapars"). It will not reproduce English spelling exactly; downstream
similarity measures (character n-grams, phonetic keys) absorb the remainder.
"""

import re
import unicodedata
from functools import lru_cache

INDIC_SCRIPTS = (
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
    "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
)
# U+0900..U+0D7F covers all nine scripts in one contiguous range.
INDIC_RE = re.compile("[ऀ-ൿ]")
# A run also absorbs zero-width (non-)joiners, which only steer glyph shaping.
_INDIC_RUN_RE = re.compile("[ऀ-ൿ‌‍]+")
_LONG_VOWEL_RE = re.compile(r"([aiu])\1")

# Vowel names as Unicode spells them -> romanisation.
_VOWELS = {
    "A": "a", "AA": "aa", "I": "i", "II": "ii", "U": "u", "UU": "uu",
    "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li", "VOCALIC LL": "li",
    "E": "e", "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au",
    "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o",
    "CANDRA A": "a",
}
# Consonant names whose spelling is not simply "<sound>A".
_CONSONANT_FIX = {
    "NNA": "n", "NGA": "ng", "NYA": "ny", "TTA": "t", "TTHA": "th", "DDA": "d",
    "DDHA": "dh", "LLA": "l", "LLLA": "l", "RRA": "r", "RRRA": "r", "NNNA": "n",
    "SSA": "sh", "SHA": "sh", "YYA": "y", "FA": "f", "ZA": "z", "QA": "q",
    "KHHA": "kh", "GHHA": "gh", "DDDHA": "d", "RHA": "rh", "VA": "v", "WA": "v",
}
_NASALS = ("SIGN ANUSVARA", "SIGN CANDRABINDU", "SIGN BINDI", "TIPPI")


@lru_cache(maxsize=None)
def _classify(ch):
    """Return (kind, roman) for one character.

    kind is one of: 'cons', 'cons_dead', 'vowel', 'sign', 'virama', 'mark'
    (anything else inside the Indic block: nasal/visarga signs, digits, danda,
    nukta, length marks).
    """
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "mark", ""
    script, _, rest = name.partition(" ")
    if script not in INDIC_SCRIPTS:
        return "mark", ch  # zero-width joiners and anything unexpected
    if rest.startswith("LETTER CHILLU "):  # Malayalam dead consonants
        return "cons_dead", rest[len("LETTER CHILLU "):].lower()
    if rest.startswith("LETTER "):
        body = rest[len("LETTER "):]
        if body in _VOWELS:
            return "vowel", _VOWELS[body]
        if body in _CONSONANT_FIX:
            return "cons", _CONSONANT_FIX[body]
        if body.endswith("A"):
            return "cons", body[:-1].lower()
        return "mark", ""
    if rest.startswith("VOWEL SIGN "):
        return "sign", _VOWELS.get(rest[len("VOWEL SIGN "):], "")
    if rest in ("SIGN VIRAMA", "SIGN HALANT"):
        return "virama", ""
    if rest in _NASALS:
        return "mark", "n"
    if rest == "SIGN VISARGA":
        return "mark", "h"
    if rest.startswith("DIGIT "):
        return "mark", str(unicodedata.digit(ch))
    if rest in ("DANDA", "DOUBLE DANDA"):
        return "mark", " "
    return "mark", ""  # nukta, addak, avagraha, length marks: silent


def _romanise_run(run):
    out = []
    pending_a = False  # a consonant is waiting for its inherent vowel
    for ch in run:
        kind, rom = _classify(ch)
        if kind == "cons":
            if pending_a:
                out.append("a")
            out.append(rom)
            pending_a = True
        elif kind == "cons_dead":
            if pending_a:
                out.append("a")
            out.append(rom)
            pending_a = False
        elif kind == "sign":
            out.append(rom)
            pending_a = False
        elif kind == "virama":
            pending_a = False
        elif kind == "vowel":
            if pending_a:
                out.append("a")
            out.append(rom)
            pending_a = False
        elif rom:  # nasal, visarga, digit, danda: the inherent vowel comes first
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(rom)
        # silent marks leave pending_a untouched
    # A consonant still pending at the end of the word keeps no vowel. This is
    # Hindi-style final-schwa deletion; Dravidian scripts spell final consonants
    # with an explicit virama, so the rule is harmless there.
    #
    # Long vowels are collapsed (aa -> a) so "praaivet" reads "praivet", closer
    # to the English spelling these names were transliterated from.
    return _LONG_VOWEL_RE.sub(r"\1", "".join(out))


def has_indic(text):
    return INDIC_RE.search(text) is not None


def romanise(text):
    """Replace every run of Indic characters in text with a Latin romanisation."""
    if INDIC_RE.search(text) is None:
        return text
    return _INDIC_RUN_RE.sub(lambda m: _romanise_run(m.group()), text)
