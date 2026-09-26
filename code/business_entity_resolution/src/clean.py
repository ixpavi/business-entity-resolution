"""Stage 1: normalise every record into matching-ready text.

The same functions run on Source 1, 2 and 3, train and test, so any rewrite is
applied identically to both sides of a candidate pair. Nothing is dropped or
merged: every input row yields exactly one output row, the raw columns are kept
untouched, and the cleaned views are added alongside them.

Every rule below targets a noise pattern measured in the training data (see
work/reports/cleaning_report.md). Nothing branches on the value of `country`, so
France (absent from training) goes through exactly the same code as US/India.

Output columns (per record)
    entity_id, source, country, business_name, business_address   raw, untouched
    name_norm     full cleaned name, legal forms canonicalised ("pvt ltd")
    name_core     name_norm without legal forms, honorifics and stop words
    name_legal    sorted canonical legal forms found ("ltd pvt")
    name_alias    text before a d/b/a / aka / formerly / t/a / nee marker
    name_is_web   name was a bare domain or @handle / #hashtag
    name_had_indic, addr_had_indic   field was (partly) in an Indic script
    addr_norm     cleaned address, components kept in order, joined by ", "
    addr_numbers  numeric tokens of the address in order (leading zeros dropped)
    addr_state    canonical state code if one was recognised, else ""
    addr_pobox    PO box / PMB / BP token pulled out of the address, else ""

Usage:  python -m src.clean [--workers 8]      (from code/business_entity_resolution)
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.translit import INDIC_RE, romanise

# --------------------------------------------------------------------------- #
# Shared text folding
# --------------------------------------------------------------------------- #

# Letters NFKD does not decompose, plus typographic punctuation.
_SPECIAL = str.maketrans({
    "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "ð": "d", "þ": "th",
    "’": "'", "‘": "'", "`": "'", "´": "'", "–": "-", "—": "-", "“": '"', "”": '"',
})


def fold(text):
    """Romanise Indic scripts, strip accents, lower-case."""
    if not text.isascii():
        text = romanise(text).translate(_SPECIAL)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b\.?")  # p.l.l.c. -> pllc
_APOSTROPHE_RE = re.compile(r"'")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _join_dotted(m):
    return m.group().replace(".", "")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

# "<made-up name> d/b/a <real name>": the real name is always AFTER the marker
# (token overlap with the Source 1 name: 1.000 after vs 0.000 before, 121k pairs).
_WRAPPER_RE = re.compile(
    r"\s+(?:d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a"
    r"|formerly(?:\s+known\s+as)?|trading\s+as|n[eé]e)\s*:?\s+",
    re.IGNORECASE,
)
_PIPE_SUFFIX_RE = re.compile(r"\s*\|.*$")                         # "X | www.x.com"
_PHONE_RE = re.compile(r"\s*-?\s*\(?\+?\d{10,}\)?")                # "X - 8984535630"
_ID_RE = re.compile(r"\s*\(?\bid\s*[:#]\s*\d+\s*\)?", re.IGNORECASE)  # "X (ID: 95627)"
_LEADING_JUNK_RE = re.compile(r"^[^a-z0-9@#]+")                    # --, <<, >>, ", ~
_MS_PREFIX_RE = re.compile(r"^m\s*/\s*s\.?\s+")                    # "M/s Hr Power"
_WEB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9_.-]*?)"
    r"\.(?:com|net|org|co\.in|co|in|fr|io|biz|us|info)$"
)
_HANDLE_RE = re.compile(r"^[@#]([a-z0-9_.]+)$")

LEGAL_FORMS = {
    "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "pvte": "pvt", "prv": "pvt",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp", "plc": "plc",
    "pc": "pc", "opc": "opc",
    # French forms (test only): the same generator writes them in the same slots.
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "ei": "ei", "eirl": "eirl", "selarl": "selarl",
    "scop": "scop", "gie": "gie", "cie": "cie",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
# Romanised Indic spellings of legal forms, applied only to names that were in an
# Indic script (प्रा. लि. -> "pra li", प्राइवेट -> "praivet", প্রাইভেট -> "praibhet").
# A fallback: the learned lexicon below covers these and much more.
ROMANISED_LEGAL = {
    "praivet": "pvt", "praibhet": "pvt", "pra": "pvt", "prai": "pvt",
    "limitet": "ltd", "li": "ltd",
}

# Romanised-token -> English-token lexicon learned from training pairs by
# build_lexicon.py ("epeks" -> "apex", "elaelapi" -> "llp"). Applied only to names
# that were written in an Indic script. Missing file = no lexicon.
LEXICON_PATH = Path(__file__).with_name("resources") / "indic_lexicon.json"


def _load_lexicon():
    if not LEXICON_PATH.is_file():
        return {}
    with open(LEXICON_PATH, encoding="utf-8") as f:
        return json.load(f)["lexicon"]


LEXICON = _load_lexicon()
# Injected in front of names; never the first token of a Source 1 name.
HONORIFICS = {"the", "m", "ms", "mr", "mrs", "smt", "sri", "shri", "dr"}
CORE_STOPWORDS = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les"}

# OCR-style digit-for-letter swaps measured on matched pairs:
# 0->o 14.8k, 1->l 10.9k, 5->s 5.8k, 8->b 2.7k, 6->g 2.1k (all others < 20).
_OCR_DIGITS = str.maketrans("01568", "olsgb")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|hr)$")
_L_FOR_I_RE = re.compile(r"^l[bcdfgkmnprstvz]")  # "lnc", "lndia": capital I read as l


def _repair_token(tok):
    if not tok.isalpha():
        if tok.isdigit() or _ORDINAL_RE.match(tok):
            return tok
        letters = sum(c.isalpha() for c in tok)
        if letters < 2 or any(c.isdigit() and c not in "01568" for c in tok):
            return tok
        tok = tok.translate(_OCR_DIGITS)
    if (len(tok) >= 4 or tok == "lnc") and _L_FOR_I_RE.match(tok):
        tok = "i" + tok[1:]
    return tok


def clean_name(raw, lexicon=None):
    """Return (name_norm, name_core, name_legal, name_alias, is_web, had_indic).

    lexicon defaults to the learned LEXICON; build_lexicon passes {} to see the
    bare romanisation.
    """
    if lexicon is None:
        lexicon = LEXICON
    text = raw.strip()
    alias = ""
    parts = _WRAPPER_RE.split(text, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        alias, text = parts[0].strip(), parts[1]

    text = _PIPE_SUFFIX_RE.sub("", text)
    text = _ID_RE.sub("", text)
    text = _PHONE_RE.sub("", text)

    had_indic = INDIC_RE.search(text) is not None
    text = fold(text).strip()
    text = _LEADING_JUNK_RE.sub("", text)
    text = _MS_PREFIX_RE.sub("", text)

    is_web = False
    m = _WEB_RE.match(text) or _HANDLE_RE.match(text)
    if m:
        is_web = True
        text = m.group(1)

    text = _DOTTED_ACRONYM_RE.sub(_join_dotted, text)
    text = _APOSTROPHE_RE.sub("", text).replace("&", " and ")
    tokens = _NON_ALNUM_RE.sub(" ", text).split()

    norm, core, legal = [], [], set()
    for tok in tokens:
        tok = _repair_token(tok)
        if had_indic:
            tok = lexicon.get(tok, tok)
        if norm and norm[-1] == tok:  # "Estate Estate", "(Shop) (Shop)"
            continue
        canon = LEGAL_FORMS.get(tok)
        if canon is None and had_indic:
            canon = ROMANISED_LEGAL.get(tok)
        if canon is not None:
            norm.append(canon)
            legal.add(canon)
            continue
        norm.append(tok)
        if tok in CORE_STOPWORDS or (not core and tok in HONORIFICS):
            continue
        core.append(tok)

    name_norm = " ".join(norm)
    name_core = " ".join(core) or name_norm  # a name made only of legal words
    alias_norm = " ".join(_NON_ALNUM_RE.sub(" ", fold(alias)).split()) if alias else ""
    return name_norm, name_core, " ".join(sorted(legal)), alias_norm, is_web, had_indic


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #

PLACEHOLDERS = {"", "null", "<null>", "n/a", "na", "none", "nan", "nil", "-", "--"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
# The 16 Indian states in the data, with the codes Source 3 uses. Romanised forms
# are what translit.romanise() makes of the native-script state names that appear
# in Source 2/3 (e.g. महाराष्ट्र -> "maharashtr", পশ্চিমবঙ্গ -> "pashcimabangg").
INDIA_STATES = {
    "maharashtra": "mh", "maharashtr": "mh",
    "delhi": "dl", "dilli": "dl", "nct of delhi": "dl",
    "uttar pradesh": "up",
    "karnataka": "ka", "karnatak": "ka",
    "tamil nadu": "tn", "tamilnadu": "tn", "tamilnatu": "tn",
    "gujarat": "gj",
    "west bengal": "wb", "pashcimabangg": "wb",
    "telangana": "tg", "telangan": "tg",
    "haryana": "hr", "hariyana": "hr",
    "rajasthan": "rj",
    "kerala": "kl", "keralan": "kl",
    "bihar": "br",
    "madhya pradesh": "mp", "madhy pradesh": "mp",
    "andhra pradesh": "ap", "andhrapradesh": "ap",
    "punjab": "pb", "panjab": "pb",
    "odisha": "od", "orissa": "od",
}
STATE_NAMES = {**US_STATES, **INDIA_STATES}
STATE_CODES = set(US_STATES.values())
# Only codes Source 3 actually writes for India; "na", "ar" etc. would collide.
INDIA_CODES = {"mh", "dl", "up", "ka", "tn", "gj", "wb", "tg", "hr", "rj", "kl",
               "br", "mp", "ap", "pb", "od"}

# One canonical spelling per address word. Street types (US/India/France),
# directions, ordinals and unit words. Applied token by token.
ADDRESS_WORDS = {
    # street types
    "road": "rd", "rd": "rd", "rod": "rd",
    "street": "st", "st": "st", "str": "st", "saint": "st",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr", "driv": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "circle": "cir", "cir": "cir", "circ": "cir",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",
    "place": "pl", "pl": "pl", "plz": "plz", "plaza": "plz",
    "trail": "trl", "trl": "trl",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "highway": "hwy", "hwy": "hwy",
    "terrace": "ter", "ter": "ter", "terr": "ter",
    "square": "sq", "sq": "sq",
    "point": "pt", "pt": "pt",
    "mount": "mt", "mt": "mt",
    "heights": "hts", "hts": "hts",
    "expressway": "expy", "expy": "expy",
    "crossing": "xing", "xing": "xing",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "cove": "cv", "cv": "cv",
    "loop": "loop", "path": "path", "way": "way", "run": "run", "pass": "pass",
    "marg": "marg", "salai": "salai", "nagar": "nagar", "colony": "colony",
    "cross": "cross", "main": "main",
    # French street types
    "rue": "rue", "allee": "all", "all": "all", "aelee": "all",
    "impasse": "imp", "imp": "imp",
    "chemin": "ch", "ch": "ch", "chem": "ch",
    "route": "rte", "rte": "rte",
    "quai": "quai", "cours": "crs", "crs": "crs",
    "residence": "res", "res": "res", "lotissement": "lot",
    "sainte": "ste", "ste": "ste",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # ordinals written as words
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th",
    # units and buildings
    "apartment": "apt", "apt": "apt", "apts": "apt", "appt": "apt", "flat": "flat",
    "suite": "ste", "building": "bldg", "bldg": "bldg",
    "floor": "fl", "flr": "fl", "fl": "fl",
    "opposite": "opp", "opp": "opp", "near": "near", "nr": "near",
    # "city", "city of": pure filler ("CITY OF MENOMONIE", "NEWPORT NEWS CITY")
    "city": "", "of": "",
}
# House-number markers carry no information once the number itself is kept.
_NUMBER_MARKER_RE = re.compile(
    r"\b(?:h\s*\.?\s*no|house\s+no|d\s*\.?\s*no|door\s+no|dor\s+no|no|num|number)\b\.?"
    r"|\bn\s*[°º]|#"
)
_POBOX_RE = re.compile(
    r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s+box|pmb|bp|cs)\s*#?\s*(\d+)\b"
)
_ORDINAL_FIX_RE = re.compile(r"\b(\d+)\s+(st|nd|rd|th)\b")  # "2 nd" -> "2nd"
_R_STREET_RE = re.compile(r"\b(\d+[a-z]?\s+)r\b")         # French "63 r de dieppe"


def _norm_component(comp):
    comp = _NUMBER_MARKER_RE.sub(" ", comp)
    comp = _R_STREET_RE.sub(r"\1rue", comp)
    comp = _APOSTROPHE_RE.sub("", comp).replace("&", " and ")
    comp = _NON_ALNUM_RE.sub(" ", comp)
    comp = _ORDINAL_FIX_RE.sub(r"\1\2", comp)
    comp = " ".join(comp.split())
    state = STATE_NAMES.get(comp)
    if state is not None:
        return state, state
    if comp in STATE_CODES or comp in INDIA_CODES:
        return comp, comp
    out = []
    for tok in comp.split():
        tok = ADDRESS_WORDS.get(tok, tok)
        if tok.isdigit():
            tok = tok.lstrip("0") or "0"
        if tok:
            out.append(tok)
    return " ".join(out), ""


def clean_address(raw):
    """Return (addr_norm, addr_numbers, addr_state, addr_pobox, had_indic)."""
    had_indic = INDIC_RE.search(raw) is not None
    text = fold(raw)
    pobox = ""
    m = _POBOX_RE.search(text)
    if m:
        pobox = m.group(1).lstrip("0") or "0"
        text = text[: m.start()] + text[m.end():]

    comps, state = [], ""
    for comp in text.split(","):
        comp = comp.strip()
        if comp in PLACEHOLDERS:
            continue
        norm, comp_state = _norm_component(comp)
        if norm and norm not in PLACEHOLDERS and norm not in comps:  # "PB, Punjab"
            comps.append(norm)
            state = state or comp_state
    addr_norm = ", ".join(comps)
    numbers = " ".join(t for t in _NON_ALNUM_RE.sub(" ", addr_norm).split() if t.isdigit())
    return addr_norm, numbers, state, pobox, had_indic


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #

OUTPUT_SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("source", pa.string()), ("country", pa.string()),
    ("business_name", pa.string()), ("business_address", pa.string()),
    ("name_norm", pa.string()), ("name_core", pa.string()),
    ("name_legal", pa.string()), ("name_alias", pa.string()),
    ("name_is_web", pa.bool_()), ("name_had_indic", pa.bool_()),
    ("addr_norm", pa.string()), ("addr_numbers", pa.string()),
    ("addr_state", pa.string()), ("addr_pobox", pa.string()),
    ("addr_had_indic", pa.bool_()),
])


def clean_batch(batch):
    ids = batch.column("entity_id").to_pylist()
    names = batch.column("business_name").to_pylist()
    addrs = batch.column("business_address").to_pylist()
    countries = batch.column("country").to_pylist()
    name_cols = list(zip(*(clean_name(n) for n in names)))
    addr_cols = list(zip(*(clean_address(a) for a in addrs)))
    return pa.table(
        [
            ids, [i[:2] for i in ids], [c.strip() for c in countries], names, addrs,
            *name_cols, *addr_cols,
        ],
        schema=OUTPUT_SCHEMA,
    )


def _batches(path, rows):
    for rb in pq.ParquetFile(path).iter_batches(batch_size=rows):
        yield pa.Table.from_batches([rb])


def clean_file(src, dst, workers, rows=100_000):
    t0 = time.time()
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(dst, OUTPUT_SCHEMA, compression="zstd") as writer, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        for table in pool.map(clean_batch, _batches(src, rows)):
            writer.write_table(table)
            n += table.num_rows
    print(f"  {dst.name:<28} {n:>10,} rows  {time.time() - t0:6.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--only", help="clean one file, e.g. train_source1")
    args = parser.parse_args()
    for split in config.SPLITS:
        for source in config.SOURCES:
            if args.only and args.only != f"{split}_{source}":
                continue
            clean_file(config.raw_parquet(split, source), config.clean_parquet(split, source),
                       args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
