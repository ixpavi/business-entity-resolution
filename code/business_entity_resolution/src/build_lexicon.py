"""Learn a romanised-Indic -> English token lexicon from the training pairs.

About a quarter of Indian Source 2/3 names are Indic-script transliterations of an
English name ("एपेक्स कन्स्ट्रक्शन्स प्राइवेट लिमिटेड" = Apex Constructions Private
Limited). translit.romanise() turns them into Latin letters, but not into English
spelling ("epeks kanstrakshans praivet limited"). Training data has millions of
matched pairs, so the English spelling can be read off directly:

    for every true pair whose S2/S3 name was Indic-script
        romanise that name (no lexicon), clean the Source 1 name
        if both have the same number of tokens, pair them position by position
    keep r -> e when it is seen often enough and wins a clear majority

Positional pairing is safe here: 95% of these pairs have equal token counts, and
wherever a romanised token already equals a Source 1 token it sits at the same
position 100% of the time (47k sampled pairs). Similarity-based pairing was
tried first and is biased towards short tokens ("pavar" = power -> "pvt").

Uses only the provided training files. Writes src/resources/indic_lexicon.json,
which clean.py loads automatically.

Usage:  python -m src.build_lexicon        (after src.ingest)
"""

import json
import sys
import time
from collections import Counter, defaultdict

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src import config
from src.clean import LEXICON_PATH, clean_name
from src.translit import INDIC_RE

MIN_COUNT = 3    # aligned pairs supporting r -> e
MIN_SHARE = 0.6  # e's share of all alignments of r


def _indic_names(source):
    t = pq.read_table(config.raw_parquet("train", source), columns=["entity_id", "business_name"])
    mask = pc.match_substring_regex(t["business_name"], INDIC_RE.pattern)
    t = t.filter(mask)
    return dict(zip(t["entity_id"].to_pylist(), t["business_name"].to_pylist()))


def main():
    t0 = time.time()
    indic = {**_indic_names("source2"), **_indic_names("source3")}
    print(f"  Indic-script S2/S3 names: {len(indic):,}")

    gt = pq.read_table(config.RAW_DIR / "train_ground_truth.parquet").to_pandas()
    s1_of = {}
    for s1, ids in zip(gt.source1_entity_id, gt.matched_entity_ids):
        if ids:
            for m in ids.split(","):
                if m in indic:
                    s1_of[m] = s1
    need = set(s1_of.values())
    s1 = pq.read_table(config.raw_parquet("train", "source1"), columns=["entity_id", "business_name"])
    s1_tokens = {
        i: clean_name(n, lexicon={})[0].split()
        for i, n in zip(s1["entity_id"].to_pylist(), s1["business_name"].to_pylist())
        if i in need
    }
    print(f"  matched pairs to align: {len(s1_of):,}  ({time.time() - t0:.0f}s)")

    votes = defaultdict(Counter)
    pairs = []
    for m, s1_id in s1_of.items():
        roman = clean_name(indic[m], lexicon={})[0].split()
        english = s1_tokens[s1_id]
        pairs.append((roman, english))
        if len(roman) == len(english):
            for r, e in zip(roman, english):
                if r != e:
                    votes[r][e] += 1

    lexicon, rows = {}, []
    for r, cnt in votes.items():
        e, n = cnt.most_common(1)[0]
        total = sum(cnt.values())
        if n >= MIN_COUNT and n / total >= MIN_SHARE:
            lexicon[r] = e
            rows.append((r, e, n, round(n / total, 3)))

    # Coverage over ALL pairs (equal length or not): share of romanised tokens
    # that equal some token of the matching Source 1 name.
    total = before = after = 0
    for roman, english in pairs:
        eng = set(english)
        total += len(roman)
        before += sum(r in eng for r in roman)
        after += sum(lexicon.get(r, r) in eng for r in roman)
    print(f"  romanised tokens equal to an S1 token, no lexicon:  {before / total:.1%}")
    print(f"  ... with the learned lexicon:                       {after / total:.1%}")
    print(f"  lexicon entries: {len(lexicon):,}  ({time.time() - t0:.0f}s)")

    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda x: -x[2])
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "about": "romanised Indic token -> English token, learned from "
                         "train_ground_truth.tsv pairs by src/build_lexicon.py",
                "params": {"min_count": MIN_COUNT, "min_share": MIN_SHARE},
                "lexicon": dict(sorted(lexicon.items())),
                "top_entries": rows[:200],
            },
            f, ensure_ascii=False, indent=1,
        )
    print("  top entries:", rows[:25])
    return 0


if __name__ == "__main__":
    sys.exit(main())
