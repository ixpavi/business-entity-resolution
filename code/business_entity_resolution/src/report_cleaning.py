"""Check the cleaned data and measure what cleaning did, on training labels.

1. Integrity: every cleaned file has exactly the raw rows, in the same order.
2. Leftovers: empty cleaned names, placeholder/Indic text surviving cleaning.
3. Effect: for true pairs and for hard non-pairs, name/address similarity on the
   raw text (lower-cased) vs the cleaned text. Hard non-pairs are true matches of a
   *different* Source 1 entity whose name starts with the same core word, in the
   same country ("apex constructions" vs the matches of "apex foods"). Cleaning is
   only useful if it raises similarity for true pairs more than for these.
   Separation is summarised as ROC-AUC (probability a true pair outscores a hard
   non-pair).

Writes work/reports/cleaning_report.md.

Usage:  python -m src.report_cleaning        (after src.clean)
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src import config

N_ENTITIES = 60_000
SEED = 13


def sims_with(scorer, left, right):
    return np.fromiter((scorer(x, y) for x, y in zip(left, right)), float, len(left))


def auc(pos, neg):
    """ROC-AUC via the rank-sum formula (ties count half)."""
    scores = np.concatenate([pos, neg])
    ranks = pd.Series(scores).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def integrity(lines):
    lines.append("## 1. Integrity\n")
    lines.append("| file | raw rows | clean rows | ids identical, same order |")
    lines.append("|---|---:|---:|---|")
    ok = True
    for split in config.SPLITS:
        for source in config.SOURCES:
            raw = pq.read_table(config.raw_parquet(split, source), columns=["entity_id"])["entity_id"]
            cln = pq.read_table(config.clean_parquet(split, source), columns=["entity_id"])["entity_id"]
            same = raw.num_chunks >= 0 and len(raw) == len(cln) and pc.all(pc.equal(raw, cln)).as_py()
            ok &= bool(same)
            lines.append(f"| {split}_{source} | {len(raw):,} | {len(cln):,} | {'yes' if same else '**NO**'} |")
    lines.append("")
    return ok


def leftovers(lines):
    lines.append("## 2. Leftovers after cleaning\n")
    lines.append("| file | empty name_norm | empty addr_norm (raw empty) | Indic chars left | "
                 "name_is_web | d/b/a alias split | state recognised |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split in config.SPLITS:
        for source in config.SOURCES:
            t = pq.read_table(config.clean_parquet(split, source)).to_pandas()
            raw_empty = (t.business_address.str.strip() == "").mean()
            indic_left = (t.name_norm.str.contains("[ऀ-ൿ]") |
                          t.addr_norm.str.contains("[ऀ-ൿ]")).mean()
            lines.append(
                f"| {split}_{source} | {(t.name_norm == '').mean():.4%} | "
                f"{(t.addr_norm == '').mean():.2%} ({raw_empty:.2%}) | {indic_left:.4%} | "
                f"{t.name_is_web.mean():.2%} | {(t.name_alias != '').mean():.2%} | "
                f"{(t.addr_state != '').mean():.1%} |"
            )
    lines.append("")


def pair_effect(lines):
    gt = pq.read_table(config.RAW_DIR / "train_ground_truth.parquet").to_pandas()
    gt = gt[gt.matched_entity_ids != ""].sample(N_ENTITIES, random_state=SEED)
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
    cols = ["entity_id", "country", "business_name", "business_address", "name_norm",
            "name_core", "addr_norm", "name_had_indic", "name_is_web", "name_alias"]
    s1 = pq.read_table(config.clean_parquet("train", "source1"), columns=cols).to_pandas().set_index("entity_id")
    other = pd.concat(
        pq.read_table(config.clean_parquet("train", s), columns=cols).to_pandas()
        for s in ("source2", "source3")
    ).set_index("entity_id")

    # Hard non-pairs: shuffle match lists among S1 entities that share country and
    # first core word, keeping only reassignments to a different entity.
    a = s1.loc[pairs.source1_entity_id].reset_index()
    b = other.loc[pairs.m].reset_index()
    key = a.country + "|" + a.name_core.str.split().str[0].fillna("")
    rng = np.random.default_rng(SEED)
    perm = np.arange(len(a))
    for idx in pd.Series(np.arange(len(a))).groupby(key.values).indices.values():
        perm[idx] = rng.permutation(idx)
    neg_ok = (a.entity_id.values != a.entity_id.values[perm]) & (perm != np.arange(len(a)))
    # a non-pair must not be a true pair either
    neg_ok &= pairs.source1_entity_id.values[perm] != pairs.source1_entity_id.values

    def sims(left, right):
        return sims_with(fuzz.token_set_ratio, left, right)

    lines.append("## 3. Effect on matched vs hard non-matched pairs (train)\n")
    lines.append(f"{len(a):,} true pairs from {N_ENTITIES:,} Source 1 entities; "
                 f"{neg_ok.sum():,} hard non-pairs (same country, same first core word).\n")
    lines.append("Similarity = rapidfuzz `token_set_ratio` (0-100). Raw = lower-cased raw text.\n")
    lines.append("| field | segment | true pairs: mean raw -> clean | exact equal raw -> clean | "
                 "non-pairs: mean raw -> clean | AUC raw -> clean |")
    lines.append("|---|---|---|---|---|---|")

    raw_nm = sims(a.business_name.str.lower(), b.business_name.str.lower())
    cln_nm = sims(a.name_core, b.name_core)
    raw_ad = sims(a.business_address.str.lower(), b.business_address.str.lower())
    cln_ad = sims(a.addr_norm, b.addr_norm)
    b_neg = b.iloc[perm]
    raw_nm_n = sims(a.business_name.str.lower()[neg_ok], b_neg.business_name.str.lower()[neg_ok])
    cln_nm_n = sims(a.name_core[neg_ok], b_neg.name_core[neg_ok])
    raw_ad_n = sims(a.business_address.str.lower()[neg_ok], b_neg.business_address.str.lower()[neg_ok])
    cln_ad_n = sims(a.addr_norm[neg_ok], b_neg.addr_norm[neg_ok])
    eq_raw_nm = (a.business_name.str.lower().values == b.business_name.str.lower().values)
    eq_cln_nm = (a.name_core.values == b.name_core.values)
    eq_raw_ad = (a.business_address.str.lower().values == b.business_address.str.lower().values)
    eq_cln_ad = (a.addr_norm.values == b.addr_norm.values)

    segments = {
        "all": np.ones(len(a), bool),
        "India": (a.country == "India").to_numpy(),
        "US": (a.country == "US").to_numpy(),
        "Indic-script name": b.name_had_indic.to_numpy(),
        "web/handle name": b.name_is_web.to_numpy(),
        "d/b/a-wrapped name": (b.name_alias != "").to_numpy(),
    }
    for field, r, c, eqr, eqc, rn, cn in (
        ("name", raw_nm, cln_nm, eq_raw_nm, eq_cln_nm, raw_nm_n, cln_nm_n),
        ("address", raw_ad, cln_ad, eq_raw_ad, eq_cln_ad, raw_ad_n, cln_ad_n),
    ):
        for seg, mask in segments.items():
            if field == "address" and seg not in ("all", "India", "US"):
                continue
            nmask = mask[neg_ok]
            if mask.sum() == 0 or nmask.sum() == 0:
                continue
            lines.append(
                f"| {field} | {seg} ({mask.sum():,}) | {r[mask].mean():.1f} -> {c[mask].mean():.1f} | "
                f"{eqr[mask].mean():.1%} -> {eqc[mask].mean():.1%} | "
                f"{rn[nmask].mean():.1f} -> {cn[nmask].mean():.1f} | "
                f"{auc(r[mask], rn[nmask]):.3f} -> {auc(c[mask], cn[nmask]):.3f} |"
            )
    lines.append("")

    # token_set_ratio on name_core is one view; the matcher gets several. Two that
    # matter: word-order-free comparison with legal forms kept, and space-free
    # comparison for domain / handle names ("indriyaclub" vs "indriya club").
    lines.append("Other views of the same pairs (AUC, raw text -> cleaned view):\n")
    lines.append("| view | segment | AUC raw | AUC cleaned |")
    lines.append("|---|---|---:|---:|")
    raw_sort = sims_with(fuzz.token_sort_ratio, a.business_name.str.lower(), b.business_name.str.lower())
    raw_sort_n = sims_with(fuzz.token_sort_ratio, a.business_name.str.lower()[neg_ok],
                           b_neg.business_name.str.lower()[neg_ok])
    norm_sort = sims_with(fuzz.token_sort_ratio, a.name_norm, b.name_norm)
    norm_sort_n = sims_with(fuzz.token_sort_ratio, a.name_norm[neg_ok], b_neg.name_norm[neg_ok])
    for seg in ("all", "India", "US"):
        mask = segments[seg]
        lines.append(f"| name_norm, token_sort_ratio | {seg} | {auc(raw_sort[mask], raw_sort_n[mask[neg_ok]]):.3f} | "
                     f"{auc(norm_sort[mask], norm_sort_n[mask[neg_ok]]):.3f} |")
    web = segments["web/handle name"]
    compact = lambda s: s.str.replace(" ", "", regex=False)
    raw_web = sims_with(fuzz.partial_ratio, compact(a.business_name.str.lower())[web],
                        compact(b.business_name.str.lower())[web])
    raw_web_n = sims_with(fuzz.partial_ratio, compact(a.business_name.str.lower())[neg_ok],
                          compact(b_neg.business_name.str.lower())[neg_ok])
    cln_web = sims_with(fuzz.partial_ratio, compact(a.name_core)[web], compact(b.name_core)[web])
    cln_web_n = sims_with(fuzz.partial_ratio, compact(a.name_core)[neg_ok], compact(b_neg.name_core)[neg_ok])
    lines.append(f"| name_core without spaces, partial_ratio | web/handle name | "
                 f"{auc(raw_web, raw_web_n):.3f} | {auc(cln_web, cln_web_n):.3f} |")
    lines.append("")


def main():
    lines = ["# Cleaning report\n",
             "Generated by `src/report_cleaning.py` from the cleaned parquet files.\n"]
    ok = integrity(lines)
    leftovers(lines)
    pair_effect(lines)
    out = config.REPORT_DIR / "cleaning_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
