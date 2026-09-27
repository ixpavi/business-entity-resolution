"""Baseline matcher: blocking score only, no model. Submission #1.

Rule:
  1. Each S2/S3 record goes to the one S1 entity that scores it highest among
     all S1 entities that have it as a candidate. (Every S2/S3 record matches at
     most one S1 entity: 0 exceptions in 7.6M training pairs.)
  2. That match is kept if its blocking score >= TAU.
  3. S1 entities left with nothing get an empty list.

TAU is chosen by the exact competition metric (macro per-entity F0.5, singletons
included) over every training S1 entity.

Usage:  python -m src.baseline            (after src.blocking for train and test)
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config, ids
from src.blocking import candidate_path, countries, true_pairs
from src.metrics import per_entity_fbeta
from src.submit import write_submission

TAU_GRID = np.round(np.arange(0.20, 0.81, 0.05), 2)


def load_candidates(split):
    return pd.concat([pd.read_parquet(candidate_path(split, c)) for c in countries(split)],
                     ignore_index=True)


def best_s1_per_record(cands, score_col="score"):
    """Keep, for each cand_id, only its highest-scoring (s1_id, cand_id) row."""
    order = np.lexsort((cands[score_col].to_numpy(), cands["cand_id"].to_numpy()))
    c = cands["cand_id"].to_numpy()[order]
    last = np.r_[c[1:] != c[:-1], True]  # last row of each cand_id group = max score
    return cands.iloc[order[last]]


def s1_ids(split):
    t = pq.read_table(config.clean_parquet(split, "source1"), columns=["entity_id", "country"])
    return pd.DataFrame({"s1_id": ids.encode(t["entity_id"]),
                         "country": t["country"].to_numpy(zero_copy_only=False)})


def tune(best, truth, entities, grid=TAU_GRID):
    rows = []
    for tau in grid:
        pred = best[best.score >= tau]
        f = per_entity_fbeta(pred, truth, entities.s1_id)
        by_country = f.groupby(entities.set_index("s1_id").country.reindex(f.index).to_numpy()).mean()
        rows.append({"tau": tau, "F0.5": f.mean(), **by_country.round(4).to_dict()})
        print(f"  tau={tau:.2f}  F0.5={f.mean():.4f}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in by_country.items()), flush=True)
    return pd.DataFrame(rows)


def main():
    print("Tuning on train ...")
    train = load_candidates("train")
    best = best_s1_per_record(train)
    del train
    entities = s1_ids("train")
    table = tune(best, true_pairs(), entities)
    tau = float(table.loc[table["F0.5"].idxmax(), "tau"])
    print(f"best tau = {tau:.2f}  (train F0.5 = {table['F0.5'].max():.4f})")
    (config.REPORT_DIR / "baseline_tau.csv").write_text(table.to_csv(index=False))

    print("Predicting test ...")
    test = load_candidates("test")
    best = best_s1_per_record(test)
    matches = best[best.score >= tau]
    print(f"  {len(matches):,} matches for {matches.s1_id.nunique():,} of "
          f"{len(s1_ids('test')):,} test S1 entities")
    ok = write_submission(ids.decode_pairs(matches), ids.decode_pairs(test))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
