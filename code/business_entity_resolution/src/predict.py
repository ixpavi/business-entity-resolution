"""Stage 4: score every test candidate with the trained model and write the submission.

Per test country: context features over the full candidate table, then string and
record features and model probabilities in chunks (so memory stays bounded).
Decision rule as in training: each S2/S3 record goes to its most probable S1
entity, kept if the probability >= tau from work/models/decision.json.

candidate_pairs.tsv = every pair the model scored (the final blocking output);
matching_results.tsv = the kept matches, a subset of it.

Usage:  python -m src.predict [--tau 0.5]      (after src.blocking --split test, src.train)
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import ids
from src.baseline import best_s1_per_record
from src.blocking import candidate_path, countries
from src.features import add_context, load_attrs, pair_features
from src.submit import all_s1_ids, write_submission
from src.train import FEATURE_DIR, MODEL_DIR

CHUNK = 4_000_000


def score_country(model, cols, country):
    t0 = time.time()
    cands = add_context(pd.read_parquet(candidate_path("test", country)))
    s1_attr = load_attrs("test", ["source1"], country)
    cand_attr = load_attrs("test", ["source2", "source3"], country)
    prob = np.empty(len(cands), np.float32)
    for i in range(0, len(cands), CHUNK):
        X = pair_features(cands.iloc[i:i + CHUNK], s1_attr, cand_attr)
        prob[i:i + CHUNK] = model.predict(X[cols], num_threads=0)
    print(f"  {country}: {len(cands):,} pairs scored ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame({"s1_id": cands.s1_id.to_numpy(), "cand_id": cands.cand_id.to_numpy(),
                         "score": prob})


def main():
    parser = argparse.ArgumentParser(description="Score test candidates and write the submission")
    parser.add_argument("--tau", type=float, help="override the tuned threshold")
    args = parser.parse_args()

    decision = json.loads((MODEL_DIR / "decision.json").read_text())
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
    cols = decision["features"]
    seen = set(decision.get("train_countries", []))

    def tau_for(country):
        if args.tau is not None:
            return args.tau
        return decision["tau"] if country in seen else decision.get("tau_unseen", decision["tau"])

    parts = []
    for country in countries("test"):
        scored = score_country(model, cols, country)
        scored["tau"] = np.float32(tau_for(country))
        parts.append(scored)
    scored = pd.concat(parts, ignore_index=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(FEATURE_DIR / "test_scored.parquet", index=False)

    best = best_s1_per_record(scored)
    matches = best[best.score >= best.tau]
    n_s1 = len(all_s1_ids("test"))
    print(f"thresholds: " + ", ".join(f"{c}={tau_for(c):.4f}" for c in countries("test")))
    print(f"{len(matches):,} matches for {matches.s1_id.nunique():,} of {n_s1:,} test S1 entities")
    ok = write_submission(ids.decode_pairs(matches), ids.decode_pairs(scored))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
