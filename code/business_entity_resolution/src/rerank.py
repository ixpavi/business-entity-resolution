"""Stage 3c: second-stage reranker that sees the first model's verdict on competitors.

Why: 35% of the true pairs the first model rejects have an EMPTY S2/S3 address and
(85% of them) exactly the right name. Such a record is listed by ~27 S1 entities
with near-equal blocking scores, and blocking-score competition features cannot
tell the one exact name from 26 similar ones. The first model can: rescoring the
competition with its probabilities ("is this S1 the model's clear favourite for
the record, by how much?") separates them.

  pass 1  score EVERY candidate with the stage-1 model -> p1
          + competition features recomputed from p1 (features.add_prob_context)
  pass 2  stage-2 LightGBM on stage-1 features + p1 context

Stage 2 trains on a fresh sample of training entities that stage 1 never trained
on, so their p1 values are out-of-sample, as they will be on test.

Usage:  python -m src.rerank train      (after src.train)
        python -m src.rerank predict    (writes and validates the submission)
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, ids
from src.baseline import best_s1_per_record
from src.blocking import candidate_path, countries, true_pairs
from src.features import PROB_CONTEXT, add_context, add_prob_context, load_attrs, pair_features
from src.submit import all_s1_ids, write_submission
from src.train import FEATURE_DIR, MODEL_DIR, fit, tune_tau

CHUNK = 4_000_000
STAGE2_MODEL = MODEL_DIR / "lgbm_stage2.txt"
STAGE2_DECISION = MODEL_DIR / "decision_stage2.json"


def load_stage1():
    d1 = json.loads((MODEL_DIR / "decision.json").read_text())
    return lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt")), d1


def stage1_pass(split, country, model1, cols1):
    """All candidates of a country with blocking context, p1 and p1 context."""
    t0 = time.time()
    cands = add_context(pd.read_parquet(candidate_path(split, country)))
    s1_attr = load_attrs(split, ["source1"], country)
    cand_attr = load_attrs(split, ["source2", "source3"], country)
    p1 = np.empty(len(cands), np.float32)
    for i in range(0, len(cands), CHUNK):
        X = pair_features(cands.iloc[i:i + CHUNK], s1_attr, cand_attr)
        p1[i:i + CHUNK] = model1.predict(X[cols1], num_threads=0)
    add_prob_context(cands, p1)
    print(f"  {split} {country}: stage 1 on {len(cands):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return cands, s1_attr, cand_attr


def stage2_features(cands, s1_attr, cand_attr):
    parts = []
    for i in range(0, len(cands), CHUNK):
        c = cands.iloc[i:i + CHUNK]
        parts.append(pd.concat([pair_features(c, s1_attr, cand_attr), c[PROB_CONTEXT]], axis=1))
    return pd.concat(parts)


def train_main(per_country, seed=1):
    model1, d1 = load_stage1()
    cols1 = d1["features"]
    cols2 = cols1 + PROB_CONTEXT
    s1_sample = pd.read_parquet(FEATURE_DIR / "train_sample.parquet", columns=["s1_id"]).s1_id
    stage1_fit_ids = pd.unique(s1_sample[s1_sample % 5 != 0])  # entities stage 1 was fitted on
    truth = true_pairs()

    frames = []
    for country in countries("train"):
        cands, a1, a2 = stage1_pass("train", country, model1, cols1)
        s1 = pd.unique(cands.s1_id)
        eligible = s1[~np.isin(s1, stage1_fit_ids)]
        pick = np.random.default_rng(seed).choice(eligible, min(per_country, len(eligible)), replace=False)
        sub = cands[cands.s1_id.isin(pick)]
        X = stage2_features(sub, a1, a2)
        X["s1_id"], X["cand_id"], X["country"] = sub.s1_id.to_numpy(), sub.cand_id.to_numpy(), country
        X = X.merge(truth.assign(label=np.int8(1)), on=["s1_id", "cand_id"], how="left")
        X["label"] = X.label.fillna(0).astype(np.int8)
        frames.append(X)
        del cands, a1, a2
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(FEATURE_DIR / "stage2_sample.parquet", index=False)

    is_valid = (data.s1_id % 5 == 0).to_numpy()
    train, valid = data[~is_valid], data[is_valid]
    print(f"Stage 2: training on {len(train):,} pairs, validating on {len(valid):,} ...", flush=True)
    model2 = fit(train, valid, cols2)
    prob = model2.predict(valid[cols2], num_threads=0)
    tau, f05, table = tune_tau(valid, prob, truth, "stage 2, all: ")
    for country in sorted(valid.country.unique()):
        m = (valid.country == country).to_numpy()
        tune_tau(valid[m], prob[m], truth, f"stage 2, {country}: ")
    # the same entities scored by stage 1 alone, for a like-for-like comparison
    tune_tau(valid, valid.p1.to_numpy(), truth, "stage 1 on the same entities: ")

    model2.save_model(str(STAGE2_MODEL), num_iteration=model2.best_iteration)
    imp = pd.Series(model2.feature_importance("gain"), index=cols2).sort_values(ascending=False)
    print("  top features by gain:", ", ".join(imp.index[:12]))
    # Unseen countries: keep the stricter margin measured for stage 1 by the
    # US<->India transfer runs (train.py), on top of stage 2's own threshold.
    shift = d1.get("tau_unseen", d1["tau"]) - d1["tau"]
    decision = {"tau": tau, "valid_f05": f05, "features": cols2,
                "train_countries": d1.get("train_countries", []),
                "tau_unseen": float(min(0.95, tau + shift)),
                "best_iteration": model2.best_iteration}
    STAGE2_DECISION.write_text(json.dumps(decision, indent=1))
    table.to_csv(MODEL_DIR / "stage2_tau.csv", index=False)


def predict_main():
    model1, d1 = load_stage1()
    d2 = json.loads(STAGE2_DECISION.read_text())
    model2 = lgb.Booster(model_file=str(STAGE2_MODEL))
    seen = set(d2["train_countries"])
    parts = []
    for country in countries("test"):
        cands, a1, a2 = stage1_pass("test", country, model1, d1["features"])
        p2 = np.empty(len(cands), np.float32)
        for i in range(0, len(cands), CHUNK):
            X = stage2_features(cands.iloc[i:i + CHUNK], a1, a2)
            p2[i:i + CHUNK] = model2.predict(X[d2["features"]], num_threads=0)
        tau = d2["tau"] if country in seen else d2["tau_unseen"]
        parts.append(pd.DataFrame({"s1_id": cands.s1_id.to_numpy(), "cand_id": cands.cand_id.to_numpy(),
                                   "score": p2, "tau": np.float32(tau)}))
        print(f"    {country}: threshold {tau:.4f}", flush=True)
        del cands, a1, a2
    scored = pd.concat(parts, ignore_index=True)
    scored.to_parquet(FEATURE_DIR / "test_scored_stage2.parquet", index=False)

    best = best_s1_per_record(scored)
    matches = best[best.score >= best.tau]
    print(f"{len(matches):,} matches for {matches.s1_id.nunique():,} of "
          f"{len(all_s1_ids('test')):,} test S1 entities")
    return write_submission(ids.decode_pairs(matches), ids.decode_pairs(scored))


def main():
    parser = argparse.ArgumentParser(description="Second-stage reranker")
    parser.add_argument("step", choices=["train", "predict"])
    parser.add_argument("--per-country", type=int, default=150_000)
    args = parser.parse_args()
    if args.step == "train":
        train_main(args.per_country)
        return 0
    return 0 if predict_main() else 1


if __name__ == "__main__":
    sys.exit(main())
