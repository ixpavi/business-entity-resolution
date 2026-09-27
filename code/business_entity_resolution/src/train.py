"""Stage 3b: train the pair classifier (LightGBM, MIT) and choose the threshold.

1. Per training country: context features over the FULL candidate table (every S1
   entity competes for every record), then a random sample of S1 entities, their
   30 candidates each, string/record features and labels.
2. Hold out 20% of sampled entities (s1_id % 5 == 0) for early stopping and for
   choosing the decision threshold with the exact metric.
3. Decision rule (same as the baseline, on model probability instead of blocking
   score): each S2/S3 record goes to its most probable S1 entity; kept if >= TAU.
4. --transfer: also train on one country and evaluate on the other (US <-> India),
   to see how far the best threshold moves on an unseen country (France proxy).

Usage:  python -m src.train [--per-country 150000] [--transfer]
Writes work/models/lgbm.txt and work/models/decision.json
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.baseline import best_s1_per_record
from src.blocking import candidate_path, countries, true_pairs
from src.features import add_context, load_attrs, pair_features
from src.metrics import per_entity_fbeta

MODEL_DIR = config.RUN_DIR / "models"
FEATURE_DIR = config.RUN_DIR / "features"
TAU_GRID = np.round(np.arange(0.10, 0.91, 0.025), 3)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)


def build_training_set(per_country, seed=0):
    truth = true_pairs()
    truth["label"] = np.int8(1)
    frames = []
    for country in countries("train"):
        t0 = time.time()
        cands = add_context(pd.read_parquet(candidate_path("train", country)))
        s1 = pd.unique(cands.s1_id)
        pick = np.random.default_rng(seed).choice(s1, min(per_country, len(s1)), replace=False)
        cands = cands[cands.s1_id.isin(pick)].reset_index(drop=True)
        X = pair_features(cands, load_attrs("train", ["source1"], country),
                          load_attrs("train", ["source2", "source3"], country))
        X["s1_id"], X["cand_id"], X["country"] = cands.s1_id, cands.cand_id, country
        X = X.merge(truth, on=["s1_id", "cand_id"], how="left")
        X["label"] = X.label.fillna(0).astype(np.int8)
        frames.append(X)
        print(f"  {country}: {len(pick):,} S1, {len(X):,} pairs, {X.label.mean():.1%} positive "
              f"({time.time() - t0:.0f}s)", flush=True)
    data = pd.concat(frames, ignore_index=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(FEATURE_DIR / "train_sample.parquet", index=False)
    return data


def feature_columns(data):
    return [c for c in data.columns if c not in ("s1_id", "cand_id", "country", "label")]


def fit(train, valid, cols, rounds=2000):
    dtrain = lgb.Dataset(train[cols], train.label, free_raw_data=True)
    dvalid = lgb.Dataset(valid[cols], valid.label, reference=dtrain)
    return lgb.train(PARAMS, dtrain, rounds, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])


def tune_tau(valid, prob, truth, label=""):
    """Macro F0.5 on the validation entities for every threshold in TAU_GRID."""
    scored = valid[["s1_id", "cand_id"]].assign(score=prob)
    best = best_s1_per_record(scored)
    entities = pd.unique(valid.s1_id)
    t = truth[truth.s1_id.isin(entities)]
    rows = [(tau, per_entity_fbeta(best[best.score >= tau], t, entities).mean()) for tau in TAU_GRID]
    table = pd.DataFrame(rows, columns=["tau", "f05"])
    top = table.loc[table.f05.idxmax()]
    print(f"  {label}best tau {top.tau:.3f} -> F0.5 {top.f05:.4f}  "
          f"(tau 0.5 -> {table.loc[(table.tau - 0.5).abs().idxmin(), 'f05']:.4f})", flush=True)
    return float(top.tau), float(top.f05), table


def unseen_country_tau(decision, train_countries):
    """Threshold for countries absent from training (France in test).

    Trained on one country and scored on the other, the model is overconfident and
    the best threshold rises (US->India 0.725 -> 0.900, India->US 0.725 -> 0.825).
    Countries not seen in training get the mean of those transferred optima;
    without a transfer run they fall back to the main threshold.
    """
    moved = [v["tau_dst"] for k, v in decision.items() if k.startswith("transfer_")]
    return {"train_countries": list(train_countries),
            "tau_unseen": float(np.mean(moved)) if moved else decision["tau"]}


def main():
    parser = argparse.ArgumentParser(description="Train the pair classifier")
    parser.add_argument("--per-country", type=int, default=150_000)
    parser.add_argument("--reuse", action="store_true", help="reuse saved training features")
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()

    print("Building training features ...")
    path = FEATURE_DIR / "train_sample.parquet"
    data = pd.read_parquet(path) if args.reuse and path.exists() else build_training_set(args.per_country)
    cols = feature_columns(data)
    truth = true_pairs()
    is_valid = (data.s1_id % 5 == 0).to_numpy()
    train, valid = data[~is_valid], data[is_valid]

    print(f"Training on {len(train):,} pairs, validating on {len(valid):,} ...")
    t0 = time.time()
    model = fit(train, valid, cols)
    prob = model.predict(valid[cols], num_threads=0)
    print(f"  {model.best_iteration} trees, {time.time() - t0:.0f}s")
    tau, f05, table = tune_tau(valid, prob, truth, "all: ")
    for country in sorted(valid.country.unique()):
        m = (valid.country == country).to_numpy()
        tune_tau(valid[m], prob[m], truth, f"{country}: ")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "lgbm.txt"), num_iteration=model.best_iteration)
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("  top features by gain:", ", ".join(imp.index[:12]))
    decision = {"tau": tau, "valid_f05": f05, "features": cols,
                "best_iteration": model.best_iteration}

    if args.transfer:
        print("Country transfer (France proxy) ...")
        for src_c, dst_c in (("US", "India"), ("India", "US")):
            tr = train[train.country == src_c]
            va_src, va_dst = valid[valid.country == src_c], valid[valid.country == dst_c]
            m = fit(tr, va_src, cols)
            t_src, _, _ = tune_tau(va_src, m.predict(va_src[cols]), truth, f"train {src_c} -> {src_c}: ")
            p_dst = m.predict(va_dst[cols])
            t_dst, f_dst, tab = tune_tau(va_dst, p_dst, truth, f"train {src_c} -> {dst_c}: ")
            f_at_src = tab.loc[(tab.tau - t_src).abs().idxmin(), "f05"]
            print(f"    {dst_c} F0.5 using {src_c}'s tau {t_src:.3f}: {f_at_src:.4f}")
            decision[f"transfer_{src_c}_to_{dst_c}"] = {"tau_src": t_src, "tau_dst": t_dst,
                                                        "f05_dst_at_src_tau": float(f_at_src),
                                                        "f05_dst_best": f_dst}
    decision.update(unseen_country_tau(decision, sorted(data.country.unique())))

    (MODEL_DIR / "decision.json").write_text(json.dumps(decision, indent=1))
    table.to_csv(MODEL_DIR / "model_tau.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
