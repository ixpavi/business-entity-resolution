"""Where is F0.5 lost? Held-out training entities (s1_id % 5 == 0 of the sample).

  1. Ceiling: F0.5 if the matcher were perfect on the candidates we have
     (only blocking misses remain).
  2. Loss split per entity: wrong merges on true singletons, entities left empty
     although they have matches, and partial errors.
  3. Pair level: missed true pairs split into "never a candidate" (blocking),
     "lost the record to another S1" (competition) and "probability below tau";
     wrong merges split by the flags of the record involved.
  4. Examples of each, with the cleaned text.

Usage:  python -m src.error_analysis        (after src.train)
Writes work/reports/error_analysis.md
"""

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, ids
from src.baseline import best_s1_per_record
from src.blocking import true_pairs
from src.metrics import per_entity_fbeta
from src.train import FEATURE_DIR, MODEL_DIR

N_EXAMPLES = 12


def texts(pairs):
    """Attach cleaned name/address of both sides (train split)."""
    cols = ["entity_id", "country", "name_norm", "addr_norm"]
    s1 = pd.read_parquet(config.clean_parquet("train", "source1"), columns=cols)
    s1.index = ids.encode(s1.pop("entity_id").to_numpy())
    need = set(pairs.cand_id)
    other = []
    for s in ("source2", "source3"):
        t = pd.read_parquet(config.clean_parquet("train", s), columns=cols + ["business_name"])
        t.index = ids.encode(t.pop("entity_id").to_numpy())
        other.append(t[t.index.isin(need)])
    other = pd.concat(other)
    out = pairs.copy()
    out["s1_text"] = (s1.name_norm + " | " + s1.addr_norm).reindex(out.s1_id).to_numpy()
    out["cand_text"] = (other.name_norm + " | " + other.addr_norm).reindex(out.cand_id).to_numpy()
    out["cand_raw_name"] = other.business_name.reindex(out.cand_id).to_numpy()
    return out


def main():
    decision = json.loads((MODEL_DIR / "decision.json").read_text())
    tau, cols = decision["tau"], decision["features"]
    data = pd.read_parquet(FEATURE_DIR / "train_sample.parquet")
    valid = data[data.s1_id % 5 == 0].reset_index(drop=True)
    del data
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
    valid["prob"] = model.predict(valid[cols], num_threads=0)

    entities = pd.unique(valid.s1_id)
    truth = true_pairs()
    truth = truth[truth.s1_id.isin(entities)]
    best = best_s1_per_record(valid[["s1_id", "cand_id", "prob"]].rename(columns={"prob": "score"}))
    pred = best[best.score >= tau][["s1_id", "cand_id"]]
    oracle = valid[valid.label == 1][["s1_id", "cand_id"]]

    f = per_entity_fbeta(pred, truth, entities)
    f_oracle = per_entity_fbeta(oracle, truth, entities)
    n_true = truth.groupby("s1_id").size().reindex(entities, fill_value=0)
    n_pred = pred.groupby("s1_id").size().reindex(entities, fill_value=0)

    lines = ["# Error analysis (held-out training entities)\n",
             f"{len(entities):,} entities, {len(truth):,} true pairs, tau = {tau}.\n",
             "| | F0.5 |", "|---|---:|",
             f"| model | {f.mean():.4f} |",
             f"| perfect matcher on our candidates (blocking ceiling) | {f_oracle.mean():.4f} |",
             f"| perfect everything | 1.0000 |", ""]

    loss = 1 - f
    groups = {
        "true singleton, but we predicted matches": (n_true == 0) & (n_pred > 0),
        "has matches, but we predicted none": (n_true > 0) & (n_pred == 0),
        "has matches, partial errors": (n_true > 0) & (n_pred > 0) & (f < 1),
    }
    lines += ["## Where the F0.5 loss sits\n", "| entity group | entities | share of total loss |",
              "|---|---:|---:|"]
    for name, mask in groups.items():
        lines.append(f"| {name} | {int(mask.sum()):,} | {loss[mask].sum() / loss.sum():.1%} |")
    lines.append("")

    # pair-level misses
    hit = truth.merge(valid[["s1_id", "cand_id", "prob", "rec_rank"]], on=["s1_id", "cand_id"], how="left")
    hit = hit.merge(pred.assign(kept=True), on=["s1_id", "cand_id"], how="left")
    hit["kept"] = hit.kept.fillna(False).astype(bool)
    top = best.set_index("cand_id").s1_id
    hit["lost_to_other"] = top.reindex(hit.cand_id).to_numpy() != hit.s1_id.to_numpy()
    fn = hit[~hit.kept]
    kinds = {
        "never a candidate (blocking)": fn.prob.isna(),
        "record went to another S1 entity": fn.prob.notna() & fn.lost_to_other,
        "best S1 but probability < tau": fn.prob.notna() & ~fn.lost_to_other,
    }
    lines += [f"## Missed true pairs: {len(fn):,} of {len(truth):,} ({len(fn) / len(truth):.1%})\n",
              "| reason | pairs | share |", "|---|---:|---:|"]
    for name, mask in kinds.items():
        lines.append(f"| {name} | {int(mask.sum()):,} | {mask.mean():.1%} |")
    lines.append("")

    # wrong merges
    fp = pred.merge(truth.assign(t=1), on=["s1_id", "cand_id"], how="left")
    fp = fp[fp.t.isna()].drop(columns="t").merge(valid, on=["s1_id", "cand_id"])
    flags = ["cand_web", "cand_indic", "cand_alias", "cand_addr_empty", "name_exact",
             "num_first_eq", "state_conflict"]
    base = valid[valid.label == 1]
    lines += [f"## Wrong merges: {len(fp):,} of {len(pred):,} predicted pairs "
              f"({len(fp) / len(pred):.2%})\n",
              "| flag | share of wrong merges | share of true pairs |", "|---|---:|---:|"]
    for fl in flags:
        lines.append(f"| {fl} | {fp[fl].mean():.1%} | {base[fl].mean():.1%} |")
    lines.append("")

    ex_fp = texts(fp.sample(min(N_EXAMPLES, len(fp)), random_state=0)[["s1_id", "cand_id", "prob"]])
    ex_fn = texts(fn[fn.prob.notna()].sample(min(N_EXAMPLES, int(fn.prob.notna().sum())),
                                             random_state=0)[["s1_id", "cand_id", "prob"]])
    ex_bl = texts(fn[fn.prob.isna()].sample(min(N_EXAMPLES, int(fn.prob.isna().sum())),
                                            random_state=0)[["s1_id", "cand_id"]])
    for title, ex in (("Wrong merges", ex_fp), ("Missed, candidate but rejected", ex_fn),
                      ("Missed, never a candidate", ex_bl)):
        lines += [f"## Examples: {title}\n", "| p | S1 | S2/S3 (cleaned) | S2/S3 raw name |",
                  "|---:|---|---|---|"]
        for r in ex.itertuples():
            p = f"{r.prob:.2f}" if hasattr(r, "prob") else "—"
            lines.append(f"| {p} | {r.s1_text} | {r.cand_text} | {r.cand_raw_name} |")
        lines.append("")

    out = config.REPORT_DIR / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
