"""The competition metric: F-beta (beta = 0.5) per Source 1 entity, macro-averaged.

Per entity, with T = true matches and P = predicted matches:
    T empty, P empty      -> 1.0   (a correctly predicted singleton)
    exactly one empty     -> 0.0
    otherwise             -> (1 + b^2) * prec * rec / (b^2 * prec + rec), 0 if no overlap
The score is the mean over ALL Source 1 entities being evaluated.

Usage:  python -m src.metrics      (runs the README's worked example as a self-test)
"""

import sys

import numpy as np
import pandas as pd

BETA = 0.5


def per_entity_fbeta(pred, truth, entities, beta=BETA):
    """F-beta for each entity.

    pred, truth: DataFrames with columns s1_id, cand_id (one row per pair).
    entities:    every Source 1 id to score (singletons included).
    Returns a Series indexed by entity.
    """
    entities = pd.Index(pd.unique(np.asarray(entities)), name="s1_id")
    pred = pred[["s1_id", "cand_id"]].drop_duplicates()
    truth = truth[["s1_id", "cand_id"]].drop_duplicates()
    n_pred = pred.groupby("s1_id").size().reindex(entities, fill_value=0).to_numpy()
    n_true = truth.groupby("s1_id").size().reindex(entities, fill_value=0).to_numpy()
    tp = (pred.merge(truth, on=["s1_id", "cand_id"]).groupby("s1_id").size()
          .reindex(entities, fill_value=0).to_numpy())

    score = np.zeros(len(entities))
    score[(n_pred == 0) & (n_true == 0)] = 1.0
    ok = tp > 0
    precision = tp[ok] / n_pred[ok]
    recall = tp[ok] / n_true[ok]
    b2 = beta * beta
    score[ok] = (1 + b2) * precision * recall / (b2 * precision + recall)
    return pd.Series(score, index=entities)


def macro_fbeta(pred, truth, entities, beta=BETA):
    return float(per_entity_fbeta(pred, truth, entities, beta).mean())


def _self_test():
    pairs = lambda rows: pd.DataFrame(rows, columns=["s1_id", "cand_id"])
    # README example: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
    pred = pairs([("S1-00001", "S2-00047"), ("S1-00001", "S2-00193"), ("S1-00001", "S3-00812")])
    truth = pairs([("S1-00001", "S2-00047"), ("S1-00001", "S3-00812")])
    s = per_entity_fbeta(pred, truth, ["S1-00001"])
    assert abs(s.iloc[0] - 0.714) < 1e-3, s
    # singleton predicted empty = 1, singleton with a prediction = 0, missed entity = 0
    pred = pairs([("S1-2", "S2-9")])
    truth = pairs([("S1-3", "S2-5")])
    s = per_entity_fbeta(pred, truth, ["S1-1", "S1-2", "S1-3"])
    assert s.tolist() == [1.0, 0.0, 0.0], s.tolist()
    print("metrics self-test passed: README example = %.3f" % 0.714)


if __name__ == "__main__":
    _self_test()
    sys.exit(0)
