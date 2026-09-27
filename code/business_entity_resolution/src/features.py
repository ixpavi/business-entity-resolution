"""Stage 3a: features for every candidate pair (S1 entity, S2/S3 record).

Three kinds, all country-agnostic (country itself is never a feature):

  context  where the pair sits in the candidate lists: blocking score, its rank in
           the S1 entity's list, how many S1 entities list the same record and
           how far this S1 is ahead of (or behind) the best other one. Each
           S2/S3 record matches at most one S1 entity, so this competition is
           what separates branches of a chain.
  string   rapidfuzz similarities of the cleaned name and address columns,
           computed in C++ over whole arrays (process.cpdist).
  record   flags and frequencies: S2 or S3, web/Indic/d-b-a names, empty address,
           how common the name is (chains), state and legal-form agreement.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src import config, ids

ATTR_COLUMNS = ["entity_id", "name_norm", "name_core", "name_legal", "name_alias",
                "name_is_web", "name_had_indic", "addr_norm", "addr_numbers", "addr_state"]
PAIR_CHUNK = 4_000_000


# --------------------------------------------------------------------------- #
# Record attributes
# --------------------------------------------------------------------------- #

def load_attrs(split, sources, country):
    """Cleaned columns of one country's records, indexed by int64 id."""
    df = pd.concat(
        [pq.read_table(config.clean_parquet(split, s), columns=ATTR_COLUMNS,
                       filters=[("country", "=", country)]).to_pandas() for s in sources],
        ignore_index=True)
    df.index = pd.Index(ids.encode(df.pop("entity_id").to_numpy()), name="id")
    df["compact"] = df.name_core.str.replace(" ", "", regex=False)
    df["first_word"] = df.name_core.str.split(" ", n=1).str[0].fillna("")
    first_num = df.addr_numbers.str.split(" ", n=1).str[0]
    df["first_num"] = pd.to_numeric(first_num, errors="coerce").fillna(-1).astype(np.int64)
    df["n_words"] = df.name_core.str.count(" ").add(1).where(df.name_core != "", 0).astype(np.int16)
    df["name_freq"] = df.groupby("name_core").name_core.transform("size").astype(np.int32)
    return df


# --------------------------------------------------------------------------- #
# Context features (need the whole candidate table of a country)
# --------------------------------------------------------------------------- #

def group_rank_gap(keys, values):
    """Within each key group: descending rank of each value, group size, and the
    gap to the best OTHER member (value - max of the rest; value if alone)."""
    order = np.lexsort((-values, keys))
    k_sorted = keys[order]
    start = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    group_start = np.maximum.accumulate(np.where(start, np.arange(len(order)), 0))
    rank = np.empty(len(order), np.int32)
    rank[order] = np.arange(len(order)) - group_start + 1
    group_id = np.cumsum(start) - 1
    size = np.bincount(group_id)
    v_sorted = values[order]
    top1 = v_sorted[start]
    second = np.minimum(np.flatnonzero(start) + 1, len(order) - 1)
    top2 = np.where(size > 1, v_sorted[second], 0.0)
    gid = np.empty(len(order), np.int64)
    gid[order] = group_id
    best_other = np.where(rank == 1, top2[gid], top1[gid])
    return rank, size[gid].astype(np.int32), (values - best_other).astype(np.float32)


def add_context(cands):
    """Add competition features in place. cands: s1_id, cand_id, score, rank."""
    score = cands.score.to_numpy()
    s1 = cands.s1_id.to_numpy()

    # per S2/S3 record: rank of this S1 among all S1 entities listing it
    cands["rec_rank"], cands["rec_n"], cands["rec_gap"] = group_rank_gap(
        cands.cand_id.to_numpy(), score)

    # per S1 entity: score relative to its best candidate
    s1_top = pd.Series(score).groupby(s1).transform("max").to_numpy()
    cands["s1_rel"] = (score / s1_top).astype(np.float32)
    cands["s1_n"] = pd.Series(s1).groupby(s1).transform("size").to_numpy().astype(np.int16)
    return cands


PROB_CONTEXT = ["p1", "p1_rec_rank", "p1_rec_gap", "p1_rec_sum", "p1_s1_rank",
                "p1_s1_rel", "p1_s1_n50"]


def add_prob_context(cands, p1):
    """Competition features recomputed from the first model's probabilities p1:
    is this S1 entity the model's clear favourite for the record, and is the
    record among this S1 entity's likely matches?"""
    p1 = np.asarray(p1, np.float32)
    cand = cands.cand_id.to_numpy()
    s1 = cands.s1_id.to_numpy()
    cands["p1"] = p1
    cands["p1_rec_rank"], _, cands["p1_rec_gap"] = group_rank_gap(cand, p1)
    cands["p1_rec_sum"] = pd.Series(p1).groupby(cand).transform("sum").to_numpy().astype(np.float32)
    cands["p1_s1_rank"], _, _ = group_rank_gap(s1, p1)
    s1_top = pd.Series(p1).groupby(s1).transform("max").to_numpy()
    cands["p1_s1_rel"] = (p1 / np.maximum(s1_top, 1e-6)).astype(np.float32)
    cands["p1_s1_n50"] = (pd.Series(p1 >= 0.5).groupby(s1).transform("sum")
                          .to_numpy().astype(np.int16))
    return cands


# --------------------------------------------------------------------------- #
# String and record features
# --------------------------------------------------------------------------- #

def _sim(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def _pair_block(c, s1_attr, cand_attr):
    ia = s1_attr.index.get_indexer(c.s1_id.to_numpy())
    ib = cand_attr.index.get_indexer(c.cand_id.to_numpy())
    A = lambda col: s1_attr[col].to_numpy()[ia]
    B = lambda col: cand_attr[col].to_numpy()[ib]

    f = {}
    f["name_sort"] = _sim(fuzz.token_sort_ratio, A("name_norm"), B("name_norm"))
    f["name_set"] = _sim(fuzz.token_set_ratio, A("name_core"), B("name_core"))
    f["name_ratio"] = _sim(fuzz.ratio, A("name_core"), B("name_core"))
    f["name_partial"] = _sim(fuzz.partial_ratio, A("compact"), B("compact"))
    f["name_jw"] = _sim(JaroWinkler.normalized_similarity, A("name_core"), B("name_core"))
    f["addr_set"] = _sim(fuzz.token_set_ratio, A("addr_norm"), B("addr_norm"))
    f["addr_sort"] = _sim(fuzz.token_sort_ratio, A("addr_norm"), B("addr_norm"))
    f["addr_partial"] = _sim(fuzz.partial_ratio, A("addr_norm"), B("addr_norm"))
    f["num_set"] = _sim(fuzz.token_set_ratio, A("addr_numbers"), B("addr_numbers"))

    na, nb = A("first_num"), B("first_num")
    f["num_first_eq"] = ((na >= 0) & (na == nb)).astype(np.int8)
    f["num_tail_eq"] = ((na >= 0) & (nb >= 0) & (na % 1000 == nb % 1000)).astype(np.int8)
    f["num_missing"] = ((na < 0) | (nb < 0)).astype(np.int8)
    sa, sb = A("addr_state"), B("addr_state")
    f["state_eq"] = ((sa != "") & (sa == sb)).astype(np.int8)
    f["state_conflict"] = ((sa != "") & (sb != "") & (sa != sb)).astype(np.int8)
    la, lb = A("name_legal"), B("name_legal")
    f["legal_eq"] = ((la != "") & (la == lb)).astype(np.int8)
    f["legal_conflict"] = ((la != "") & (lb != "") & (la != lb)).astype(np.int8)
    f["first_word_eq"] = (A("first_word") == B("first_word")).astype(np.int8)
    f["name_exact"] = (A("name_core") == B("name_core")).astype(np.int8)
    f["words_diff"] = (A("n_words") - B("n_words")).astype(np.int16)
    f["s1_name_freq"] = A("name_freq")
    f["cand_name_freq"] = B("name_freq")
    f["cand_is_s3"] = (c.cand_id.to_numpy() >= ids.S3_OFFSET).astype(np.int8)
    f["cand_web"] = B("name_is_web").astype(np.int8)
    f["cand_indic"] = B("name_had_indic").astype(np.int8)
    f["cand_alias"] = (B("name_alias") != "").astype(np.int8)
    f["cand_addr_empty"] = (B("addr_norm") == "").astype(np.int8)
    return pd.DataFrame(f, index=c.index)


CONTEXT = ["score", "rank", "rec_rank", "rec_n", "rec_gap", "s1_rel", "s1_n"]


def pair_features(cands, s1_attr, cand_attr):
    """Feature frame (context + string + record) aligned with cands' index."""
    parts = pd.concat([_pair_block(cands.iloc[i:i + PAIR_CHUNK], s1_attr, cand_attr)
                       for i in range(0, len(cands), PAIR_CHUNK)])
    return pd.concat([cands[CONTEXT], parts], axis=1)
