"""Stage 2: candidate generation (blocking).

For every Source 1 entity, find the K most similar Source 2/3 records in the same
country. No true pair crosses countries (0 of 7.6M training pairs), and country is
treated as an open set of labels, so France is blocked like any other country.

Similarity is rarity-weighted keyword overlap (TF-IDF cosine), computed as one
sparse matrix product per country with sparse_dot_topn. Each record becomes a
bag of keys from its cleaned columns:

    name     every word of name_core               "keystone", "robotics"
    compact  name_core with spaces removed         "keystoneroboticsstudios"
                                                   (web/handle names, joined words)
    addr     every word of addr_norm               "welshwood", "nashville", "441"
    house    house number + following word         "441_welshwood"

Keys found in more than MAX_DF records of the country are dropped; the rest get
weight log(N / df). Vectors are L2-normalised, so the score is a cosine in [0, 1].

Measured on 20k-entity samples per country against the full S2/S3 pool
(share of true pairs kept):
                          @10     @20     @30     @50
    India, MAX_DF=2000   0.899   0.929   0.940   0.953
    India, MAX_DF=20000  0.924   0.953   0.962   0.969   <- used
    US,    MAX_DF=20000  0.932   0.958   0.969   0.975   <- used
    US,    MAX_DF=200000 0.941   0.967   0.976   0.981   (6-10x slower)
Tried and dropped: separate name-only / address-only searches (no gain at equal
budget), one-deletion typo keys for name words (0.962 -> 0.886: the variants swamp
the address keys), last-3-digit house-number keys (+0.1 pt India, -0.1 pt US).
Name character trigrams ("trigram" group, off by default) at weight 0.3 / 0.5 / 1.0:
India @30 0.9616 -> 0.9634 / 0.9652 / 0.9633, US 0.9685 -> 0.9690 / 0.9690 / 0.9615.
About +0.17 pt recall overall for twice the blocking time and a full pipeline
re-run, so not adopted.

Usage:  python -m src.blocking --split train --sample 20000   (quick recall check)
        python -m src.blocking --split train                   (all, for training)
        python -m src.blocking --split test
Writes work/candidates/<split>_<country>.parquet with int64 ids (see ids.py):
    s1_id, cand_id, score, rank   (rank 1 = most similar for that S1 entity)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from src import config, ids

COLUMNS = ["entity_id", "country", "name_core", "addr_norm"]
TOP_K = 30        # candidates kept per Source 1 entity
MAX_DF = 20_000   # keys in more records than this are dropped
MIN_SCORE = 0.05  # candidates below this cosine are not kept
CHUNK = 50_000    # Source 1 rows per matrix product
GROUP_WEIGHTS = {"name": 1.0, "compact": 1.0, "addr": 1.0, "house": 1.0}
CANDIDATE_DIR = config.WORK_DIR / "candidates"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def countries(split):
    t = pq.read_table(config.clean_parquet(split, "source1"), columns=["country"])
    return sorted(pc.unique(t["country"]).to_pylist())


def load_records(split, sources, country):
    tables = [
        pq.read_table(config.clean_parquet(split, s), columns=COLUMNS,
                      filters=[("country", "=", country)])
        for s in sources
    ]
    return pa.concat_tables(tables).combine_chunks()


def candidate_path(split, country):
    return CANDIDATE_DIR / f"{split}_{country}.parquet"


# --------------------------------------------------------------------------- #
# Keys -> sparse matrices
# --------------------------------------------------------------------------- #

def _words(arr):
    """Split each string on whitespace: (row index per word, flat word array)."""
    lists = pc.utf8_split_whitespace(arr)
    return pc.list_parent_indices(lists).to_numpy(), pc.list_flatten(lists)


def _trigrams(strings):
    """(row index, 3-character substring) for every position of every string."""
    lens = pc.utf8_length(strings).to_numpy(zero_copy_only=False)
    rows, grams = [], []
    for i in range(max(0, int(lens.max(initial=0)) - 2)):
        sel = lens >= i + 3
        rows.append(np.flatnonzero(sel))
        grams.append(pc.utf8_slice_codeunits(strings.filter(pa.array(sel)), i, i + 3)
                     .cast(pa.large_string()))
    if not rows:
        return np.empty(0, np.int64), pa.array([], pa.large_string())
    return np.concatenate(rows), pa.concat_arrays(grams)


def _key_groups(table, groups):
    """Yield (group, row_indices, keys) for each requested key group, in a fixed order."""
    name = table.column("name_core").combine_chunks()
    compact = pc.replace_substring(name, " ", "")
    if "name" in groups:
        yield ("name", *_words(name))
    if "compact" in groups:
        keep = pc.greater_equal(pc.utf8_length(compact), 4).to_numpy(zero_copy_only=False)
        yield "compact", np.flatnonzero(keep), compact.filter(pa.array(keep))
    if "trigram" in groups:  # typo-tolerant name keys: "robotics" -> rob, obo, bot, ...
        yield ("trigram", *_trigrams(compact))

    addr = pc.replace_substring(table.column("addr_norm").combine_chunks(), ",", "")
    rows, words = _words(addr)
    if "addr" in groups:
        yield "addr", rows, words
    if "house" in groups:  # house number followed by the next word: "441_welshwood"
        if len(words) > 1:
            first, second = words.slice(0, len(words) - 1), words.slice(1)
            sel = (rows[:-1] == rows[1:]) & pc.utf8_is_digit(first).to_numpy(zero_copy_only=False)
            mask = pa.array(sel)
            yield ("house", rows[:-1][sel],
                   pc.binary_join_element_wise(first.filter(mask), second.filter(mask), "_"))
        else:
            yield "house", np.empty(0, np.int64), pa.array([], pa.string())


def build_matrices(s1, cand, max_df=MAX_DF, weights=None):
    """TF-IDF matrices for Source 1 rows (A) and candidate rows (B), same columns.

    weights: {group: multiplier} (default GROUP_WEIGHTS); a group's idf weights are
    scaled by its multiplier, so a many-key group can be kept from swamping others.
    """
    weights = weights or GROUP_WEIGHTS
    r1, c1, r2, c2 = [], [], [], []
    offset = 0
    col_weight = []
    for (group, rows1, keys1), (_, rows2, keys2) in zip(_key_groups(s1, weights),
                                                        _key_groups(cand, weights)):
        enc = pc.dictionary_encode(pa.concat_arrays([keys1.cast(pa.large_string()),
                                                     keys2.cast(pa.large_string())]))
        codes = enc.indices.to_numpy().astype(np.int64) + offset
        r1.append(rows1); c1.append(codes[: len(keys1)])
        r2.append(rows2); c2.append(codes[len(keys1):])
        col_weight.append(np.full(len(enc.dictionary), weights[group], np.float32))
        offset += len(enc.dictionary)

    def binary(rows, cols, n):
        m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, offset))
        m.sum_duplicates()
        m.data[:] = 1.0
        return m

    A = binary(np.concatenate(r1), np.concatenate(c1), s1.num_rows)
    B = binary(np.concatenate(r2), np.concatenate(c2), cand.num_rows)

    df = np.bincount(B.indices, minlength=offset)
    keep = (df > 0) & (df <= max_df)
    idf = np.zeros(offset, np.float32)
    idf[keep] = np.log(cand.num_rows / df[keep])
    idf *= np.concatenate(col_weight)

    for m in (A, B):
        m.data *= idf[m.indices]
        m.eliminate_zeros()
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        m.data /= np.repeat(norms, np.diff(m.indptr)).astype(np.float32)
    return A, B


# --------------------------------------------------------------------------- #
# Top-K
# --------------------------------------------------------------------------- #

def top_k(A, B, k=TOP_K, min_score=MIN_SCORE, n_threads=None):
    """Return (s1_row, cand_row, score, rank) arrays of each A row's top-k B rows."""
    n_threads = n_threads or os.cpu_count()
    BT = B.T.tocsr()
    out = []
    for start in range(0, A.shape[0], CHUNK):
        C = sp_matmul_topn(A[start:start + CHUNK], BT, top_n=k, threshold=min_score,
                           sort=True, n_threads=n_threads)
        counts = np.diff(C.indptr)
        rows = np.repeat(np.arange(C.shape[0]), counts) + start
        rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts) + 1
        out.append((rows, C.indices.astype(np.int64), C.data, rank))
    return tuple(np.concatenate(x) for x in zip(*out))


def block_country(split, country, s1_ids=None, k=TOP_K, max_df=MAX_DF, verbose=True):
    """Candidates for one country: DataFrame of int64 s1_id, cand_id, score, rank."""
    t0 = time.time()
    s1 = load_records(split, ["source1"], country)
    if s1_ids is not None:
        s1 = s1.filter(pc.is_in(s1["entity_id"], value_set=pa.array(list(s1_ids))))
    cand = load_records(split, ["source2", "source3"], country)
    A, B = build_matrices(s1, cand, max_df=max_df)
    t1 = time.time()
    rows, cols, score, rank = top_k(A, B, k=k)
    cands = pd.DataFrame({
        "s1_id": ids.encode(s1["entity_id"])[rows],
        "cand_id": ids.encode(cand["entity_id"])[cols],
        "score": score.astype(np.float32),
        "rank": rank.astype(np.int16),
    })
    if verbose:
        empty = s1.num_rows - pd.unique(rows).size
        print(f"  {split} {country}: {s1.num_rows:,} S1 x {cand.num_rows:,} S2/S3 | "
              f"{len(cands):,} candidates ({len(cands) / max(1, s1.num_rows):.1f}/S1, "
              f"{empty:,} S1 with none) | matrices {t1 - t0:.0f}s, top-k {time.time() - t1:.0f}s",
              flush=True)
    return cands


# --------------------------------------------------------------------------- #
# Evaluation on training labels
# --------------------------------------------------------------------------- #

def true_pairs():
    """All training (s1_id, cand_id) pairs as int64 ids."""
    gt = pq.read_table(config.RAW_DIR / "train_ground_truth.parquet").to_pandas()
    gt = gt[gt.matched_entity_ids != ""]
    pairs = gt.assign(cand_id=gt.matched_entity_ids.str.split(",")).explode("cand_id")
    return pd.DataFrame({"s1_id": ids.encode(pairs.source1_entity_id.to_numpy()),
                         "cand_id": ids.encode(pairs.cand_id.to_numpy())})


def blocking_recall(cands, s1_ids=None, ks=(1, 5, 10, 20, 30, 50)):
    """Share of true pairs (of the given int S1 ids, default all) within the top k."""
    truth = true_pairs()
    if s1_ids is not None:
        truth = truth[truth.s1_id.isin(np.asarray(list(s1_ids)))]
    hit = truth.merge(cands[["s1_id", "cand_id", "rank"]], on=["s1_id", "cand_id"], how="left")
    top = int(cands["rank"].max())
    return {k: float((hit["rank"] <= k).mean()) for k in ks if k <= top}, len(truth)


def main():
    parser = argparse.ArgumentParser(description="Blocking / candidate generation")
    parser.add_argument("--split", choices=config.SPLITS, required=True)
    parser.add_argument("--sample", type=int, default=0,
                        help="train only: block this many random S1 entities per country "
                             "and report recall instead of writing files")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--max-df", type=int, default=MAX_DF)
    args = parser.parse_args()

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for country in countries(args.split):
        t = pq.read_table(config.clean_parquet(args.split, "source1"),
                          columns=["entity_id"], filters=[("country", "=", country)])
        all_ids = t["entity_id"].to_numpy(zero_copy_only=False)
        s1_ids = None
        if args.sample:
            s1_ids = set(np.random.default_rng(0).choice(all_ids, min(args.sample, len(all_ids)),
                                                         replace=False))
        cands = block_country(args.split, country, s1_ids, k=args.k, max_df=args.max_df)
        if args.split == "train":
            scored = list(s1_ids) if s1_ids is not None else all_ids
            recall, n_true = blocking_recall(cands, ids.encode(scored))
            print(f"    pair recall ({n_true:,} true pairs): "
                  + "  ".join(f"@{k}={v:.4f}" for k, v in recall.items()), flush=True)
        if not args.sample:
            cands.to_parquet(candidate_path(args.split, country), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
