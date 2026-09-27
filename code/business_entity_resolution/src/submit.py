"""Write matching_results.tsv / candidate_pairs.tsv and run the organisers' validator.

Both files: one row per test Source 1 entity (empty list allowed), tab-separated,
IDs comma-joined with no quoting, no duplicate IDs within a row.
"""

import subprocess
import sys

import pyarrow.parquet as pq

from src import config

OUTPUT_DIR = config.OUTPUT_DIR
VALIDATOR = config.DATA_DIR.parent / "utils" / "validate_submission.py"


def all_s1_ids(split="test"):
    t = pq.read_table(config.clean_parquet(split, "source1"), columns=["entity_id"])
    return t["entity_id"].to_pylist()


def write_id_lists(pairs, s1_ids, path, header):
    """pairs: DataFrame s1_id, cand_id. Every id in s1_ids gets exactly one row."""
    lists = (pairs[["s1_id", "cand_id"]].drop_duplicates()
             .groupby("s1_id")["cand_id"].agg(",".join).to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{lists.get(s1, '')}\n")


def write_submission(matches, candidates, out_dir=OUTPUT_DIR):
    """Write both files for the test split and validate them. Returns True on PASS."""
    s1_ids = all_s1_ids("test")
    missing = matches.merge(candidates[["s1_id", "cand_id"]], how="left", indicator=True)
    if (missing["_merge"] == "left_only").any():
        raise ValueError("some matches are not among the candidates")
    write_id_lists(matches, s1_ids, out_dir / "matching_results.tsv",
                   ["source1_entity_id", "matched_entity_ids"])
    write_id_lists(candidates, s1_ids, out_dir / "candidate_pairs.tsv",
                   ["source1_entity_id", "candidate_entity_ids"])
    return validate(out_dir)


def validate(out_dir=OUTPUT_DIR):
    cmd = [sys.executable, str(VALIDATOR),
           "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"),
           "--test-dir", str(config.DATA_DIR / "test")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:], result.stderr[-2000:], sep="")
    return result.returncode == 0
