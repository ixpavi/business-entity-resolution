"""Entity ids as int64, so tens of millions of candidate pairs fit in memory.

"S1-925783039" -> 925783039, "S2-49942811" -> 49942811,
"S3-202863386" -> 202863386 + S3_OFFSET. S2 and S3 numbers can collide, so S3 is
shifted past every id in the data (ids have at most 9 digits).
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

S3_OFFSET = 10**10


def encode(ids):
    """Array/list of 'S?-123' strings -> int64 numpy array."""
    arr = pa.array(ids) if not isinstance(ids, (pa.Array, pa.ChunkedArray)) else ids
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy(zero_copy_only=False)
    is_s3 = pc.equal(pc.utf8_slice_codeunits(arr, 0, 2), "S3").to_numpy(zero_copy_only=False)
    return num + is_s3.astype(np.int64) * S3_OFFSET


def decode_pairs(df):
    """DataFrame with int s1_id, cand_id -> same pairs as id strings."""
    return pd.DataFrame({"s1_id": decode(df["s1_id"].to_numpy(), prefix="S1-"),
                         "cand_id": decode(df["cand_id"].to_numpy())})


def decode(nums, prefix=None):
    """int64 array -> numpy array of id strings. prefix forces e.g. 'S1-'."""
    nums = np.asarray(nums, dtype=np.int64)
    if prefix is not None:
        return (prefix + pd.Series(nums).astype(str)).to_numpy()
    is_s3 = nums >= S3_OFFSET
    text = pd.Series(np.where(is_s3, nums - S3_OFFSET, nums)).astype(str)
    return np.where(is_s3, "S3-" + text, "S2-" + text)
