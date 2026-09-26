"""Stage 0: faithful TSV -> parquet conversion.

Nothing is cleaned here. Every value is kept as the exact string in the file so the
cleaning stage (clean.py) always works from an untouched copy.

Two parser defaults would silently corrupt this data, so both are overridden:
  * NA detection is off. pandas/pyarrow would otherwise turn business names such as
    "NA" or "None" and the literal "null" inside addresses into missing values.
  * Quote handling stays ON. Some fields are CSV-quoted with doubled quotes
    (e.g. "9Th Floor, ""Niagara"" Building"); a quote-aware parser restores them to
    9Th Floor, "Niagara" Building.

Each output's row count is checked against the file's line count.

Usage:  python -m src.ingest            (from code/business_entity_resolution)
"""

import sys
import time

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src import config


def _count_data_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # minus header


def read_tsv(path, columns):
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(
            delimiter="\t", quote_char='"', double_quote=True, newlines_in_values=False
        ),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            null_values=[],
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
        ),
    )
    if table.column_names != list(columns):
        raise ValueError(f"{path}: unexpected header {table.column_names}")
    return table


def ingest_file(src, dst, columns):
    t0 = time.time()
    table = read_tsv(src, columns)
    expected = _count_data_lines(src)
    if table.num_rows != expected:
        raise ValueError(f"{src}: parsed {table.num_rows} rows, file has {expected}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="zstd")
    print(f"  {src.name:<28} {table.num_rows:>10,} rows  {time.time() - t0:5.1f}s")


def main():
    for split in config.SPLITS:
        for source in config.SOURCES:
            ingest_file(
                config.raw_tsv(split, source),
                config.raw_parquet(split, source),
                config.RECORD_COLUMNS,
            )
    ingest_file(
        config.ground_truth_tsv(),
        config.RAW_DIR / "train_ground_truth.parquet",
        ("source1_entity_id", "matched_entity_ids"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
