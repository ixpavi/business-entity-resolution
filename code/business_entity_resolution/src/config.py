"""Paths shared by every pipeline stage.

Override the defaults with environment variables so the pipeline can be run from
any checkout:
    ER_DATA_DIR  folder holding train/ and test/ (the organisers' dataset/ folder)
    ER_WORK_DIR  scratch folder for parquet caches, features and reports
    ER_RUN_DIR   folder for one pipeline run's candidates, features, models and
                 output files (default: WORK_DIR, with output/ at the project root),
                 so an experimental run cannot overwrite the current submission
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(
    os.environ.get(
        "ER_DATA_DIR",
        _PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
    )
)
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", _PROJECT_ROOT / "work"))

RAW_DIR = WORK_DIR / "raw"
CLEAN_DIR = WORK_DIR / "clean"
REPORT_DIR = WORK_DIR / "reports"

_run = os.environ.get("ER_RUN_DIR")
RUN_DIR = Path(_run) if _run else WORK_DIR
OUTPUT_DIR = RUN_DIR / "output" if _run else _PROJECT_ROOT / "output"

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")
RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def raw_tsv(split, source):
    return DATA_DIR / split / f"{split}_{source}.tsv"


def ground_truth_tsv():
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def raw_parquet(split, source):
    return RAW_DIR / f"{split}_{source}.parquet"


def clean_parquet(split, source):
    return CLEAN_DIR / f"{split}_{source}.parquet"
