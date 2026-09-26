# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline code for the team. Data is not in this repo: everyone uses their own
copy of the organisers' dataset and regenerates the cleaned files locally
(about 3 minutes, identical output on every machine).

## Setup

1. Clone this repo.
2. Extract the organisers' `6ab10eb3b23ba_student_resource.zip` into the repo
   root, so `6ab10eb3b23ba_student_resource/student_resource/dataset/` sits next
   to `code/`. (Or keep it elsewhere and set `ER_DATA_DIR` to its `dataset/` folder.)
3. Build the cleaned data, from `code/business_entity_resolution/`:

   ```bash
   pip install -r requirements.txt
   python -m src.ingest              # TSV -> work/raw (~40 s)
   python -m src.clean --workers 8   # work/raw -> work/clean (~2.5 min)
   ```

The cleaned parquet files land in `work/clean/` at the repo root (not versioned).
`code/business_entity_resolution/README.md` documents every column and cleaning rule.

## Layout

```
code/business_entity_resolution/   the submission's code folder (src/, README, requirements)
work/                              generated locally: raw/, clean/, reports/ (git-ignored)
6ab10eb3b23ba_student_resource/    organisers' dataset (git-ignored)
```

## Working together

- Pull before you start, and re-run `src.clean` if anything under `src/` changed.
- Add new stages (blocking, matching, submission) as modules in `src/`, run the
  same way: `python -m src.<module>`.
- Never commit data files; `.gitignore` keeps everything except code out.
