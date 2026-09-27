# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline status: **end to end**. Cleaning → blocking → LightGBM matcher → reranker → submission.
Local validation F0.5: baseline 0.8286, model 0.9661, model + reranker 0.9688.

## Setup

```bash
pip install -r requirements.txt
```

By default the code reads the organisers' dataset from
`../../6ab10eb3b23ba_student_resource/student_resource/dataset` and writes
intermediate files to `../../work`. Override with `ER_DATA_DIR` / `ER_WORK_DIR`.

Run every command from this folder (`code/business_entity_resolution`).

## Steps

| # | command | reads | writes | time* |
|---|---|---|---|---|
| 0 | `python -m src.ingest` | dataset TSVs | `work/raw/*.parquet` | ~40 s |
| 1 | `python -m src.build_lexicon` | `work/raw` (train only) | `src/resources/indic_lexicon.json` | ~30 s |
| 2 | `python -m src.clean --workers 8` | `work/raw` | `work/clean/*.parquet` | ~2.5 min |
| 3 | `python -m src.blocking --split train` then `--split test` | `work/clean` | `work/candidates/*.parquet` | ~12 + 10 min |
| 4 | `python -m src.train --transfer` | candidates, clean | `work/models/lgbm.txt`, `decision.json` | ~20 min |
| 5 | `python -m src.predict` | test candidates, model | `../../output/*.tsv` (validated) | ~20 min |
| 6 | `python -m src.rerank train` then `predict` | step 4 model, candidates | `lgbm_stage2.txt`; final `../../output/*.tsv` | ~30 + 30 min |
| 7 | `python -m src.package --team "<team name>"` | `../../output`, code, docs | `../../<team>_submission.zip` (validated) | ~2 min |
| — | `python -m src.error_analysis` | step 4 model | `work/reports/error_analysis.md` | ~5 min |
| — | `python -m src.baseline` | candidates | baseline submission (no model) | ~10 min |
| — | `python -m src.report_cleaning` | `work/raw`, `work/clean` | `work/reports/cleaning_report.md` | ~4 min |

\*20-thread laptop, 24 GB RAM.

Step 1's output is committed, so step 2 reproduces without re-running it.

### 3. Blocking (`src/blocking.py`)
Per country (no true pair crosses countries), each record becomes a TF-IDF vector
of keys: `name_core` words, the name without spaces, `addr_norm` words, and
house number + next word. Keys in more than 20,000 records are dropped. A top-30
sparse product (`sparse_dot_topn`) gives each S1 entity its 30 most similar S2/S3
records. On the full training set this keeps **96.2% (India) / 96.7% (US)** of
true pairs, with 30 candidates per S1 entity out of 4–6M records per country.

### 4–5. Matching (`src/features.py`, `src/train.py`, `src/predict.py`)
33 features per candidate pair: competition context (rank of this S1 among all S1
entities listing the same record, gap to the best other S1), rapidfuzz name and
address similarities, house-number/state/legal-form agreement, and record flags.
LightGBM (MIT) is trained on 9M pairs from 300k sampled training entities, with 20%
of entities held out. Decision: each S2/S3 record goes to its most probable S1
entity (every record matches at most one), kept if probability ≥ τ. τ = 0.75 is
chosen on the held-out entities with the exact metric (`src/metrics.py`).
Countries absent from training (France) use τ = 0.8625: trained on one country and
scored on the other, the best τ rose to 0.90 (US→India) and 0.825 (India→US).

### 6. Reranker (`src/rerank.py`)
Error analysis (`src/error_analysis.py`) showed 35% of rejected true pairs have an
empty S2/S3 address and an exact name, but compete with ~27 S1 entities of similar
names at near-equal blocking scores. Stage 1 scores every candidate (66M training
pairs); competition features are recomputed from its probabilities (is this S1 the
model's clear favourite for the record?); a second LightGBM decides on stage-1
features + those. Stage 2 trains on entities stage 1 never saw, so its inputs are
out-of-sample as on test. Same held-out entities: stage 1 0.9670 → stage 2 0.9688.
Thresholds: 0.675 (seen countries), 0.7875 for unseen (stage 1's +0.1125 margin).

### 0. Ingest (`src/ingest.py`)
A faithful TSV→parquet copy of all 7 files: no cleaning, every value kept as a
string, and each file's row count checked against its line count. Two parser
defaults are overridden because they would corrupt this data. NA detection is off,
so names like "NA" and the literal "null" in addresses survive. CSV quote handling
stays on, so `"9Th Floor, ""Niagara"" Building"` becomes `9Th Floor, "Niagara" Building`.

### 1–2. Clean (`src/clean.py`, `src/translit.py`, `src/build_lexicon.py`)
One function per field, applied identically to all three sources and both splits.
No rows are dropped, merged or reordered. Raw columns are kept and cleaned views
added:

| column | what it is |
|---|---|
| `name_norm` | cleaned full name, legal forms canonical (`private limited` → `pvt ltd`) |
| `name_core` | `name_norm` minus legal forms, injected honorifics (`the`, `m/s`, `shri`, `dr`…) and stop words |
| `name_legal` | legal forms found, sorted (`ltd pvt`) |
| `name_alias` | the made-up name before `d/b/a`, `aka`, `t/a`, `formerly`, `trading as`, `née` (the real name is always after) |
| `name_is_web` | name was a bare domain or @handle / #hashtag (compare it space-free) |
| `name_had_indic`, `addr_had_indic` | field contained Indic script before romanisation |
| `addr_norm` | cleaned address; comma components kept in order, de-duplicated |
| `addr_numbers` | numeric tokens of the address in order, leading zeros dropped |
| `addr_state` | state code if recognised (US 50+DC, India's 16 states); empty for France |
| `addr_pobox` | PO box / PMB / BP number, removed from `addr_norm` (noise injected by S2/S3) |

What the cleaning undoes (every rule was measured on training data first):

- **Indic scripts.** About 25% of Indian S2/S3 names are in Devanagari, Bengali,
  Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada or Malayalam. `translit.py`
  romanises all nine using only Python's `unicodedata`. `build_lexicon.py` then
  learns 542 romanised→English word mappings (`epeks`→`apex`, `elaelapi`→`llp`)
  from positionally aligned training pairs. That raises the share of romanised
  words equal to a word in the matching Source 1 name from 50.6% to 95.4%.
- **Name junk.** Trade-name wrappers, `| www.x.com` suffixes, `- 8984535630`
  phones, `(ID: 95627)`, leading `--`/`<<`/`"`, `M/s`, brackets, hyphens between
  words, `&`→`and`, dotted acronyms (`P.L.L.C.`→`pllc`), repeated words, and
  domains/handles (`indriyaclub.com`→`indriyaclub`).
- **OCR swaps**, only the ones the generator uses (measured on pairs): `0→o`,
  `1→l`, `5→s`, `8→b`, `6→g` inside words, and `l` for a capital `I` (`lnc`→`inc`).
- **Accents** (injected and genuine) are stripped via NFKD.
- **Addresses.** Placeholders (`NULL`, `<NULL>`, `N/A`) are dropped; state
  names become codes (`Texas`→`tx`, `Maharashtra`/`महाराष्ट्र`→`mh`); street types,
  directions, ordinals and units get one spelling each (US, India and French
  forms: `Road`/`Rd`→`rd`, `R.`/`Rue`→`rue`, `second`→`2nd`); house-number markers
  (`#`, `No.`, `H.No`, `N°`) are removed while the numbers are kept; leading zeros
  are dropped.

Nothing branches on the `country` value, so France (test only) runs through the
same code as US and India.

### Results (`work/reports/cleaning_report.md`)
On 220k true training pairs vs 145k hard non-pairs (same country, same first name
word):

| | raw | cleaned |
|---|---:|---:|
| true-pair names exactly equal | 10.8% | 64.8% |
| true-pair addresses exactly equal | 7.2% | 28.4% |
| name AUC, all (token_sort, `name_norm`) | 0.865 | 0.936 |
| name AUC, India | 0.794 | 0.930 |
| name AUC, Indic-script names (token_set, `name_core`) | 0.316 | 0.977 |

## Facts about the data that shape the next stages
From `train_ground_truth.tsv`:
- 2.2M S1 entities; **only 5.6% are singletons** (identical in US and India). Most
  have 2–5 matches (max 11). Predicting nothing everywhere scores ≈ 0.056.
- **Each S2/S3 record matches at most one S1 entity** (0 exceptions in 7.6M pairs).
- **No true pair crosses countries**: country can partition blocking.
- About 26% of S2/S3 records match nothing (distractors).
- The validator only rejects duplicate IDs *within* a row.
