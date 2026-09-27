# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Pavitra Sharma, Naman, Lakshya, Pranav Kumar Maurya  
**Submission Date:** [Date]

---

## 1. Executive Summary
We clean all 24.2M records into matching-ready text (including romanising nine
Indic scripts), generate 30 candidates per Source 1 entity with a per-country
TF-IDF keyword search (96.5% of true pairs kept, 99.9992% reduction), and score
pairs with a two-stage LightGBM whose second stage sees the first stage's verdict
on competing entities. A decision rule that gives every Source 2/3 record to at
most one Source 1 entity reaches a macro F0.5 of **0.969** on held-out training
entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data before any modelling:

- **Scale:** train 2.2M S1 / 5.0M S2 / 5.3M S3 records; test 1.7M / 4.9M / 5.1M.
- **Matches:** 7.6M true pairs. Only **5.6%** of S1 entities are singletons (same in
  US and India); most have 2–5 matches, at most 11. Predicting nothing scores ≈ 0.056.
- **Every S2/S3 record matches at most one S1 entity** (0 exceptions), and
  **no true pair crosses countries**. About 26% of S2/S3 records match nothing.
- **The noise is synthetic and systematic** (S2/S3 only; S1 is nearly clean):
  - *Scripts:* about 25% of Indian S2/S3 names are Devanagari, Bengali, Gurmukhi,
    Gujarati, Oriya, Tamil, Telugu, Kannada or Malayalam transliterations; state
    names too.
  - *Names:* case changes, hyphens for spaces, OCR digit swaps (0→o 14.8k, 1→l
    10.9k, 5→s 5.8k, 8→b 2.7k, 6→g 2.1k pairs; all others < 20), injected accents,
    character typos, shuffled word order, legal suffixes moved or bracketed,
    injected honorifics (the, m/s, shri, dr), filler words (Center, Group),
    trade-name wrappers ("X d/b/a Y", aka, t/a, formerly, née; in S3 the real
    name is always after the marker), bare domains and @handles, appended phone
    numbers and "(ID: n)" tags, and names replaced by unrelated made-up words.
  - *Addresses:* reordered components, US states as codes (S1/S2) vs full names
    (S3), Indian states the reverse, NULL / <NULL> / N/A placeholders (≈3%), empty
    addresses (≈3%), street-type abbreviations and typos, house numbers
    zero-padded or missing their first digit, city replaced by county/township.
  - *France (test only)* uses the same generator in French: SARL/SAS/EURL, Rue/R.,
    Allée, N°, BP boxes, region vs département.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (hybrid)  
**Core Innovation:** Competition-aware matching. Because each Source 2/3 record
belongs to at most one Source 1 entity, the model is given features describing
how this entity compares with every other entity that wants the same record,
first from blocking scores, then (stage 2) from stage-1 probabilities. The final
decision assigns each record to its single most probable entity.

Pipeline: ingest → clean → block (per country) → features → LightGBM stage 1 →
stage-1 competition features → LightGBM stage 2 → one-entity-per-record
assignment with a tuned threshold.

**Compliance.** Only the provided files are used: no external data, APIs or
lookups. Transliteration uses Python's Unicode database; the transliteration
lexicon is learned from training pairs. Models are LightGBM (MIT), far below 8B
parameters. Libraries: pyarrow (Apache-2.0), sparse_dot_topn (Apache-2.0),
rapidfuzz (MIT), numpy/pandas/scipy (BSD-3).

---

## 3. Candidate Generation (Blocking)
Records are blocked **within country**: no training pair crosses countries, and
country is treated as an open set of labels, so France runs through the same code.

Each record becomes a TF-IDF vector of keys from the cleaned columns:

| key group | example |
|---|---|
| words of the cleaned core name | `keystone`, `robotics` |
| the core name with spaces removed | `keystoneroboticsstudios` (matches domain/handle names) |
| words of the cleaned address | `welshwood`, `nashville`, `441` |
| house number + next word | `441_welshwood` |

Keys found in more than 20,000 records of a country are dropped; the rest get
weight log(N/df); vectors are L2-normalised. One sparse top-k product per country
(`sparse_dot_topn`) returns each S1 entity's 30 most similar S2/S3 records.

- **Blocking keys used:** cleaned name words, space-free name, address words, house number + street word; country as the partition.
- **Candidate pairs generated:** 51,972,999 for the test set (30.0 per S1 entity).
  Reduction ratio 99.99923% against all within-country pairs (6.72 × 10¹²) and
  99.99970% against the full cross product.
- **How you ensured true matches were not lost:**
  - Cleaning first: exact name agreement on true pairs rises from 10.8% to 64.8%,
    so matches share keys.
  - Several independent key types: a match found by name *or* address still ranks high.
  - K and the key cut-off were chosen by measured recall on the full training set:
    **96.2% (India) and 96.7% (US)** of true pairs are in the top 30.
  - Tried and rejected: a 2,000-record key cut-off (94.0%), separate name-only and
    address-only searches (no gain at equal budget), one-deletion typo keys (88.6%:
    they swamp the address keys), last-3-digit house-number keys (no net gain),
    name character trigrams at reduced weight (+0.17 pt recall for twice the
    blocking time), and K = 50 (+0.7 pt recall for 67% more candidates).

---

## 4. Matching Model

**Features used** (33 in stage 1, 40 in stage 2):
- Name features: rapidfuzz token-sort (legal forms kept), token-set, ratio,
  Jaro-Winkler, partial ratio on the space-free name (domain/handle names), exact
  name, first-word match, word-count difference, legal-form agreement and conflict.
- Address features: token-set, token-sort and partial ratio; house-number set
  similarity; first number equal; last-3-digits equal (first digit dropped);
  missing number; state agreement and conflict.
- Other:
  - *Competition (blocking):* blocking score, rank in the S1 list, rank of this S1
    among all S1 entities listing the record, number of such entities, and gap to
    the best other one.
  - *Competition (stage 2):* the same, recomputed from stage-1 probabilities:
    p1, its rank and gap for the record, total probability mass on the record,
    rank and relative p1 within the S1 list, count of p1 ≥ 0.5.
  - *Record flags:* S2 or S3, Indic-origin name, domain/handle name, d/b/a
    wrapper, empty address; name frequency of both sides (chains).
  - Country is never a feature.

**Model type:** LightGBM gradient-boosted trees (binary objective, 127 leaves,
learning rate 0.05, early stopping), in two stages. Stage 1: 1,714 trees on 9M
pairs of 300k sampled entities. Stage 2: 619 trees on 9M pairs of 300k *different*
entities, so its stage-1 inputs are out-of-sample as on test.

**Threshold selection method:** each S2/S3 record is assigned to its single most
probable S1 entity; the pair is kept if the probability ≥ τ. τ maximises the
exact competition metric (macro per-entity F0.5, singletons included) on 20% of
entities held out by id: τ = 0.675 (stage 2). For countries absent from training
(France), τ is raised by the margin measured with leave-one-country-out runs
(trained on US and scored on India, the best τ rose from 0.725 to 0.90; India →
US, to 0.825): τ = 0.7875.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9688 on held-out training entities (India 0.965, US 0.973).

  | stage | held-out F0.5 |
  |---|---:|
  | blocking-score baseline | 0.8286 |
  | LightGBM stage 1 | 0.9661 |
  | + stage-2 reranker (same entities: 0.9670 → 0.9688) | 0.9688 |
  | perfect matcher on our candidates (blocking ceiling) | 0.9885 |

  Public leaderboard: [score].
- **Common false positives (wrong merges):** 0.79% of predicted pairs. Mostly
  same-name entities at neighbouring numbers of the same street (28 vs 29
  Atlantic Ave), sister companies sharing a building ("IP Fuels Pvt Ltd" vs
  "IP Fuels Public Ltd"), and near-identical names differing by one generic word
  ("Legacy Income Group" vs "Legacy Income Corp").
- **Common false negatives (missed matches):** 7.9% of true pairs (stage 1).
  44% never become candidates: names replaced by made-up words or typo'd together
  with a damaged house number, and domain names whose address lost a digit
  ("womenshealthvalley.com | 454 cross keys rd" vs "4545 Cross Keys Rd"). 53% are
  candidates rejected by the model; 35% of those have an empty S2/S3 address and
  an exact but common name contested by ~27 S1 entities. This motivated the
  stage-2 reranker.

---

## 6. Conclusion
Careful, measured cleaning (especially Indic transliteration) plus a
country-partitioned TF-IDF blocker gives a high-recall, 30-per-entity candidate
set. A LightGBM matcher that reasons about competition between entities, and a
one-entity-per-record assignment, turn it into 0.969 macro F0.5. The main lessons:
measure every rule on labelled pairs before keeping it, and exploit the problem's
structure (country partition, one entity per record) rather than add generic
model capacity.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`, run from `code/business_entity_resolution/`
(full commands and timings in its README):

| step | module | output |
|---|---|---|
| 0 | `ingest.py` | faithful TSV → parquet copy, row counts checked |
| 1 | `build_lexicon.py` | romanised-Indic → English lexicon (learned from train pairs) |
| 2 | `clean.py` (+ `translit.py`) | cleaned name/address columns |
| 3 | `blocking.py` | top-30 candidates per S1 entity, per country |
| 4 | `train.py` (+ `features.py`, `metrics.py`) | stage-1 model, threshold, transfer runs |
| 5 | `rerank.py train` / `rerank.py predict` | stage-2 model; `output/matching_results.tsv`, `output/candidate_pairs.tsv`, validated |

Also: `baseline.py` (no-model baseline), `error_analysis.py`, `report_cleaning.py`,
`submit.py` (writer + validator), `ids.py`, `config.py`.

### B. Additional Results
Cleaning effect on 220k true training pairs vs 145k hard non-pairs (same country,
same first name word):

| measure | raw | cleaned |
|---|---:|---:|
| true-pair names exactly equal | 10.8% | 64.8% |
| true-pair addresses exactly equal | 7.2% | 28.4% |
| name AUC (token-sort) | 0.865 | 0.936 |
| name AUC, Indic-script names | 0.316 | 0.977 |

Transliteration lexicon: romanised Indic words equal to the matching Source 1
word rise from 50.6% to 95.4% (542 learned entries).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
