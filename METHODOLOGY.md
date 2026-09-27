# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Annie are you okay  
**Team Members:** Rhuthvija Allavala, Vennela Indukuri  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a scalable, precision-first entity resolution pipeline in five steps:

1. **Normalise** names and addresses. Indic-script words are romanised with a dictionary *learned from the training matches*.
2. **Block** with *composite rare keys*: pairs of rare name words, 4-letter name fragments, street words and house numbers. This keeps **97.4%** of true matches.
3. **Stage 1:** a LightGBM classifier scores each candidate pair on 47 similarity features. It also acts as a learned filter: only pairs with stage-1 probability ≥ 0.01 go on to the final model, which is **3.9 candidates per Source 1 entity**.
4. **Stage 2 (final model):** a second LightGBM re-scores those pairs. It uses *group* evidence: sibling copies of the same business, twins, and competing entities. It also uses *"what differs"* features that recognise which kinds of differences are generator noise and which ones indicate a different business.

5. **Stage 3 (cross-encoder):** a small transformer (MiniLM, 33M parameters, Apache-2.0) fine-tuned on our training candidate pairs reads both records' text side by side. Its score is blended with the stage-2 probability.

Each S2/S3 record is then assigned to at most one S1 entity, the one with the highest probability, if that probability is above a tuned threshold.

Honest out-of-fold validation on all 2.2M training entities gives **macro F0.5 = 0.9813** for stages 1–2, with 99.6% pair precision and 95.6% pair recall. On a separate 15% holdout of S1 entities, stage 3 raises macro F0.5 from 0.9816 to **0.9874**. Public leaderboard: **0.979**.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training set (2.21M S1, 5.03M S2 and 5.29M S3 records) found these patterns:

| Observation | Number | Consequence for the design |
|---|---|---|
| S1 entities with no match (singletons) | 5.6% | an empty prediction is worth 1.0, so we need a confident threshold |
| True matches per S1 entity | mean 3.46, 0–11 | recall per entity matters as well as precision |
| S2/S3 records that match nothing (decoys) | 26% in train, **37% in test** (estimated) | the test set is harder, and it needs a stricter threshold (confirmed on the leaderboard) |
| S2/S3 records matched to more than one S1 | **0** | each S2/S3 record has at most one owner, which is used in the assignment step |
| True pairs whose country labels agree | 100% | blocking is done within the country label, treated as an open set |
| S2/S3 addresses that are empty | 3.3% | name-only evidence must be possible |
| India S2 / S3 names written in an Indic script | ~24% / ~12% (9 scripts) | romanisation is needed before comparing |
| Decoys with a near-identical twin elsewhere in S2/S3 | 3.7% (vs 83% of exact true copies) | "group" evidence separates real businesses from lone decoys |

Noise that we observed and handle explicitly:

- **Names**
  - Typos
  - OCR-style digit substitutions ("Appare1s", "6roup", "1ndia")
  - Shuffled word order
  - Brackets and junk prefixes (`***`, `<<`)
  - Honorifics (Mr, Dr, Sri, Smt)
  - Generic words added or dropped ("Center", "Services")
  - Website-style names (`lifeinvestments.com`)
  - `DBA:` and `formerly` names
  - ID tags "(ID: 58156)" and phone numbers inside names
  - Legal forms in many spellings (Pvt/Private/Privte, L.L.C., S.A.S)
- **Addresses**
  - Abbreviations (Rd/Road, St/Street/"Saint", R./Rue, BD/Boulevard)
  - State names vs codes (Maharashtra/MH/महाराष्ट्र, New York/NY)
  - Dropped or reordered components
  - House-number changes and ranges ("6601-6605")
  - Mailbox and unit numbers ("PMB 2727", "Unit 414", "Fl 0")
  - Landmark and filler words

**Key insights:**

1. The generator reuses a limited vocabulary of names and streets. Single words are therefore weak blocking keys, even the rarest word of a record (median ~30 S1 records), so blocking uses *combinations* of rare features.
2. True copies and decoys differ in the **kind** of differences they show, not only in how much they differ. For example:
   - An extra "services" in the name is generator noise (true match ~12%), while an extra "group" or "holdings" means a different business (~0%).
   - A house number that lost its first or last digit is typical noise (~50% true), while a changed last digit usually means a neighbouring business (~5%).

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (LightGBM) + constrained assignment (hybrid)

**Core Innovation:**

1. **Composite rare-key blocking.** Pairs of a record's rarest name keys, 4-letter name fragments, address words and house numbers are combined into keys. Budgets are asymmetric: S1 records emit more keys than S2/S3 records, because an S2/S3 record is usually a degraded copy of its S1 record.
2. **A learned transliteration dictionary** built from the training matches, so no external resource is needed.
3. **Stage-2 "group" and "what differs" features:**
   - sibling agreement among the confident candidates of an S1 entity
   - twin counts
   - competition margins
   - noise-word scores for extra or missing words, learned (cross-fitted) from the training pairs
   - an edit-type code for house numbers and names
4. **A one-owner assignment rule**, tuned directly for macro F0.5.

```
TSV -> normalise -> composite-key blocking (DuckDB)            ~24 pairs / S1, 97.4 % recall
    -> stage 1: 47 pair features (RapidFuzz) -> LightGBM p1   (learned filter: keep p1 >= 0.01)
    -> stage 2: p1 + group + "what differs" features -> LightGBM p2      ~3.9 pairs / S1
    -> each S2/S3 record keeps only its best S1 -> accept if p2 >= threshold -> output TSVs
```

**Generalisation to France (unseen country).** No step hard-codes the country set:

- `country` is only used to keep comparisons inside the same label. It is never a model feature.
- All features are generic similarity or difference measures.
- IDF statistics and twin counts are computed per split, unsupervised.
- French legal forms (SARL, SAS, SASU, EURL, SCI, SA), street words (Rue/R., Av, Bd, Allée, Impasse, Chemin) and stopwords (de, du, des, la, le) are handled by the normaliser as plain linguistic knowledge.
- **Unseen vocabulary.** Words that never occur in training, such as French département names (Gironde, Nord) and French business words (Fils, Associés, Collège), would otherwise get the pessimistic default noise score. Instead, their scores are estimated on the test candidate pairs **without labels**: we take the mean stage-1 probability of the pairs in which the word appears on only one side, and map it onto the training scale with a monotone calibration fitted on training words. On training words this label-free estimate correlates 0.99 with the true label-based score. Stage 1 does not use these word scores, so the estimate is not circular. After the fix, harmless French noise words ("Et Fils", "& Associés", "Développement") are accepted, and French business-type swaps ("Collège" → "Culturelle", "Loisirs" → "Sport", typical decoys) are rejected, which mirrors the US/India pattern. This improved the public leaderboard score from 0.972 to 0.973.

---

## 3. Candidate Generation (Blocking)

**Single keys per record:**

| Key | What it is |
|---|---|
| `W:` | core-name tokens (legal forms, honorifics and stopwords removed; DBA/formerly names included) |
| `C:` | the whole core name with spaces removed |
| `P:` | the first 5 letters of long name tokens |
| `G:` | 4-letter fragments of the space-free core name, so a typo only breaks the fragments around it |
| `A:` | address words |
| `N:` | address numbers with ≥2 digits |

**Composite keys.**

- Single keys are ranked by rarity, their document frequency among S1 records of the same country.
- An **S2/S3 record** keeps its 2 rarest name keys, 4 rarest address keys and 3 rarest name fragments.
- An **S1 record** keeps 3 name keys, 7 address keys and 6 fragments.
- Composite keys are pairs among the kept keys: name×address, address×address, name×name and fragment×fragment. Single keys are also used on their own when they are very rare (df ≤ 3).
- Keys shared by more than 30 S1 or more than 150 S2/S3 records are dropped.

**Scoring and pruning.**

- A pair's blocking score is the sum of the IDF-style weights of the shared composite keys.
- We keep the top 4 S1 per S2/S3 record and the top 18 S2/S3 per S1 (union).
- All of this runs as SQL joins in DuckDB, in hash slices, with disk spilling, so it fits in 8 GB RAM.
- Top-K selection uses `max_by(…, k)` aggregation instead of a full sort.

**Stage-1 learned filter.** The stage-1 classifier is a cheap first model that scores every blocking candidate. Only pairs with p1 ≥ 0.01 reach the final model. This removes 84% of the blocking candidates and loses only 0.05% of true pairs. By the rules' definition, `candidate_pairs.tsv` is exactly this final set.

**Blocking keys used:** composite name × address, address × address, name × name and 4-gram × 4-gram keys, plus compact-name, prefix and house-number keys.

**Candidate pairs generated:**

| | Blocking candidates | Final candidates (`candidate_pairs.tsv`) |
|---|---|---|
| Training | 53.6M (24.3 / S1) | 8.62M (**3.9 / S1**) |
| Test | 47.3M (27.3 / S1) | 6.83M (**3.9 / S1**) |

The within-country cross product is about 1.2×10¹³ pairs, so the final candidate set is a reduction ratio of 99.99993%.

**How you ensured true matches were not lost:** blocking recall was measured on the training ground truth at every iteration (`eval_blocking.py`):

| Version | Candidates / S1 | Recall (all) | US | India |
|---|---|---|---|---|
| Single-token inverted index | too expensive (2.6B raw pairs) | n/a | n/a | n/a |
| Composite keys, symmetric budgets | 14.9 | 95.2% | 96.6% | 92.8% |
| Composite keys, asymmetric budgets | 21.2 | 97.1% | 97.5% | 96.5% |
| **+ name 4-gram keys, unit/ID/range normalisation (final)** | 24.3 → **3.9 after stage-1 filter** | **97.4%** (97.38% after filter) | **97.8%** | **96.9%** |

Most of the remaining misses are S2/S3 records with an empty address whose name is shared by several S1 entities. For example, "Delta Project" with no address has three identical S1 candidates in three states. These cannot be resolved from the record alone, and a precision-first metric does not reward guessing.

---

## 4. Matching Model

### Stage 1: pair classifier (47 features, LightGBM, 500 trees, 127 leaves, trained on 7M sampled pairs)

- **Name:**
  - RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio` and `WRatio`
  - Jaro-Winkler
  - Compact-name ratios
  - Full-name token-sort ratio
  - Best ratio against the DBA/formerly name
  - Token Jaccard and containment
  - Typo-tolerant plain and IDF-weighted word coverage, both directions
  - Legal-form agreement
  - Word counts
  - Website-name flag
- **Address:**
  - Token-set, token-sort and partial ratios
  - Jaccard and containment
  - Typo-tolerant plain and IDF-weighted coverage
  - State agreement
  - Empty-address flag
- **Numbers:** first (house) number equal, number-set Jaccard and containment.
- **Blocking and competition:** blocking score, number of shared keys, rank of this S1 among the record's candidates and vice versa, relative scores, and candidate counts.

### Stage 2: group-aware and "what differs" re-scoring (37 features, LightGBM, 700 trees, 255 leaves, learning rate 0.03)

- **Competition:** stage-1 probability and logit, the best competing S1 probability for this record, the margin to it, the number of plausible S1s, and per-S1 counts and sums of confident candidates.
- **Siblings:** similarity of this record to the *other* confident candidates of the same S1: max name, address and combined similarity, the number of close siblings and their probability, and a probability-weighted similarity.
- **Twins:** how many S2/S3 records share this exact core name (and house number).
- **"What differs":**
  - The number of extra and missing name words and address words after typo-tolerant matching.
  - A noise-word score (min and mean) for the extra and missing words. This is P(true match | this word appears on only one side), learned from training candidate pairs and **cross-fitted** (each fold uses scores learned on the other fold), with smoothing towards a prior for unseen words.
  - An edit-type code for the house number (equal, first/last digit dropped or added, first/middle/last digit changed, …).
  - An edit-type code for single-word name changes (operation and position).
  - Whether the first and last name words match.

**Unseen words at test time:** see Section 2.2. Noise scores for words absent from training are estimated without labels (`test_vocab.py`).

**Model type:** two LightGBM binary classifiers (MIT license). Both are tiny, far below the 8B parameter limit.

**Threshold selection method:**

1. S1 entities are split into 2 folds by hash. Both stages are trained 2-fold, which gives out-of-fold probabilities for every training pair. Stage 2 is trained on stage-1 *out-of-fold* probabilities, so its inputs look like they will at test time.
2. The exact challenge metric is swept over the threshold after the one-owner rule. The out-of-fold optimum is t = 0.675, and the curve is flat within ±0.0002 between 0.60 and 0.75.
3. The test set has about 40% more decoys than training. Leaderboard probes confirmed that a stricter threshold is better there: +0.002 when moving from 0.65 to 0.75–0.85. The stage-2-only submission used 0.775. With stage 3, the blended score is thresholded at 0.75 (holdout optimum 0.725, nudged stricter for the same reason).

---

### Stage 3: cross-encoder re-scoring (GPU step)

The hand-built features describe *how much* and *what kind* of text differs, but some decoys still look like noisy copies: "PCJ Innovative Corp" vs a decoy "PCJ Innovate Corp", or a one-word business-type swap. A model that reads the raw text pair can learn these generator patterns directly.

- **Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (Apache-2.0, 33M parameters), with one output logit. It is fine-tuned with binary cross-entropy for one pass (70-minute budget, batch 128, learning rate 4e-5, fp16) on a Google Colab T4. Notebook: `colab/ber_cross_encoder.ipynb`.
- **Input:** `"<name> | <address>"` for the S1 record and for the S2/S3 record, romanised with the same learned transliteration + ASCII folding as the rest of the pipeline, max 96 tokens.
- **Training data:** 2.76M stage-2 candidate pairs from 700k training S1 entities. 15% of S1 entities are held out, and no training pair shares an S1 *or* an S2/S3 record with the holdout.
- **Blending:** a LightGBM with 10 features (stage-2 logit, cross-encoder logit, and each score's gap to the best competitor for the same S2/S3 record and within the same S1 entity) is fitted on the holdout. `ce_blend.py` evaluates it 2-fold before fitting on the whole holdout.
- **Holdout macro F0.5:** stage 2 alone 0.9816 → cross-encoder alone 0.9828 → **blend 0.9874** (best threshold 0.725, curve flat from 0.65 to 0.80). Adding all 35 stage-2 features to the blender gave only +0.0002, so the simpler 10-feature blender is used.
- **Test:** the cross-encoder scores the same 6.8M stage-2 pairs (no new candidates), so `candidate_pairs.tsv` is unchanged. Submitted threshold: 0.75.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9813** out-of-fold on all 2.21M training S1 entities, with pair precision **99.6%** and pair recall **95.6%**.

**Progress of the design** (out-of-fold validation, then public leaderboard):

| Version | OOF macro F0.5 | Public LB |
|---|---|---|
| v1: composite blocking + 37-feature LightGBM | 0.9650 | 0.954 |
| v2: asymmetric blocking + coverage/IDF features | 0.9742 | 0.963 (t = 0.65), 0.965 (t = 0.75 / 0.85) |
| v2 + stage 2 (group features) | 0.9763 | 0.966 |
| v3: 4-gram blocking, better normalisation, stage 2 with "what differs" features | 0.9813 | 0.972 (t = 0.775 and t = 0.85) |
| v3 + label-free noise scores for unseen (French) words | 0.9813 (training unchanged) | 0.973 |
| **v5: + stage-3 cross-encoder blend, t = 0.75 (final)** | 0.9874 (15% S1 holdout; stage 2 alone 0.9816 on the same holdout) | **0.979** |

**Common false positives (wrong merges).** These are mostly decoys that copy an entity's address and change a short name by one or two letters ("NO Energy" vs "UO Energy"). There are also records with the same name and street but a neighbouring house number. The "what differs" features were added specifically for these cases, and they cut pair-level false positives by about half (precision 99.2% → 99.6%).

**Common false negatives (missed matches).**

- About half are never candidates. These are mostly empty-address records whose name matches several S1 entities, which is inherently ambiguous.
- The rest score below the threshold. Typical cases:
  - the name replaced by a brand or DBA name
  - a heavily trimmed address combined with a name typo
  - a house-number change combined with a changed business-type word

Loss decomposition (out-of-fold): most of the remaining macro-F0.5 loss comes from entities where all but one match are found (recall 1 − 1/n). Wrong merges and fully missed entities account for the rest.

---

## 6. Conclusion

A precision-first ER pipeline works well on this data. It combines:

- careful normalisation, including a transliteration dictionary learned from the data itself
- composite rare-key blocking that stays scalable and still keeps 97.4% of true matches
- a two-stage LightGBM whose second stage looks at the whole group of records around each pair and at *what kind* of differences separate two records

The final model sees only 3.9 candidates per entity and reaches 0.981 out-of-fold. A small fine-tuned cross-encoder that reads the raw text pair adds about +0.006 on a holdout (0.979 on the public leaderboard). The main lesson: on this data, the *type* of disagreement (which word, which digit) is more informative than the *amount* of disagreement. Learning that from the training pairs gave the largest single improvement.

---

## Appendix

### A. Code Artefacts

The code ships in `code/business_entity_resolution/`. `run_pipeline.sh` runs everything, `requirements.txt` pins the packages, and `artifacts/` holds the learned files:

- `translit.json`: transliteration dictionary
- `noise_vocab.json`: noise-word scores
- `soft_calibration.json`: calibration for the label-free word estimate
- `model.txt` + `model_meta.json`: stage 1
- `model_stage2.txt` + `model_stage2_meta.json`: stage 2
- `blend.txt`: stage-3 blender (stage-2 probability + cross-encoder score)

`colab/ber_cross_encoder.ipynb` is the GPU step: it fine-tunes and scores the cross-encoder.

The `src/` modules, in pipeline order:

| Module | Role |
|---|---|
| `learn_translit.py` | learns the transliteration dictionary |
| `normalize.py`, `preprocess.py` | normalisation |
| `blocking.py`, `eval_blocking.py` | composite-key candidate generation (DuckDB) and its recall |
| `features.py`, `build_features.py` | 47 stage-1 features |
| `train.py` | stage-1 OOF training, threshold sweep, final model |
| `predict.py` | stage-1 scores for test |
| `stage2.py` | group features, stage-2 training and prediction |
| `diff_features.py`, `stage2_diff.py` | "what differs" features and the cross-fitted noise vocabulary |
| `test_vocab.py` | label-free noise scores for words unseen in training |
| `ce_export.py`, `ce_blend.py` | stage 3: exports pairs + text for the cross-encoder notebook, then blends its scores (holdout-validated) |
| `final_output.py` | one-owner assignment, threshold, and both TSVs |
| `evaluate.py`, `metrics.py` | the exact challenge metric |

Entry points:

```bash
bash run_pipeline.sh <dataset_dir> <output_dir> [work_dir]                  # full retrain + inference
SKIP_TRAIN=1 bash run_pipeline.sh <dataset_dir> <output_dir> [work_dir]     # inference with shipped models
CE_DIR=<ce_dir> SKIP_TRAIN=1 bash run_pipeline.sh ...                       # + stage 3, after running the notebook
```

Every matched ID is also a candidate ID, and the official validator prints PASS for both files.

### B. Additional Results

Stage-2 threshold sweep (out-of-fold macro F0.5):

| t | 0.60 | 0.65 | **0.675** | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|
| F0.5 | 0.9811 | 0.9812 | **0.9813** | 0.9813 | 0.9811 | 0.9809 | 0.9804 |

Top stage-2 features by gain: competition margin, stage-1 probability and logit, noise score of the extra name words, house-number edit type, sibling address similarity, per-S1 probability mass, twin count, close-sibling probability, and noise score of the missing name words.

**Compliance:**

- No external data, APIs, geocoding or internet lookups are used. The dictionary, noise vocabulary and all statistics are learned from the provided files. The unseen-word scores are computed from the provided test records without any labels.
- Models: LightGBM (MIT).
- Libraries: polars (MIT), DuckDB (MIT), RapidFuzz (MIT), anyascii (ISC), numpy (BSD), pyarrow (Apache-2.0).
