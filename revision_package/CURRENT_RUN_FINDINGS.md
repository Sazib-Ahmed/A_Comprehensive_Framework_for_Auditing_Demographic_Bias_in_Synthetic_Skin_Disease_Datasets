# Findings from notebooks 00 and 01

Source archive: `runs.zip`, SHA-256 `90a9cee9d692c72a5a6ce9e64b5ebedafee050706d8c411e91b754fdf0cd4cba`.

## Validated inputs

- `complete_metadata_m.csv` is the correct historical multimodal metadata file.
- Its 565 relative paths all resolve under the hierarchical `Train/Test/Val` dataset tree.
- The tree contains 2,750 decodable image files. The metadata cohort contains 565 unique images; 2,185 archive images are outside that cohort.
- Reconstructed cohort counts are Train 438, Test 103, Val 24.
- The `skin` environment passed `pip check`; PyTorch detected the RTX 5060 Ti.
- The official pretrained DINOv3 snapshot is not yet present in the local Hugging Face cache.

## Duplicate and relationship screen

The pHash threshold was Hamming distance <= 6. A pHash hit is a screening candidate, not evidence of the same patient, source case, or generated parent.

### Entire 2,750-file archive

- 396 candidate pairs involving 520 unique files.
- 16 byte-identical and decoded-pixel-identical pairs.
- 380 pHash-only pairs requiring visual/source review.
- 193 candidate pairs cross the archive's Train/Test/Val directories.
- Two of those cross-split pairs are exact duplicates: one Test-Train Monkeypox pair and one Train-Val Herpes pair.
- One exact within-Train pair has conflicting folder labels (Herpes versus Varicela), which is a data-quality conflict.

Therefore, the current audit does not certify full-archive benchmark independence. Full-archive performance should not remain central unless affected results are handled under a corrected protocol.

### 565-image proxy-defined cohort

- 13 pHash candidate pairs have both images in the cohort; they involve 21 images and form 9 candidate components.
- None of the 16 archive-wide exact duplicates is in the 565-image cohort.
- All 13 cohort pairs are pHash-only and remain unresolved.
- Their pHash-distance distribution is 6 pairs at distance 0, 3 at distance 2, and 4 at distance 6.
- Eight pairs cross the archive's Train/Test/Val directories.
- All 13 pairs cross Train/Val/Test roles in at least one reconstructed seven-fold split; the exported 44 rows repeat the same 13 pairs across affected folds.

The cohort result is promising but not conclusive. Only visual/source verification can determine whether any of these pairs are truly duplicate or related variants. Missing patient, case, parent-image and acquisition-session identifiers mean that a negative pHash review cannot prove patient independence.

## Proxy and provenance fields

- Age proxy counts: 0-17 = 61, 18-59 = 493, 60+ = 11.
- Gender-proxy counts: female = 210, male = 355.
- Appearance-proxy counts: white = 338, asian = 70, indian = 50, latino hispanic = 50, black = 38, middle eastern = 19.
- Annotation route, source status and body site are unknown for all 565 cohort rows.

Consequences for the response:

- Reviewer comments on real-versus-synthetic effects, route selection bias and body-site/source confounding cannot be answered with new stratified results from these files. Resolve them by narrowing the title, abstract, aims and conclusions and by stating the missing provenance explicitly.
- The corrected mutually exclusive proxy counts and deterministic observed label mapping directly support the dataset-accounting and mapping responses.
- Use the terms `proxy-defined subgroup performance` and `appearance-proxy group`; do not call these verified demographic fairness results.

## Minimal next run order

1. Skip notebook 02 because the historical `.pth` checkpoints were deleted.
2. Run the revised notebook 04 once. It displays four contact-sheet pages for only the 13 cohort pairs and creates `reviewed_duplicate_pairs_cohort.csv`.
3. Fill only the `verified_related` column with `yes` or `no`, using the displayed images and any source evidence. Rerun notebook 04.
4. Request access to the official gated DINOv3 model and run notebook 02A once to cache it.
5. Run notebook 03. It performs one frozen DINOv3 inference pass and fits small fixed logistic probes. If notebook 04 created grouped folds, notebook 03 uses those folds automatically.
6. Run notebook 05 only if two people can produce genuinely independent, blinded new ratings.

No neural model was trained to produce these findings.
