# Revision package: start here

**Windows/Jupyter users:** begin with `JUPYTER_RUN_GUIDE.md`, then run the single-cell notebooks in `notebooks/`. Their defaults already use the supplied repository, hierarchical `Synthetic_Skin_Disease_Dataset` and `complete_metadata_m.csv` paths. The command-line sections below remain as a reproducible reference.

Read `REVISION_PLAN.md` first. It contains the full 18-major/15-minor response matrix, verified errors, proposed framing, recovered results and four-day schedule. `REVIEW_RESPONSE_DRAFT.md` is an author working draft, not a completed rebuttal. `MANUSCRIPT_REPLACEMENTS.tex` contains proposed insertions, not a revised submission-ready paper.

## What has actually run

- `01_audit_repository.py` ran against the actual GitHub snapshot at commit `c81c8917afab02f04d9ceb56cef04a6474f8fd99`. Its outputs are in `verified_results/repository_audit/`.
- The CPU scripts were checked with unit/failure-case tests and an artificial end-to-end fixture. Test data/results are not included as study evidence.
- Actual image hashing, annotation-route comparisons, checkpoint inference, real disease confusion matrices and new baselines still require the author's files. Scripts 04 and 08 cannot be validated end to end here because the release contains neither images nor checkpoints and this runtime has no PyTorch installation. They do not train anything.

No manuscript values were silently overwritten, no original notebook was executed, and no changes were pushed to GitHub.

## Installation and checks

Use a new analysis environment for CPU work:

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

For historical checkpoint recovery, use the author's original compatible Python/PyTorch/TorchVision/Transformers environment and keep its `pip freeze`. Do not upgrade it blindly. The exact original package versions are not in the repository; do not invent them. Scripts 04 and 08 use local model/config files and never automatically download weights.

Commands below run from the extracted package directory. Replace every example path with an actual path. Windows paths may be written as `D:/AIUB/Neural/...` and quoted. Every script requires a new or empty output directory and refuses to overwrite nonempty results.

## A. Reproduce the work already completed: no training

```bash
python scripts/01_audit_repository.py --repo "/path/to/repository" --out runs/repository_audit
```

Outputs include sample-vs-population SD, fold metrics, all exact saved subgroup counts, marginal Wilson intervals, threshold sensitivity, historical notebook metrics, copied text classification reports, and a provisional paired comparison. The fold pairing is not identity-verified. Old EOD numbers are not recertified by this script.

## B. Audit actual data and reconstruct/check splits

Keep the original metadata CSV unchanged. Script 02 creates a separate audit copy, preserving row order. It derives new audit age bins from numeric age but does not change old model inputs.

```bash
python scripts/02_audit_data.py --data-root "/path/to/Synthetic_Skin_Disease_Dataset" --metadata "/path/to/complete_metadata_m.csv" --reconstruct-legacy-splits --confirm-original-row-order --out runs/data_audit
```

This combined mode scans/hashes the image tree once, creates `reconstructed_fold_manifest.csv`, and writes the cross-role duplicate audit in the same output directory. Use script 09 separately only when reconstruction must be decoupled from image scanning.

Prefer historically saved folds to reconstruction. If paths were saved with an old absolute prefix, use `--strip-prefix "D:/AIUB/Neural/Synthetic_Skin_Disease_Dataset"` on scripts 02/04/08. The default mode strips only that explicitly supplied prefix and does not guess by basename. `--flat-images` is available only for an explicitly flattened copy and first rejects ambiguous basenames; it is not used by the supplied Jupyter notebooks.

Optional route manifest: `sample_id,annotation_route`, with `sample_id` equal to the image's full relative path from the dataset root. Pass `--route-manifest "/path/to/verified_routes.csv"`. If original metadata already has an annotation-route column, reconcile it there rather than supplying a conflicting second source. Preserve route provenance in the response. Do not infer source/route from filenames or simply rerun DeepFace and assume the old decisions are reproduced.

The default 64-bit pHash Hamming threshold is 6. This is a screening parameter, not a validated medical duplication threshold. Inspect candidates, preferably compare thresholds 4/6/8, and document decisions. Exact byte and decoded-pixel matches are also recorded. Candidates are never deleted or automatically labeled as common patients. Cross-role overlap must include training–validation, training–test, and validation–test; the original archive split alone does not describe the proxy CV folds.

Important outputs: `image_inventory.csv`, `duplicate_candidates_for_review.csv`, `dataset_accounting.csv`, `observed_mapping.csv`, `audit_metadata.csv`, disease/route/proxy/source cross-tabs, and, with a manifest, `duplicate_candidates_crossing_roles.csv` and `known_related_cases_crossing_roles.csv`.

To merge other verified source/body-site/patient metadata, first add those columns by exact sample ID to a copy of the original table with a validated one-to-one join. The provided scanner does not infer them. Unknown values are an analysis result, not blanks to fill speculatively.

## C. Recover original OOF predictions: no training

Need original ordered metadata, image tree, exact local DINOv3 config, and seven checkpoints per model. The `--confirm-original-row-order` flag is an explicit provenance assertion; do not use it if the CSV was sorted or changed. If original folds exist, compare them to reconstructed folds before trusting recovered outputs.

```bash
python scripts/04_recover_oof.py --repo "/path/to/repository" --metadata "/path/to/complete_metadata_m.csv" --data-root "/path/to/Synthetic_Skin_Disease_Dataset" --kind concat --checkpoint-dir "/path/to/concat_checkpoints" --backbone-config "/path/to/original_hf_snapshot" --confirm-original-row-order --permutations 20 --save-features --out runs/recovered_concat
python scripts/04_recover_oof.py --repo "/path/to/repository" --metadata "/path/to/complete_metadata_m.csv" --data-root "/path/to/Synthetic_Skin_Disease_Dataset" --kind dsaf --checkpoint-dir "/path/to/dsaf_checkpoints" --backbone-config "/path/to/original_hf_snapshot" --confirm-original-row-order --gate-scale 0.5 --meta-dropout 0.3 --permutations 20 --out runs/recovered_dsaf
```

Expected names are `best_model_fold_1.pth` … `_7.pth` for concat and `best_multimodal_dsaf_fold_1.pth` … `_7.pth` for DSAF. The original code saves full state dictionaries, including the frozen backbone. Gate scale is an ordinary Python value, not a saved tensor: choose the actual historical setting. This script targets the released standard DSAF run, not the unsupported 0.9239 tuned run.

Recovery loads only selected definitions from the **first notebook code cell**, avoiding the later visualization cells that redefine classes/configuration. Full state dictionaries are loaded with `weights_only=True` and strict key/shape checks. Only known THOP profiling-counter fields may be removed. The recovered marginal group counts and each fold accuracy must match the release; otherwise the script stops and writes a failure record rather than silently accepting different predictions. Small numerical differences near decision boundaries may still cause a stop; investigate the environment rather than weakening the guard to force agreement.

`oof_predictions.csv` is written only after all folds pass. `classes.json` gives the exact probability order. `reconstructed_fold_manifest.csv` remains a reconstruction. `inference_metadata_permutations.csv` includes joint-tuple and independent-column shuffles separately. These test conditional reliance of the trained head; they are not retrained baselines or tests of verified demographic causation.

`--save-features` saves CLS features only after checking that frozen backbone weights are identical across recovered folds. These are usable for new fixed-feature probes only if the author confirms the backbone was independent of dataset disease fitting. Do not use features from a disease-fine-tuned model as a global cache for newly assigned folds.

## D. Correct metrics and uncertainty: no training

```bash
python scripts/03_evaluate_predictions.py --predictions runs/recovered_concat/oof_predictions.csv runs/recovered_dsaf/oof_predictions.csv --classes runs/recovered_concat/classes.json --metadata runs/data_audit/audit_metadata.csv --manifest runs/recovered_concat/reconstructed_fold_manifest.csv --compare concat,dsaf --out runs/prediction_audit
```

Check both recovery directories have identical class lists and fold manifests before this command. The evaluator checks matching true labels, fold identities and unique sample IDs. Predictions must use `sample_id,model,fold,y_true,y_pred,p_0,...,p_4`; labels are the disease strings in `classes.json`, not a guessed alphabet. If probabilities are unavailable, omit **all** p-columns; accuracy/rate metrics can run but AUC/calibration will be unavailable.

Outputs: count confusion matrices; per-disease recall/specificity/precision/F1; pooled and annotation-route subgroup reports; per-group/class TPR/FPR with positive/negative support; threshold sensitivity; corrected EOD; calibration tables; paired fold differences and an approximate dependence-corrected comparison when full train/test roles are supplied.

Undefined rates stay undefined. A single eligible group cannot produce a valid gap. Macro EOD across all five classes is unavailable if a class cannot be compared; available-class means are separately labeled. Repeat with `--minimum-rate-n 5` in a different directory as a class-denominator sensitivity. Group total thresholds and class-positive/negative thresholds are different settings.

Intervals are conditional sampling summaries of held-out predictions and independent image units, not uncertainty from repeated model training. If meaningful patient/parent clusters exist, use a grouped evaluation and add cluster-aware uncertainty for population claims; these scripts do not pretend ordinary image-level intervals solve unknown dependence. All current claims should remain exploratory.

## E. Cheap missing baselines: only small classifier fitting

If original checkpoints are unavailable, obtain an existing independent local pretrained DINOv3 snapshot and cache features once:

```bash
python scripts/08_cache_frozen_features.py --model-path "/path/to/local_pretrained_dinov3_snapshot" --metadata runs/data_audit/audit_metadata.csv --data-root "/path/to/Synthetic_Skin_Disease_Dataset" --out runs/feature_cache
```

Otherwise use `runs/recovered_concat/frozen_cls_features.npz` from section C. Then fit the matched probe family:

```bash
python scripts/05_train_frozen_probes.py --features runs/recovered_concat/frozen_cls_features.npz --metadata runs/data_audit/audit_metadata.csv --manifest runs/data_audit/reconstructed_fold_manifest.csv --confirm-independent-frozen-features --out runs/probes
python scripts/03_evaluate_predictions.py --predictions runs/probes/probe_oof_predictions.csv --classes runs/probes/classes.json --metadata runs/data_audit/audit_metadata.csv --manifest runs/data_audit/reconstructed_fold_manifest.csv --compare probe_concat,probe_image_only --out runs/probe_evaluation
```

Substitute the cache from script 08 when appropriate. Use actual verified folds if available. Three fixed L2 logistic models × seven folds = 21 small fits: image-only, metadata-only, concatenation. Scaling and one-hot vocabularies are learned using training rows only. Validation rows remain unused because C=1 is fixed in advance; no test-driven search is performed. No stochastic image augmentation is used in this **new diagnostic probe** protocol.

Optional `--train-control-repeats 3` adds six randomized-training probe variants per fold. Joint shuffles preserve the metadata tuple distribution; column shuffles also destroy inter-attribute dependence. These controls randomize metadata in training only and use authentic held-out metadata. Do not count repeated shuffles as independent patients or as multiple original model-training seeds.

**Interpretation boundary:** these probes directly compare information content within one matched linear classifier family. They are not replacements for matched nonlinear image-only/metadata-only DSAF ablations. Keep their table separate from the original neural fusion table and state this limitation. If the reviewer insists on the exact original head family, those additional heads must be fitted; no script can make that a training-free experiment.

## F. If leakage is detected

Review every candidate in `duplicate_candidates_for_review.csv`; fill `verified_related` with `yes` or `no` using evidence, not model scores. Preserve both decisions and reasons. Then:

```bash
python scripts/06_make_grouped_splits.py --metadata runs/data_audit/audit_metadata.csv --reviewed-pairs "/path/to/reviewed_pairs.csv" --out runs/grouped_protocol
```

The script links related images transitively across verified duplicates and recorded patient/case/parent/session identifiers. It refuses unresolved candidate decisions and checks clusters do not cross training/validation/test roles. It creates **new** folds. Use them with newly fitted probes; do not reuse old disease heads on newly held-out images that they may previously have seen. Report exact grouped split counts. Grouping on verified available links does not prove all patient/source relationships are known.

If architecture comparisons remain central after leakage is found, rerun both retained neural heads under the same corrected protocol. That rerun is not automated in this package; an integrity problem cannot be fixed by relabeling old test predictions. The four-day low-compute alternative is a narrower paper centered on the correctly grouped probe audit with old contaminated comparisons removed.

## G. Independent annotation agreement, only if genuinely collected

Each rater file needs `sample_id,appearance_proxy,gender_proxy,age_proxy` (or pass an explicit `--columns` list). Both must contain the same prespecified sampled images, labeled independently without seeing previous labels or model outputs. Use a scientifically defensible nonidentifying appearance rubric and permit abstention. Existing collaborative consensus labels are not two independent ratings.

```bash
python scripts/07_annotation_agreement.py --rater-a "/path/to/rater_a.csv" --rater-b "/path/to/rater_b.csv" --confirm-independent --out runs/agreement
```

The script retains “unclear” as a category, reports missing pairs, disagreement tables, raw agreement and nominal Cohen kappa with a conditional bootstrap interval when estimable. For continuous age, do not mislabel categorical kappa as continuous-age agreement; specify buckets or separately report absolute differences. Report the sampling design because prevalence and stratified oversampling affect kappa. Agreement is not proof of demographic or clinical validity.

## Source/ethics and result discipline

The programs do not infer race, gender identity, true age, skin phototype, patient identity, body site, real/synthetic status or original-image lineage. They analyze supplied labels and simple image-file characteristics. Never invent missing provenance, replace missing event rates with zero, choose group thresholds because they produce favorable numbers, or turn pending commands into past-tense results in a rebuttal.

Use `verified_results` for already completed aggregate calculations. Keep all new real-data outputs in dated, separately named directories, and cite their exact scripts, inputs, folds, environment and checksums in the supplement.
