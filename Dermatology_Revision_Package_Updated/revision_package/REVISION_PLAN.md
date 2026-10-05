# Four-day revision plan: proxy-defined dermatology audit

Prepared 4 October 2026. This is an evidence audit and execution plan, not a claim that the revision has been completed or that the journal will accept it.

**Recommendation:** make the paper an exploratory, reproducible audit of image-derived proxy groups in a provider-described mixed real/synthetic dataset. Make label provenance, disease composition, and support-dependent interpretation the central contribution. Retain simple versus adaptive fusion as a bounded secondary comparison. Do not start a new generative-model study or fine-tune a new visual backbone merely to preserve the current title.

The reviewer explicitly offers this reframing in major comment 1. Several other comments allow limitations or optional experiments. However, reframing cannot repair incorrect results, unknown image overlap, or inconsistent metrics. Those require corrections and targeted analysis.

## 1. What was inspected and what remains unavailable

- All nine pages of the supplied review: 18 numbered major comments and 15 minor bullets.
- The complete `frontiers.tex`, bibliography, included manuscript PDF, and seven figure assets from the uploaded ZIP. The PDF and LaTeX do not have identical correspondence details; recompile the actual revised source before submission.
- GitHub repository at commit `c81c8917afab02f04d9ceb56cef04a6474f8fd99`, including the primary notebooks, stored text outputs, four released fold CSVs/JSON summaries, and subgroup CSVs. Notebook cells were read, not executed as training programs.
- Primary documentation for DINOv3 token structure, equalized odds, and dependence-corrected cross-validation comparisons; links are in section 12.

The release does **not** contain the image dataset, `complete_metadata_m.csv`, independently recorded annotation-route membership, saved `.pth` checkpoints, per-image probabilities, or identity-level fold CSVs. The annotation-generation notebook described in the manuscript is not present among the primary notebooks inspected. Consequently, the claimed 306 automatic and 259 manual records are manuscript statements, not independently verified annotation counts in this audit. Source/parent/patient information cannot be reconstructed from aggregate scores.

The repository audit script has run on the actual release. Data auditing, prediction recovery, and new scientific experiments have **not** run on the author's images. CPU analysis workflows were tested on artificial fixtures; those fixture results are not paper results. GPU recovery/caching scripts were syntax checked and their original split extraction tested, but cannot be validated end to end without the actual checkpoints/images and a compatible PyTorch environment.

## 2. Corrections required before submission

| Issue | Verified evidence | Required action |
|---|---|---|
| 2,740 versus 2,750 | Full DINOv3 notebooks print Train=2,000, Val=250, Test=500. The 0.9260 result is accuracy on the 500-image test partition. | For these recorded runs report the 2,750-file development dataset and 500-image test denominator. Do not invent ten exclusions. Reconcile the actual manifest before declaring a different run used 2,740. |
| Full-set evaluation called seven-fold CV | Full-data DINOv3 notebooks use the supplied train/val/test folders; seven-fold CV is in the 565-row metadata notebooks. | Describe two different protocols. Do not put the full-data test result and metadata CV result into a single purported CV experiment. |
| Wrong simple-head metrics in full-data table | `full_Dataset_DINOv3_Simple.ipynb`, first cell output: accuracy 0.9020, AUC 0.9880, kappa 0.8775, MCC 0.8779. The manuscript uses 0.9700, 0.8875, 0.8980 for the last three. | Replace with values from one documented run, or recover the run that supports the manuscript row. Do not mix metrics across runs. |
| Main proxy DSAF row unverified | Released DSAF fold CSV gives mean accuracy 0.920304, sample SD 0.046398; stored JSON gives population SD 0.042956. The manuscript emphasizes 0.9239 ± 0.0322 from a proposed tuned run. No matching performance artifact was found. | Use the released standard configuration as the provisional traceable run; retain the tuned run only if its complete fold outputs, settings and checkpoints are recovered. Do not invent a seed explanation. |
| Standard deviations use different conventions | Concat uses pandas sample SD (`ddof=1`); DSAF uses `np.nanstd` default `ddof=0`; manuscript equation specifies sample SD. | Recompute all model summaries with sample SD and identify the convention. Already done in `verified_results/repository_audit/model_summary_recomputed.csv`. |
| Age count and boundary errors | Saved counts are 61 child, 501 combined nonchild/nonsenior, 3 senior, summing to 565. Model input bins end at 17,29,44,64,200. Code displays middle_aged within “18–59” and senior as “60+”. | Do not simply change 11 to 3 and leave age definitions untouched. If retaining legacy aggregates, describe 0–17,18–64,65+ **conditional on verification of original numeric-age binning**. Prefer recomputing genuine 0–17,18–59,60+ audit groups from original ages, keeping old model inputs unchanged. |
| Appearance counts | 338+70+50+50+38+19 = 565, consistent with the released fold counts. Gender counts 355+210=565. | Clarify that the confirmed arithmetic conflict is the age figure; show all counts from a single manifest. Do not claim every axis is inconsistent. |
| “Skin tone” labels misleading | Code passes normalized `dominant_race` strings as `skin_tone`. Figures 2 and 7 display Type I–II/III/V–VI labels, and Figure1 says “Fitzpatrick-inspired,” despite prose denying phototype interpretation. | Remove clinical phototype equivalences from the figures themselves. Prefer source-category strings with a prominent proxy disclaimer or neutral codes with an explicit key. A race-category model output is not a skin-tone measurement. |
| EOD is not conventional equalized odds | Both original metric functions use TPR only. Concat sets absent-class TPR to zero via `denominator or 1`; DSAF uses NA. Both can return zero when no comparable group pair exists. | Recompute one consistent OVR TPR/FPR definition from held-out predictions. Do not just relabel the existing numbers as corrected EOD. If predictions cannot be recovered, remove EOD results and explain why. |
| Support rule removes much of evaluation population | Original n≥15 rule applies separately within each held-out fold, to composite groups. It retains only one or two groups; average image coverage is 49.21%. | Report coverage beside WGA and distinguish within-fold filtering from pooled subgroup support. Do not describe the numerical increase as improved fairness. |
| “Metadata gates disabled” is not what full-data DSAF does | Full-data `DSAFHead` concatenates 384-dimensional global and local vectors; no metadata gates exist in that head. | Give the image-only dual-stream head a distinct name. It is not the same gated architecture with gates disabled. |
| DSAF diagram disagrees with code | Code uses vocabulary-dependent embedding dimensions, vector gates and all non-CLS tokens. Figure 3 shows fixed 16/8/16 embeddings and 197 hidden tokens. | Replace diagram with the actual dimensions below. Do not silently change the model when recovering old results. |
| Saliency provenance is questionable | Figures6/7 themselves say “Attention Map,” whereas manuscript captions/text say Grad-CAM. Concat visualization cells implement attention maps. They select the first fold's test rows while loading a fold-5 checkpoint; the vocabulary also comes from a different train/validation construction. | Do not present these as verified held-out Grad-CAM evidence. Trace exact figure-generation code and checkpoint; regenerate correctly or remove figures from the main claims. |
| Availability statement overpromises | Manuscript says split files/final metadata are available; none were found in the released snapshot. | Release verified artifacts where permitted, or state precisely what is and is not available. Replace short URL with versioned repository link. |

The source's first comment says “Frontiers in Computer Science, Computer Vision,” while the requested revision is for Frontiers in Medicine. This is a template comment, not proof of a submission error, but check the journal, article type, correspondence block, word count and final compiled file.

### Traceable results recovered now

All rows below refer to released artifacts, not fresh model execution. “±” is sample SD across seven folds.

| Proxy-subset released run | Accuracy | Macro AUC | Kappa | MCC |
|---|---:|---:|---:|---:|
| DINOv3 + concat | 0.938117 ± 0.034829 | 0.993180 ± 0.007118 | 0.913599 ± 0.049200 | 0.914370 ± 0.048686 |
| DINOv3 + standard DSAF | 0.920304 ± 0.046398 | 0.990172 ± 0.005945 | 0.888577 ± 0.065359 | 0.889564 ± 0.065373 |
| ResNet50 released concat directory | 0.900970 ± 0.033982 | 0.983552 ± 0.010474 | 0.862898 ± 0.046794 | 0.864455 ± 0.046301 |
| VGG16 released concat directory | 0.860185 ± 0.028197 | 0.960919 ± 0.034584 | 0.807673 ± 0.039095 | 0.810111 ± 0.039167 |

The manuscript's ResNet50 value 0.8673 ± 0.0393 **does** occur in the separate root notebook `by_Yamlick_Multimodal_ResNet-50_with_Metadata_.ipynb`. It is not fabricated merely because it differs from the released concat directory. Identify the exact run and its trainable layers; do not mix the two ResNet variants. The released CNN concat logs describe last-block fine-tuning, so they are not all frozen-backbone controls.

The full-data frozen DSAF accuracy 0.9260 is traceable. The 0.9239 full-data ablation, 0.9000 single-split DSAF comparison, and tuned DSAF rows are not supported by matching final performance artifacts found here. Remove or relocate unsupported tables rather than preserving them with “separate runs” as the only explanation. “Single split versus CV improved performance” is especially misleading because the evaluated samples, inputs and architecture differ.

### Support sensitivity recovered without training

These are **concat composite groups using the historical saved category definitions**, not corrected-age groups. Threshold 1 includes all observed groups. Rows with no eligible group are undefined, not zero.

| Minimum group n | Pooled groups retained | Pooled images retained / 565 | Pooled WGA | Mean within-fold WGA | Mean within-fold image coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 565 | 0.0000 | 0.3429 | 100.00% |
| 5 | 13 | 553 | 0.8333 | 0.7928 | 76.46% |
| 10 | 12 | 545 | 0.8333 | 0.9197 | 51.68% |
| 15 | 11 | 532 | 0.8333 | 0.9197 | 49.21% |
| 20 | 9 | 499 | 0.8974 | 0.9356 | 33.44% |
| 30 | 6 | 425 | 0.8974 | undefined | 0% |
| 50 | 2 | 292 | 0.9337 | undefined | 0% |

This is a concrete central result: the support rule changes **which groups and images the statistic describes**. It does not repair any model errors. The threshold-dependent rankings, group counts, coverage and conservative uncertainty bounds are saved in the package.

Selected marginal Wilson 95% intervals, calculated from exact released correct/total counts:

| Source-category proxy | Correct / n | Accuracy | Wilson 95% interval |
|---|---:|---:|---:|
| `middle eastern` | 16 / 19 | 84.21% | 62.43–94.48% |
| `black` | 37 / 38 | 97.37% | 86.51–99.53% |
| `asian` | 63 / 70 | 90.00% | 80.77–95.07% |
| `white` | 318 / 338 | 94.08% | 91.04–96.14% |

These labels are inherited algorithm/manual categories, not verified identities. These intervals are descriptive and conditional on predictions; they do not account for unobserved patient clustering, annotation error, model-selection uncertainty or full training variability. Do not interpret interval overlap as a formal pairwise significance test.

### Provisional comparison of fusion heads

Released fold differences (concat minus standard DSAF) are 0.024691, −0.024691, 0, 0.012346, 0.012346, 0, 0.100000. Mean difference = **1.7813 percentage points**. With the known nominal test/train ratios (81/399 in five folds; 80/400 in two), an approximate dependence-corrected CV interval is **−3.8781 to +7.4407 percentage points**, two-sided p≈0.4704. An uncorrected fold sign-flip sensitivity is p=0.375, but overlapping CV training sets weaken its exchangeability assumption.

**These inferential numbers are provisional:** fold numbers and marginal group counts match, and the source splitter definitions match, but actual sample IDs cannot be confirmed without the ordered CSV/manifests. Use the paired result in the response only after identity-level verification. Neither nonsignificance nor a negative mean DSAF difference proves equivalence or universal superiority of concatenation.

## 3. Recommended framing and contribution

Suggested title:

**Auditing Proxy-Defined Subgroup Performance in a Mixed-Source Dermatology Dataset: Label Provenance, Group Support, and Fusion Baselines**

Core contribution statement:

> We present an exploratory case study of subgroup evaluation when audit attributes are inferred from the images being classified. The audit links the origin and meaning of proxy labels, their association with disease composition, and the coverage–precision trade-off introduced by subgroup support rules. A comparison of simple and metadata-conditioned fusion provides a secondary test of whether added architectural complexity is supported under the same internal evaluation protocol.

State this as the study objective until route/composition analyses are actually completed. Afterward, report what those analyses found; do not promise a result. The novelty is the integrated, auditable empirical case and its demonstrated interpretation failure modes, not a new fairness definition or proof that small-group filtering is novel.

Keep these claim boundaries throughout title, abstract, figures, discussion and response:

- Mixed-source performance does not estimate the effect of synthetic augmentation.
- Image-derived metadata can encode disease, source, body site or label-route information. It is not an independent patient modality.
- Predictive utility beyond a frozen CLS embedding is not evidence of demographic validity or causal demographic effects.
- A within-fold support filter and a pooled support filter define different evaluation populations.
- “No demonstrated DSAF advantage under this protocol” is defensible; “simple fusion is superior” and “equivalent models” require stronger evidence.
- Clinical deployment, population fairness and skin-phototype fairness are outside the supported claims.

## 4. Every major review comment: minimum response and evidence

Status key: **W** = writing/reporting route available; **A** = data or prediction analysis, no model training; **H** = a small new classifier/head experiment; **C** = conditional on a substantive problem or retained claim. A writing route addresses a comment but does not guarantee the reviewer will regard it as sufficient.

| Review | Minimum defensible action | Training need | Current evidence/status |
|---|---|---|---|
| M1 Synthetic augmentation | Adopt reviewer's narrower title/abstract/conclusion. Describe provider-reported mixture and unavailable source provenance. If reliable source labels exist, add source-stratified held-out results. | W; A optional. Real-only vs mixed training is C only if preserving augmentation-effect claim. | Writing route explicitly offered by reviewer. No source labels available here. |
| M2 Leakage/duplicates | Audit file SHA256, decoded-pixel identity and perceptual-hash candidates; visually/source-verify near matches; check Train/Val/Test overlap for every actual fold. Record patient/case/parent/session availability. | A first. C if leakage found: grouped rerun of retained comparisons, or remove invalid claims. | Script 02 + 06; actual images required. No hash test proves absence of common patients or synthetic lineage. |
| M3 Complementary information | Same-sample image-only, metadata-only and concat diagnostic probes on frozen features; retain original DSAF/concat comparison; two distinct inference controls (joint metadata-tuple permutation; independent column permutation). Explain what each tests. | H for missing baselines, A for checkpoint permutations. No backbone training needed. | Scripts 04/05/08. Logistic probes are explicitly a new matched probe experiment, not matched nonlinear DSAF ablations. Exact original-head ablation remains a possible reviewer request. |
| M4 Proxy–disease/source association | Disease×route, disease×each proxy, source×proxy, route×proxy, and body-site×proxy where recorded. Show within-group class supports before interpreting accuracy gaps. | A | Script 02 plus group-class rate tables in 03. Unknown provenance/body site must remain unknown. |
| M5 Small groups | Counts, marginal accuracy intervals, thresholds 5/10/15/20/30/50, retained coverage and ranks; separate pooled and within-fold results. Include sparse groups in complete supplement. | A + W | Much is already recovered from exact aggregate counts by 01. Corrected-age/route analysis needs raw data. |
| M6 EOD definition | Explain and correct the TPR-only calculation and missing-class bug. Use explicit OVR TPR/FPR gaps and declared reduction. Report undefined comparisons/coverage. | A + W | 03 implements new definition. Old EOD values cannot be repaired from overall accuracy counts. Remove if no trustworthy predictions. |
| M7 Manual reliability | Best: independently reannotate a prespecified, documented subset; report agreement, kappa and disagreements. Fallback explicitly allowed: describe consensus annotation, no independent agreement, and soften reliability claims. | W fallback; human effort + A preferred. No model training. | 07 computes statistics for real independent labels; it cannot manufacture agreement from consensus labels. |
| M8 Annotation-route selection | Compare disease, proxy, image-size/brightness descriptors, source and body site; repeat subgroup analyses within each route and combined. | A | 02/03. Route membership must come from original annotation records, not assumed face detection reruns. |
| M9 Sample counts | Build file→eligible dataset→proxy subset accounting, exclusion reasons, missing/uncertain labels. Reconcile 2,750/2,740 and 61/501/11; synchronize figures. | A + W | Logs support 2,750 and aggregate proxy counts support 565. Author manifest still needed. Never invent exclusion reasons. |
| M10 Raw-to-final mapping | Publish exact raw strings, normalization, unmapped/unclear/missing policies, conflict resolution and independent model-input versus audit-label definitions. | W from verified records + A to count observed values | 02 outputs actual mapping. Use raw source-category names or neutral codes; no fabricated phototype mapping. |
| M11 Architecture | Provide full implementation-derived supplement, vector gate dimensions, regularization, checkpoint, token indexing, initialization and actual training settings. Correct diagrams. | W/code inspection | Details below. Fold vocab sizes and exact software/checkpoint revision require author artifacts. |
| M12 Two DSAF values | Distinguish 0.9260 image-only fixed split from proxy CV. Trace 0.9239 to an actual run or remove it. Name canonical artifacts and settings. | W/provenance recovery; no new training needed just to explain numbers | 0.9260 traceable; released proxy DSAF is 0.920304. Tuning explanation cannot be asserted without files. |
| M13 Statistical comparison | Verify paired test IDs/labels; show all fold differences, mean difference and exploratory dependence-aware uncertainty/test. Mention one CV run and model-selection limitations. | A | 01 provisional; 03 paired analysis after recovery. Multiple seeds optional, not mandatory within four days. |
| M14 Hyperparameter selection | Report true selection history. Separate exploratory tuning from an unbiased test claim. Acknowledge reuse of outer results if it occurred. Freeze additional probe settings before running them. | W fallback explicitly offered; C if strong unbiased superiority claim retained | Renaming folds or bootstrapping cannot retrospectively create nested CV. New clean folds do not erase prior dataset-level selection either. |
| M15 Baseline matrix | Explicitly separate full-data historical models, original proxy fusion models and matched new frozen-feature probes. Include image-only and metadata-only probe baselines. | H only for missing small baselines; no DINO fine-tuning required | Optional fine-tuned DINO is not a priority; “unfreeze” notebook titles alone do not establish which layers actually trained. |
| M16 Disease metrics/calibration | Export held-out labels/probabilities; count confusion matrix, per-class recall/specificity/precision/F1, and log loss/Brier/reliability table if probabilities available. | A | 03. Stored full-data classification reports provide some historical precision/recall/F1 immediately; do not reconstruct exact confusion counts from rounded heatmaps. |
| M17 Explanations | Minimum: remove reliance/clinical-localization claims and describe maps only as selected qualitative visualizations; verify attention vs Grad-CAM and checkpoint/fold alignment. Optional: verified lesion masks plus equal-area/random-mask controls, all on held-out data. | W fallback for exploratory scope; A + manual masks if pursuing quantitative explanation claim | No lesion masks supplied. A center crop or arbitrary mask is not a lesion localization test. Correct provenance before adding more figures. |
| M18 Fairness terminology | Consistent proxy-defined subgroup disparity / appearance-proxy performance; verified demographics and phototypes explicitly unavailable; no causal or clinical equity claims. | W | Apply to figure text, not only prose. Current Figure 2 contradicts the caveat. |

### Exact list of comments addressable primarily by writing

**M1, M11, M12 and M18** can mainly be addressed by a narrower claim, exact methods and traceable result reporting. M12 requires removing unverifiable runs if provenance cannot be recovered. **M14** has an explicit reporting-and-limitation fallback. **M7** has an explicit no-independent-agreement fallback. **M17** has a reasonable scope-reduction response because its quantitative experiment is requested “if possible,” but acknowledging a limitation is not the same as completing the requested experiment. **M10** is documentation work only after the actual mapping is verified.

Writing can improve interpretation in M3–M6, M8, M9, M13, M15 and M16, but it cannot substitute for their missing tables or analyses. M2 is especially important: writing alone cannot establish that images are independent across splits.

## 5. Every minor comment

| Minor bullet, in review order | Action |
|---|---|
| 1 Counts consistent everywhere | One manifest and generated tables; update abstract, methods, captions, figures, supplement and response together. |
| 2 Abbreviations at first use | Define Dual-Stream Attention Fusion (DSAF), receiver operating characteristic (ROC), area under the curve (AUC), Matthews correlation coefficient (MCC), equalized odds difference (EOD), true/false positive rates (TPR/FPR), cross-validation (CV). |
| 3 Exact DINO checkpoint | `facebook/dinov3-vitb16-pretrain-lvd1689m`; add actual snapshot revision/checksum and Transformers version from the original environment. |
| 4 Frozen vs fine-tuned | DINO primary notebooks freeze the backbone; CNN released concat variants fine-tune a final block. Separate optional “unfreeze” experiments and verify actual `requires_grad` settings. |
| 5 Preprocessing/resolution | RGB, 224×224 resize, normalization; specify interpolation and actual augmentations, including ColorJitter in proxy experiments. |
| 6 Exact splits | Fixed folder split for full data; proxy CV seven folds, seed42, validation remainder fraction 0.175, seed42+fold for validation; report actual counts. |
| 7 Seeds | Split seeds are recorded. Primary proxy training cells do not set complete Python/NumPy/PyTorch training seeds; do not claim 42 governs all model initialization/augmentation. New scripts seed their stochastic controls. |
| 8 Missing proxies | Show counts and route-specific missingness; retain explicit missing/unknown/unclear/unmapped categories and abstentions. Do not silently delete them or impute sensitive labels. |
| 9 Labels before/after split | Training notebooks load an existing CSV before creating CV folds; original annotation timing/blinding is not independently established. Recover records or state the limit. |
| 10 Subgroup n in table | Exact correct/n plus intervals and disease-class support, including excluded groups in supplement. |
| 11 Subgroup uncertainty | Wilson intervals for accuracy/rates, conservative simultaneous bounds for WGA/EOD; explain conditional scope and cluster limitation. |
| 12 Complete confusion matrix | Use count matrix from authentic OOF predictions; optional row-normalized view beside it, with class order. |
| 13 Move details to supplement | Full architecture/config, raw label mapping, all small groups, complete threshold curves, all per-fold scores and checksum manifest. |
| 14 Exploratory vs confirmatory | Explicit in abstract and limitations; this retrospective revision is exploratory, not preregistered validation. |
| 15 Avoid causal conclusions | Describe associations and observed conditional performance, not effects of synthetic augmentation, demographic causes or shortcut causation without supporting experiment. |

## 6. Architecture and protocol supplement from the actual code

For the **released standard proxy DSAF** (first code cell of its notebook):

- Backbone: DINOv3 ViT-B/16, hidden dimension 768, 12 attention heads, frozen. At 224×224 the published checkpoint returns 201 tokens: one CLS, four registers, 196 patches. The source uses `hidden_states[:,1:]`, so its local branch processes **200 non-CLS tokens including registers**, not only 196 spatial patches. Preserve this for old-checkpoint recovery and describe it honestly; removing registers changes the model computation and requires a separately labeled experiment.
- Inputs: categorical age-group, gender-proxy and source appearance-category index. Vocabularies fitted on each fold's training rows, with `<UNK>` for unseen categories. Embedding dimensions are `min(16,max(4,V_age))`, `min(8,max(3,V_gender))`, and `min(16,max(4,V_appearance))`, where each V includes `<UNK>`. Exact values depend on training vocabulary; do not report 16/8/16 as constants. A common vocabulary with V=6/3/7 would produce 6/3/7 embeddings (16 concatenated dimensions), but verify each fold before claiming these actual sizes.
- Metadata MLP: sum-of-embedding-dimensions → 128 → 256, GELU after each linear; its internal dropout is **0.1** in DSAF because the constructor default is used. A separate metadata dropout of **0.3** is applied just before the gates. The concat metadata encoder instead receives configured dropout 0.3.
- Global stream: CLS 768 → linear384 → GELU → dropout0.1.
- Local stream: 12-head attention at dimension768, dropout0.1; residual+LayerNorm; MLP 768→1536→768 with GELU/dropout0.1, residual+LayerNorm; average over non-CLS tokens; linear768→384, GELU, dropout0.1.
- Each gate: metadata256 → linear384 → GELU → linear384 → sigmoid, then `0.5 + s*(sigmoid_output−0.5)`. Gates are **384-dimensional feature-wise vectors**, not scalar weights and not a softmax pair. With released s=0.5, each component lies between 0.25 and 0.75. With s=0.25 it would lie between 0.375 and 0.625; that setting needs its own run provenance.
- Fusion: elementwise weighted sum of the two 384-vectors; classifier LayerNorm384→linear192→GELU→dropout0.1→linear5. Gates do not have to sum to one.
- New layers use PyTorch module defaults; no custom initialization routine is defined. Report this and the PyTorch version rather than inventing a Xavier scheme for every layer.
- Optimizer/loss: AdamW LR1e−4, weight decay2e−5, batch128, maximum200 epochs, minimum validation loss, patience20, 10-epoch warmup then cosine schedule, label smoothing0.1, AMP. Fold-level initialization/augmentation seeds are not fixed in these cells. Full-data DINO weight decay is 1e−4, so one blanket optimization paragraph is inaccurate.

**Simple concat:** CLS768 plus metadata256 =1024 → linear512 → BatchNorm512 → GELU → dropout0.5 → linear5. Metadata MLP uses hidden128/out256 and dropout0.3. Do not label classifier dropout as 0.3 simply because the metadata dropout is 0.3.

**Image-only full-data dual-stream model:** same 768-dimensional visual backbone, separate global/local projections to384, concatenate to768, then LayerNorm768→linear384→GELU→dropout0.1→linear5. There are no metadata gates. Its stored 0.9260 is from the original 500-image test set.

Original proxy split counts are 399 Train /85 Val /81 Test for folds1–5 and 400/85/80 for folds6–7. These follow the source splitter for N=565. Matching seeds do not establish matching identities if CSV ordering differed; save the actual manifest.

## 7. Metric definitions to use in the revision

For group g and disease c, use one-vs-rest counts:

\[
TPR_{gc}=TP_{gc}/(TP_{gc}+FN_{gc}),\quad
FPR_{gc}=FP_{gc}/(FP_{gc}+TN_{gc}).
\]

A denominator of zero means **undefined**, not zero. For eligible groups with estimable rates, define per-disease gaps as the largest minus smallest TPR and FPR. At least two groups are needed for each comparison. Define

\[
EOD_c=\max(\max_gTPR_{gc}-\min_gTPR_{gc},\;
                 \max_gFPR_{gc}-\min_gFPR_{gc}).
\]

Report every per-disease value and coverage. Report an unweighted five-class macro EOD only when all five class gaps are estimable; otherwise mark that total as unavailable. If reporting an available-class average, label it explicitly and give the number of classes. Also show the maximum available class gap. Do not imply this is the only possible multiclass extension. The new script uses this definition consistently across models.

Total group thresholds and class-conditioned denominator thresholds are different. The script defaults to positive/negative denominator≥1 and accepts `--minimum-rate-n 5` as a declared sensitivity. Keep zero-support cells visible. Extremely sparse rates may remain uninformative despite being mathematically computable.

For WGA, state whether groups are filtered using pooled n or each test fold's n. A WGA confidence interval obtained by treating the selected worst group's ordinary interval as the interval for the minimum ignores selection. The scripts instead include conservative simultaneous binomial bounds. All supplied uncertainty summaries remain conditional on the trained predictions and do not resolve missing patient/source clustering.

## 8. Minimum-compute execution order and stop rules

1. **Freeze evidence first:** preserve exact ordered metadata, all existing checkpoint/config files and repository commit. Export `pip freeze` from the author's original training environment. Name every table's dataset, inputs, split, checkpoint, seed and selection history.
2. **Run data/duplicate audit before trusting model comparisons.** Use actual fold manifests where available; reconstructed folds are weaker evidence. Review near-duplicate candidates without looking at which deletion improves accuracy.
3. **If original checkpoints exist:** recover OOF probabilities for concat and standard DSAF, verify the expected fold accuracies and metadata-count fingerprints, then run disease, group, source and route analysis. Recovery and metadata permutations do not train anything.
4. **Add cheap missing baselines:** cache independent frozen CLS features once; fit fixed logistic probes for image-only, metadata-only and concatenation on the same folds (21 small fits for seven folds). Optional randomized-training controls add small fits. These address information-content questions within a transparent matched probe family; do not mix them into the old nonlinear-head table as though training procedures were identical.
5. **If leakage is found:** group exact/verified-related cases, create new shared folds, and refit retained models. At minimum obtain group-aware frozen-probe results and make those the primary reliable diagnostic experiment. If keeping the original DSAF-versus-concat architecture conclusion, rerun those two heads on the corrected folds; cached features may reduce compute, but fixed features without the original stochastic augmentations define a revised training protocol that must be disclosed. The supplied scripts do not falsely treat old checkpoints as valid on new folds.
6. **If no historical checkpoints exist:** do not invent probabilities from summaries. Use aggregate count corrections now, derive only metrics the artifacts support, and run the cheap frozen-probe experiment if images are available. Remove unsupported original EOD/calibration/paired identity claims. This is a narrower revision and may not fully satisfy M2/M3/M8/M16 without more evidence.
7. **If independent manual labels cannot be obtained:** take the explicit M7 limitations route, not fabricated kappa. If source labels cannot be recovered, keep unknown and take M1's reframing route.

Do not spend the deadline on synthetic-image generation, new CNN training, DINO fine-tuning, a large hyperparameter search or cherry-picked saliency examples. None is necessary for the recommended narrower claim. New grouping-based training becomes necessary only if leakage makes the retained evaluations invalid.

## 9. Four-day schedule

Use the journal portal's actual deadline and timezone; “four days left” does not identify the precise cutoff. Starting from the user's local evening of 4 October (Asia/Dhaka):

| Work block | Deliverable | Decision at end |
|---|---|---|
| Day1 / first 24h | Freeze original files; run inventory/hash and mapping audits; reconcile counts; establish annotation-route membership; identify canonical runs. | Is original evaluation recoverable and is contamination present? If provenance cannot be recovered promptly, contact editor about time rather than claim unperformed checks. |
| Day2 | Recover OOF outputs or cache independent features; run uncertainty, threshold, route and disease analyses; fit only missing small probes. Begin independent annotation subset if feasible. | Which claims survive? If leakage exists, switch to verified grouped protocol immediately. |
| Day3 | Finalize corrected tables/figures, architecture supplement, label mapping and response; integrate independently measured agreement if available. | Remove unsupported tables, old EOD numbers, false phototypes and unjustified explanation claims. |
| Day4, finishing before portal cutoff | Recompile LaTeX, visually check every figure/table and correspondence details, reconcile every response with actual changes, archive scripts/manifests/environment and upload revision files. | No placeholders, pending analyses or unverified results in submitted manuscript/response. |

A modest reannotation sample is an operational choice, not a powered reliability validation. If attempting it, select the subset before examining predictions, balance disease/route coverage as appropriate, record sampling probabilities, allow “unclear,” blind raters to prior annotations and model outputs, and report exactly the subset agreement. Do not ask annotators to establish race, gender identity or clinical phototype from lesion images. Agreement on image appearance does not validate demographic identity.

## 10. Manuscript and figure edit checklist

- Abstract: remove augmentation-effect framing; state two datasets/protocols correctly; include only canonical results; emphasize coverage/uncertainty instead of extreme WGA change without denominators.
- Introduction: lead with absent verified audit metadata and the difficulty of interpreting image-derived labels; synthetic provenance is context. Fix run-together sentences in the current source.
- Methods: exact dataset accounting; original annotation protocol; deterministic label mapping; model-input versus audit-label distinction; source/duplicate grouping availability; architecture; two evaluation protocols; correct EOD and uncertainty definitions; explicit exploratory status.
- Results: start with accounting/provenance and disease×proxy composition; present coverage-sensitive subgroup results, then bounded model comparisons and disease metrics. Separate historical fixed-split results from new probes.
- Discussion: remove unsupported causal statements and negative-architecture overreach; distinguish selected maps from functional reliance tests; discuss route selection and missing source lineage.
- Figure1: align workflow with actual performed analyses; separate audit labels from model inputs and remove the unsupported “Fitzpatrick-inspired” label.
- Figure2: regenerate from final metadata; remove Type I–VI references and misleading “skin tone” ordering; fix age boundaries/counts.
- Figures3/4: correct embedding sizes, 201 tokens and register inclusion, gate vector dimension, dropout and classifier details. Supplement placement is reasonable.
- Figure5: replace mixed-metric bars labeled “Accuracy (%)”; use separate classification and coverage/support displays. Remove obsolete EOD values.
- Figures6/7: verify method, checkpoint, exact test fold, image selection and any train-set exposure. Current image-only Grad-CAM claim is contradicted by the panels' own “Attention Map” labels and not established by the available code. Remove Figure7's unsupported Type I–II/Type V–VI labels. Regenerate or remove from main evidence.
- Tables6/7/8: remove unsupported full-data ablation, split comparison and tuned configurations unless matching artifacts are recovered.
- Availability: only claim files actually released. Preserve annotation privacy and dataset license; a file manifest/IDs may be releasable when images cannot be redistributed.
- Ethics: the provided statement is the authors' institutional claim. Verify it with the institution; this audit cannot assert exemption or license compliance on their behalf.

## 11. Files needed from the author to finish the scientific analyses

1. Exact original `complete_metadata_m.csv` with original row order, paths and raw proxy fields; preserve a checksum before any cleanup.
2. The actual dataset image tree or a reliable local path on the machine where the scripts will run.
3. Original automatic/manual annotation record files or a verified sample-ID route manifest; parent/patient/case/source/body-site fields if they genuinely exist.
4. Seven concat and seven standard DSAF checkpoints, plus any claimed tuned-run outputs/checkpoints supporting 0.9239/0.9133; the original backbone config/snapshot and software versions.
5. Historically saved fold assignments and per-image predictions if available. A prior export is preferable to reconstruction.

These are requests for existing evidence, not a request to retrain everything. Until they are available, the recovered aggregate corrections are usable as audit findings, while image-level conclusions remain conditional.

## 12. Technical sources and limits

- Repository inspected: https://github.com/Sazib-Ahmed/A_Comprehensive_Framework_for_Auditing_Demographic_Bias_in_Synthetic_Skin_Disease_Datasets/tree/c81c8917afab02f04d9ceb56cef04a6474f8fd99
- DINOv3 model card (architecture, register/patch tokens): https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- DINOv3 Transformers documentation (separating CLS/register/patch tokens): https://huggingface.co/docs/transformers/en/model_doc/dinov3
- Fairlearn definition (both TPR and FPR components): https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html
- scikit-learn statistical comparison example (overlapping folds, corrected variance): https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html

The uncertainty methodology in this package is explicitly chosen for the revision; it is not attributed to the original authors' execution. The full bibliography and clinical/dataset provenance have not been independently authenticated item by item. No claim is made that a new controlled experiment, independent annotation, real-only evaluation or leakage-free external validation has already been completed.
