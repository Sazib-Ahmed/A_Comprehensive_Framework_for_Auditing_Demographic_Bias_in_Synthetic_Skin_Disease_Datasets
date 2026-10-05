# Working response to the reviewer

**Not ready to submit.** Bracketed actions/results must be completed, verified and replaced. The manuscript itself has not been edited by this package. Do not say an analysis was performed merely because a script is supplied. The text below separates response wording from evidence still required. Add final page/line/table references after compiling the actual revision.

Suggested opening after the completed revision:

> We thank the reviewer for distinguishing the value of an exploratory proxy-defined subgroup audit from claims that the available dataset cannot establish. The revision narrows the study to the interpretation of image-derived subgroup labels in a mixed-source dermatology dataset. We have reconciled the reported experiments with their underlying artifacts, clarified the two evaluation protocols, and [INSERT ONLY COMPLETED ANALYSES]. The revised contribution concerns label provenance, disease composition and subgroup support; it does not establish the effect of synthetic augmentation or verified demographic fairness.

## Major comments

**1. Synthetic augmentation.**

Proposed response: “We agree that our experiments do not isolate the effect of synthetic augmentation. We have adopted the reviewer's suggested narrower framing and changed the title to [FINAL TITLE]. We now describe the study as an exploratory audit of proxy-defined subgroup performance in a provider-described mixed real/synthetic dataset. We make no claim that synthetic data improved or worsened subgroup performance. [IF VERIFIED SOURCE LABELS EXIST, ADD THE ACTUAL SOURCE-STRATIFIED EVALUATION; OTHERWISE STATE THAT SOURCE-SPECIFIC LABELS/GENERATION LINEAGE WERE UNAVAILABLE.]”

Complete: title, abstract, introduction, conclusion, captions and keywords; do not claim real-only training/evaluation unless run on verified source labels.

**2. Leakage and duplicates.**

Proposed response: “We agree that image-level splitting alone cannot establish case independence. We audited [ACTUAL NUMBER] image files using byte hashes, decoded-pixel hashes and perceptual-hash screening, followed by [ACTUAL REVIEW PROCEDURE]. We found [ACTUAL COUNTS] exact/verified-related image pairs crossing [SPECIFY WHICH] roles in [WHICH PROTOCOL]. [DESCRIBE GROUPED RERUN IF NEEDED.] Patient/case identifiers and synthetic-parent lineage were [ACTUAL AVAILABILITY]. The absence of perceptual matches would not exclude shared underlying patients or unknown synthetic lineage, and this residual limitation is now explicit.”

Complete: image audit + actual fold overlap + preserved candidate decisions. If no audit can be performed, state so plainly; a stronger paragraph does not replace evidence for this comment.

**3. Image-derived metadata and complementary information.**

Proposed response: “We now describe the auxiliary inputs as image-derived proxy features, rather than independent patient metadata. To test incremental information beyond the frozen representation, we added a matched fixed-feature probe comparison of image-only, metadata-only and concatenated features on [VERIFIED SHARED FOLDS]. We also evaluated the trained fusion heads with jointly permuted metadata tuples and independently permuted metadata columns within each held-out fold. The first control preserves inter-attribute relationships; the second also disrupts them. [INSERT EFFECT SIZES, UNCERTAINTY AND NUMBER OF PERMUTATIONS.] These are conditional predictive/reliance tests and do not validate demographic identity or demonstrate a causal demographic effect.”

Complete: baseline fitting and/or checkpoint inference. Explain that logistic probes are a distinct diagnostic family; do not claim they are exact original nonlinear-head ablations. If only a subset was possible, name the missing experiments and narrow the claim accordingly. “Shuffled” and “randomly permuted” are otherwise overlapping descriptions, so define the implemented controls explicitly.

**4. Disease and source information in proxies.**

Proposed response: “We added disease-by-annotation-route and disease-by-proxy distributions, with exact counts and missingness. [ADD BODY-SITE/SOURCE TABLES ONLY WHERE VERIFIED.] We also report class-conditioned errors so that differences in subgroup accuracy can be assessed alongside disease composition. [INSERT ACTUAL FINDINGS.] These associations are not interpreted as demographic causation.”

Complete: source/route provenance and tables. Unknown body site or real/synthetic status must be reported as unavailable rather than inferred from appearance.

**5. Small groups.**

Proposed response: “We agree that the n≥15 rule is not a principled fairness cutoff. We report all observed groups with correct/total counts and uncertainty, and provide sensitivity at n≥5,10,15,20,30,50 alongside retained groups and image coverage. We distinguish filtering within each fold from filtering on pooled out-of-fold support. In the archived concat results, n≥15 within a fold retained one or two composite groups and approximately 49.2% of test images on average; pooled filtering retained 11 groups and 532/565 images. These calculations use the historical category definitions [REPLACE WITH FINAL CORRECTED-AGE VALUES IF RECOMPUTED]. Increasing WGA under exclusion does not represent an improvement in the model.”

Complete: ensure numbers correspond to final groups and explicitly label legacy versus corrected-age analyses. Add the full threshold/ranking supplement rather than only top/bottom groups.

**6. Equalized odds.**

Proposed response if corrected: “Inspection of the released implementation identified that the previous quantity summarized only TPR differences, with inconsistent treatment of absent disease classes. It was therefore not the conventional TPR-and-FPR equalized-odds difference. We have replaced it with an explicitly defined one-vs-rest analysis: per disease, EOD is the maximum of the between-group TPR and FPR ranges. Missing denominators remain undefined; each range requires at least two eligible groups. We report class-level values, estimable coverage and [ACTUAL AGGREGATION/INTERVALS]. The old values are withdrawn.”

Alternative if authentic predictions are unavailable: “We have removed the EOD values because their interpretation was not supported by the released implementation and the available aggregates do not allow a valid recomputation. The audit now focuses on supported accuracy/count/uncertainty analyses. This omission is stated explicitly.”

Complete: implement one choice consistently in text, table, Figure5 and response. Do not preserve the old 0.868/0.469 under a new definition.

**7. Manual-label reliability.**

Preferred response after actual independent reannotation: “The original labels were produced by collaborative consensus, so they do not support an independent-rater agreement statistic. We additionally obtained independent annotations on a prespecified subset of [N] images selected by [SAMPLING METHOD], with raters blinded to [ACTUAL BLINDING]. We report agreement, kappa, uncertainty and disagreement tables. [ACTUAL RESULTS.] Agreement on the stated appearance rubric is not evidence of verified demographic identity or clinical phototype.”

Explicit fallback offered by reviewer: “The original annotations were not recorded independently, and their independence cannot be retrospectively reconstructed. We therefore do not report kappa from consensus labels. We have made the absence of independent agreement a prominent limitation and removed wording implying validated reliability.”

Complete: choose one honest route. Do not describe a small stratified subset as representative without explaining its selection and weighting.

**8. Selection by annotation route.**

Proposed response: “We reconstructed annotation-route membership from [ORIGINAL RECORDS] and compared disease and proxy distributions, [RECORDED IMAGE/BODY-SITE/SOURCE VARIABLES], and missingness. Subgroup performance is now reported separately for the automatic-label subset, manual-label subset and combined set. [ACTUAL COUNTS/RESULTS.] Differences between these populations cannot be attributed solely to annotation quality because route selection is associated with what is visible in the image.”

Complete: route manifest + comparisons. The number of automatic versus manual images alone cannot reconstruct route membership or route-specific performance.

**9. Counts.**

Proposed response: “We checked the reported counts against [ACTUAL MANIFEST AND ARCHIVED OUTPUTS]. The traceable full-data notebooks record 2,000 training, 250 validation and 500 test images, totaling 2,750; the unsupported 2,740 figure has been [CORRECTED OR EXPLAINED WITH DOCUMENTED EXCLUSIONS]. The released proxy counts sum to 565. We also identified that the old age display labels did not match the input-bin boundaries, so the final audit age groups and all counts were [EXACT RESOLUTION]. The dataset accounting table now lists exclusions, missing/uncertain labels and analysis denominators.”

Complete: raw manifest before final claim. Do not claim ten corrupt images were excluded without evidence. Do not change the senior count from 11 to 3 while keeping incorrect 60+ semantics.

**10. Deterministic mapping.**

Proposed response: “We now publish the raw-to-analysis mapping, including normalization, missing, unknown, unclear and unmapped values and any conflict resolution. The six inherited appearance-category strings are retained as source proxy labels [OR EXPLICIT NEUTRAL CODES], without correspondence to Fitzpatrick types. Model-input age categories and audit age categories are defined separately to preserve checkpoint compatibility and make the corrected audit reproducible.”

Complete: verify original mapping or state that old display mapping cannot be recovered and replace display labels. Preserve original raw values and do not invent a light-to-dark ordering.

**11. Architecture.**

Proposed response: “The architecture supplement now gives all layer dimensions, activations, normalization, dropout, gate scaling, optimization and available random-seed information. The metadata gates are 384-dimensional feature-wise vectors. We also distinguish the full-data ungated dual-stream visual head from the metadata-gated DSAF head. The released DINOv3 local branch consumes all non-CLS tokens, including register tokens; the diagram now reflects the actual computation. Fold-specific vocabulary dimensions and exact model/software versions are [DOCUMENTED LOCATION].”

Complete: update diagrams and supply actual snapshot/config/version. Do not claim a nonexistent custom initialization or a training seed that only belongs to the splitter.

**12. DSAF values.**

Proposed response: “The 0.9260 accuracy corresponds to the image-only, ungated dual-stream model evaluated on the original 500-image test split. It is not directly comparable to a proxy-subset seven-fold result. The released standard proxy DSAF artifact gives mean accuracy 0.920304. [IF THE 0.9239 RUN IS RECOVERED, DOCUMENT ITS DATASET, INPUTS, FOLDS, SETTINGS AND ARTIFACT. OTHERWISE STATE IT HAS BEEN REMOVED.] We have established one traceable canonical result per defined experiment.”

Complete: distinguish correction from new result. “Separate runs” without provenance is not an adequate explanation.

**13. Statistical comparison.**

Proposed response after pairing verification: “We verified that the compared models predict the same test images under the same fold assignments. We report every fold difference and the mean concat-minus-DSAF difference, with an exploratory dependence-corrected cross-validation interval/test and the limitations of one CV run and configuration selection. [INSERT FINAL VALUES.] We describe the result as no demonstrated DSAF advantage under this protocol, not evidence of universal superiority or equivalence.”

Complete: validate sample identities before using provisional statistics in the plan. Repeated folds share training observations, so seven folds are not seven independent studies.

**14. Hyperparameter selection.**

Proposed response: “We clarify the actual configuration-selection history: [FACTUAL HISTORY]. [IF OUTER FOLDS WERE CONSULTED] Because those folds informed configuration exploration, their results are exploratory and may be optimistically selected. We do not present this as nested or unbiased confirmatory evaluation. Additional diagnostic probes use fixed settings stated before those runs. We acknowledge that this does not retrospectively remove selection bias from prior experiments.”

Complete: recover history; do not label the current protocol nested. The reviewer explicitly permits a limitation if nested CV is impractical.

**15. Baselines.**

Proposed response: “The revised baseline matrix distinguishes image-only DINOv3, metadata-only, simple concatenation and DSAF, and identifies each model's feature extractor, input set, trainable components, evaluation sample set and protocol. [DESCRIBE COMPLETED NEW PROBES.] Historical CNN and full-data results are separated from the proxy-subset comparison. We did not undertake additional backbone fine-tuning because it is not required for the narrower audit question.”

Complete: do not present unmatched full-data and proxy-subset results as ablations on the same samples. If new probes are linear, identify that limitation.

**16. Disease-specific measures.**

Proposed response: “We report a complete count confusion matrix and per-disease precision, sensitivity, specificity and F1 with denominators, together with [ONLY ACTUALLY COMPUTED CALIBRATION/AUC MEASURES]. We additionally report disease-conditioned subgroup rates to expose sparse or absent group–disease cells. These internal results do not establish clinical deployment performance.”

Complete: authentic predictions/probabilities. Stored classification reports can support their own rounded historical precision/recall/F1, but not an invented exact confusion matrix or new calibration values.

**17. Explanations.**

Minimum scope response: “We agree that qualitative maps do not establish clinically relevant feature reliance or exclude shortcuts. We have [CORRECTED/REMOVED] the method and checkpoint/fold attribution and removed claims that selected maps demonstrate causal reliance or fairness. Quantitative localization/shortcut claims would require verified masks and suitable perturbation controls; these data were not available in the current study. The visualizations, if retained, are explicitly illustrative.”

Complete: inspect actual generation provenance. If perturbation is performed, specify masks, held-out selection, replacement baseline, equal-area/random controls and paired performance change; do not use arbitrary center masks as lesion ground truth.

**18. Fairness terminology.**

Proposed response: “We now consistently use proxy-defined subgroup performance or disparity. We distinguish this from verified demographic fairness and clinical skin-phototype analysis, neither of which is established by the available labels. We removed phototype equivalences in figures as well as demographic/causal overclaims in the text.”

Complete: global source and figure search; changing captions without changing Figure2 is insufficient.

## Minor comments

Answer all 15 bullets separately in the final response using section 5 of `REVISION_PLAN.md`. For each, give the actual revised page/line/table reference. In particular, distinguish the known split seed from unavailable model-training seeds, the already supplied CSV from unverified annotation timing, and a rounded historical classification report from a complete new OOF confusion matrix.

Final check: every past-tense claim in the submitted rebuttal must correspond to an actual edit, recorded analysis or released artifact. Delete all bracketed placeholders and any branch of an alternative response that was not completed.
