# Jupyter run guide for the Windows `skin` environment

## Paths already configured

The notebooks use these defaults:

```text
Repository: D:\AIUB\Neural\A_Comprehensive_Framework_for_Auditing_Demographic_Bias_in_Synthetic_Skin_Disease_Datasets
Package:    D:\AIUB\Neural\A_Comprehensive_Framework_for_Auditing_Demographic_Bias_in_Synthetic_Skin_Disease_Datasets\revision_package
Images:     D:\AIUB\Neural\Synthetic_Skin_Disease_Dataset
Metadata:   D:\AIUB\Neural\complete_metadata_m.csv
Fallback:   D:\AIUB\Neural\complete_metadata.csv
```

Use `complete_metadata_m.csv`. The released multimodal notebooks explicitly used that filename. In the two uploaded CSVs, all 565 rows, row order, labels, ages, gender-proxy values and appearance-proxy values are identical. The only difference is `filepath`: `complete_metadata.csv` starts with the obsolete Colab prefix `/content/skindata/`, whereas `complete_metadata_m.csv` stores paths such as `Train/Herpes/file.png`.

The `_m.csv` paths align directly with your dataset tree. For example, `Train/Herpes/file.png` resolves to `D:\AIUB\Neural\Synthetic_Skin_Disease_Dataset\Train\Herpes\file.png`. The preflight checks every one of the 565 paths. It does not move files or guess by basename.

## Environment commands

Open **Anaconda Prompt** and run:

```bat
conda activate skin
cd /d D:\AIUB\Neural\A_Comprehensive_Framework_for_Auditing_Demographic_Bias_in_Synthetic_Skin_Disease_Datasets
python -m pip check
python -m ipykernel install --user --name skin --display-name "Python (skin)"
jupyter lab
```

The uploaded `skin` environment already contains the packages needed by every supplied script: NumPy, pandas, SciPy, scikit-learn, Pillow, JupyterLab, PyTorch, TorchVision and Transformers. Do **not** upgrade PyTorch or Transformers before checkpoint recovery. If notebook 00 reports a missing CPU-analysis package, run only this command and then restart the kernel:

```bat
python -m pip install -r revision_package\requirements-jupyter-minimal.txt
```

In JupyterLab, select the kernel **Python (skin)**.

## Run order

Every notebook contains one executable code cell. Open it and use **Run Cell**.

1. `notebooks/00_Preflight_and_Paths.ipynb` — mandatory; no training. Checks the environment, both CSVs, hashes, all dataset paths and GPU visibility.
2. `notebooks/01_Data_Audit_and_Splits.ipynb` — mandatory; no training. Hashes/scans images, screens duplicates, derives corrected audit age bins and reconstructs the released seven folds in one pass.
3. `notebooks/04_Grouped_Leakage_Protocol_if_Needed.ipynb` — run its first pass next. It filters the archive-wide screen to pairs for which **both** images are in the 565-image cohort, displays side-by-side contact sheets, and creates `reviewed_duplicate_pairs_cohort.csv`. Fill only `verified_related` with `yes` or `no`, save it, and rerun notebook 04. If any pair is verified, it creates new grouped folds without fitting a model.
4. `notebooks/02_Recover_Checkpoints_and_Evaluate.ipynb` — run only if the seven concat and seven DSAF `.pth` checkpoints still exist. No training. If the checkpoints were deleted, skip this notebook.
5. `notebooks/02A_DINOv3_One_Time_Cache.ipynb` — run once if notebook 00 says DINOv3 is not locally cached. First request access to the official gated model in your browser, then the notebook accepts a hidden Hugging Face read-token prompt and downloads the pretrained snapshot. It does not read the study images or train a model.
6. `notebooks/03_Frozen_Feature_Probes.ipynb` — recommended after the duplicate review. It makes one frozen DINOv3 inference pass, then fits fixed logistic probes only: image-only, metadata-only, concatenated and randomized-metadata controls. If notebook 04 created grouped folds, notebook 03 selects them automatically; otherwise it uses the reconstructed historical folds. It does not train DINOv3.
7. `notebooks/05_Annotation_Agreement_if_New_Ratings.ipynb` — conditional. First run creates two blank, matched templates. Two people must label them independently and blindly. A later run calculates agreement; it cannot reconstruct historical inter-rater reliability.

For the current no-checkpoint case, the shortest path is:

```text
00 (already done) → 01 (already done) → 04 first pass → review 13 cohort pairs
→ 04 second pass → 02A once → 03
```

The cohort-only review supports the 565-image proxy-defined analysis. It does not certify the separate 2,750-file archive or any full-archive benchmark. If full-archive performance remains a central result, its exact and near-duplicate findings must be handled separately.

Outputs go to:

```text
revision_package\runs\review_revision_run
```

Completed outputs are reused on a second run. If a run was interrupted and left a partial directory, inspect it and start Jupyter with a new label:

```bat
set DERM_RUN_LABEL=review_revision_retry1
jupyter lab
```

## Checkpoint locations

Notebook 02 first checks the original model folders inside the repository for:

```text
best_model_fold_1.pth ... best_model_fold_7.pth
best_multimodal_dsaf_fold_1.pth ... best_multimodal_dsaf_fold_7.pth
```

If checkpoints live elsewhere, set the directories before starting Jupyter:

```bat
set DERM_CONCAT_CHECKPOINT_DIR=D:\path\to\concat_checkpoints
set DERM_DSAF_CHECKPOINT_DIR=D:\path\to\dsaf_checkpoints
jupyter lab
```

The DINOv3 model identifier is `facebook/dinov3-vitb16-pretrain-lvd1689m`. Analysis notebook 03 always loads locally and will never silently download a replacement. The separate notebook 02A is the only supplied notebook authorized to download the official pretrained snapshot, and it does so only after interactive authentication.

## What not to do

- Do not flatten or move the existing Train/Test/Val disease folders. They already match `complete_metadata_m.csv`.
- Do not rename/sort the metadata before historical fold recovery. The supplied SHA-256 guard protects the original row order.
- Do not describe pHash candidates as duplicate patients without human/source evidence.
- Do not run grouped folds and then score them with old checkpoints. A changed split requires refitting the affected heads/models.
- Do not claim agreement unless two genuinely independent ratings were collected.
