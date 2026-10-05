# Validation record

## Windows/Jupyter update (2026-10-05)

- Added seven `.ipynb` wrappers. Each contains exactly one executable code cell and declares the `Python (skin)` kernel. The seventh is an explicit one-time downloader for the official gated pretrained DINOv3 snapshot; analysis notebooks remain local-only.
- Default paths now match the author's confirmed layout: repository under `D:\AIUB\Neural`, `complete_metadata_m.csv`, and the hierarchical `Synthetic_Skin_Disease_Dataset\{Train,Test,Val}\<disease>` tree.
- Verified from the supplied files that `complete_metadata.csv` and `complete_metadata_m.csv` have the same 565 rows, row order and all non-path values. After removing `/content/skindata/` from the former, every row matches. The released multimodal notebooks reference `complete_metadata_m.csv`.
- The supplied `skin` export already lists NumPy, pandas, SciPy, scikit-learn, Pillow, JupyterLab, ipykernel, PyTorch, TorchVision and Transformers. No upgrade is prescribed; notebook 00 checks the live kernel and reports any discrepancy.
- The revised combined data-audit command was executed on a hierarchical 70-image integration fixture. It produced 70 audit rows, 490 fold-role rows (7 folds), exactly one held-out assignment per image and cross-role duplicate output.
- All seven notebook JSON files parse and every executable cell compiles.
- All delivered Python modules compile; all 13 unit tests pass.
- The author's completed notebook 00/01 outputs were inspected from `runs.zip` (SHA-256 `90a9cee9d692c72a5a6ce9e64b5ebedafee050706d8c411e91b754fdf0cd4cba`). The audit contains 2,750 decodable archive files and a 565-image metadata cohort. It reports 16 exact archive-wide pairs, including two across archive splits; none of the exact pairs is in the cohort. Thirteen unresolved pHash-only pairs have both images in the cohort.
- Notebook 04 was integration-tested against those real audit tables with synthetic placeholder pixels at the 21 required paths. It correctly reduced 396 archive-wide candidates to 13 cohort candidates, created four review pages, and, in a simulated one-verified-pair branch, produced 3,955 grouped fold-role rows across 564 clusters with no cluster crossing a fold role. Placeholder pixels were used only to test rendering and are not study evidence.

The real 2,750-image directory, historical checkpoints and local Hugging Face cache are not available in this workspace. Real-image hashing was run by the author and its output was analyzed here; checkpoint inference cannot be recovered because the checkpoints were deleted. The notebooks fail explicitly if a required path, hash, checkpoint set or local model cache is missing.

- Repository analyzed: commit `c81c8917afab02f04d9ceb56cef04a6474f8fd99`.
- Real-data execution completed: aggregate repository recovery only (`01_audit_repository.py`). Per-fold subgroup sums were checked across attributes and against reported fold accuracy; all four released model CSVs passed.
- Targeted unit tests: 13 tests passed, covering unsafe serialized input rejection, impossible counts, small perfect-group uncertainty, correct audit age boundaries, absent-class rates, FPR-sensitive EOD, undefined single-group EOD, probability order, repeated OOF samples, hierarchical and flat path resolution, cross-role duplicates and transitive cluster links.
- Artificial integration fixture: 70 generated test images and five arbitrary labels; data audit, grouped splitting, 35 small probe fits, prediction evaluation including paired output, and independent-label agreement all completed. These are software checks, not dermatology findings.
- Original first-cell notebook split definitions were loaded without executing training and compared on an ordered 565-row artificial fixture: concat and DSAF produced identical assignments and the expected 399/85/81 and 400/85/80 sizes.
- All delivered Python files passed syntax compilation. CLI help and revised multi-file prediction input were checked.
- CPU environment used: Python with NumPy2.3.5, pandas2.2.3, SciPy1.17.0, scikit-learn1.8.0, Pillow12.3.0. These are analysis-test versions, not the authors' historical training environment.

Limitations: scripts04/08 were not run through actual neural-model inference because PyTorch/Transformers and the author's checkpoint/image assets were unavailable here. Successful syntax/split checks do not prove checkpoint compatibility. Their strict recovery guards intentionally stop on provenance or metric mismatches. No actual DeepFace rerun, patient grouping, real/synthetic classification, independent reannotation, external validation or new dermatology model training was performed in this workspace.
