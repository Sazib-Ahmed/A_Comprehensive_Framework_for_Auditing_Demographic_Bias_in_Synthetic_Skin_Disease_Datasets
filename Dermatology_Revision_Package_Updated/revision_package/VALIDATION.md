# Validation record

## Windows/Jupyter update (2026-10-05)

- Added six `.ipynb` wrappers. Each contains exactly one executable code cell and declares the `Python (skin)` kernel.
- Default paths now match the author's confirmed layout: repository under `D:\AIUB\Neural`, `complete_metadata_m.csv`, and the hierarchical `Synthetic_Skin_Disease_Dataset\{Train,Test,Val}\<disease>` tree.
- Verified from the supplied files that `complete_metadata.csv` and `complete_metadata_m.csv` have the same 565 rows, row order and all non-path values. After removing `/content/skindata/` from the former, every row matches. The released multimodal notebooks reference `complete_metadata_m.csv`.
- The supplied `skin` export already lists NumPy, pandas, SciPy, scikit-learn, Pillow, JupyterLab, ipykernel, PyTorch, TorchVision and Transformers. No upgrade is prescribed; notebook 00 checks the live kernel and reports any discrepancy.
- The revised combined data-audit command was executed on a hierarchical 70-image integration fixture. It produced 70 audit rows, 490 fold-role rows (7 folds), exactly one held-out assignment per image and cross-role duplicate output.
- All six notebook JSON files parse and every executable cell compiles.
- All delivered Python modules compile; the original 11 unit tests still pass.

The real 2,750-image directory, historical checkpoints and local Hugging Face cache are not available in this workspace, so real-image hashing and checkpoint inference remain author-run steps. The notebooks fail explicitly if a required path, hash, checkpoint set or local model cache is missing.

- Repository analyzed: commit `c81c8917afab02f04d9ceb56cef04a6474f8fd99`.
- Real-data execution completed: aggregate repository recovery only (`01_audit_repository.py`). Per-fold subgroup sums were checked across attributes and against reported fold accuracy; all four released model CSVs passed.
- Targeted unit tests: 11 tests passed, covering unsafe serialized input rejection, impossible counts, small perfect-group uncertainty, correct audit age boundaries, absent-class rates, FPR-sensitive EOD, undefined single-group EOD, probability order, repeated OOF samples, cross-role duplicates and transitive cluster links.
- Artificial integration fixture: 70 generated test images and five arbitrary labels; data audit, grouped splitting, 35 small probe fits, prediction evaluation including paired output, and independent-label agreement all completed. These are software checks, not dermatology findings.
- Original first-cell notebook split definitions were loaded without executing training and compared on an ordered 565-row artificial fixture: concat and DSAF produced identical assignments and the expected 399/85/81 and 400/85/80 sizes.
- All delivered Python files passed syntax compilation. CLI help and revised multi-file prediction input were checked.
- CPU environment used: Python with NumPy2.3.5, pandas2.2.3, SciPy1.17.0, scikit-learn1.8.0, Pillow12.3.0. These are analysis-test versions, not the authors' historical training environment.

Limitations: scripts04/08 were not run through actual neural-model inference because PyTorch/Transformers and the author's checkpoint/image assets were unavailable here. Successful syntax/split checks do not prove checkpoint compatibility. Their strict recovery guards intentionally stop on provenance or metric mismatches. No actual DeepFace rerun, patient grouping, real/synthetic classification, independent reannotation, external validation or new dermatology model training was performed in this workspace.
