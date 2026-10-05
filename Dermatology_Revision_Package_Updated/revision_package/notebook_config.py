"""Paths for the single-cell Jupyter revision notebooks.

Defaults exactly match the Windows folders supplied by the author.  Optional
DERM_* environment variables make the same notebooks testable elsewhere
without editing their code cells.
"""
from __future__ import annotations

import os
from pathlib import Path


NEURAL_DIR = Path(os.environ.get("DERM_NEURAL_DIR", r"D:\AIUB\Neural"))
REPO_DIR = Path(os.environ.get(
    "DERM_REPO_DIR",
    r"D:\AIUB\Neural\A_Comprehensive_Framework_for_Auditing_Demographic_Bias_in_Synthetic_Skin_Disease_Datasets",
))
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DERM_DATA_ROOT", str(NEURAL_DIR / "Synthetic_Skin_Disease_Dataset")))

METADATA_M = Path(os.environ.get("DERM_METADATA_M", str(NEURAL_DIR / "complete_metadata_m.csv")))
METADATA_ORIGINAL = Path(os.environ.get("DERM_METADATA_ORIGINAL", str(NEURAL_DIR / "complete_metadata.csv")))

# Hashes of the two files supplied in this conversation.  They have identical
# row order and non-path values; only the filepath prefix differs.
EXPECTED_METADATA_M_SHA256 = "2ff2c29654ae2966226bff495b951432a7392b0ea92632d6f2726c7f9abb07e6"
EXPECTED_METADATA_ORIGINAL_SHA256 = "6df1fd913e3905719c74b0c9ea81b1f91bf2d63be01763fdc9ffb17c0ca48025"

RUN_LABEL = os.environ.get("DERM_RUN_LABEL", "review_revision_run")
RUN_ROOT = PACKAGE_DIR / "runs" / RUN_LABEL

HF_MODEL_SOURCE = os.environ.get(
    "DERM_HF_MODEL_SOURCE",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
)
CONCAT_CHECKPOINT_DIR = Path(os.environ.get(
    "DERM_CONCAT_CHECKPOINT_DIR",
    str(REPO_DIR / "metadata_Multimodal_DINOv3_Concat_KFold"),
))
DSAF_CHECKPOINT_DIR = Path(os.environ.get(
    "DERM_DSAF_CHECKPOINT_DIR",
    str(REPO_DIR / "metadata_Multimodal_DINOv3_DSAF_KFold"),
))

PHASH_DISTANCE = 6
RECOVERY_PERMUTATIONS = 20
PROBE_CONTROL_REPEATS = 3
