"""Small helpers used by the single-cell Jupyter wrappers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from notebook_config import (
    DATA_ROOT,
    EXPECTED_METADATA_M_SHA256,
    EXPECTED_METADATA_ORIGINAL_SHA256,
    METADATA_M,
    METADATA_ORIGINAL,
    PACKAGE_DIR,
    REPO_DIR,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_relative_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    prefix = "/content/skindata/"
    if text.casefold().startswith(prefix):
        text = text[len(prefix):]
    return text


def compare_metadata_files() -> dict:
    result = {
        "metadata_m_exists": METADATA_M.is_file(),
        "metadata_original_exists": METADATA_ORIGINAL.is_file(),
    }
    if not (METADATA_M.is_file() and METADATA_ORIGINAL.is_file()):
        return result
    relative = pd.read_csv(METADATA_M)
    original = pd.read_csv(METADATA_ORIGINAL)
    if list(relative.columns) != list(original.columns) or len(relative) != len(original):
        result.update(equal_after_path_normalization=False, reason="schema or row-count difference")
        return result
    left = relative.copy()
    right = original.copy()
    left["filepath"] = left["filepath"].map(_normalized_relative_path)
    right["filepath"] = right["filepath"].map(_normalized_relative_path)
    result.update(
        equal_after_path_normalization=bool(left.astype(str).equals(right.astype(str))),
        rows=len(left),
        columns=list(left.columns),
        metadata_m_sha256=sha256(METADATA_M),
        metadata_original_sha256=sha256(METADATA_ORIGINAL),
    )
    return result


def select_metadata(require_known_hash: bool = True) -> Path:
    """Prefer the relative-path file used by the released multimodal notebooks."""
    candidates = [
        (METADATA_M, EXPECTED_METADATA_M_SHA256, "complete_metadata_m.csv"),
        (METADATA_ORIGINAL, EXPECTED_METADATA_ORIGINAL_SHA256, "complete_metadata.csv"),
    ]
    for path, expected, label in candidates:
        if not path.is_file():
            continue
        observed = sha256(path)
        if require_known_hash and observed != expected:
            raise ValueError(
                f"{label} exists but SHA-256 differs from the uploaded file. "
                f"Expected {expected}; observed {observed}. Stop before reconstructing historical folds."
            )
        return path
    raise FileNotFoundError(
        f"Neither metadata file exists. Expected {METADATA_M} (preferred) or {METADATA_ORIGINAL}."
    )


def validate_hierarchical_image_mapping(metadata_path: Path) -> dict:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_ROOT}")
    files = [
        path for path in DATA_ROOT.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    metadata = pd.read_csv(metadata_path)
    if "filepath" not in metadata:
        raise ValueError("Metadata must contain filepath")
    relative = metadata["filepath"].map(_normalized_relative_path)
    repeated = relative.str.casefold()[relative.str.casefold().duplicated(keep=False)]
    if len(repeated):
        raise ValueError(f"Metadata relative paths are not unique: {sorted(repeated.unique())[:10]}")
    resolved = [(DATA_ROOT / Path(value)).resolve() for value in relative]
    outside = [path for path in resolved if not path.is_relative_to(DATA_ROOT.resolve())]
    if outside:
        raise ValueError(f"Metadata resolves outside the dataset root: {outside[:5]}")
    missing = [relative.iloc[index] for index, path in enumerate(resolved) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} metadata images are absent from {DATA_ROOT}. First examples: {missing[:10]}"
        )
    file_ids = {path.relative_to(DATA_ROOT).as_posix().casefold() for path in files}
    metadata_ids = {value.casefold() for value in relative}
    return {
        "metadata_rows": int(len(metadata)),
        "metadata_unique_relative_paths": int(relative.str.casefold().nunique()),
        "image_files_found": int(len(files)),
        "matched_metadata_images": int(len(relative)),
        "extra_image_files": int(len(file_ids - metadata_ids)),
        "mapping": "metadata filepath relative to the Train/Test/Val dataset root",
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def run_once(script_name: str, arguments: list[str], out_dir: Path, expected: list[str]) -> Path:
    out_dir = Path(out_dir)
    expected_paths = [out_dir / item for item in expected]
    if expected_paths and all(path.exists() for path in expected_paths):
        print(f"Reusing completed output: {out_dir}")
        return out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete/nonempty output directory: {out_dir}. "
            "Inspect it, then choose a new DERM_RUN_LABEL before restarting Jupyter."
        )
    script = PACKAGE_DIR / "scripts" / script_name
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, str(script), *map(str, arguments), "--out", str(out_dir)]
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(PACKAGE_DIR), check=True)
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Script completed without expected outputs: {missing}")
    return out_dir


def expected_checkpoint_names(kind: str) -> list[str]:
    if kind == "concat":
        return [f"best_model_fold_{fold}.pth" for fold in range(1, 8)]
    if kind == "dsaf":
        return [f"best_multimodal_dsaf_fold_{fold}.pth" for fold in range(1, 8)]
    raise ValueError(kind)


def find_checkpoint_dir(preferred: Path, kind: str) -> Path:
    names = expected_checkpoint_names(kind)
    preferred = Path(preferred)
    if all((preferred / name).is_file() for name in names):
        return preferred
    first = names[0]
    parents = []
    if REPO_DIR.is_dir():
        parents = sorted({path.parent for path in REPO_DIR.rglob(first)})
    complete = [parent for parent in parents if all((parent / name).is_file() for name in names)]
    if len(complete) == 1:
        return complete[0]
    missing = [name for name in names if not (preferred / name).is_file()]
    detail = f"Default directory: {preferred}; missing: {missing}."
    if complete:
        detail += f" Multiple complete sets found: {[str(path) for path in complete]}."
    raise FileNotFoundError(
        f"Cannot identify the seven {kind} checkpoints. {detail} "
        f"Set DERM_{kind.upper()}_CHECKPOINT_DIR before launching Jupyter if they are elsewhere."
    )


def ensure_core_outputs(run_root: Path) -> tuple[Path, Path]:
    audit = Path(run_root) / "data_audit"
    metadata = audit / "audit_metadata.csv"
    manifest = audit / "reconstructed_fold_manifest.csv"
    if not metadata.is_file() or not manifest.is_file():
        raise FileNotFoundError("Run 01_Data_Audit_and_Splits.ipynb first")
    return metadata, manifest
