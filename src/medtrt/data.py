"""
Phase 1 Data & Bundle Manager (`src/medtrt/data.py`).

Responsibilities:
1. Ensure the pinned MONAI `spleen_ct_segmentation` bundle is present in
   `./bundles/spleen_ct_segmentation`, and verify the SHA-256 of its weights.
2. Download and extract `Task09_Spleen.tar` from official MONAI AWS S3 bucket:
   `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar`
   into `./data/Task09_Spleen/`, verifying `imagesTr/`, `labelsTr/`, and `dataset.json`.
3. Deterministically resolve the 9 evaluation cases (`configs/train.json` `@images[-9:]`) and write
   `results/heldout_cases.json`.

About that split: these 9 cases never entered a gradient update, but they are the bundle's
validation set, and its `CheckpointSaver` selected `model.pt` by Dice on exactly these cases. MSD
test labels are not public, so no untouched test set exists here. Dice on these cases is an
optimistic estimate: it is valid for comparing backends, not for estimating generalization.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import torch
from monai.bundle import download
from monai.data import MetaTensor

from medtrt.config import BUNDLES_DIR, DATA_DIR, MEDTRT_ROOT, RESULTS_DIR

BUNDLE_NAME = "spleen_ct_segmentation"
BUNDLE_PATH = BUNDLES_DIR / BUNDLE_NAME
BUNDLE_VERSION = "0.6.1"
BUNDLE_MODEL_SHA256 = "502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe"
MSD_SPLEEN_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
TASK09_DIR = DATA_DIR / "Task09_Spleen"
UNCOMPRESSED_IMAGES_DIR = TASK09_DIR / "imagesTr_uncompressed"
HELDOUT_JSON_PATH = RESULTS_DIR / "heldout_cases.json"


def ensure_uncompressed_nifti(image_path: str | Path) -> Path:
    """
    Ensure an uncompressed `.nii` file exists for `image_path` so `nibabel.load(..., mmap=True)`
    can memory-map the volume without on-the-fly gzip decompression overhead.
    """
    p = Path(image_path)
    if p.suffix == ".nii" and p.exists():
        return p
    if p.name.endswith(".nii.gz"):
        UNCOMPRESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        out_nii = UNCOMPRESSED_IMAGES_DIR / p.name[:-3]  # strip .gz -> .nii
        if not out_nii.exists() or out_nii.stat().st_size == 0:
            with gzip.open(p, "rb") as f_in, open(out_nii, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out_nii
    return p


def load_nifti_mmap(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memory-mapped NIfTI reader (`nibabel.load(..., mmap=True)`) on uncompressed `.nii` volumes.
    Constructs a channel-first `MetaTensor` `(1, X, Y, Z)` with original affine and spatial metadata
    compatible with MONAI's `Orientationd`, `Spacingd`, and `Invertd`.

    This read is lazy: it maps the file and returns before the voxels are touched. The bytes are
    read when the tensor is first used, normally inside `.to(device)`, so time this together with
    the host-to-device copy. Timing it alone measures a file open, not a read.
    """
    d = dict(data)
    raw_path = Path(d["image"])
    nii_path = ensure_uncompressed_nifti(raw_path)
    img = nib.load(str(nii_path), mmap=True)
    arr = np.asarray(img.dataobj)
    affine_np = np.asarray(img.affine, dtype=np.float64)
    meta = {
        "spatial_shape": np.asarray(arr.shape[:3], dtype=np.int64),
        "original_channel_dim": "no_channel",
        "original_affine": affine_np,
        "filename_or_obj": str(nii_path),
    }
    tensor = torch.from_numpy(arr).unsqueeze(0)
    d["image"] = MetaTensor(
        tensor,
        affine=torch.as_tensor(affine_np, dtype=torch.float64),
        meta=meta,
    )
    return d


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_bundle() -> Path:
    """
    Ensure `./bundles/spleen_ct_segmentation` holds the pinned bundle, downloading it if needed.
    Raises if the weights do not match `BUNDLE_MODEL_SHA256`, because every published number is
    valid only for these exact weights.
    """
    model_pt = BUNDLE_PATH / "models" / "model.pt"
    if not model_pt.exists():
        BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MONAI bundle '{BUNDLE_NAME}' v{BUNDLE_VERSION} into {BUNDLES_DIR}...")
        download(name=BUNDLE_NAME, version=BUNDLE_VERSION, bundle_dir=str(BUNDLES_DIR), progress=True)
        if not model_pt.exists():
            raise FileNotFoundError(f"Bundle model.pt not found at {model_pt}")

    actual = _sha256(model_pt)
    if actual != BUNDLE_MODEL_SHA256:
        raise RuntimeError(
            f"Bundle weights do not match the pinned SHA-256.\n  expected {BUNDLE_MODEL_SHA256}\n  found    {actual}"
        )
    return BUNDLE_PATH


def ensure_msd_task09() -> Path:
    """
    Download and extract `Task09_Spleen.tar` into `./data/Task09_Spleen/`.
    Verifies presence of `imagesTr/` (41 NIfTI volumes), `labelsTr/` (41 NIfTI volumes), and `dataset.json`.
    """
    images_tr = TASK09_DIR / "imagesTr"
    labels_tr = TASK09_DIR / "labelsTr"
    dataset_json = TASK09_DIR / "dataset.json"

    existing_images = (
        [p for p in images_tr.glob("spleen_*.nii.gz") if not p.name.startswith("._")]
        if images_tr.exists()
        else []
    )
    existing_labels = (
        [p for p in labels_tr.glob("spleen_*.nii.gz") if not p.name.startswith("._")]
        if labels_tr.exists()
        else []
    )

    if len(existing_images) == 41 and len(existing_labels) == 41 and dataset_json.exists():
        print(f"MSD Task09_Spleen already verified at {TASK09_DIR} (41 imagesTr, 41 labelsTr).")
        return TASK09_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming & extracting MSD Task09_Spleen from {MSD_SPLEEN_URL} to {DATA_DIR}...")
    req = urllib.request.urlopen(MSD_SPLEEN_URL, timeout=120)
    extracted_count = 0
    with tarfile.open(fileobj=req, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if any(p.startswith("._") for p in parts):
                continue
            # Extract dataset.json, imagesTr/*, and labelsTr/* (skip imagesTs/* to save disk/time)
            rel_path = Path(*parts)
            if "imagesTr" in parts or "labelsTr" in parts or parts[-1] == "dataset.json":
                dest_path = DATA_DIR / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(dest_path, "wb") as dst:
                    if src is not None:
                        shutil.copyfileobj(src, dst)
                extracted_count += 1
                if extracted_count % 10 == 0:
                    print(f"  Extracted {extracted_count} files...")

    # Verify extracted counts
    final_images = sorted(
        [p for p in images_tr.glob("spleen_*.nii.gz") if not p.name.startswith("._")]
    )
    final_labels = sorted(
        [p for p in labels_tr.glob("spleen_*.nii.gz") if not p.name.startswith("._")]
    )
    if not dataset_json.exists() or len(final_images) != 41 or len(final_labels) != 41:
        raise RuntimeError(
            f"Dataset verification failed: dataset.json={dataset_json.exists()}, "
            f"imagesTr={len(final_images)}, labelsTr={len(final_labels)}"
        )
    print(f"Verified Task09_Spleen: {len(final_images)} imagesTr, {len(final_labels)} labelsTr, dataset.json OK.")
    return TASK09_DIR


def resolve_heldout_split() -> Dict[str, Any]:
    """
    Deterministically resolve the held-out validation split:
    1. Inspect `dataset.json` and `bundles/spleen_ct_segmentation/configs/train.json`.
    2. `train.json` specifies `sorted(glob.glob(@dataset_dir + '/imagesTr/*.nii.gz'))[-9:]`.
       In accordance with both `train.json` and the Phase 1 Fallback Rule, sort all 41 cases
       in `imagesTr` alphanumerically and reserve the final 9 cases strictly for evaluation.
    3. Write `results/heldout_cases.json`.
    """
    ensure_bundle()
    ensure_msd_task09()

    images_tr = TASK09_DIR / "imagesTr"
    labels_tr = TASK09_DIR / "labelsTr"
    train_json_path = BUNDLE_PATH / "configs" / "train.json"
    metadata_json_path = BUNDLE_PATH / "configs" / "metadata.json"

    all_images = sorted(
        [p for p in images_tr.glob("spleen_*.nii.gz") if not p.name.startswith("._")],
        key=lambda p: p.name,
    )
    assert len(all_images) == 41, f"Expected 41 cases in imagesTr, found {len(all_images)}"

    # Inspect train.json for split expression
    split_source = "alphanumeric_final_9_and_bundle_train_json"
    train_val_expr = None
    if train_json_path.exists():
        with open(train_json_path, "r", encoding="utf-8") as f:
            train_cfg = json.load(f)
        train_val_expr = (
            train_cfg.get("validate", {}).get("dataset", {}).get("data", "")
        )

    bundle_ref_dice = 0.961
    if metadata_json_path.exists():
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        bundle_ref_dice = float(meta.get("eval_metrics", {}).get("mean_dice", 0.961))

    heldout_images = all_images[-9:]
    cases: List[Dict[str, str]] = []
    for img_path in heldout_images:
        case_id = img_path.name.replace(".nii.gz", "")
        lbl_path = labels_tr / img_path.name
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing ground-truth label for {case_id}: {lbl_path}")
        uncompressed_path = ensure_uncompressed_nifti(img_path)
        cases.append(
            {
                "case_id": case_id,
                "image": str(img_path.relative_to(MEDTRT_ROOT).as_posix()),
                "image_uncompressed": str(uncompressed_path.relative_to(MEDTRT_ROOT).as_posix()),
                "label": str(lbl_path.relative_to(MEDTRT_ROOT).as_posix()),
            }
        )

    payload = {
        "dataset": "MSD Task09_Spleen",
        "total_training_cases": len(all_images),
        "heldout_count": len(cases),
        "split_rule": split_source,
        "split_role": "bundle_validation_split_used_for_checkpoint_selection",
        "bundle_version": BUNDLE_VERSION,
        "bundle_validate_expr": train_val_expr,
        "bundle_reference_mean_dice": bundle_ref_dice,
        "case_ids": [c["case_id"] for c in cases],
        "cases": cases,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HELDOUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 84)
    print(f"RESOLVED DETERMINISTIC HELD-OUT SPLIT ({len(cases)} / {len(all_images)} CASES)")
    print("=" * 84)
    print(f"Split Rule        : {split_source}")
    print(f"Bundle Val Config : {train_val_expr}")
    print(f"Bundle Ref Dice   : {bundle_ref_dice:.4f}")
    print(f"Held-Out Case IDs : {payload['case_ids']}")
    print(f"Saved             : {HELDOUT_JSON_PATH}")
    print("=" * 84)
    return payload


if __name__ == "__main__":
    resolve_heldout_split()
