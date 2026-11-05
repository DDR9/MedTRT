"""
Phase 1 Data & Bundle Manager (`src/medtrt/data.py`).

Responsibilities:
1. Ensure MONAI `spleen_ct_segmentation` bundle is present in `./bundles/spleen_ct_segmentation`.
   Supports `monai.bundle.download` with direct fallback to MONAI Model Zoo GitHub release zip.
2. Download and extract `Task09_Spleen.tar` from official MONAI AWS S3 bucket:
   `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar`
   into `./data/Task09_Spleen/`, verifying `imagesTr/`, `labelsTr/`, and `dataset.json`.
3. Deterministically resolve the 9 held-out validation cases (per `configs/train.json` `@images[-9:]`
   and alphanumeric sort fallback of all 41 `imagesTr` cases) and write `results/heldout_cases.json`.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from monai.bundle import download

from medtrt.config import BUNDLES_DIR, DATA_DIR, MEDTRT_ROOT, RESULTS_DIR

BUNDLE_NAME = "spleen_ct_segmentation"
BUNDLE_PATH = BUNDLES_DIR / BUNDLE_NAME
FALLBACK_BUNDLE_URL = (
    "https://github.com/Project-MONAI/model-zoo/releases/download/"
    "hosting_storage_0.4.0/spleen_ct_segmentation_v0.4.0.zip"
)
MSD_SPLEEN_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
TASK09_DIR = DATA_DIR / "Task09_Spleen"
HELDOUT_JSON_PATH = RESULTS_DIR / "heldout_cases.json"


def ensure_bundle() -> Path:
    """
    Ensure `./bundles/spleen_ct_segmentation` is present.
    Reuses existing local bundle if available, otherwise runs `monai.bundle.download`
    with fallback to the GitHub release archive.
    """
    model_pt = BUNDLE_PATH / "models" / "model.pt"
    if model_pt.exists():
        return BUNDLE_PATH

    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded in sibling ../medical-ai/models/bundle/spleen_ct_segmentation
    sibling_bundle = MEDTRT_ROOT.parent / "medical-ai" / "models" / "bundle" / BUNDLE_NAME
    if (sibling_bundle / "models" / "model.pt").exists():
        print(f"Copying cached MONAI bundle from {sibling_bundle} to {BUNDLE_PATH}...")
        shutil.copytree(sibling_bundle, BUNDLE_PATH, dirs_exist_ok=True)
        return BUNDLE_PATH

    try:
        print(f"Downloading MONAI bundle '{BUNDLE_NAME}' into {BUNDLES_DIR}...")
        download(name=BUNDLE_NAME, bundle_dir=str(BUNDLES_DIR), progress=True)
    except Exception as exc:
        print(f"monai.bundle.download failed ({exc}); falling back to {FALLBACK_BUNDLE_URL}...")
        zip_path = BUNDLES_DIR / "spleen_ct_segmentation.zip"
        urllib.request.urlretrieve(FALLBACK_BUNDLE_URL, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(BUNDLES_DIR))
        zip_path.unlink(missing_ok=True)

    if not model_pt.exists():
        raise FileNotFoundError(f"Bundle model.pt not found at {model_pt}")
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
        cases.append(
            {
                "case_id": case_id,
                "image": str(img_path.relative_to(MEDTRT_ROOT).as_posix()),
                "label": str(lbl_path.relative_to(MEDTRT_ROOT).as_posix()),
            }
        )

    payload = {
        "dataset": "MSD Task09_Spleen",
        "total_training_cases": len(all_images),
        "heldout_count": len(cases),
        "split_rule": split_source,
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
