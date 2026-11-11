#!/usr/bin/env python3
"""5-fold CV splits for paired datasets + single-modality data.

Paired datasets (8):
    Layout: split_data_5fold/<dataset>/
        test.csv                     (20% shared test)
        fold_0..4/{train,val}.csv    (5-fold train/val on remaining 80%)

Single-modality datasets (Blood/Proteomics — no MRI, used for pseudo-pairing):
    Layout: split_data_5fold/_single/<name>/train.csv
    These are whole-dataset copies (no fold split needed).
    They provide auxiliary training signal via pseudo-pairing.

After running, protect subjects with scripts/prepare_data.py before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch  # noqa: F401 -- import before pandas/sklearn for the supported OpenMP environment.
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import DATA_ROOT, SPLIT_ROOT as OUT_ROOT  # noqa: E402
SEED = 42
N_FOLDS = 5
TEST_RATIO = 0.20

# ---- Paired datasets (MRI + tabular) ----
PAIRED_DATASETS = [
    ("MRI-Clinic-ADNI",        "MRI-Clinic/ADNI.csv",        ["mri", "clinic"]),
    ("MRI-Clinic-AIBL",        "MRI-Clinic/AIBL.csv",        ["mri", "clinic"]),
    ("MRI-Clinic-NIFD",        "MRI-Clinic/NIFD.csv",        ["mri", "clinic"]),
    ("MRI-Clinic-OASIS",       "MRI-Clinic/OASIS.csv",       ["mri", "clinic"]),
    ("MRI-Blood-ADNI",         "MRI-Blood/ADNI.csv",         ["mri", "blood"]),
    ("MRI-Blood-ANMerge",      "MRI-Blood/ANMerge.csv",      ["mri", "blood"]),
    ("MRI-Proteomics-ADNI",    "MRI-Proteomics/ADNI.csv",    ["mri", "proteomics"]),
    ("MRI-Proteomics-ANMerge", "MRI-Proteomics/ANMerge.csv", ["mri", "proteomics"]),
]

# ---- Single-modality datasets (tabular-only, for pseudo-pairing) ----
SINGLE_DATASETS = [
    ("Blood/ANMerge",      "Blood/ANMerge.csv",      "blood"),
    ("Proteomics/ANMerge", "Proteomics/ANMerge.csv", "proteomics"),
    ("Proteomics/ADNI",    "Proteomics/ADNI.csv",    "proteomics"),
]


def _can_stratify(labels: np.ndarray, min_per_class: int) -> bool:
    if len(np.unique(labels)) < 2:
        return False
    return min(Counter(labels.tolist()).values()) >= min_per_class


def split_paired(df: pd.DataFrame, name: str) -> dict:
    """Stratified 5-fold split for paired MRI-tabular data."""
    df = df.copy()
    if "label" not in df.columns:
        raise ValueError(f"{name}: no 'label' column")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    labels = df["label"].values
    idx = np.arange(len(df))
    rng = np.random.RandomState(SEED)

    test_ratio = TEST_RATIO
    use_strat_test = _can_stratify(labels, 2)
    if use_strat_test:
        dev_idx, test_idx = train_test_split(
            idx, test_size=test_ratio, random_state=SEED, stratify=labels)
    else:
        shuffled = idx.copy(); rng.shuffle(shuffled)
        n_test = max(1, int(round(len(df) * test_ratio)))
        dev_idx, test_idx = shuffled[n_test:], shuffled[:n_test]

    test_df = df.iloc[test_idx].reset_index(drop=True)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    dev_labels = dev_df["label"].values

    strategy = "stratified"
    if _can_stratify(dev_labels, N_FOLDS):
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(len(dev_df)), dev_labels))
    else:
        strategy = "kfold(no_strat)"
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(len(dev_df))))

    folds_meta = []
    ds_dir = OUT_ROOT / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(ds_dir / "test.csv", index=False)

    for f, (tr_local, va_local) in enumerate(splits):
        fdir = ds_dir / f"fold_{f}"
        fdir.mkdir(parents=True, exist_ok=True)
        dev_df.iloc[tr_local].reset_index(drop=True).to_csv(fdir / "train.csv", index=False)
        dev_df.iloc[va_local].reset_index(drop=True).to_csv(fdir / "val.csv", index=False)
        folds_meta.append({
            "fold": f,
            "train": len(tr_local), "val": len(va_local),
            "train_class_dist": dict(Counter(dev_df.iloc[tr_local]["label"].tolist())),
            "val_class_dist": dict(Counter(dev_df.iloc[va_local]["label"].tolist())),
        })

    return {
        "n": len(df), "test": len(test_df), "dev": len(dev_df),
        "test_class_dist": dict(Counter(test_df["label"].tolist())),
        "dev_class_dist": dict(Counter(dev_df["label"].tolist())),
        "n_folds": N_FOLDS, "fold_strategy": strategy,
        "folds": [{**m, "train_class_dist": dict(m["train_class_dist"]),
                   "val_class_dist": dict(m["val_class_dist"])} for m in folds_meta],
    }


def copy_single(name: str, rel: str, modality: str, summary: dict) -> None:
    """Copy single-modality CSV into _single/<name>/train.csv (whole dataset)."""
    src = DATA_ROOT / rel
    if not src.exists():
        print(f"  [skip single] {name}: {src} missing")
        return
    df = pd.read_csv(src)
    out_dir = OUT_ROOT / "_single" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "train.csv", index=False)
    print(f"  [single] {name}: {len(df)} samples -> _single/{name}/train.csv")
    summary["_single"][f"single_{name.replace('/', '_')}"] = {
        "name": name, "csv_path": str(rel), "modality": modality, "n": len(df)}


def main(argv=None) -> None:
    global DATA_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)
    DATA_ROOT, OUT_ROOT = args.data_root.resolve(), args.output_dir.resolve()
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise FileExistsError(f'Refusing to replace existing splits: {OUT_ROOT}')
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "seed": SEED, "n_folds": N_FOLDS, "test_ratio": TEST_RATIO,
        "datasets": {}, "_single": {},
    }

    # Paired datasets
    for name, rel, modalities in PAIRED_DATASETS:
        csv_path = DATA_ROOT / rel
        if not csv_path.exists():
            print(f"[skip] {name}: {csv_path} missing")
            continue
        df = pd.read_csv(csv_path)
        lbls = dict(Counter(df["label"].dropna().astype(int).tolist()))
        print(f"[{name}] rows={len(df)}  classes={lbls}")
        info = split_paired(df, name)
        info["csv_path"] = str(rel)
        info["modalities"] = modalities
        summary["datasets"][name] = info
        print(f"    -> test={info['test']}, dev={info['dev']}, folds=",
              ", ".join(f"[{f['train']},{f['val']}]" for f in info["folds"]),
              f", strategy={info['fold_strategy']}")

    # Single-modality datasets
    for name, rel, modality in SINGLE_DATASETS:
        copy_single(name, rel, modality, summary)

    (OUT_ROOT / "split_info_5fold.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary -> {OUT_ROOT/'split_info_5fold.json'}")


if __name__ == "__main__":
    main()
