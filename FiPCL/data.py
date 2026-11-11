"""Strict real-MRI paired datasets and train-fitted tabular preprocessing."""
import torch  # noqa: I001 — import torch before scikit-learn: with some Intel-OpenMP
import numpy as np  # builds, loading sklearn first and torch second makes libiomp abort
import pandas as pd  # while importing torch ("OMP: Error #179: Function Can't open SHM2")
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from configs.paths import MRI_BASE_PATH, MRI_CACHE_DIR
from torch.utils.data import Dataset


# ============================================================
# MRI helpers
# ============================================================

def mri_cache_path(filename, mri_folder, cache_dir=None):
    """Resolve one cohort cache. Missing filenames can never become zero images."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"Missing MRI filename for {mri_folder}")
    safe = Path(filename).stem.replace('/', '_').replace('\\', '_').replace('.nii', '').replace('.gz', '')
    return Path(cache_dir or MRI_CACHE_DIR) / Path(mri_folder).name / f'{safe}.npy'


class CachedMRI:
    """Lazy, worker-safe cached MRI reads; no silent image replacement.

    Only one batch is materialized instead of retaining tens of GB per run.
    NPY files must contain finite nonzero slices; raw NIfTI must be preprocessed
    separately using the same pipeline that produced the experiment's cache.
    """

    def __init__(self, df, mri_folder, cache_dir=None, target_slices=39,
                 target_h=182, target_w=182):
        if 'filename' not in df:
            raise ValueError('MRI dataset needs a filename column')
        self.paths = [mri_cache_path(f, mri_folder, cache_dir) for f in df['filename']]
        self.shape = (target_slices, 1, target_h, target_w)
        missing = [p for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} MRI caches missing; first: {missing[0]}. '
                'Set FIPCL_MRI_CACHE to the preprocessed cache root (<root>/<cohort>/*.npy).'
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        array = np.load(path, allow_pickle=False)
        if array.ndim == 3:
            array = array[:, None]
        if array.ndim != 4 or array.shape[1] != 1 or min(array.shape) == 0:
            raise ValueError(f'Invalid MRI shape {array.shape}: {path}')
        if not np.isfinite(array).all() or not np.any(array):
            raise ValueError(f'Nonfinite or all-zero MRI: {path}')
        n, _, h, w = self.shape
        if array.shape[2:] != (h, w):
            import cv2
            array = np.stack([cv2.resize(s[0], (w, h), interpolation=cv2.INTER_LINEAR)
                              for s in array])[:, None]
        if len(array) > n:
            start = (len(array) - n) // 2
            array = array[start:start + n]
        elif len(array) < n:
            padded = np.zeros(self.shape, dtype=np.float32)
            padded[:len(array)] = array
            array = padded
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))


def _preload_mri_data(df, mri_folder, cache_dir=None, target_slices=39,
                      target_h=182, target_w=182, desc=""):
    """Compatibility entry point returning strict lazy MRI storage."""
    return CachedMRI(df, mri_folder, cache_dir, target_slices, target_h, target_w)


def _detect_mri_folder(csv_path, mri_base_path=None):
    mri_base = Path(mri_base_path) if mri_base_path else MRI_BASE_PATH
    if mri_base.name in ['ANMerge', 'ADNI', 'AIBL', 'NACC', 'NIFD', 'OASIS']:
        return mri_base
    for part in reversed(Path(csv_path).parts[:-1]):
        for key in ['ANMerge', 'ADNI', 'AIBL', 'NACC', 'NIFD', 'OASIS']:
            if key in part:
                return mri_base / key
    # Synthetic cohorts have a named cache as well (e.g. MRI-Clinic-Demo).
    for part in reversed(Path(csv_path).parts[:-1]):
        if part.startswith('MRI-'):
            return mri_base / part
    raise ValueError(f'Cannot identify MRI cohort from {csv_path}; pass mri_base_path explicitly')


# ============================================================
# Feature scanning (matches single_modality scripts exactly)
# ============================================================

def _scan_feature_columns(df, exclude_cols, fallback_to_before=False):
    """Detect feature columns between filename and label columns.

    Matches the column detection logic in scripts/single_modality.
    When fallback_to_before=True (proteomics), falls back to columns
    before 'filename' if the between-segment is empty.
    """
    cols = list(df.columns)

    if 'filename' in cols and 'label' in cols:
        fname_idx = cols.index('filename')
        label_idx = cols.index('label')
        between_cols = cols[fname_idx + 1:label_idx]

        if fallback_to_before and len(between_cols) == 0:
            candidate_cols = cols[:fname_idx]
        else:
            candidate_cols = between_cols
    else:
        candidate_cols = [c for c in df.columns if c not in ['label', 'filename']]

    feature_cols = [c for c in candidate_cols
                    if c not in exclude_cols
                    and not c.startswith('Unnamed')
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    return feature_cols


def scan_blood_features(csv_path):
    """Matches Blood_train.py _scan_feature_info."""
    df = pd.read_csv(csv_path)
    exclude_cols = {'Unnamed: 0', 'Phase', 'Visit', 'SubjectID', 'Subject', 'Sample'}
    blood_cols = _scan_feature_columns(df, exclude_cols)
    return blood_cols, len(df)


def scan_clinic_features(csv_path):
    """Matches Clinic_train.py _scan_feature_info."""
    df = pd.read_csv(csv_path)
    exclude_cols = {'Unnamed: 0', 'Phase', 'Visit', 'SubjectID', 'Subject', 'Sample',
                    'Sex', 'APOE', 'gender', 'race', 'hispanic'}
    clinic_cols = _scan_feature_columns(df, exclude_cols)
    return clinic_cols, len(df)


def scan_proteomics_features(csv_path):
    """Matches Proteomics_train.py _scan_feature_info."""
    df = pd.read_csv(csv_path)
    exclude_cols = {'Unnamed: 0', 'RID', 'Visit_Code', 'Phase', 'Visit', 'SubjectID', 'filename', 'label'}
    proteomics_cols = _scan_feature_columns(df, exclude_cols, fallback_to_before=True)
    return proteomics_cols, len(df)


# ============================================================
# Blood Paired Dataset (matches Blood_train.py preprocessing)
# ============================================================

class BloodPairedDataset(Dataset):
    """Blood + MRI paired dataset. Preprocessing matches Blood_train.py."""

    def __init__(self, csv_path, mode='train', scaler=None, blood_cols=None,
                 low_var_mask=None, variance_threshold=0.001, pad_to_dim=None,
                 mri_base_path=None, cache_dir=None):
        self.csv_path = Path(csv_path)
        self.mode = mode
        self.mri_base_path = Path(mri_base_path) if mri_base_path else MRI_BASE_PATH
        self.mri_folder = _detect_mri_folder(csv_path, mri_base_path)

        df = pd.read_csv(csv_path)

        # Feature column detection (matches Blood_train.py)
        exclude_cols = {'Unnamed: 0', 'Phase', 'Visit', 'SubjectID', 'Subject', 'Sample'}
        if blood_cols is not None:
            self.blood_cols = [c for c in blood_cols if c in df.columns]
        else:
            self.blood_cols = _scan_feature_columns(df, exclude_cols)

        self.labels = df['label'].values
        blood_data = df[self.blood_cols].values.astype(np.float32)
        blood_data = np.nan_to_num(blood_data, nan=0.0)
        blood_data = np.clip(blood_data, -1e6, 1e6)

        # Low-variance filter (matches Blood_train.py)
        if mode == 'train':
            feature_vars = np.var(blood_data, axis=0)
            self.low_var_mask = feature_vars > variance_threshold
            n_removed = np.sum(~self.low_var_mask)
            if n_removed > 0:
                print(f"    -> Removed {n_removed} low-variance features")
            blood_data = blood_data[:, self.low_var_mask]
        else:
            self.low_var_mask = low_var_mask
            if self.low_var_mask is not None:
                blood_data = blood_data[:, self.low_var_mask]

        self.native_dim = blood_data.shape[1]

        # StandardScaler (matches Blood_train.py)
        if mode == 'train':
            self.scaler = StandardScaler()
            blood_data = self.scaler.fit_transform(blood_data)
        elif scaler is not None:
            blood_data = scaler.transform(blood_data)
            self.scaler = scaler

        # Zero-pad to global max dim (matches Blood_train.py)
        if pad_to_dim is not None and blood_data.shape[1] < pad_to_dim:
            padded = np.zeros((blood_data.shape[0], pad_to_dim), dtype=np.float32)
            padded[:, :blood_data.shape[1]] = blood_data
            blood_data = padded
            print(f"    -> Padded: {self.native_dim} -> {pad_to_dim}")
        else:
            print(f"    -> Native dim: {self.native_dim}")

        self.blood_data = blood_data
        self.filenames = df['filename'].values if 'filename' in df.columns else [None] * len(df)
        # Preload MRI (参考 MRI_Clinic_train.py 预加载策略)
        self._mri_data = _preload_mri_data(df, self.mri_folder, cache_dir, desc=" [Blood]")

        n_feat = len(self.blood_cols)
        print(f"  [{Path(csv_path).parent.name}] {len(df)} samples, "
              f"{n_feat} raw features -> {self.native_dim} after low-var -> {blood_data.shape[1]} final")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        blood = torch.from_numpy(self.blood_data[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'mri_slices': self._mri_data[idx], 'blood': blood, 'label': label}


def collate_blood(batch):
    return {
        'mri_slices': torch.stack([item['mri_slices'] for item in batch]),
        'blood': torch.stack([item['blood'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
    }


# ============================================================
# Clinic Paired Dataset (matches Clinic_train.py preprocessing)
# ============================================================

class ClinicPairedDataset(Dataset):
    """Clinic + MRI paired dataset. Preprocessing matches Clinic_train.py."""

    def __init__(self, csv_path, mode='train', scaler=None, clinic_cols=None,
                 low_var_mask=None, variance_threshold=0.001, pad_to_dim=None,
                 mri_base_path=None, cache_dir=None, clinic_feature_union=None,
                 clinic_missing_indicator=False):
        self.csv_path = Path(csv_path)
        self.mode = mode
        self.mri_base_path = Path(mri_base_path) if mri_base_path else MRI_BASE_PATH
        self.mri_folder = _detect_mri_folder(csv_path, mri_base_path)

        df = pd.read_csv(csv_path)

        # Feature column detection (matches Clinic_train.py)
        exclude_cols = {'Unnamed: 0', 'Phase', 'Visit', 'SubjectID', 'Subject', 'Sample',
                        'Sex', 'APOE', 'gender', 'race', 'hispanic'}
        if clinic_cols is not None:
            missing = set(clinic_cols) - set(df.columns)
            if missing:
                raise ValueError(f'Missing saved Clinic source columns in {csv_path}: {sorted(missing)}')
            self.clinic_cols = list(clinic_cols)
        else:
            self.clinic_cols = _scan_feature_columns(df, exclude_cols)
        canonical = [c.casefold() for c in self.clinic_cols]
        if len(canonical) != len(set(canonical)):
            raise ValueError(f'Ambiguous case-insensitive Clinic source columns: {self.clinic_cols}')
        self.clinic_feature_union = list(clinic_feature_union) if clinic_feature_union is not None else None
        self.clinic_missing_indicator = clinic_missing_indicator
        if clinic_missing_indicator and self.clinic_feature_union is None:
            raise ValueError('Clinic missing indicators require a semantic feature union')
        if mode != 'train' and self.clinic_feature_union is not None and (scaler is None or low_var_mask is None):
            raise ValueError('Clinic semantic evaluation requires the saved training scaler and variance mask')

        self.labels = df['label'].values
        clinic_data = df[self.clinic_cols].values.astype(np.float32)
        clinic_data = np.nan_to_num(clinic_data, nan=0.0)
        clinic_data = np.clip(clinic_data, -1e6, 1e6)

        # Low-variance filter (matches Clinic_train.py)
        if mode == 'train':
            feature_vars = np.var(clinic_data, axis=0)
            self.low_var_mask = feature_vars > variance_threshold
            n_removed = np.sum(~self.low_var_mask)
            if n_removed > 0:
                print(f"    -> Removed {n_removed} low-variance features")
            clinic_data = clinic_data[:, self.low_var_mask]
        else:
            self.low_var_mask = low_var_mask
            if self.low_var_mask is not None:
                clinic_data = clinic_data[:, self.low_var_mask]

        self.native_dim = clinic_data.shape[1]

        # StandardScaler (matches Clinic_train.py)
        if mode == 'train':
            self.scaler = StandardScaler() if clinic_data.shape[1] > 0 else None
            if self.scaler is not None:
                clinic_data = self.scaler.fit_transform(clinic_data)
        elif scaler is not None:
            clinic_data = scaler.transform(clinic_data)
            self.scaler = scaler
        else:
            self.scaler = None

        # Scatter retained, source-standardized features into train-defined semantic slots.
        # AIBL/NIFD/OASIS have different column orders and cannot share positional slots.
        if self.clinic_feature_union is not None:
            union = self.clinic_feature_union
            if union != [c.casefold() for c in union] or len(union) != len(set(union)):
                raise ValueError('Clinic union must have unique casefolded feature names')
            if set(canonical) - set(union):
                raise ValueError('Clinic source columns are absent from the saved feature union')
            retained = np.ones(len(canonical), dtype=bool) if self.low_var_mask is None else np.asarray(self.low_var_mask, dtype=bool)
            positions = {name: index for index, name in enumerate(union)}
            indices = [positions[name] for name, keep in zip(canonical, retained) if keep]
            aligned = np.zeros((len(df), len(union)), dtype=np.float32)
            aligned[:, indices] = clinic_data
            self.feature_mask = np.zeros(len(union), dtype=np.float32)
            self.feature_mask[indices] = 1.
            if clinic_missing_indicator:
                aligned = np.concatenate([aligned, np.broadcast_to(self.feature_mask, aligned.shape)], axis=1)
            if pad_to_dim is not None and aligned.shape[1] != pad_to_dim:
                raise ValueError('Clinic semantic layout does not match the saved input dimension')
            clinic_data = aligned
        # Historical positional layout retained for existing checkpoints.
        elif pad_to_dim is not None and clinic_data.shape[1] < pad_to_dim:
            padded = np.zeros((clinic_data.shape[0], pad_to_dim), dtype=np.float32)
            padded[:, :clinic_data.shape[1]] = clinic_data
            clinic_data = padded
            print(f"    -> Padded: {self.native_dim} -> {pad_to_dim}")
        else:
            print(f"    -> Native dim: {self.native_dim}")

        self.clinic_data = clinic_data
        self.filenames = df['filename'].values if 'filename' in df.columns else [None] * len(df)
        # Preload MRI (参考 MRI_Clinic_train.py 预加载策略)
        self._mri_data = _preload_mri_data(df, self.mri_folder, cache_dir, desc=" [Clinic]")

        n_feat = len(self.clinic_cols)
        print(f"  [{Path(csv_path).parent.name}] {len(df)} samples, "
              f"{n_feat} raw features -> {self.native_dim} after low-var -> {clinic_data.shape[1]} final")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        clinic = torch.from_numpy(self.clinic_data[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'mri_slices': self._mri_data[idx], 'clinic': clinic, 'label': label}


def collate_clinic(batch):
    return {
        'mri_slices': torch.stack([item['mri_slices'] for item in batch]),
        'clinic': torch.stack([item['clinic'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
    }


# ============================================================
# Proteomics Paired Dataset (matches Proteomics_train.py preprocessing)
# ============================================================

class ProteomicsPairedDataset(Dataset):
    """Proteomics + MRI paired dataset. Preprocessing matches Proteomics_train.py."""

    def __init__(self, csv_path, mode='train', scaler=None, proteomics_cols=None,
                 pad_to_dim=None, mri_base_path=None, cache_dir=None):
        self.csv_path = Path(csv_path)
        self.mode = mode
        self.mri_base_path = Path(mri_base_path) if mri_base_path else MRI_BASE_PATH
        self.mri_folder = _detect_mri_folder(csv_path, mri_base_path)

        df = pd.read_csv(csv_path)

        # Feature column detection (matches Proteomics_train.py)
        exclude_cols = {'Unnamed: 0', 'RID', 'Visit_Code', 'Phase', 'Visit', 'SubjectID', 'filename', 'label'}
        if proteomics_cols is not None:
            self.proteomics_cols = list(proteomics_cols)
        else:
            self.proteomics_cols = _scan_feature_columns(df, exclude_cols, fallback_to_before=True)

        self.labels = df['label'].values
        valid_features = np.array([c in df.columns for c in self.proteomics_cols], dtype=np.float32)
        proteomics_data = df.reindex(columns=self.proteomics_cols, fill_value=0).values.astype(np.float32)
        proteomics_data = np.nan_to_num(proteomics_data, nan=0.0)

        self.native_dim = proteomics_data.shape[1]

        # StandardScaler (matches Proteomics_train.py)
        if mode == 'train':
            self.scaler = StandardScaler()
            proteomics_data = self.scaler.fit_transform(proteomics_data)
        else:
            self.scaler = scaler
            if self.scaler is not None:
                proteomics_data = self.scaler.transform(proteomics_data)

        # Zero-pad to global max dim (matches Proteomics_train.py)
        if pad_to_dim is not None and proteomics_data.shape[1] < pad_to_dim:
            padded = np.zeros((proteomics_data.shape[0], pad_to_dim), dtype=np.float32)
            padded[:, :proteomics_data.shape[1]] = proteomics_data
            proteomics_data = padded
            print(f"    -> Padded: {self.native_dim} -> {pad_to_dim}")
        else:
            print(f"    -> Native dim: {self.native_dim}")

        # Binary mask: 1 for real features, 0 for zero-padded (matches Proteomics_train.py)
        self.feature_mask = np.zeros(proteomics_data.shape[1], dtype=np.float32)
        self.feature_mask[:len(valid_features)] = valid_features

        self.proteomics_data = proteomics_data
        self.filenames = df['filename'].values if 'filename' in df.columns else [None] * len(df)
        # Preload MRI (参考 MRI_Clinic_train.py 预加载策略)
        self._mri_data = _preload_mri_data(df, self.mri_folder, cache_dir, desc=" [Proteomics]")

        n_feat = len(self.proteomics_cols)
        print(f"  [{Path(csv_path).parent.name}] {len(df)} samples, "
              f"{n_feat} raw features -> {self.native_dim} native -> {proteomics_data.shape[1]} final")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prot = torch.from_numpy(self.proteomics_data[idx]).float()
        mask = torch.from_numpy(self.feature_mask).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'mri_slices': self._mri_data[idx], 'prot': prot, 'mask': mask, 'label': label}


def collate_proteomics(batch):
    return {
        'mri_slices': torch.stack([item['mri_slices'] for item in batch]),
        'prot': torch.stack([item['prot'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
    }


# ============================================================
# Single-Modality Datasets (for pseudo-pairing)
# ============================================================

class SingleModalityBloodDataset(Dataset):
    """Blood-only dataset from single-modality directories (no MRI pairing).

    Used for pseudo-pairing: blood samples paired with MRI prototypes.
    """

    def __init__(self, csv_path, scaler=None, blood_cols=None,
                 low_var_mask=None, variance_threshold=0.001, pad_to_dim=None):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        exclude_cols = {'Unnamed: 0', 'Phase', 'Visit', 'SubjectID', 'Subject', 'Sample'}
        if blood_cols is not None:
            self.blood_cols = [c for c in blood_cols if c in df.columns]
        else:
            self.blood_cols = _scan_feature_columns(df, exclude_cols)

        self.labels = df['label'].values

        if len(self.blood_cols) == 0:
            n_samples = len(df)
            dim = pad_to_dim if pad_to_dim else 1
            blood_data = np.zeros((n_samples, dim), dtype=np.float32)
            self.native_dim = 0
        else:
            blood_data = df[self.blood_cols].values.astype(np.float32)
            blood_data = np.nan_to_num(blood_data, nan=0.0)
            blood_data = np.clip(blood_data, -1e6, 1e6)

            if low_var_mask is not None and len(low_var_mask) == blood_data.shape[1]:
                blood_data = blood_data[:, low_var_mask]

            self.native_dim = blood_data.shape[1]

            if scaler is not None:
                blood_data = scaler.transform(blood_data)

        if pad_to_dim is not None and blood_data.shape[1] < pad_to_dim:
            padded = np.zeros((blood_data.shape[0], pad_to_dim), dtype=np.float32)
            if blood_data.shape[1] > 0:
                padded[:, :blood_data.shape[1]] = blood_data
            blood_data = padded

        self.blood_data = blood_data
        print(f"  [Single Blood] {len(df)} samples, {len(self.blood_cols)} feat cols, {blood_data.shape[1]} dims")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        blood = torch.from_numpy(self.blood_data[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'blood': blood, 'label': label, 'is_pseudo': True}


class SingleModalityProteomicsDataset(Dataset):
    """Proteomics-only dataset from single-modality directories (no MRI pairing).

    Used for pseudo-pairing: proteomics samples paired with MRI prototypes.
    """

    def __init__(self, csv_path, scaler=None, proteomics_cols=None, pad_to_dim=None):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        exclude_cols = {'Unnamed: 0', 'RID', 'Visit_Code', 'Phase', 'Visit', 'SubjectID',
                        'filename', 'label', 'Month', 'Site', 'Sex', 'Age', 'APOE', 'MMSE'}
        if proteomics_cols is not None:
            self.proteomics_cols = list(proteomics_cols)
        else:
            self.proteomics_cols = _scan_feature_columns(df, exclude_cols, fallback_to_before=True)

        self.labels = df['label'].values

        if len(self.proteomics_cols) == 0:
            n_samples = len(df)
            dim = pad_to_dim if pad_to_dim else 1
            prot_data = np.zeros((n_samples, dim), dtype=np.float32)
            self.native_dim = 0
        else:
            prot_data = df.reindex(columns=self.proteomics_cols, fill_value=0).values.astype(np.float32)
            prot_data = np.nan_to_num(prot_data, nan=0.0)
            self.native_dim = prot_data.shape[1]

            if scaler is not None:
                prot_data = scaler.transform(prot_data)

            self.feature_mask = np.zeros(prot_data.shape[1], dtype=np.float32)
            self.feature_mask[:self.native_dim] = 1.0

        if pad_to_dim is not None and prot_data.shape[1] < pad_to_dim:
            padded = np.zeros((prot_data.shape[0], pad_to_dim), dtype=np.float32)
            if prot_data.shape[1] > 0:
                padded[:, :prot_data.shape[1]] = prot_data
            prot_data = padded
            self.feature_mask = np.zeros(prot_data.shape[1], dtype=np.float32)
            self.feature_mask[:self.native_dim] = 1.0
        else:
            self.feature_mask = np.ones(prot_data.shape[1], dtype=np.float32)

        self.feature_mask = np.zeros(prot_data.shape[1], dtype=np.float32)
        self.feature_mask[:len(self.proteomics_cols)] = [c in df.columns for c in self.proteomics_cols]
        self.prot_data = prot_data
        print(f"  [Single Proteomics] {len(df)} samples, {len(self.proteomics_cols)} feat cols, {prot_data.shape[1]} dims")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prot = torch.from_numpy(self.prot_data[idx]).float()
        mask = torch.from_numpy(self.feature_mask).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'prot': prot, 'mask': mask, 'label': label, 'is_pseudo': True}


def collate_single_blood(batch):
    return {
        'blood': torch.stack([item['blood'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'is_pseudo': True,
    }


def collate_single_proteomics(batch):
    return {
        'prot': torch.stack([item['prot'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'is_pseudo': True,
    }
