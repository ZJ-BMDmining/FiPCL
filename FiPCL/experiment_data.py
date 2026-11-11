"""Train-fitted preprocessing and strict lazy loaders for the sharing study."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

import data
from reproducibility import sha256

PAIRS = {
    'clinic': ('Clinic', ('ADNI', 'AIBL', 'NIFD', 'OASIS'), data.ClinicPairedDataset,
               data.scan_clinic_features, data.collate_clinic, 'clinic_cols'),
    'blood': ('Blood', ('ADNI', 'ANMerge'), data.BloodPairedDataset,
              data.scan_blood_features, data.collate_blood, 'blood_cols'),
    'prot': ('Proteomics', ('ADNI', 'ANMerge'), data.ProteomicsPairedDataset,
             data.scan_proteomics_features, data.collate_proteomics, 'proteomics_cols'),
}
SINGLE = {
    'blood': (data.SingleModalityBloodDataset, data.collate_single_blood, ('ANMerge',)),
    'prot': (data.SingleModalityProteomicsDataset, data.collate_single_proteomics, ('ADNI', 'ANMerge')),
}


def preprocessing_record(ds, column_attr):
    scaler = ds.scaler
    record = {
        'feature_columns': list(getattr(ds, column_attr)), 'native_dim': ds.native_dim,
        'low_var_mask': getattr(ds, 'low_var_mask', None),
        'scaler': scaler,  # saved with torch.save, not an approximation/refit at test time
    }
    if column_attr == 'clinic_cols':
        record.update(clinic_feature_union=ds.clinic_feature_union,
                      clinic_missing_indicator=ds.clinic_missing_indicator)
    return record


def make_loader(dataset, collate, batch_size, workers, seed, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(seed))


def build_training_data(args):
    root = Path(args.split_root)
    paired: dict[str, dict[str, DataLoader]] = {}
    single: dict[str, DataLoader | None] = {}
    dimensions: dict[str, int] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for task_index, (mk, (category, cohorts, cls, scan, collate, key)) in enumerate(PAIRS.items()):
        folders = {cohort: root / f'MRI-{category}-{cohort}' / f'fold_{args.fold}_linked'
                   for cohort in cohorts}
        columns = {cohort: scan(folder / 'train.csv')[0] for cohort, folder in folders.items()}
        union = sorted(set().union(*map(set, columns.values()))) if mk == 'prot' else None
        clinic_union = None
        if mk == 'clinic' and getattr(args, 'clinic_feature_layout', 'positional') == 'union':
            clinic_union = sorted({name.casefold() for names in columns.values() for name in names})
        dimension = len(union) if union is not None else max(map(len, columns.values()))
        if clinic_union is not None:
            dimension = len(clinic_union) * (2 if getattr(args, 'clinic_missing_indicator', False) else 1)
        if dimension == 0:
            raise ValueError(f'No numeric training features for {mk}')
        dimensions[mk] = dimension
        splits: dict[str, list[Any]] = {'train': [], 'val': []}
        artifacts[mk], references = {}, {}
        for cohort, folder in folders.items():
            kwargs = {'pad_to_dim': dimension, key: union if union is not None else columns[cohort]}
            if mk == 'clinic':
                kwargs.update(clinic_feature_union=clinic_union,
                              clinic_missing_indicator=getattr(args, 'clinic_missing_indicator', False))
            training = cls(folder / 'train.csv', mode='train', **kwargs)
            if training.native_dim == 0 or training.scaler is None:
                raise ValueError(f'No valid features for {mk}/{cohort}')
            references[cohort] = training
            splits['train'].append(training)
            artifacts[mk][cohort] = preprocessing_record(training, key)
            kwargs[key] = list(getattr(training, key))
            if mk != 'prot':
                kwargs['low_var_mask'] = training.low_var_mask
            splits['val'].append(cls(folder / 'val.csv', mode='test', scaler=training.scaler, **kwargs))
            for split in ('train', 'val'):
                hashes[str(folder / f'{split}.csv')] = sha256(folder / f'{split}.csv')
        paired[mk] = {
            split: make_loader(ConcatDataset(datasets), collate, args.batch_size, args.workers,
                               args.seed + task_index * 10 + (split == 'val'), split == 'train')
            for split, datasets in splits.items()
        }
        single[mk] = None
        if mk in SINGLE and not args.no_pseudo:
            single_cls, single_collate, aux_cohorts = SINGLE[mk]
            items = []
            for cohort in aux_cohorts:
                source = root / '_single' / f'fold_{args.fold}' / category / cohort / 'train.csv'
                reference = references[cohort]
                kw = {key: list(getattr(reference, key)), 'pad_to_dim': dimension,
                      'scaler': reference.scaler}
                if mk != 'prot':
                    kw['low_var_mask'] = reference.low_var_mask
                ds = single_cls(source, **kw)
                if ds.native_dim == 0:
                    raise ValueError(f'No matched single-modality features: {source}')
                items.append(ds)
                hashes[str(source)] = sha256(source)
            single[mk] = make_loader(ConcatDataset(items), single_collate,
                                     args.aux_batch_size, 0, args.seed + 100 + task_index, True)
    return paired, single, dimensions, artifacts, hashes


def build_test_loader(args, mk, artifacts, dimension):
    category, cohorts, cls, _, collate, key = PAIRS[mk]
    items, hashes = [], {}
    for cohort in cohorts:
        saved = artifacts[mk][cohort]
        source = Path(args.split_root) / f'MRI-{category}-{cohort}' / 'test.csv'
        kw = {key: saved['feature_columns'], 'pad_to_dim': dimension, 'scaler': saved['scaler']}
        if mk != 'prot':
            kw['low_var_mask'] = saved['low_var_mask']
        if mk == 'clinic':
            kw.update(clinic_feature_union=saved.get('clinic_feature_union'),
                      clinic_missing_indicator=saved.get('clinic_missing_indicator', False))
        items.append(cls(source, mode='test', **kw))
        hashes[str(source)] = sha256(source)
    return make_loader(ConcatDataset(items), collate, args.batch_size, args.workers, args.seed), hashes


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()
            if torch.is_tensor(value)}


def checkpoint_config(args, dimensions):
    return {'input_dims': dimensions, 'embed_dim': args.embed_dim,
            'num_classes': 3, 'dropout': args.dropout, 'pretrained_mri': False,
            'mri_sharing': args.mri_sharing, 'tabular_sharing': args.tabular_sharing,
            'clinic_encoder_type': getattr(args, 'clinic_encoder_type', None)}
