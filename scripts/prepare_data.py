#!/usr/bin/env python3
"""Protect held-out subjects across all three tasks before sharing encoders.

Preserves existing test sets. Removes global test subjects from train/val and
global validation subjects from training. Auxiliary sources without recoverable
subject identifiers remain explicitly unverified; these are exploratory runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import torch  # Import before pandas on the supported OpenMP environment.
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'FiPCL'), str(ROOT)]
from configs.paths import SPLIT_ROOT
from reproducibility import sha256, subject_key, write_json

COHORTS = {
    'Clinic': ('ADNI', 'AIBL', 'NIFD', 'OASIS'),
    'Blood': ('ADNI', 'ANMerge'),
    'Proteomics': ('ADNI', 'ANMerge'),
}


def identities(frame, cohort):
    return frame['filename'].map(lambda name: subject_key(name, cohort))


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError(f'Refusing to replace split directory: {output}')
    output.mkdir(parents=True)
    datasets = [(f'MRI-{category}-{cohort}', cohort)
                for category, cohorts in COHORTS.items() for cohort in cohorts]
    held_test, tests, hashes = set(), {}, {}
    for name, cohort in datasets:
        path = source / name / 'test.csv'
        # Test labels are not used to make data policy/model selection decisions.
        tests[name] = pd.read_csv(path, usecols=['filename'])
        held_test.update(identities(tests[name], cohort))
        hashes[str(path)] = sha256(path)
        (output / name).mkdir()
        shutil.copy2(path, output / name / 'test.csv')
    report: dict[str, Any] = {'source': str(source), 'output': str(output), 'source_hashes': hashes,
              'preparation_torch': torch.__version__,
              'global_test_subjects': len(held_test), 'folds': {},
              'single_modality_subject_audit': 'PARTIAL_UNVERIFIED_ANMERGE_TABULAR',
              'paper_evidence_eligible': False,
              'interpretation': 'Post-test-known exploratory architecture comparison; '
                                'original fixed tests, cross-task subject protection.'}
    for fold in range(5):
        held_val: set[str] = set()
        frames: dict[str, dict[str, pd.DataFrame]] = {}
        for name, cohort in datasets:
            frames[name] = {}
            for split in ('train', 'val'):
                path = source / name / f'fold_{fold}' / f'{split}.csv'
                frames[name][split] = pd.read_csv(path)
                hashes[str(path)] = sha256(path)
            val = frames[name]['val']
            keys = identities(val, cohort)
            held_val.update(keys[~keys.isin(held_test)])
        fold_report: dict[str, Any] = {'global_validation_subjects': len(held_val), 'datasets': {}, 'auxiliary': {}}
        train_keys: set[str] = set()
        val_keys: set[str] = set()
        for name, cohort in datasets:
            folder = output / name / f'fold_{fold}_linked'
            folder.mkdir()
            counts = {}
            for split in ('train', 'val'):
                frame = frames[name][split]
                excluded = held_test | held_val if split == 'train' else held_test
                keys = identities(frame, cohort)
                clean = frame.loc[~keys.isin(excluded)].reset_index(drop=True)
                if clean.empty:
                    raise ValueError(f'Empty protected split: {name}/{fold}/{split}')
                clean.to_csv(folder / f'{split}.csv', index=False)
                (train_keys if split == 'train' else val_keys).update(identities(clean, cohort))
                counts[split] = {'before': len(frame), 'after': len(clean),
                                 'classes': {str(k): int(v) for k, v in clean['label'].value_counts().items()}}
            (folder / 'test.csv').symlink_to(Path('../test.csv'))
            fold_report['datasets'][name] = counts
        if train_keys & (held_test | val_keys) or val_keys & held_test:
            raise ValueError('Cross-task paired subject leakage remained after protection')
        for path in sorted((source / '_single').glob('*/*/train.csv')):
            category, cohort = path.parent.parent.name, path.parent.name
            if category not in ('Blood', 'Proteomics'):
                continue
            frame = pd.read_csv(path)
            known = False
            if cohort == 'ADNI' and 'RID' in frame:
                keys = frame['RID'].map(lambda rid: f'ADNI:{int(rid)}')
                known = True
            else:
                keys = pd.Series('', index=frame.index)
            clean = frame.loc[~keys.isin(held_test | held_val)]
            dest = output / '_single' / f'fold_{fold}' / category / cohort / 'train.csv'
            dest.parent.mkdir(parents=True, exist_ok=True)
            clean.to_csv(dest, index=False)
            hashes[str(path)] = sha256(path)
            fold_report['auxiliary'][f'{category}/{cohort}'] = {
                'before': len(frame), 'after': len(clean),
                'subject_identity': 'verified' if known else 'unresolved',
            }
        report['folds'][str(fold)] = fold_report
    write_json(output / 'data_manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SPLIT_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print('Prepared five protected folds:', args.output)
    print('Auxiliary identity:', report['single_modality_subject_audit'])
    for fold, info in report['folds'].items():
        print('fold', fold, {k: (v['train']['after'], v['val']['after'])
                             for k, v in info['datasets'].items()})


if __name__ == '__main__':
    main()
