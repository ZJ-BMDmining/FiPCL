"""Shared experiment provenance, parameter ownership and subject contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from typing import TypedDict

import torch
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require_finite(tensor, context):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'Nonfinite {context}')


def subject_key(filename, cohort):
    """Canonical main-task participant identity, consistent across modalities."""
    name = str(filename)
    patterns = {
        'ADNI': r'ADNI_\d+_S_(\d+)',
        'AIBL': r'AIBL_(\d+)_',
        'NIFD': r'NIFD_(\d+_S_\d+)',
        'OASIS': r'(?:sub-)?(OAS\d+)',
        'ANMerge': r'^([A-Za-z]+\d+)_(?:m|M)\d+',
    }
    if cohort not in patterns:
        raise ValueError(f'No verified participant parser for {cohort}')
    match = re.search(patterns[cohort], Path(name).name)
    if not match:
        raise ValueError(f'Cannot recover {cohort} participant identity from MRI filename')
    value = match.group(1)
    if cohort in ('ADNI', 'AIBL'):
        value = str(int(value))
    return f'{cohort}:{value}'


def make_optimizer(model, learning_rate=2e-4, mri_scale=0.1, weight_decay=1e-4):
    """Every shared parameter appears once and receives all task gradients."""
    regular: list[torch.nn.Parameter] = []
    mri: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f'Duplicate optimizer parameter: {name}')
        seen.add(id(parameter))
        (mri if name.startswith(('mri_encoder.', 'mri_encoders.')) else regular).append(parameter)
    groups = [{'params': regular, 'lr': learning_rate, 'name': 'task_and_tabular'},
              {'params': mri, 'lr': learning_rate * mri_scale, 'name': 'mri'}]
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    owned = [id(p) for g in optimizer.param_groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(owned) != len(set(owned)) or set(owned) != expected:
        raise ValueError('Optimizer ownership must cover every trainable parameter exactly once')
    return optimizer


class BatchNormAudit(TypedDict):
    layers: int
    min_variance: float | None
    max_variance: float | None


def audit_batchnorm(model):
    result: BatchNormAudit = {'layers': 0, 'min_variance': None, 'max_variance': None}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.running_var is None:
                continue
            require_finite(module.running_mean, f'{name} running mean')
            require_finite(module.running_var, f'{name} running variance')
            lo, hi = float(module.running_var.min()), float(module.running_var.max())
            if lo < 0:
                raise FloatingPointError(f'Negative BatchNorm running variance: {name}')
            result['layers'] += 1
            result['min_variance'] = lo if result['min_variance'] is None else min(result['min_variance'], lo)
            result['max_variance'] = hi if result['max_variance'] is None else max(result['max_variance'], hi)
    return result
