#!/usr/bin/env python3
"""Evaluate sealed FiPCL checkpoints using saved training preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'FiPCL'), str(ROOT)]
from experiment_data import build_test_loader
from trainer import predict
from models.shared_fusion_model import MODALITY_KEYS, create_independent_fipcl_model
from reproducibility import seed_everything, sha256, write_json


def confusion_metrics(labels, probabilities):
    """Independent arithmetic check of the four SOTA-table metrics."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.shape != (len(labels), 3) or not len(labels):
        raise ValueError('Expected nonempty labels[N] and probabilities[N,3]')
    if not np.isin(labels, [0, 1, 2]).all() or not np.isfinite(probabilities).all():
        raise ValueError('Invalid labels or nonfinite probabilities')
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid class probabilities')
    matrix = np.bincount(3 * labels.astype(int) + probabilities.argmax(1), minlength=9).reshape(3, 3)
    tp = matrix.diagonal().astype(float)
    true, predicted = matrix.sum(1), matrix.sum(0)
    recall = np.divide(tp, true, out=np.zeros(3), where=true > 0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * tp, true + predicted, out=np.zeros(3), where=true + predicted > 0)
    return {'accuracy': float(tp.sum() / matrix.sum()), 'f1': float(f1.mean()),
            'recall': float(recall.mean()), 'precision': float(precision.mean())}, matrix.tolist()


def evaluate(run_dir, output_dir, selection_manifest, workers=2):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    selection_manifest = Path(selection_manifest)
    selection = json.loads(selection_manifest.read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    result = json.loads((run_dir / 'result.json').read_text())
    if not selection.get('validation_selection_sealed'):
        raise ValueError('Complete and seal the validation comparison before test evaluation')
    expected = selection['run_result_sha256'].get(str(run_dir.resolve()))
    if expected != sha256(run_dir / 'result.json'):
        raise ValueError('Run result does not match the sealed validation manifest')
    if result.get('diagnostic_only'):
        raise ValueError('Smoke runs cannot produce formal test metrics')
    for name, expected in result['artifact_sha256'].items():
        if sha256(run_dir / name) != expected:
            raise ValueError(f'Sealed training artifact changed: {name}')
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    data_manifest_path = Path(config['split_root']) / 'data_manifest.json'
    if sha256(data_manifest_path) != manifest['data_manifest_sha256']:
        raise ValueError('Protected split manifest changed after training')
    data_manifest = json.loads(data_manifest_path.read_text())
    if output_dir.exists():
        raise FileExistsError(f'Refusing to replace evaluation: {output_dir}')
    output_dir.mkdir(parents=True)
    args = SimpleNamespace(**config)
    args.workers = workers
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessing = torch.load(run_dir / 'preprocessing.pt', map_location='cpu', weights_only=False)
    report = {'run_dir': str(run_dir), 'selection_manifest_sha256': sha256(selection_manifest),
              'selection_uses_test': False, 'per_modality': {}, 'checkpoints': {},
              'test_data_sha256': {}, 'confusion_matrices': {}, 'prediction_sha256': {}}
    for mk in MODALITY_KEYS:
        checkpoint = run_dir / f'best_{mk}.pt'
        if sha256(checkpoint) != result['checkpoint_sha256'][checkpoint.name]:
            raise ValueError(f'Sealed checkpoint changed: {checkpoint.name}')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = create_independent_fipcl_model(**saved['model_config'], device=str(device)).to(device)
        model.load_state_dict(saved['state_dict'], strict=True)
        loader, hashes = build_test_loader(args, mk, preprocessing, saved['model_config']['input_dims'][mk])
        metrics, labels, probs = predict(model, loader, device, args.amp)
        for source, digest in hashes.items():
            relative = Path(source).relative_to(args.split_root)
            original = str(Path(data_manifest['source']) / relative)
            if digest != data_manifest['source_hashes'][original]:
                raise ValueError(f'Test split changed since data protection: {source}')
        # Check the serialized arrays with confusion-matrix arithmetic, rather
        # than merely invoking the reporting implementation a second time.
        path = output_dir / f'test_{mk}.npz'
        np.savez_compressed(path, labels=labels, probabilities=probs)
        with np.load(path) as exported:
            verified, confusion = confusion_metrics(exported['labels'], exported['probabilities'])
            for key, value in verified.items():
                if not np.isclose(value, metrics[key], atol=1e-12, rtol=0):
                    raise AssertionError(f'Prediction metric mismatch: {mk}/{key}')
        report['per_modality'][mk] = metrics
        report['checkpoints'][mk] = {'sha256': sha256(checkpoint), 'selected_epoch': saved['epoch']}
        report['test_data_sha256'].update(hashes)
        report['confusion_matrices'][mk] = confusion
        report['prediction_sha256'][mk] = sha256(path)
        del loader, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(output_dir / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir, args.output_dir, args.selection_manifest, args.workers)), flush=True)


if __name__ == '__main__':
    main()
