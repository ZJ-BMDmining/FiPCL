#!/usr/bin/env python3
"""Train FiPCL with independent encoders and progressive Clinic regularization."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'FiPCL'), str(ROOT)]
from configs.paths import MRI_CACHE_DIR, SPLIT_ROOT
from configs.training import TRAINING_DEFAULTS
from experiment_data import (batch_to_device, build_training_data,
                             checkpoint_config)
from models.shared_fusion_model import MRI_SHARING_GROUPS, MODALITY_KEYS, create_independent_fipcl_model
from reproducibility import (audit_batchnorm, make_optimizer, require_finite,
                             seed_everything, sha256, write_json)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def autocast(device, enabled):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if enabled and device.type == 'cuda' else nullcontext()


def compute_metrics(labels, probabilities):
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite probabilities; metrics are not valid')
    prediction = probabilities.argmax(1)
    metrics: dict[str, float | None] = {'accuracy': float(accuracy_score(labels, prediction)),
               'f1': float(f1_score(labels, prediction, average='macro', labels=[0, 1, 2], zero_division=0)),
               'recall': float(recall_score(labels, prediction, average='macro', labels=[0, 1, 2], zero_division=0)),
               'precision': float(precision_score(labels, prediction, average='macro', labels=[0, 1, 2], zero_division=0))}
    auc, aupr = [], []
    for c in range(3):
        binary = (labels == c)
        if binary.any() and not binary.all():
            auc.append(roc_auc_score(binary, probabilities[:, c]))
            aupr.append(average_precision_score(binary, probabilities[:, c]))
    metrics['auc'] = float(np.mean(auc)) if auc else None
    metrics['aupr'] = float(np.mean(aupr)) if aupr else None
    return metrics


@torch.no_grad()
def predict(model, loader, device, amp=False, max_batches=None):
    model.eval()
    labels, probabilities = [], []
    for i, cpu_batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        with autocast(device, amp):
            logits = model(batch, return_all=True)['fusion_logits']
        require_finite(logits, 'validation/test logits')
        labels.append(batch['label'].cpu().numpy())
        probabilities.append(logits.float().softmax(1).cpu().numpy())
    yt, yp = np.concatenate(labels), np.concatenate(probabilities)
    return compute_metrics(yt, yp), yt, yp


def refresh_fisher(model, loaders, device, num_batches, amp=False):
    # Reliability estimation must not update BatchNorm running statistics.
    model.eval()
    for mk in MODALITY_KEYS:
        tracker = model.fisher_trackers[mk]
        tracker.reset()
        params = [p for module in (getattr(model, f'mri_projector_{mk}'),
                                  getattr(model, f'{mk}_projector')) for p in module.parameters()]
        split = len(list(getattr(model, f'mri_projector_{mk}').parameters()))
        for i, cpu_batch in enumerate(loaders[mk]['train']):
            if i >= num_batches:
                break
            batch = batch_to_device(cpu_batch, device)
            with autocast(device, amp):
                logits = model(batch, return_all=True)['fusion_logits']
                loss = F.cross_entropy(logits.float(), batch['label'])
            require_finite(loss, f'{mk} Fisher loss')
            gradients = torch.autograd.grad(loss, params)
            values = []
            for gradient in gradients:
                require_finite(gradient, f'{mk} Fisher gradient')
                values.append(float(gradient.float().square().sum()))
            tracker.accumulate([sum(values[:split]), sum(values[split:])])
        tracker.get_trace()  # exactly one EMA update per estimate
        model.update_fisher_weights(mk)


@torch.no_grad()
def refresh_prototypes(model, loaders, device, num_batches, amp=False):
    model.eval()
    for mk in MODALITY_KEYS:
        bank = model.mri_prototype_bank(mk)
        bio_bank = getattr(model, f'{mk}_prototypes') if mk != 'clinic' else None
        for i, cpu_batch in enumerate(loaders[mk]['train']):
            # On first use, extend the budget until each training class has
            # actual anchors. Never pair an auxiliary row with random weights.
            if i >= num_batches and bank.initialized_flags.all():
                break
            batch = batch_to_device(cpu_batch, device)
            with autocast(device, amp):
                mri, bio, _ = model.get_embeddings(batch)
            require_finite(mri, f'{mk} prototype MRI embeddings')
            require_finite(bio, f'{mk} prototype tabular embeddings')
            bank.update(F.normalize(mri.float(), dim=1), batch['label'])
            if bio_bank is not None:
                bio_bank.update(F.normalize(bio.float(), dim=1), batch['label'])
        if not bank.initialized_flags.all():
            raise ValueError(f'{mk} training data cannot initialize all three MRI prototype classes')


def pseudo_loss(model, batch, mk, epoch, warmup):
    labels = batch['label']
    bio = model._encode_bio(batch[mk], mk, batch.get('mask'))
    bank = model.mri_prototype_bank(mk)
    if not bank.initialized_flags[labels].all():
        raise ValueError('Pseudo-pair label has no train-initialized MRI prototype')
    indices = torch.randint(bank.num_protos, (len(labels),), device=labels.device)
    mri = bank.prototypes[labels, indices].detach().to(bio.dtype)
    out = model.forward_pseudo_pair(mri, bio, labels, mk)
    cosine = F.cosine_similarity(bio.float(), mri.float(), dim=1)
    confidence = torch.sigmoid(cosine / .5)
    base = .1 + .3 * min(1., (epoch - warmup) / 20.)
    # Apply the manuscript's Fisher-conditioned threshold, with the same policy
    # and scalar coefficients for every architecture.
    traces = model.fisher_trackers[mk].peek_trace()
    balance = 4 * traces[0] * traces[1] / ((sum(traces)) ** 2 + 1e-8)
    threshold = (1 - .5 * balance) * base
    confidence = torch.where(confidence >= threshold, confidence, .05)
    loss = sum(weight * (F.cross_entropy(out[key].float(), labels, reduction='none',
                                        label_smoothing=.1 if key == 'fusion_logits' else 0.) * confidence).mean()
               for key, weight in [('fusion_logits', 1.), ('mri_logits', .5), ('bio_logits', .5)])
    distance = (F.normalize(bio.float(), dim=1) - F.normalize(mri.float(), dim=1)).square().sum(1)
    loss = loss + .1 * (confidence * distance).mean()
    loss = loss + .1 * model.compute_prototype_loss(mri.float(), bio.float(), labels, mk)
    return loss


def stage2_weight(args, epoch, modality):
    if epoch <= args.warmup:
        return 0.
    ramp = getattr(args, 'stage2_ramp_epochs', 0)
    factor = min(1., (epoch - args.warmup) / ramp) if ramp else 1.
    return factor * (getattr(args, 'clinic_stage2_scale', 1.) if modality == 'clinic' else 1.)


def train_epoch(model, paired, single, optimizer, args, device, epoch):
    model.train()
    stage2 = epoch > args.warmup
    aux = {mk: iter(loader) for mk, loader in single.items() if loader is not None}
    sums = {mk: {'loss': 0., 'batches': 0, 'samples': 0, 'pseudo_batches': 0,
                 'stage2_weight': stage2_weight(args, epoch, mk)} for mk in MODALITY_KEYS}
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0

    def step(count):
        if count < args.grad_accum:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(args.grad_accum / count)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        require_finite(norm, 'gradient norm')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Interleave tasks while covering each real training row once per epoch.
    iterators = [iter(paired[mk]['train']) for mk in MODALITY_KEYS]
    for batches in itertools.zip_longest(*iterators):
        for mk, cpu_batch in zip(MODALITY_KEYS, batches):
            if cpu_batch is None or (args.max_train_batches and sums[mk]['batches'] >= args.max_train_batches):
                continue
            batch = batch_to_device(cpu_batch, device)
            for key, value in batch.items():
                require_finite(value, f'{mk} training input {key}')
            with autocast(device, args.amp):
                out = model(batch, return_all=True)
                for key in ('fusion_logits', 'mri_logits', 'bio_logits'):
                    require_finite(out[key], f'{mk} {key}')
                loss = (F.cross_entropy(out['fusion_logits'].float(), batch['label'], label_smoothing=.1)
                        + .5 * F.cross_entropy(out['mri_logits'].float(), batch['label'])
                        + .5 * F.cross_entropy(out['bio_logits'].float(), batch['label']))
                fusion = getattr(model, f'fusion_{mk}')
                loss = loss + .05 * fusion.compute_modality_balance_loss(
                    out['mri_logits'].float(), out['bio_logits'].float(), out['fusion_logits'].float(), batch['label'])
                if stage2:
                    factor = stage2_weight(args, epoch, mk)
                    loss = loss + (.1 * factor) * model.compute_fisher_guided_fusion_loss(
                        out['mri_logits'].float(), out['bio_logits'].float(), batch['label'], mk)
                    loss = loss + (.005 * factor) * model.compute_fisher_regularization(
                        out['mri_emb'].float(), out['bio_emb'].float(), mk)
                    loss = loss + (.1 * factor) * model.compute_prototype_loss(
                        out['mri_emb'].float(), out['bio_emb'].float(), batch['label'], mk)
                    if mk in aux:
                        try:
                            extra = next(aux[mk])
                        except StopIteration:
                            aux[mk] = iter(single[mk])
                            extra = next(aux[mk])
                        extra = batch_to_device(extra, device)
                        weight = .2 + .3 * min(1., (epoch - args.warmup) / 40.)
                        loss = loss + (weight * factor) * pseudo_loss(model, extra, mk, epoch, args.warmup)
                        sums[mk]['pseudo_batches'] += 1
            require_finite(loss, f'{mk} loss')
            (loss / args.grad_accum).backward()
            accumulation += 1
            if accumulation == args.grad_accum:
                step(accumulation)
                accumulation = 0
            sums[mk]['loss'] += float(loss.detach())
            sums[mk]['batches'] += 1
            sums[mk]['samples'] += len(batch['label'])
        if args.max_train_batches and all(sums[mk]['batches'] >= args.max_train_batches for mk in MODALITY_KEYS):
            break
    if accumulation:
        step(accumulation)
    for values in sums.values():
        values['loss'] /= max(values['batches'], 1)
    return sums


def save_state(path, model, model_config, epoch, metrics):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for name, tensor in state.items():
        require_finite(tensor, f'checkpoint tensor {name}')
    payload = {'state_dict': state, 'model_config': model_config, 'epoch': epoch,
               'validation': metrics, 'fisher': {mk: model.fisher_trackers[mk].peek_trace()
                                                for mk in MODALITY_KEYS}}
    tmp = Path(str(path) + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='Release defaults: independent MRI/tabular encoders; Clinic MLP; '
                                            'named feature union with availability indicators; '
                                            '5 warmup epochs, 10 ramp epochs, Clinic stage-2 scale 0.1.')
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    parser.add_argument('--clinic-stage2-scale', type=float, default=1.)
    parser.add_argument('--stage2-ramp-epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--aux-batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mri-lr-scale', type=float, default=.1)
    parser.add_argument('--dropout', type=float, default=.4)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--fisher-batches', type=int, default=15)
    parser.add_argument('--prototype-batches', type=int, default=15)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-pseudo', action='store_true')
    parser.add_argument('--allow-unverified-auxiliary', action='store_true',
                        help='Label the run exploratory when auxiliary subject IDs are unavailable')
    parser.add_argument('--split-root', type=Path, default=SPLIT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-train-batches', type=int, default=0, help='Diagnostic smoke only')
    parser.add_argument('--max-val-batches', type=int, default=0, help='Diagnostic smoke only')
    parser.set_defaults(**TRAINING_DEFAULTS)
    args = parser.parse_args(argv)
    if args.epochs <= args.warmup or min(args.batch_size, args.aux_batch_size, args.grad_accum) < 1:
        parser.error('epochs must exceed warmup, and batch sizes/gradient accumulation must be positive')
    if args.patience < 1 or args.warmup < 0 or args.fisher_batches < 1 or args.prototype_batches < 1:
        parser.error('Invalid training schedule or reliability refresh budget')
    if args.workers < 0 or args.lr <= 0 or args.mri_lr_scale <= 0:
        parser.error('workers must be nonnegative and learning rates positive')
    if not math.isfinite(args.clinic_stage2_scale) or args.clinic_stage2_scale <= 0 or args.stage2_ramp_epochs < 0:
        parser.error('Clinic stage-2 scale must be finite/positive, and ramp epochs nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = args.output_dir or ROOT / 'results' / f'fold_{args.fold}_{int(time.time())}'
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {output}')
    output.mkdir(parents=True)
    manifest_path = Path(args.split_root) / 'data_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('First run scripts/prepare_data.py; a protected data manifest is required')
    data_manifest = json.loads(manifest_path.read_text())
    if not args.no_pseudo and not args.allow_unverified_auxiliary and data_manifest['single_modality_subject_audit'] != 'PASS':
        raise ValueError('Auxiliary subject IDs are unresolved; explicitly label exploratory runs '
                         'with --allow-unverified-auxiliary or run a --no-pseudo ablation')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(output / 'config.json', config)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    import sklearn
    write_json(output / 'manifest.json', {
        'started_at': timestamp(), 'source_sha256': source_hashes, 'python': sys.version,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
        'mri_cache': str(MRI_CACHE_DIR), 'data_manifest_sha256': sha256(manifest_path),
        'paper_evidence_eligible': False, 'test_metrics_used_for_selection': False,
        'single_modality_subject_audit': data_manifest['single_modality_subject_audit'],
        'protocol': 'real-MRI FiPCL; fixed test; independent MRI and tabular encoders; '
                    'task-specific adapters/projectors/fusion; explicit saved tabular architecture; '
                    'paper-direction tabular-to-MRI pseudo-pairs; one optimizer per unique parameter',
    })
    print(f'START {output} MRI={args.mri_sharing} TABULAR={args.tabular_sharing}', flush=True)
    paired, single, dims, preprocessing, hashes = build_training_data(args)
    torch.save(preprocessing, output / 'preprocessing.pt')
    write_json(output / 'training_data_hashes.json', hashes)
    mconfig = checkpoint_config(args, dims)
    model = create_independent_fipcl_model(**mconfig, device=str(device)).to(device)
    optimizer = make_optimizer(model, args.lr, args.mri_lr_scale)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    architecture = {
        'mri_sharing': args.mri_sharing, 'tabular_sharing': args.tabular_sharing,
        'clinic_encoder_type': model.clinic_encoder_type,
        'mri_groups': MRI_SHARING_GROUPS[args.mri_sharing],
        'unique_parameters': sum(p.numel() for p in model.parameters()),
        'optimizer_unique_parameters': sum(p.numel() for g in optimizer.param_groups for p in g['params']),
        'sizes': {mk: {s: len(loader.dataset) for s, loader in loaders.items()} for mk, loaders in paired.items()},
        'auxiliary_sizes': {mk: len(loader.dataset) if loader is not None else 0 for mk, loader in single.items()},
    }
    write_json(output / 'architecture.json', architecture)
    print(json.dumps(architecture), flush=True)
    best = {mk: {'f1': -1., 'epoch': None} for mk in MODALITY_KEYS}
    best_joint, bad = -1., 0
    started = time.monotonic()
    last_epoch = 0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            if epoch > args.warmup:
                refresh_fisher(model, paired, device, args.fisher_batches, args.amp)
                refresh_prototypes(model, paired, device, args.prototype_batches, args.amp)
            training = train_epoch(model, paired, single, optimizer, args, device, epoch)
            scheduler.step()
            validation, improved = {}, False
            for mk in MODALITY_KEYS:
                metrics, labels, probs = predict(model, paired[mk]['val'], device, args.amp,
                                                 args.max_val_batches or None)
                validation[mk] = metrics
                if epoch > args.warmup and metrics['f1'] > best[mk]['f1']:
                    best[mk] = {'f1': metrics['f1'], 'epoch': epoch, 'metrics': metrics}
                    save_state(output / f'best_{mk}.pt', model, mconfig, epoch, metrics)
                    np.savez_compressed(output / f'validation_{mk}.npz', labels=labels, probabilities=probs)
                    improved = improved or args.early_stopping_task == 'any' or mk == args.early_stopping_task
            joint = float(np.mean([validation[mk]['f1'] for mk in MODALITY_KEYS]))
            if epoch > args.warmup:
                if joint > best_joint:
                    best_joint = joint
                    save_state(output / 'best_joint.pt', model, mconfig, epoch, validation)
                bad = 0 if improved else bad + 1
            record = {'epoch': epoch, 'stage': 'FiPCL' if epoch > args.warmup else 'warmup',
                      'training': training, 'validation': validation, 'joint_validation_f1': joint,
                      'best': best, 'seconds': time.monotonic() - epoch_start,
                      'batchnorm': audit_batchnorm(model),
                      'fisher': {mk: list(map(float, model.fisher_trackers[mk].peek_trace())) for mk in MODALITY_KEYS}}
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            last_epoch = epoch
            write_json(output / 'status.json', {'status': 'RUNNING', 'updated_at': timestamp(),
                                               'epoch': epoch, 'best': best, 'seconds': time.monotonic() - started})
            print(f'EPOCH {epoch:3d} {record["stage"]} C={validation["clinic"]["f1"]:.4f} '
                  f'B={validation["blood"]["f1"]:.4f} P={validation["prot"]["f1"]:.4f} '
                  f'time={record["seconds"]:.1f}s', flush=True)
            if epoch > args.warmup and bad >= args.patience:
                break
        result = {'completed_at': timestamp(), 'status': 'COMPLETE', 'fold': args.fold,
                  'mri_sharing': args.mri_sharing, 'tabular_sharing': args.tabular_sharing,
                  'best_validation': best, 'selection_score': float(np.mean([b['f1'] for b in best.values()])),
                  'epochs_completed': last_epoch, 'seconds': time.monotonic() - started,
                  'diagnostic_only': bool(args.max_train_batches or args.max_val_batches),
                  'test_evaluated': False, 'paper_evidence_eligible': False}
        result['checkpoint_sha256'] = {p.name: sha256(p) for p in output.glob('best*.pt')}
        result['artifact_sha256'] = {name: sha256(output / name) for name in
                                     ('config.json', 'preprocessing.pt', 'training_data_hashes.json', 'manifest.json')}
        write_json(output / 'result.json', result)
        write_json(output / 'status.json', {'status': 'COMPLETE', 'updated_at': timestamp(), 'epoch': last_epoch})
        print('COMPLETE ' + json.dumps(result), flush=True)
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'FAILED', 'updated_at': timestamp(),
                                           'epoch': last_epoch, 'error': f'{type(error).__name__}: {error}'})
        raise
    return result


if __name__ == '__main__':
    main()
