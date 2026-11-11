#!/usr/bin/env python3
"""Train the release configuration, seal validation checkpoints, then test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'FiPCL'), str(ROOT)]
from configs.training import TRAINING_DEFAULTS
from reproducibility import sha256, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def current_sources():
    return {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.rglob('*.py')
            if '__pycache__' not in p.parts}


def run_process(command, log_path, status_path, status):
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=ROOT, env=os.environ.copy(), start_new_session=True)
        status.update(child_pid=process.pid, command=command, log=str(log_path))
        write_json(status_path, status)
        try:
            while process.poll() is None:
                status['updated_at'] = timestamp()
                write_json(status_path, status)
                time.sleep(2)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        if process.returncode:
            raise RuntimeError(f'Process failed ({process.returncode}); see {log_path}')


def validate_run(run):
    record = json.loads((run / 'result.json').read_text())
    if record['status'] != 'COMPLETE' or record['diagnostic_only'] or record['test_evaluated']:
        raise ValueError(f'Invalid validation-only experiment: {run}')
    for name, expected in {**record['artifact_sha256'], **record['checkpoint_sha256']}.items():
        if sha256(run / name) != expected:
            raise ValueError(f'Completed artifact changed: {run / name}')
    config = json.loads((run / 'config.json').read_text())
    for best in record['best_validation'].values():
        if best['epoch'] <= config['warmup']:
            raise ValueError('Warmup-only checkpoints cannot produce FiPCL test metrics')
    return record, config


def seal_validation(output, folds):
    hashes, records, configuration = {}, [], None
    for fold in folds:
        run = output / 'runs' / f'fold_{fold}'
        record, config = validate_run(run)
        recipe = {k: v for k, v in config.items() if k not in ('fold', 'output_dir')}
        if configuration is not None and recipe != configuration:
            raise ValueError('Folds must use the same training configuration')
        configuration = recipe
        hashes[str(run.resolve())] = sha256(run / 'result.json')
        records.append(record)
    payload = dict(validation_selection_sealed=True, sealed_at=timestamp(),
                   folds=list(folds), configuration=configuration,
                   test_access_for_selection=False, paper_evidence_eligible=False,
                   checkpoint_selection='per-task validation macro-F1 after warmup',
                   validation_macro_f1={task: statistics.mean(r['best_validation'][task]['f1'] for r in records)
                                        for task in ('clinic', 'blood', 'prot')},
                   run_result_sha256=hashes)
    if (output / 'selection.json').exists():
        previous = json.loads((output / 'selection.json').read_text())
        if {k: v for k, v in previous.items() if k != 'sealed_at'} != {k: v for k, v in payload.items() if k != 'sealed_at'}:
            raise ValueError('Existing validation seal disagrees with completed runs')
        return previous
    write_json(output / 'selection.json', payload)
    return payload


def aggregate_test(output, folds):
    selection_path = output / 'selection.json'
    selection = json.loads(selection_path.read_text())
    report = dict(folds=list(folds), std_ddof=0, selection_uses_test=False,
                  paper_evidence_eligible=False, configuration=selection['configuration'],
                  per_modality={})
    rows = []
    for fold in folds:
        directory = output / 'evaluation' / f'fold_{fold}'
        row = json.loads((directory / 'result.json').read_text())
        if row['selection_manifest_sha256'] != sha256(selection_path):
            raise ValueError('Evaluation does not match the sealed validation manifest')
        for task, expected in row['prediction_sha256'].items():
            if sha256(directory / f'test_{task}.npz') != expected:
                raise ValueError('Exported predictions changed')
        rows.append(row)
    for task in ('clinic', 'blood', 'prot'):
        report['per_modality'][task] = {}
        for metric in ('accuracy', 'f1', 'recall', 'precision', 'auc', 'aupr'):
            values = [r['per_modality'][task][metric] for r in rows]
            report['per_modality'][task][metric] = dict(mean=statistics.mean(values),
                                                       std=statistics.pstdev(values), values=values)
    write_json(output / 'summary.json', report)
    return report


def archive_incomplete(directory, output, category, fold, status_path):
    if not directory.exists():
        return
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        pid = previous.get('child_pid')
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists() and str(directory).encode() in cmdline.read_bytes():
            raise RuntimeError(f'A previous child process is still active: {pid}')
    archive = output / category / f'fold_{fold}_{time.time_ns()}'
    archive.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    for key in ('epochs', 'warmup', 'patience', 'workers', 'seed'):
        parser.add_argument('--' + key, type=int, default=TRAINING_DEFAULTS[key])
    parser.add_argument('--allow-unverified-auxiliary', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true', help='Two short diagnostic epochs; never evaluates tests')
    args = parser.parse_args(argv)
    if args.epochs <= args.warmup or args.warmup < 0 or args.patience < 1 or args.workers < 0:
        parser.error('epochs must exceed nonnegative warmup; patience positive, workers nonnegative')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Termination signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    folds = sorted(set(args.folds))
    sources = current_sources()
    options = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    options['folds'] = folds
    plan = dict(options=options, source_sha256=sources, training_defaults=TRAINING_DEFAULTS,
                python=sys.executable, created_at=timestamp())
    if (output / 'plan.json').exists():
        previous = json.loads((output / 'plan.json').read_text())
        if {k: v for k, v in previous.items() if k != 'created_at'} != {k: v for k, v in plan.items() if k != 'created_at'}:
            raise ValueError('Existing run has a different source/configuration fingerprint')
    else:
        write_json(output / 'plan.json', plan)
    status_path = output / 'status.json'
    status = dict(status='RUNNING', supervisor_pid=os.getpid(), started_at=timestamp(),
                  completed_training=0, total_training=len(folds), completed_evaluation=0)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)

    def check_sources():
        if current_sources() != sources:
            raise RuntimeError('Source changed during execution; refusing mixed-code results')

    try:
        for fold in folds:
            check_sources()
            run = output / 'runs' / f'fold_{fold}'
            status.update(phase='training', fold=fold)
            if (run / 'result.json').exists():
                if args.smoke_only:
                    record = json.loads((run / 'result.json').read_text())
                    if record['status'] != 'COMPLETE' or not record['diagnostic_only']:
                        raise ValueError('Invalid diagnostic run')
                else:
                    validate_run(run)
            else:
                archive_incomplete(run, output, 'interrupted_runs', fold, status_path)
                command = [sys.executable, '-u', str(ROOT / 'FiPCL/train.py'), '--fold', str(fold),
                           '--epochs', str(2 if args.smoke_only else args.epochs),
                           '--warmup', str(1 if args.smoke_only else args.warmup),
                           '--patience', str(args.patience), '--workers', str(args.workers),
                           '--seed', str(args.seed), '--split-root', str(args.split_root.resolve()),
                           '--output-dir', str(run)]
                if args.allow_unverified_auxiliary:
                    command.append('--allow-unverified-auxiliary')
                if args.smoke_only:
                    command += ['--max-train-batches', '2', '--max-val-batches', '2',
                                '--fisher-batches', '2', '--prototype-batches', '2']
                run_process(command, logs / f'train_fold{fold}.log', status_path, status)
            status['completed_training'] += 1
            write_json(status_path, status)
        if not args.smoke_only:
            check_sources()
            seal_validation(output, folds)
            if not args.validation_only:
                for fold in folds:
                    check_sources()
                    directory = output / 'evaluation' / f'fold_{fold}'
                    status.update(phase='test_evaluation', fold=fold)
                    if not (directory / 'result.json').exists():
                        archive_incomplete(directory, output, 'interrupted_evaluation', fold, status_path)
                        command = [sys.executable, '-u', str(ROOT / 'scripts/eval_from_checkpoints.py'),
                                   '--run-dir', str(output / 'runs' / f'fold_{fold}'),
                                   '--output-dir', str(directory),
                                   '--selection-manifest', str(output / 'selection.json'),
                                   '--workers', str(args.workers)]
                        run_process(command, logs / f'eval_fold{fold}.log', status_path, status)
                    status['completed_evaluation'] += 1
                    write_json(status_path, status)
                aggregate_test(output, folds)
        status.update(status='COMPLETE', phase='complete', completed_at=timestamp())
        write_json(status_path, status)
    except BaseException as error:
        status.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                      error=f'{type(error).__name__}: {error}', updated_at=timestamp())
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
