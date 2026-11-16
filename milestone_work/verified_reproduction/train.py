"""Width-configurable dense MASTER; copied upstream architecture, audited runner."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import resource
from checkpointing import CheckpointTracker

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from master import MASTER

SPLITS = {'train': ('2008-01-01', '2020-03-31'),
          'valid': ('2020-04-01', '2020-06-30'),
          'test': ('2020-07-01', '2022-12-31')}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_model(width=64, universe='csi300'):
    return MASTER(158, width, 4, 2, .5, .5, 158, 221,
                  5 if universe == 'csi300' else 2)


def daily_groups(dataset, split, limit=0, purge=True):
    index = dataset.get_index()
    if not isinstance(index, pd.MultiIndex) or not {'datetime', 'instrument'} <= set(index.names):
        raise ValueError('Expected MultiIndex with datetime and instrument')
    if not index.is_unique:
        raise ValueError('Duplicate date/instrument rows')
    dates = pd.DatetimeIndex(index.get_level_values('datetime'))
    start, end = map(pd.Timestamp, SPLITS[split])
    # Released samplers include historical context rows before split start.
    # Retain that history in the sampler, but never train/score those rows.
    in_split = (dates >= start) & (dates <= end)
    unique = dates[in_split].unique().sort_values()
    # Conservative: discard last 5 observed dates before next split, ensuring
    # five-trading-observation forward labels cannot enter the next segment.
    dropped = unique[-5:] if purge and split != 'test' else unique[:0]
    keep = unique[:-5] if len(dropped) else unique
    if limit:
        keep = keep[:limit]
    if not len(keep):
        raise ValueError(f'No dates left in {split}')
    groups = [(date, np.flatnonzero(dates == date)) for date in keep]
    return index, groups, {'actual_start': str(dates.min()), 'actual_end': str(dates.max()),
                           'rows': len(index), 'context_or_outside_rows_excluded': int((~in_split).sum()),
                           'scored_start': str(keep[0]), 'scored_end': str(keep[-1]),
                           'rows_used': sum(len(p) for _, p in groups), 'dates_used': len(groups),
                           'boundary_dates_removed': [str(x) for x in dropped]}


def get_batch(dataset, positions):
    a = np.asarray(dataset[positions], dtype=np.float32)
    if a.shape != (len(positions), 8, 222):
        raise ValueError(f'Expected [N,8,222], got {a.shape}; verify dataset version')
    x = torch.from_numpy(a[:, :, :-1].copy())
    y = torch.from_numpy(a[:, -1, -1].copy())
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite inputs: expected upstream normalized, zero-filled features')
    return x, y


def training_batch(x, y, version):
    finite = torch.isfinite(y)
    x, y = x[finite], y[finite]
    if version == 'opensource':
        order = torch.argsort(y)
        k = int(.025 * len(y))
        if k:
            order = order[k:-k]
        keep = torch.zeros(len(y), dtype=torch.bool)
        keep[order] = True
        x, y = x[keep], y[keep]
        if len(y) < 2 or y.std() == 0:
            raise ValueError('Insufficient nonconstant training labels')
        y = (y - y.mean()) / y.std()
    if len(y) < 2:
        raise ValueError('Insufficient training stocks')
    return x, y


def score_frame(frame):
    rows = []
    for date, g in frame.groupby('datetime', sort=True):
        valid = g[['prediction', 'label']].replace([np.inf, -np.inf], np.nan).dropna()
        good = len(valid) >= 2 and valid.prediction.nunique() > 1 and valid.label.nunique() > 1
        rows.append({'datetime': date, 'n_scored': len(valid),
                     'IC': valid.prediction.corr(valid.label) if good else np.nan,
                     'RankIC': valid.prediction.corr(valid.label, method='spearman') if good else np.nan})
    daily = pd.DataFrame(rows)
    metrics = {'dates': len(daily), 'undefined_dates': int(daily.IC.isna().sum())}
    for col in ['IC', 'RankIC']:
        values = daily[col].dropna()
        metrics[col] = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=0)) if len(values) else 0
        metrics[col + 'IR'] = float(values.mean() / std) if std > 0 else None
    return daily, metrics


def evaluate(model, dataset, index, groups, device):
    model.eval()
    frames = []
    with torch.no_grad():
        for _, positions in groups:
            x, y = get_batch(dataset, positions)
            pred = model(x.to(device)).cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite predictions')
            frame = index[positions].to_frame(index=False)
            frame['prediction'], frame['label'] = pred, y.numpy()
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    daily, metrics = score_frame(frame)
    return frame, daily, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data' / 'opensource')
    p.add_argument('--data-version', choices=['opensource', 'original'], default='opensource')
    p.add_argument('--universe', choices=['csi300', 'csi800'], default='csi300')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--threshold', type=float, default=.95)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--max-dates', type=int, default=0, help='Bound train/valid dates for smoke run; never uses test')
    p.add_argument('--evaluate-test', action='store_true', help='Use only after protocol is frozen')
    p.add_argument('--keep-boundary-labels', action='store_true', help='Match released split rows rather than purge forward labels')
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / time.strftime('%Y%m%d-%H%M%S'))
    a = p.parse_args()
    if Path.cwd().resolve() != ROOT:
        p.error('Run this sandbox trainer with verified_reproduction/ as the working directory')
    if not a.output.resolve().is_relative_to(ROOT) or not a.data_dir.resolve().is_relative_to(ROOT.parent / 'data'):
        p.error('Data must stay in milestone_work/data and outputs inside this package')
    if a.threads < 1:
        p.error('threads must be positive')
    torch.set_num_threads(a.threads)
    if a.width <= 0 or a.width % 4 or a.epochs < 1 or a.max_dates < 0:
        p.error('Width must be a positive multiple of 4, epochs positive, max-dates nonnegative')
    if a.evaluate_test:
        p.error('Use the separate frozen paired_test_eval.py for the approved test protocol')
    if a.max_dates and a.evaluate_test:
        p.error('A bounded smoke run must not evaluate the held-out test set')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    if a.device == 'mps' and not torch.backends.mps.is_available():
        p.error('MPS requested but unavailable')
    if a.output.exists():
        p.error('Output already exists; choose a new directory to preserve prior runs')
    a.output.mkdir(parents=True)
    write_json(a.output / 'config.json', vars(a))
    (a.output / 'environment.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())) + '\n')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    except subprocess.CalledProcessError:
        revision, dirty = 'unavailable', ''
    write_json(a.output / 'provenance.json', {'revision': revision, 'dirty_status': dirty,
               'command': sys.argv, 'python': sys.version, 'torch': torch.__version__,
               'upstream_revision': 'de8f58557096abde4216a701b35fc4368158d111',
               'target': 'close[t+5]/close[t+1]-1', 'features': 'upstream Alpha158 + 63 market features',
               'preprocessing': 'upstream fitted train robust zscore, clip [-3,3], fill 0; pickle provenance must be audited',
               'ICIR': 'mean daily IC / population standard deviation, no annualization'})
    datasets, manifests = {}, {}
    for split in ['train', 'valid'] + (['test'] if a.evaluate_test else []):
        path = a.data_dir / f'{a.universe}_dl_{split}.pkl'
        if not path.exists():
            raise FileNotFoundError(f'{path}: run download_data.py or see data/README.md')
        print(f'Loading {path}', flush=True)
        # Only use trusted authors' pickles. Loading pickle executes Python code.
        with path.open('rb') as f:
            dataset = pickle.load(f)
        index, groups, info = daily_groups(dataset, split, a.max_dates, not a.keep_boundary_labels)
        datasets[split] = dataset, index, groups
        manifests[split] = {**info, 'file': str(path.resolve()), 'sha256': digest(path), 'bytes': path.stat().st_size}
    write_json(a.output / 'data_manifest.json', manifests)
    results = []
    for seed in a.seeds:
        out = a.output / f'seed-{seed}'
        out.mkdir()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if a.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.cuda.reset_peak_memory_stats()
        model = make_model(a.width, a.universe).to(a.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
        tracker = CheckpointTracker(out)
        started = time.perf_counter()
        cpu_started = time.process_time()
        history = []
        reason = 'epoch_limit'
        for epoch in range(1, a.epochs + 1):
            model.train()
            losses = []
            ds, _, groups = datasets['train']
            for j in np.random.permutation(len(groups)):
                x, y = get_batch(ds, groups[j][1])
                x, y = training_batch(x, y, a.data_version)
                pred = model(x.to(a.device))
                loss = ((pred - y.to(a.device)) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite gradient')
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
                optimizer.step()
                losses.append(loss.item())
            frame, daily, metrics = evaluate(model, *datasets['valid'], a.device)
            row = {'epoch': epoch, 'train_mse': float(np.mean(losses)), **metrics,
                   'elapsed_seconds': time.perf_counter() - started,
                   'peak_host_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))}
            history.append(row)
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(f'seed={seed} {row}', flush=True)
            frame.to_csv(out / 'valid_predictions.csv', index=False)
            daily.to_csv(out / 'valid_daily_metrics.csv', index=False)
            if row['train_mse'] <= a.threshold:
                reason = 'reference_train_loss_threshold'
            tracker.update({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'config': vars(a), 'seed': seed,
                        'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                        'python_rng': random.getstate()}, metrics, row['train_mse'] <= a.threshold)
            if reason != 'epoch_limit':
                break
        write_json(out / 'valid_metrics.json', metrics)
        write_json(out / 'checkpoint_selection.json', tracker.summary())
        summary = {'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                   'epochs': epoch, 'stopping_reason': reason,
                   'seconds': time.perf_counter() - started,
                   'peak_cuda_bytes': torch.cuda.max_memory_allocated() if a.device == 'cuda' else None,
                   'validation': metrics, 'checkpoint_selection': tracker.summary(),
                   'peak_host_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                   'threads': a.threads, 'cpu_seconds': time.process_time() - cpu_started}
        if a.evaluate_test:
            frame, daily, metrics = evaluate(model, *datasets['test'], a.device)
            frame.to_csv(out / 'test_predictions.csv', index=False)
            daily.to_csv(out / 'test_daily_metrics.csv', index=False)
            write_json(out / 'test_metrics.json', metrics)
            summary['test'] = metrics
        if tracker.best_epoch is not None:
            best = torch.load(out / 'best_validation.pt', map_location=a.device, weights_only=False)
            model.load_state_dict(best['model'])
            best_frame, best_daily, best_metrics = evaluate(model, *datasets['valid'], a.device)
            best_frame.to_csv(out / 'best_valid_predictions.csv', index=False)
            best_daily.to_csv(out / 'best_valid_daily_metrics.csv', index=False)
            write_json(out / 'best_valid_metrics.json', best_metrics)
            summary['best_validation'] = best_metrics
        write_json(out / 'summary.json', summary)
        results.append(summary)
    write_json(a.output / 'summary.json', results)
    for split in ['validation', 'best_validation'] + (['test'] if a.evaluate_test else []):
        aggregate = {}
        for metric in ['IC', 'RankIC', 'ICIR', 'RankICIR']:
            vals = [r[split][metric] for r in results if r[split][metric] is not None]
            aggregate[metric] = {'mean': float(np.mean(vals)) if vals else None,
                                 'std_across_seeds': float(np.std(vals)) if vals else None,
                                 'n_seeds': len(vals)}
        write_json(a.output / f'{split}_aggregate.json', aggregate)


if __name__ == '__main__':
    main()
