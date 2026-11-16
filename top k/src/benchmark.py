"""Bounded training-throughput check; not a forecast-quality experiment.

Only load trusted upstream pickle files: pickle can execute code.
"""
import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .model import MASTER


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def load_days(path, limit):
    with path.open('rb') as handle:
        dataset = pickle.load(handle)
    index = dataset.get_index()
    dates = index.get_level_values('datetime')
    unique_dates = dates.unique().sort_values()[:limit]
    days = []
    for date in unique_dates:
        positions = np.flatnonzero(dates == date)
        batch = np.asarray(dataset[positions])
        if batch.ndim != 3 or batch.shape[1:] != (8, 222):
            raise ValueError(f'Expected [stocks, 8, 222], got {batch.shape}')
        days.append((str(date), torch.tensor(batch[:, :, :221], dtype=torch.float32),
                     torch.tensor(batch[:, -1, -1], dtype=torch.float32)))
    if not days:
        raise ValueError('No dates in dataset')
    return days, len(dates.unique())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Trusted CSI300 TRAIN TSDataSampler pickle (never test)')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--dates', type=int, default=10)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--stocks', type=int, default=300)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--label-policy', choices=['original', 'opensource'], default='opensource')
    parser.add_argument('--output', type=Path, default=Path('runs/benchmark.json'))
    args = parser.parse_args()
    if args.dates < 1 or args.width < 4 or args.width % 4:
        parser.error('dates must be positive; width must be a positive multiple of 4')
    if bool(args.data) == args.synthetic:
        parser.error('Choose exactly one of --data or --synthetic')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    load_start = time.perf_counter()
    if args.synthetic:
        days = [(str(i), torch.randn(args.stocks, 8, 221), torch.randn(args.stocks)) for i in range(args.dates)]
        total_dates, checksum = None, None
    else:
        days, total_dates = load_days(args.data, args.dates)
        digest = hashlib.sha256()
        with args.data.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        checksum = digest.hexdigest()
    load_seconds = time.perf_counter() - load_start
    device = torch.device(args.device)
    report = {'synthetic': args.synthetic, 'device': str(device), 'torch': torch.__version__,
              'platform': platform.platform(), 'seed': 0, 'width': args.width,
              'label_policy': args.label_policy, 'data': str(args.data), 'sha256': checksum,
              'prediction_dates': [date for date, _, _ in days],
              'config': {'lookback': 8, 'stock_features': 158, 'market_features': 63,
                         'temporal_heads': 4, 'stock_heads': 2, 'dropout': .5,
                         'beta': 5, 'lr': 1e-5, 'optimizer': 'Adam', 'cpu_threads': 4},
              'model_source_sha256': hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
              'load_seconds': load_seconds, 'total_dataset_dates': total_dates, 'results': []}
    for mode, k in [('dense', 300), ('topk', 10), ('topk', 50)]:
        torch.manual_seed(0)
        model = MASTER(158, args.width, 4, 2, .5, .5, 158, 221, 5, mode, k).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        def step(x, y):
            x, y = x.to(device), y.to(device)
            mask = torch.isfinite(y)
            if args.label_policy == 'opensource':
                finite = torch.where(mask)[0]
                order = finite[y[finite].argsort()]
                trim = int(.025 * len(order))
                kept = order[trim:len(order)-trim] if trim else order
                mask = torch.zeros_like(mask)
                mask[kept] = True
            x, y = x[mask], y[mask]
            if len(y) < 2 or not torch.isfinite(x).all():
                raise ValueError('Insufficient labels or non-finite processed features')
            if args.label_policy == 'opensource':
                y = (y - y.mean()) / y.std().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3)
            optimizer.step()
            return float(loss.detach().cpu())
        # Warm up separate weights; reset weights and optimizer before timed run.
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        step(days[0][1], days[0][2])
        synchronize(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        torch.manual_seed(0)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        start = time.perf_counter()
        losses = [step(x, y) for _, x, y in days]
        synchronize(device)
        seconds = time.perf_counter() - start
        row = {'mode': mode, 'k': None if mode == 'dense' else k, 'dates': len(days),
               'stocks_per_date': [len(y) for _, _, y in days], 'seconds': seconds,
               'seconds_per_date': seconds / len(days), 'mean_training_loss': float(np.mean(losses)),
               'estimated_training_epoch_seconds': seconds / len(days) * total_dates if total_dates else None,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
        report['results'].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
