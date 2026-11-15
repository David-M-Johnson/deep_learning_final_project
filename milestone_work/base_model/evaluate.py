"""Evaluate a saved checkpoint without retraining; writes dated predictions and IC metrics."""
import argparse
from pathlib import Path
import pickle
import torch
from train import ROOT, daily_groups, digest, evaluate, make_model, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data' / 'opensource')
    p.add_argument('--split', choices=['valid', 'test'], default='test')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--max-dates', type=int, default=0, help='Bound validation dates for a reload smoke test')
    p.add_argument('--output', type=Path, required=True, help='New evaluation directory')
    a = p.parse_args()
    if a.max_dates < 0 or (a.max_dates and a.split == 'test'):
        p.error('Bounded evaluation is supported for validation only')
    if a.output.exists():
        p.error('Output exists; choose a new directory')
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    model = make_model(config['width'], config['universe']).to(a.device)
    model.load_state_dict(state['model'])
    path = a.data_dir / f'{config["universe"]}_dl_{a.split}.pkl'
    with path.open('rb') as f:
        ds = pickle.load(f)
    index, groups, info = daily_groups(ds, a.split, limit=a.max_dates, purge=not config['keep_boundary_labels'])
    frame, daily, metrics = evaluate(model, ds, index, groups, a.device)
    a.output.mkdir(parents=True)
    frame.to_csv(a.output / f'{a.split}_predictions.csv', index=False)
    daily.to_csv(a.output / f'{a.split}_daily_metrics.csv', index=False)
    write_json(a.output / f'{a.split}_metrics.json', metrics)
    write_json(a.output / 'evaluation_manifest.json', {**info, 'checkpoint': str(a.checkpoint.resolve()),
               'checkpoint_sha256': digest(a.checkpoint), 'data_file': str(path.resolve()),
               'data_sha256': digest(path), 'device': a.device})
    print(metrics)


if __name__ == '__main__':
    main()
