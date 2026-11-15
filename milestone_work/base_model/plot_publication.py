"""Export unsmoothed single-run learning curves as vector PDF and 300-dpi PNG."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width', type=int, choices=[64, 128, 256], default=128)
    p.add_argument('--run', type=Path, help='Override the default copied run snapshot')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, help='Default: figures/width<actual run width>')
    a = p.parse_args()
    run = (a.run or ROOT / 'saved_runs' / f'csi300-width{a.width}').resolve()
    source = run / f'seed-{a.seed}' / 'history.csv'
    try:
        # A single byte snapshot keeps every exported artifact consistent.
        snapshot = source.read_bytes()
        h = pd.read_csv(io.BytesIO(snapshot))
        config = json.loads((run / 'config.json').read_text())
    except (OSError, ValueError) as exc:
        p.error(f'Cannot read run: {exc}')
    required = ['epoch', 'train_mse', 'IC', 'RankIC']
    if h.empty or not set(required) <= set(h):
        p.error('History must contain epoch, train_mse, IC and RankIC')
    if not np.isfinite(h[required].to_numpy()).all():
        p.error('History contains incomplete/nonfinite values; retry after the current write')
    if (h.epoch <= 0).any() or (h.epoch % 1 != 0).any() or (h.epoch.diff().dropna() <= 0).any():
        p.error('Epochs must be positive, unique and increasing integers')
    width = config['width']
    a.output = a.output or ROOT / 'figures' / f'width{width}'
    label = f'{config["universe"].upper()} | width {width} | seed {a.seed} | epochs 1–{int(h.epoch.iloc[-1])}'
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white'})

    def style(ax, ylabel):
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(axis='y', color='#e8e8e8', linewidth=.6)
        ax.set_axisbelow(True)

    def save(fig, name):
        for suffix in ('pdf', 'png'):
            fig.savefig(a.output / f'{name}.{suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), layout='constrained')
    ax.plot(h.epoch, h.train_mse, color='#0072B2', linewidth=1.8,
            marker='o' if len(h) == 1 else None)
    ax.set_title('Training loss', loc='left', fontweight='bold')
    style(ax, 'Training MSE (lower is better)')
    fig.suptitle(label, fontsize=10)
    save(fig, 'training_loss')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, layout='constrained')
    best = {}
    for ax, metric, color in zip(axes, ['IC', 'RankIC'], ['#0072B2', '#D55E00']):
        ax.plot(h.epoch, h[metric], color=color, linewidth=1.8)
        peak = h.loc[h[metric].idxmax()]
        best[metric] = {'epoch': int(peak.epoch), 'value': float(peak[metric])}
        ax.scatter(peak.epoch, peak[metric], s=28, color=color, zorder=3)
        ax.axhline(0, color='#999999', linewidth=.7, linestyle='--')
        ax.set_title(f'{metric} | peak {peak[metric]:.3f} at epoch {int(peak.epoch)}',
                     loc='left', fontsize=10)
        style(ax, 'Validation correlation' if metric == 'IC' else '')
    fig.suptitle(label, fontsize=10)
    save(fig, 'validation_quality')

    (a.output / 'history_snapshot.csv').write_bytes(snapshot)
    manifest = {'source': str(source), 'history_sha256': hashlib.sha256(snapshot).hexdigest(),
                'config': config, 'epochs_plotted': len(h), 'best_observed_validation': best,
                'note': 'Unsmoothed single-seed validation history; no uncertainty bands or test claims.'}
    (a.output / 'figure_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    caption = (
        f'Dense MASTER on {config["universe"].upper()}, hidden width {width}, seed {a.seed}. '
        f'Training loss and validation correlations over {len(h)} recorded epochs. '
        'IC and RankIC are daily cross-sectional Pearson and Spearman correlations, '
        'respectively, averaged over validation dates. Curves are unsmoothed; dots mark '
        'the highest observed validation values, not necessarily retained checkpoints. '
        'These are single-seed validation results, not held-out test results. '
        'The history may be a partial snapshot of an ongoing run.\n'
    )
    (a.output / 'caption.txt').write_text(caption)
    print(f'Exported PDF + PNG figures for {len(h)} epochs to {a.output.resolve()}')


if __name__ == '__main__':
    main()
