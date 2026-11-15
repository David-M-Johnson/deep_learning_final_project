"""Make two readable epoch-history figures; no training or test evaluation."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='*', help='Run roots or seed directories')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures' / 'comparison')
    args = parser.parse_args()
    paths = args.runs or [ROOT / 'saved_runs' / f'csi300-width{w}' for w in (64, 128, 256)]
    histories, sources = [], []
    for path in paths:
        seed = path if (path / 'history.csv').exists() else path / 'seed-0'
        history_path = seed / 'history.csv'
        if not history_path.exists() and not args.runs:
            print(f'Skipping missing history: {history_path}')
            continue
        # Read once: both figures use the same snapshot of an ongoing run.
        history = pd.read_csv(history_path)
        columns = ['epoch', 'train_mse', 'IC', 'RankIC']
        if history.empty or not set(columns) <= set(history):
            parser.error(f'{history_path}: expected nonempty epoch/MSE/IC/RankIC columns')
        if not np.isfinite(history[columns].to_numpy()).all():
            parser.error(f'{history_path}: nonfinite history; retry if a write is in progress')
        if history.epoch.duplicated().any() or not history.epoch.is_monotonic_increasing:
            parser.error(f'{history_path}: epochs must be unique and increasing')
        config_path = seed.parent / 'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        label = f'Width {config["width"]}, {seed.name}' if 'width' in config else f'{seed.parent.name}/{seed.name}'
        histories.append((label, history))
        sources.append({'label': label, 'history': str(history_path.resolve()),
                        'epochs_shown': history.epoch.tolist(),
                        'metric_values': history[columns].to_dict(orient='records')})
    if not histories:
        parser.error('No training histories found')
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    colors = ['#2563a6', '#c56b19', '#42865a', '#8c62a8']

    def style(ax, ylabel):
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(axis='y', color='#e5e5e5', linewidth=.7)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for i, (label, h) in enumerate(histories):
        ax.plot(h.epoch, h.train_mse, color=colors[i % len(colors)],
                linewidth=2, label=label, marker='o' if len(h) == 1 else None)
    ax.set_title('Training error across epochs', loc='left', fontweight='bold')
    style(ax, 'Training MSE (lower is better)')
    ax.legend(frameon=False)
    fig.savefig(args.output / 'training_loss.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout='constrained')
    for ax, metric in zip(axes, ['IC', 'RankIC']):
        for i, (label, h) in enumerate(histories):
            color = colors[i % len(colors)]
            ax.plot(h.epoch, h[metric], color=color, linewidth=2, label=label)
            best = h.loc[h[metric].idxmax()]
            ax.scatter(best.epoch, best[metric], color=color, s=45, zorder=3)
        ax.axhline(0, color='#999999', linewidth=.8, linestyle='--')
        ax.set_title(metric, loc='left', fontweight='bold')
        style(ax, 'Validation correlation (higher is better)' if metric == 'IC' else '')
    axes[0].legend(frameon=False)
    fig.suptitle('Does validation improve with more training?\nDots mark each curve\'s best observed epoch; validation is not a test result.',
                 fontsize=12)
    fig.savefig(args.output / 'validation_quality.png', dpi=180)
    plt.close(fig)
    (args.output / 'epoch_plot_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    for source, (_, history) in zip(sources, histories):
        print(f'{source["label"]}: {len(history)} epochs; best IC at epoch '
              f'{int(history.loc[history.IC.idxmax(), "epoch"])}; best RankIC at epoch '
              f'{int(history.loc[history.RankIC.idxmax(), "epoch"])}')
    print(f'Figures saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
