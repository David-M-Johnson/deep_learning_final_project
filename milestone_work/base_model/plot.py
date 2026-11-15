"""Plot logged learning, daily correlations, and realized-return rank diagnostics."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parent / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='seed directory containing history.csv')
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--history', type=Path, help='Matching history CSV for a separate evaluation directory')
    p.add_argument('--output', type=Path, help='Figure directory; defaults under base_model/figures/')
    a = p.parse_args()
    output = a.output or Path(__file__).resolve().parent / 'figures' / a.run.parent.name / a.run.name
    output.mkdir(parents=True, exist_ok=True)
    history_path = a.history or a.run / 'history.csv'
    if not a.history and not history_path.exists():
        history_path = a.run.parent / 'history.csv'
    history = pd.read_csv(history_path)
    daily = pd.read_csv(a.run / f'{a.split}_daily_metrics.csv', parse_dates=['datetime'])
    pred = pd.read_csv(a.run / f'{a.split}_predictions.csv')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(history.epoch, history.train_mse, marker='o', label='Training standardized-return MSE')
    ax.set(xlabel='Epoch', ylabel='MSE', title='Training loss'); ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(history.epoch, history.IC, marker='o', label='Validation IC')
    ax.plot(history.epoch, history.RankIC, marker='o', label='Validation RankIC')
    ax.axhline(0, color='gray', lw=.7)
    ax.set(xlabel='Epoch', ylabel='Correlation', title='Validation quality'); ax.legend()
    ax = axes[1, 0]
    for col in ['IC', 'RankIC']:
        ax.plot(daily.datetime, daily[col].rolling(20, min_periods=1).mean(), label=f'{col}: trailing 20 dates')
    ax.axhline(0, color='gray', lw=.7)
    ax.set(ylabel='Correlation', title=f'{a.split}: daily correlations smoothed'); ax.legend(fontsize=8)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    valid = pred.replace([np.inf, -np.inf], np.nan).dropna(subset=['prediction', 'label']).copy()
    # Rank each date separately, so large cross-sections do not define the bins.
    valid['bucket'] = valid.groupby('datetime').prediction.transform(
        lambda s: np.clip(np.ceil(s.rank(method='average', pct=True) * 10).astype(int) - 1, 0, 9))
    means = valid.groupby(['datetime', 'bucket']).label.mean().groupby('bucket').mean()
    ax = axes[1, 1]
    ax.bar(means.index + 1, means.values)
    ax.set(xlabel='Predicted-score bucket (1 low, 10 high)', ylabel='Mean released target',
           title=f'{a.split}: realized target by predicted rank')
    ax.axhline(0, color='gray', lw=.7)
    fig.suptitle('MASTER diagnostics — correlations are not classification accuracy; buckets are not a backtest')
    fig.savefig(output / f'{a.split}_diagnostics.png', dpi=160)
    plt.close(fig)
    print(output / f'{a.split}_diagnostics.png')


if __name__ == '__main__':
    main()
