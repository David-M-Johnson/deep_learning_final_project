"""Train MASTER on trusted upstream CSI300 open-source sampler pickles."""
import argparse
import hashlib
import json
import pickle
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from .model import MASTER


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class DailyData:
    def __init__(self, path, start, end, purge=0, limit=None):
        with path.open('rb') as f:
            self.sampler = pickle.load(f)
        self.index = self.sampler.get_index()
        dates = self.index.get_level_values('datetime')
        if not dates.is_monotonic_increasing or not self.index.is_unique:
            raise ValueError('Expected chronological unique stock/date index')
        unique, starts, counts = np.unique(dates, return_index=True, return_counts=True)
        selected = [(pd.Timestamp(d), int(s), int(n)) for d, s, n in zip(unique, starts, counts)
                    if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)]
        if purge:
            selected = selected[:-purge]
        if limit:
            selected = selected[:limit]
        if not selected:
            raise ValueError('Empty date selection')
        self.groups = selected
        # Audit current-day stock features throughout the underlying array. One
        # sentinel all-NaN row is expected; historical windows checked on access.
        arr = self.sampler.data_arr
        rows = arr[:-1] if np.isnan(arr[-1]).all() else arr
        self.manifest = {'path': str(path), 'sha256': checksum(path),
                         'raw_start': str(dates.min()), 'raw_end': str(dates.max()),
                         'selected_start': str(selected[0][0]), 'selected_end': str(selected[-1][0]),
                         'dates': len(selected), 'samples': sum(n for _, _, n in selected),
                         'stocks_min': min(n for _, _, n in selected),
                         'stocks_max': max(n for _, _, n in selected),
                         'step_len': self.sampler.step_len, 'fillna_type': self.sampler.fillna_type,
                         'raw_nonfinite_stock_features': int((~np.isfinite(rows[:, :158])).sum()),
                         'raw_missing_labels': int((~np.isfinite(rows[:, -1])).sum()),
                         'purged_final_dates': purge}
        if self.sampler.step_len != 8 or arr.shape[1] != 222:
            raise ValueError('Expected 8 observations and 222 columns')

    def batch(self, group):
        _, start, n = group
        values = np.asarray(self.sampler[np.arange(start, start+n)])
        x = torch.tensor(values[:, :, :221], dtype=torch.float32)
        y = torch.tensor(values[:, -1, -1], dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError(f'Nonfinite input window on {group[0]}')
        market = x[:, -1, 158:221]
        if not torch.allclose(market, market[0:1].expand_as(market), equal_nan=False):
            raise ValueError('Current market vector is not shared across stocks')
        return x, y


def training_labels(x, y):
    finite = torch.where(torch.isfinite(y))[0]
    order = finite[y[finite].argsort()]
    trim = int(.025 * len(order))
    keep = order[trim:len(order)-trim] if trim else order
    # Preserve original stock ordering, as in upstream boolean filtering.
    keep = keep.sort().values
    x, y = x[keep], y[keep]
    if len(y) < 2 or y.std() <= 0:
        raise ValueError('Insufficient nonconstant training labels')
    return x, (y-y.mean())/y.std()


def evaluate(model, data, device, prediction_path=None):
    model.eval()
    rows, daily = [], []
    with torch.no_grad():
        for group in data.groups:
            date, start, n = group
            x, y = data.batch(group)
            predictions = model(x.to(device)).cpu().numpy()
            labels = y.numpy()
            if not np.isfinite(predictions).all():
                raise ValueError('Nonfinite predictions')
            mask = np.isfinite(labels)
            ic = ric = float('nan')
            if mask.sum() > 1 and np.std(labels[mask]) > 0 and np.std(predictions[mask]) > 0:
                ic = float(np.corrcoef(predictions[mask], labels[mask])[0, 1])
                ric = float(spearmanr(predictions[mask], labels[mask]).statistic)
            daily.append({'date': str(date.date()), 'IC': ic, 'RankIC': ric, 'stocks': n,
                          'finite_labels': int(mask.sum())})
            if prediction_path:
                instruments = data.index[start:start+n].get_level_values('instrument')
                rows.extend({'date': str(date.date()), 'instrument': str(stock),
                             'prediction': float(pred), 'label': float(label)}
                            for stock, pred, label in zip(instruments, predictions, labels))
    result = {'dates': len(daily)}
    for name in ['IC', 'RankIC']:
        values = np.array([d[name] for d in daily])
        values = values[np.isfinite(values)]
        result[name] = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=0)) if len(values) else 0
        result[name+'IR'] = float(values.mean()/std) if std > 0 else None
        result[name+'_valid_dates'] = len(values)
    if prediction_path:
        pd.DataFrame(rows).to_csv(prediction_path, index=False)
        pd.DataFrame(daily).to_csv(prediction_path.with_name('daily_metrics.csv'), index=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data/opensource')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['dense','topk'], default='topk')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--stop-loss', type=float, default=.95)
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--limit-dates', type=int, help='Smoke checks only; limits train/validation dates')
    p.add_argument('--evaluate-test', action='store_true', help='Score test once after fixed training protocol')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be new/empty to avoid overwriting a run')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    train = DailyData(a.data_dir/'csi300_dl_train.pkl', '2008-01-01','2020-03-31',purge=5,limit=a.limit_dates)
    valid = DailyData(a.data_dir/'csi300_dl_valid.pkl', '2020-04-01','2020-06-30',purge=5,limit=a.limit_dates)
    # No model selection from validation: it is diagnostic because of upstream caveat.
    config = {key: str(value) if isinstance(value, Path) else value for key,value in vars(a).items()}
    config.update({'stock_features':158,'market_features':63,'lookback':8,'lr':1e-5,
                   'beta':5,'dropout':.5,'temporal_heads':4,'stock_heads':2,
                   'checkpoint_rule':'first epoch mean training loss <= threshold, otherwise last',
                   'validation_use':'diagnostic only; published processing caveat unresolved',
                   'label_policy':'opensource: drop missing/tails, cross-sectional sample-std zscore',
                   'self_attention':'eligible, counts toward k if selected; not forced',
                   'torch':torch.__version__, 'numpy':np.__version__, 'pandas':pd.__version__,
                   'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   'git_status':subprocess.check_output(['git','status','--short'],text=True),
                   'model_sha256':checksum(Path(__file__).with_name('model.py'))})
    manifest={'config':config,'train':train.manifest,'validation':valid.manifest}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
    model = MASTER(158,a.width,4,2,.5,.5,158,221,5,a.mode,a.k).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    run_start=time.perf_counter(); history=[]; reason='epoch_budget'
    for epoch in range(a.epochs):
        model.train(); start=time.perf_counter(); losses=[]
        for j in np.random.default_rng(a.seed+epoch).permutation(len(train.groups)):
            x,y=training_labels(*train.batch(train.groups[j]))
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_value_(model.parameters(),3)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        metrics=evaluate(model,valid,device)
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),
             'validation':metrics,'epoch_seconds':time.perf_counter()-start,
             'elapsed_seconds':time.perf_counter()-run_start}
        history.append(row)
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
        stop=row['train_loss']<=a.stop_loss
        reason='training_loss_threshold' if stop else 'epoch_budget'
        checkpoint={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'config':config,'epoch':epoch+1,'torch_rng_state':torch.get_rng_state(),
                    'stopping_reason':reason,'manifest':manifest}
        torch.save(checkpoint,a.output/'last.pt')
        print(json.dumps(row),flush=True)
        if stop: break
    # Always save selected model, including when threshold was never reached.
    torch.save(checkpoint,a.output/'selected.pt')
    result={'stopping_reason':reason,'epochs':len(history),'validation':history[-1]['validation'],
            'training_and_validation_seconds':time.perf_counter()-run_start,
            'smoke_only':bool(a.limit_dates)}
    if a.evaluate_test:
        if a.limit_dates: raise ValueError('Do not evaluate test in smoke mode')
        del train
        test=DailyData(a.data_dir/'csi300_dl_test.pkl','2020-07-01','2022-12-31')
        result['test_manifest']=test.manifest
        result['test']=evaluate(model,test,device,a.output/'test_predictions.csv')
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__': main()
