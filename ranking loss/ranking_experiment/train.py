"""Dense MASTER with a configurable pairwise ranking objective."""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .shared import MASTER, DailyData, training_labels, evaluate, checksum, SOURCE
from .loss import ranking_objective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data/opensource')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ranking-weight', type=float, default=0.1)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--stop-loss', type=float, default=.95, help='Threshold on MSE component, not combined loss')
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--limit-dates', type=int, help='Smoke checks only; limits train/validation dates')
    p.add_argument('--evaluate-test', action='store_true', help='Score test once after fixed training protocol')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be new/empty to avoid overwriting a run')
    if not np.isfinite(a.ranking_weight) or a.ranking_weight < 0 or a.epochs < 1:
        p.error('Finite nonnegative ranking weight and positive epochs required')
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
                   'checkpoint_rule':'first epoch mean MSE component <= threshold, otherwise last',
                   'validation_use':'diagnostic only; published processing caveat unresolved',
                   'label_policy':'opensource: drop missing/tails, cross-sectional sample-std zscore',
                   'self_attention':'dense reference attention',
                   'torch':torch.__version__, 'numpy':np.__version__, 'pandas':pd.__version__,
                   'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   'git_status':subprocess.check_output(['git','status','--short'],text=True),
                   'model_sha256':checksum(SOURCE/'model.py'),
                   'shared_training_sha256':checksum(SOURCE/'train.py'),
                   'runner_sha256':checksum(Path(__file__)),
                   'loss_sha256':checksum(Path(__file__).with_name('loss.py')),
                   'objective':'MSE + weight * mean pairwise logistic loss; ties excluded',
                   'attention_mode':'dense'})
    manifest={'config':config,'train':train.manifest,'validation':valid.manifest}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
    model = MASTER(158,a.width,4,2,.5,.5,158,221,5,'dense',10).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    run_start=time.perf_counter(); history=[]; reason='epoch_budget'
    for epoch in range(a.epochs):
        model.train(); start=time.perf_counter(); losses=[]; mse_losses=[]; rank_losses=[]
        for j in np.random.default_rng(a.seed+epoch).permutation(len(train.groups)):
            x,y=training_labels(*train.batch(train.groups[j]))
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss,mse,ranking=ranking_objective(model(x),y,a.ranking_weight)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_value_(model.parameters(),3)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
            mse_losses.append(float(mse.detach().cpu())); rank_losses.append(float(ranking.detach().cpu()))
        metrics=evaluate(model,valid,device)
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),
             'mse':float(np.mean(mse_losses)), 'pairwise_loss':float(np.mean(rank_losses)),
             'validation':metrics,'epoch_seconds':time.perf_counter()-start,
             'elapsed_seconds':time.perf_counter()-run_start}
        history.append(row)
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
        stop=row['mse']<=a.stop_loss
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
