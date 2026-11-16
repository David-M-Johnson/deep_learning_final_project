"""Author-checkpoint-only diagnostic of old sampler context rows; primary protocol unchanged."""
import importlib.util
import json
from pathlib import Path
import pickle
import statistics

import numpy as np
import pandas as pd
import torch
from baseline_runner import make_model
from paired_test_eval import sha

ROOT=Path(__file__).resolve().parent


class ContextView:
    def __init__(self,ds):
        self.ds=ds;self.positions=np.arange(ds.start_idx);self.index=ds.get_index()[self.positions]
    def __len__(self):return len(self.positions)
    def get_index(self):return self.index
    def __getitem__(self,item):return self.ds[self.positions[item]]


def main():
    torch.set_num_threads(2)
    output=ROOT/'outputs/public_release_test'
    if not (output/'comparison.json').exists():raise RuntimeError('Primary comparison must finish first')
    dest=output/'reference_context_diagnostic'
    dest.mkdir(exist_ok=False)
    with (ROOT.parent/'data/opensource/csi300_dl_test.pkl').open('rb') as f:ds=pickle.load(f)
    spec=importlib.util.spec_from_file_location('diagnostic_ref',ROOT/'reference/vendor/base_model.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    model=make_model(256)
    model.load_state_dict(torch.load(ROOT/'reference/model/csi300_opensource_0.pkl',weights_only=False,map_location='cpu'))
    wrapper=ref.SequenceModel(1,1e-5,seed=None);wrapper.device=torch.device('cpu');wrapper.model=model;wrapper.fitted=0
    view=ContextView(ds)
    predictions,_=wrapper.predict(view)
    dates=pd.DatetimeIndex(view.index.get_level_values('datetime'))
    records=[]
    for date in dates.unique():
        p=np.flatnonzero(dates==date)
        labels=view[p][:,-1,-1]
        ic,ric=ref.calc_ic(predictions.values[p],labels)
        records.append({'datetime':date,'n_inference':len(p),'n_scored':int(np.isfinite(labels).sum()),'IC':ic,'RankIC':ric})
    context=pd.DataFrame(records)
    primary=pd.read_csv(output/'authors_opensource_seed0/daily_metrics.csv',parse_dates=['datetime'])
    combined=pd.concat([context,primary],ignore_index=True).sort_values('datetime')
    assert len(context)==8 and len(combined)==619
    metrics={}
    for name in ['IC','RankIC']:
        metrics[name]=float(np.mean(combined[name]))
        metrics[name+'IR']=float(np.mean(combined[name])/np.std(combined[name]))
    config=json.loads((ROOT/'configs/public_release_test.json').read_text())
    target=config['upstream_seed0_workbook_metrics']
    checks={k:{'measured_with_context':metrics[k],'workbook':target[k],
               'difference':metrics[k]-target[k],
               'within_declared_tolerance':abs(metrics[k]-target[k])<=config['benchmark_tolerances'][k]} for k in target}
    report={'author_checkpoint_only':True,'primary_protocol_unchanged':True,
      'diagnostic':'Include all old sampler get_index() rows, including June warmup context, as the released loop does under current Qlib',
      'extra_dates':[str(x.date()) for x in dates.unique()],'extra_inference_rows':len(view),
      'total_dates':len(combined),'total_inference_rows':int(combined.n_inference.sum()),
      'metrics_with_context':metrics,'benchmark_checks':checks,
      'all_metrics_match':all(v['within_declared_tolerance'] for v in checks.values()),
      'warning':'These 8 June dates are outside the declared July-2020 test window. A numerical match does not make them legitimate test dates.'}
    context.to_csv(dest/'context_daily_metrics.csv',index=False)
    combined.to_csv(dest/'all_rows_daily_metrics.csv',index=False)
    (dest/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
