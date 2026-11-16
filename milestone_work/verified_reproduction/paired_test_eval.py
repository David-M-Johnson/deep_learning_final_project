"""Frozen public-release test comparison using the unchanged upstream predict/metric code."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

import numpy as np
import pandas as pd
import torch
from baseline_runner import make_model

ROOT=Path(__file__).resolve().parent
METRICS={'IC':'IC','ICIR':'ICIR','RIC':'RankIC','RICIR':'RankICIR'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class SegmentView:
    """Map the old sampler's intended prediction segment without losing lookback context."""
    def __init__(self,dataset,start,end):
        self.dataset=dataset
        index=dataset.get_index()
        dates=pd.DatetimeIndex(index.get_level_values('datetime'))
        self.positions=np.flatnonzero((dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end)))
        self.index=index[self.positions]
        assert self.index.is_unique
        assert pd.DatetimeIndex(self.index.get_level_values('datetime')).is_monotonic_increasing
        # This is exactly the published old sampler's segment, not a new subset.
        np.testing.assert_array_equal(self.positions,np.arange(dataset.start_idx,dataset.end_idx))
    def __len__(self):return len(self.positions)
    def get_index(self):return self.index
    def __getitem__(self,item):return self.dataset[self.positions[item]]


def main():
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Execute from verified_reproduction/')
    config_path=ROOT/'configs/public_release_test.json'
    config=json.loads(config_path.read_text())
    output=ROOT/config['output']
    if output.exists():raise FileExistsError(f'{output}: preserving existing test results; no rerun overwrite')
    for name,expected in config['input_sha256'].items():
        assert sha(ROOT/name)==expected,f'Frozen input changed: {name}'
    assert sha(Path(__file__))==config['evaluation_code_sha256']
    torch.set_num_threads(config['threads'])
    spec=importlib.util.spec_from_file_location('public_reference_wrapper',ROOT/'reference/vendor/base_model.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    with (ROOT/config['test_file']).open('rb') as f:dataset=pickle.load(f)
    view=SegmentView(dataset,config['test_start'],config['test_end'])
    dates=pd.DatetimeIndex(view.get_index().get_level_values('datetime'))
    grouped=[(date,np.flatnonzero(dates==date)) for date in dates.unique()]
    assert len(grouped)==611 and len(view)==183300
    assert all(len(p)==300 for _,p in grouped)
    labels=np.empty(len(view),dtype=np.float32)
    for date,positions in grouped:
        raw=view[positions]
        assert raw.shape==(300,8,222)
        assert np.isfinite(raw[:,:,:221]).all()
        labels[positions]=raw[:,-1,-1]
    assert sum(np.isnan(labels))==305
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    index_sha=hashlib.sha256(view.index.to_frame(index=False).to_csv(index=False).encode()).hexdigest()
    records=[]
    for checkpoint in config['checkpoints']:
        path=ROOT/checkpoint['path']
        raw=torch.load(path,map_location='cpu',weights_only=False)
        if checkpoint['kind']=='upstream':
            weights=raw;epoch=None
        else:
            assert raw['config']['width']==256
            assert raw['config']['lr']==1e-5 and raw['config']['threshold']==.95
            assert raw['epoch']==checkpoint['epoch']
            weights=raw['model'];epoch=raw['epoch']
        model=make_model(256);model.load_state_dict(weights,strict=True)
        # Avoid the upstream string-valued load_param flag bug. No training occurs.
        wrapper=reference.SequenceModel(n_epochs=1,lr=1e-5,GPU=None,seed=None,train_stop_loss_thred=.95)
        wrapper.device=torch.device('cpu');wrapper.model=model.to('cpu');wrapper.fitted=0
        started=time.perf_counter()
        predictions,upstream_metrics=wrapper.predict(view)
        elapsed=time.perf_counter()-started
        assert predictions.index.equals(view.get_index())
        assert np.isfinite(predictions.values).all()
        daily=[]
        for date,positions in grouped:
            ic,ric=reference.calc_ic(predictions.values[positions],labels[positions])
            daily.append({'datetime':date,'n_inference':len(positions),
                'n_scored':int(np.isfinite(labels[positions]).sum()),'IC':ic,'RankIC':ric})
        daily=pd.DataFrame(daily)
        assert np.isfinite(daily[['IC','RankIC']].values).all()
        metrics={new:float(upstream_metrics[old]) for old,new in METRICS.items()}
        for key in ['IC','RankIC']:
            assert np.isclose(metrics[key],daily[key].mean(),rtol=0,atol=1e-12)
            assert np.isclose(metrics[key+'IR'],daily[key].mean()/daily[key].std(ddof=0),rtol=0,atol=1e-12)
        dest=output/checkpoint['label'];dest.mkdir()
        frame=view.get_index().to_frame(index=False)
        frame['prediction']=predictions.values;frame['label']=labels
        frame.to_csv(dest/'predictions.csv',index=False)
        daily.to_csv(dest/'daily_metrics.csv',index=False)
        record={**checkpoint,'loaded_epoch':epoch,'metrics':metrics,'reference_key_metrics':upstream_metrics,
            'seconds':elapsed,'checkpoint_sha256':sha(path),'test_index_sha256':index_sha,
            'test_dates':len(grouped),'inference_rows':len(view),'scored_rows':int(np.isfinite(labels).sum()),
            'undefined_dates':0,'parameters':sum(p.numel() for p in model.parameters())}
        (dest/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
        print(checkpoint['label'],metrics,flush=True)
    author=records[0]['metrics']
    target=config['upstream_seed0_workbook_metrics']
    benchmark={metric:{'measured':author[metric],'reported':target[metric],
                'difference':author[metric]-target[metric],
                'within_tolerance':abs(author[metric]-target[metric])<=config['benchmark_tolerances'][metric]}
                for metric in target}
    ours=[r for r in records if r['kind']=='local_threshold']
    aggregate={k:{'mean':statistics.mean(r['metrics'][k] for r in ours),
                   'population_std':statistics.pstdev(r['metrics'][k] for r in ours),
                   'seeds':[r['seed'] for r in ours]} for k in ['IC','RankIC','ICIR','RankICIR']}
    result={'complete':True,'protocol_sha256':sha(config_path),'same_test_index_sha256':index_sha,
            'test_file_sha256':sha(ROOT/config['test_file']),'models':records,
            'upstream_seed0_benchmark_check':benchmark,
            'upstream_seed0_benchmark_reproduced':all(v['within_tolerance'] for v in benchmark.values()),
            'ours_threshold_three_seed_aggregate':aggregate,
            'limits':['Public renewed dataset, not unavailable corrected proprietary paper data',
                      'Only author seed 0 checkpoint is supplied; author five-seed workbook is contextual',
                      'Training still differs in boundary purging, RNG/threads and software',
                      'No portfolio evaluation and no checkpoint/hyperparameter tuning on test outcomes']}
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame([{'model':r['label'],'seed':r['seed'],'epoch':r['loaded_epoch'],**r['metrics']} for r in records]).to_csv(output/'comparison.csv',index=False)
    lines=['# Matched public-release test evaluation','',
      'Every model uses the same 611 dates (2020-07-01–2022-12-30), 300 inference stocks per date, '
      '183,300 inference rows and 182,995 labeled rows. Missing labels are masked only by the '
      'unchanged upstream correlation code; no inference stocks are removed. Legacy context rows '
      'are excluded exactly using the stored sampler segment. No final test dates are purged.','',
      '| Checkpoint | Epoch | IC | RankIC | ICIR | RankICIR |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics'];lines.append(f'| {r["label"]} | {r["loaded_epoch"] or "not supplied"} | {m["IC"]:.6f} | {m["RankIC"]:.6f} | {m["ICIR"]:.6f} | {m["RankICIR"]:.6f} |')
    a=aggregate
    lines+=['',f'Our three-seed threshold-selected mean ± population std: IC {a["IC"]["mean"]:.6f} ± {a["IC"]["population_std"]:.6f}; '
              f'RankIC {a["RankIC"]["mean"]:.6f} ± {a["RankIC"]["population_std"]:.6f}.','',
              '## Supplied checkpoint versus its workbook seed-0 result','',
              '| Metric | Measured | Workbook seed 0 | Difference | Within declared tolerance |',
              '| --- | ---: | ---: | ---: | --- |']
    for k,v in benchmark.items():lines.append(f'| {k} | {v["measured"]:.8f} | {v["reported"]:.8f} | {v["difference"]:+.8f} | {v["within_tolerance"]} |')
    lines+=['','Benchmark values come from renewed train/test block, Sheet1 B14:E14. '
       'The means over five authors\' seeds are not the target for a single supplied seed-0 checkpoint.',
       '', '## Scope', '',
       'This is a one-for-one **evaluation** on the public release. It does not establish identical '
       'training or reproduce unavailable original proprietary data. Our three checkpoints were '
       'fixed by the pre-existing training-loss threshold, not selected using test performance. '
       'The historical training sweep remains unchanged. No portfolio/backtest metrics are claimed.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Matched evaluation completed. Upstream seed-0 benchmark reproduced:',result['upstream_seed0_benchmark_reproduced'],flush=True)


if __name__=='__main__':main()
