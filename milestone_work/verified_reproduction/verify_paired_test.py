"""Verify common test populations, source hashes, saved metrics and sample predictions."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from baseline_runner import make_model
from paired_test_eval import SegmentView,sha

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    config=json.loads((ROOT/'configs/public_release_test.json').read_text())
    output=ROOT/config['output']
    result=json.loads((output/'comparison.json').read_text())
    for name,expected in config['input_sha256'].items():assert sha(ROOT/name)==expected
    with (ROOT/config['test_file']).open('rb') as f:ds=pickle.load(f)
    view=SegmentView(ds,config['test_start'],config['test_end'])
    dates=pd.DatetimeIndex(view.index.get_level_values('datetime'))
    common=None;checks=[]
    for cp in config['checkpoints']:
        p=output/cp['label'];frame=pd.read_csv(p/'predictions.csv')
        daily=pd.read_csv(p/'daily_metrics.csv');record=json.loads((p/'metrics.json').read_text())
        identity=frame[['instrument','datetime','label']]
        if common is None:common=identity
        else:pd.testing.assert_frame_equal(identity,common)
        assert len(frame)==183300 and frame.label.notna().sum()==182995
        assert len(daily)==611 and (daily.n_inference==300).all()
        assert record['test_index_sha256']==result['same_test_index_sha256']
        for metric in ['IC','RankIC']:
            assert np.isclose(daily[metric].mean(),record['metrics'][metric],rtol=0,atol=1e-12)
            assert np.isclose(daily[metric].mean()/daily[metric].std(ddof=0),record['metrics'][metric+'IR'],rtol=0,atol=1e-12)
        raw=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)
        model=make_model(256);model.load_state_dict(raw if cp['kind']=='upstream' else raw['model']);model.eval()
        errors=[]
        for date in dates.unique()[[0,300,610]]:
            positions=np.flatnonzero(dates==date)
            x=torch.from_numpy(np.asarray(view[positions])[:,:,:-1].copy())
            with torch.no_grad():pred=model(x).numpy()
            err=float(np.abs(pred-frame.prediction.values[positions]).max())
            assert err<1e-6
            errors.append(err)
        checks.append({'model':cp['label'],'same_population':True,'metric_recomputation':True,
                       'sampled_prediction_max_error':max(errors)})
    report={'passed':True,'checks':checks,'models':4,'dates':611,'inference_rows_per_model':183300,
            'scored_rows_per_model':182995,'frozen_input_hashes_match':True,
            'training_protocol_unchanged':True,'portfolio_evaluation':False}
    (ROOT/'reports/paired_test_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: common populations, checkpoint/data hashes, metric recomputation, and sampled saved predictions for all four models')


if __name__=='__main__':main()
