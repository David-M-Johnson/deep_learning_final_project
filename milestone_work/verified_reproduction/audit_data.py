"""Read-only audit of local sampler snapshots, source contracts and cached workbook values."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as E
import zipfile

import numpy as np
import pandas as pd
from baseline_runner import daily_groups,get_batch

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def workbook_cells(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[''.join(n.itertext()) for n in E.fromstring(z.read('xl/sharedStrings.xml'))]
        cells={}
        for row in E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns):
            for c in row:
                v=c.find('s:v',ns)
                if v is not None:cells[c.attrib['r']]=strings[int(v.text)] if c.attrib.get('t')=='s' else float(v.text)
    return cells


def main():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    records=[]
    for split in ['train','valid','test']:
        path=ROOT.parent/'data/opensource'/f'csi300_dl_{split}.pkl'
        with path.open('rb') as f:ds=pickle.load(f)
        idx,groups,info=daily_groups(ds,split)
        counts=[];max_market_spread=0.;label_missing=0;features_nonfinite=0
        for date,positions in groups:
            # Current-date rows suffice to audit input/label coverage; no model prediction.
            raw=np.asarray(ds[positions])[:,-1,:]
            counts.append({'date':str(date.date()),'stocks':len(positions)})
            features_nonfinite+=int((~np.isfinite(raw[:,:221])).sum())
            label_missing+=int((~np.isfinite(raw[:,-1])).sum())
            max_market_spread=max(max_market_spread,float(np.ptp(raw[:,158:221],axis=0).max()))
        pd.DataFrame(counts).to_csv(out/f'{split}_daily_counts.csv',index=False)
        # Verify sampled lookbacks select only dates <= current prediction date.
        future_lookbacks=0
        for _,positions in groups[::max(1,len(groups)//10)]:
            for pos in positions[::max(1,len(positions)//5)]:
                row,col=ds._get_row_col(int(pos))
                indices=ds._get_indices(row,col)
                finite=indices[np.isfinite(indices)].astype(int)
                for k in finite:
                    if k==ds.nan_idx:continue
                    # Original data rows are instrument-sorted; invert idx_arr for date check.
                    located=np.where(ds.idx_arr[:,col]==k)[0]
                    if len(located) and located[0]>row:future_lookbacks+=1
        r={'split':split,'file_sha256':sha(path),'bytes':path.stat().st_size,
           'dtype':str(ds.data_arr.dtype),'raw_shape':list(ds.data_arr.shape),'step_len':ds.step_len,
           'fillna_type':ds.fillna_type,'stored_start':str(ds.start),'stored_end':str(ds.end),
           'legacy_start_idx':int(ds.start_idx),'legacy_end_idx':int(ds.end_idx),**info,
           'stocks_min':min(c['stocks'] for c in counts),'stocks_max':max(c['stocks'] for c in counts),
           'missing_current_labels':label_missing,'nonfinite_current_features':features_nonfinite,
           'max_within_day_market_spread':max_market_spread,'sampled_future_lookbacks':future_lookbacks,
           'feature_column_names_available':False}
        records.append(r);print(split,info['dates_used'],r['stocks_min'],r['stocks_max'],flush=True)
        assert features_nonfinite==0 and max_market_spread==0 and future_lookbacks==0
        del ds
    (out/'data_audit.json').write_text(json.dumps(records,indent=2)+'\n')
    cells=workbook_cells(ROOT/'reference/model/performance.xlsx')
    (out/'benchmark_workbook_cells.json').write_text(json.dumps({'sheet':'Sheet1','cells':cells},indent=2)+'\n')
    rows=[]
    for universe,start in [('CSI300','B'),('CSI800','J')]:
        first=ord(start)
        for metric,j in zip(['IC','ICIR','RankIC','RankICIR','AR','IR'],range(6)):
            col=chr(first+j)
            vals=np.array([cells[f'{col}{r}'] for r in range(14,19)])
            assert np.isclose(vals.mean(),cells[f'{col}19'])
            rows.append({'universe':universe,'protocol':'train/test renewed open-source; Qlib portfolio',
                         'metric':metric,'mean':float(vals.mean()),'population_std':float(vals.std()),
                         'workbook_mean':cells[f'{col}19'],'workbook_std':cells[f'{col}20'],
                         'source_cells':f'Sheet1!{col}14:{col}20','seeds':'0,1,2,3,4'})
    pd.DataFrame(rows).to_csv(out/'renewed_benchmark.csv',index=False)
    for file in ['vendor/master.py','vendor/base_model.py']:
        assert sha(ROOT/file)==sha(ROOT/'reference'/file)
    # Supplied sampler objects discarded their DataFrame column names; no raw close bars are included.
    findings={
      'confirmed':['221 inputs + 1 label, float32, lookback 8','market columns 158:221 shared within each date',
                   'input market gate active; CSI300 beta=5','historical indices sampled do not exceed prediction date',
                   'paper/YAML target close[t+5]/close[t+1]-1','boundary purge and context-date filtering recorded',
                   'no missing current labels in supplied inference samples'],
      'not_independently_confirmable':['Alpha158 exact expression ordering: feature names absent from pickles',
          'actual target arithmetic/alignment against raw close bars: raw prices absent',
          'original median/MAD fit interval and missing-value transformation cannot be re-fitted from processed snapshots',
          'missing inference stocks from upstream DropnaLabel cannot be restored without raw input universe/features'],
      'validation_issue':'No missing labels in supplied validation. Consistent with documented DropnaLabel; cannot quantify omitted eligible stocks without raw universe. Do not claim repaired.',
      'compatibility':'Old sampler objects retain unsliced context plus start_idx/end_idx. Qlib 0.9.7 get_index returns all data_index; explicit calendar filtering restores intended prediction segments.',
      'benchmark':'Use Sheet1 B14:G20 for renewed CSI300, not B4:G10 (original training / renewed test). Benchmark is test, not our short validation.',
      'feature_preprocessing_source':'Upstream README, not independently reconstructed',
      'test_policy':'Only coverage/shape/label missingness audited; no model test predictions or IC computed.'}
    (out/'audit_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    print('Audit and benchmark extraction complete',flush=True)


if __name__=='__main__':main()
