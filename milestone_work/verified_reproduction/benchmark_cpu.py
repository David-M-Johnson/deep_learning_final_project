"""Bounded training-only timing probe; never used to tune predictive settings."""
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from train import daily_groups,get_batch,training_batch,make_model

ROOT=Path(__file__).resolve().parent
with (ROOT.parent/'data/opensource/csi300_dl_train.pkl').open('rb') as f:ds=pickle.load(f)
_,groups,_=daily_groups(ds,'train')
batches=[training_batch(*get_batch(ds,p),'opensource') for _,p in groups[:20]]
rows=[]
for threads in [1,2]:
 torch.set_num_threads(threads)
 for width in [64,128,256]:
  torch.manual_seed(0);m=make_model(width);o=torch.optim.Adam(m.parameters(),lr=1e-5)
  started=time.perf_counter()
  for x,y in batches:
   o.zero_grad();loss=(m(x)-y).square().mean();loss.backward()
   torch.nn.utils.clip_grad_value_(m.parameters(),3.);o.step()
  seconds=time.perf_counter()-started
  r={'threads':threads,'width':width,'training_dates':len(batches),'seconds':seconds,
     'rough_seconds_per_full_epoch':seconds/len(batches)*len(groups)}
  print(r,flush=True);rows.append(r)
(ROOT/'reports/cpu_timing.json').write_text(json.dumps(rows,indent=2)+'\n')
