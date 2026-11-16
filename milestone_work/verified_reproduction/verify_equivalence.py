"""Compare actual reference train_epoch with our one-step mechanics on real daily batches."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import baseline_runner as ours

ROOT=Path(__file__).resolve().parent


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def maxdiff(a,b):return float((a-b).abs().max())


def main():
    torch.set_num_threads(2)
    reference=load_module('reference_wrapper',ROOT/'reference/vendor/base_model.py')
    ref_arch=load_module('reference_architecture',ROOT/'reference/vendor/master.py')
    with (ROOT.parent/'data/opensource/csi300_dl_train.pkl').open('rb') as f:ds=pickle.load(f)
    _,groups,_=ours.daily_groups(ds,'train')
    chosen=[groups[k] for k in [0,len(groups)//2,len(groups)-1]]
    results=[]
    for width in [64,128,256]:
      for date,positions in chosen:
        raw=torch.from_numpy(np.asarray(ds[positions]).copy())
        x,y=ours.get_batch(ds,positions)
        mask,ref_y=reference.drop_extreme(raw[:,-1,-1])
        ref_y=reference.zscore(ref_y)
        our_x,our_y=ours.training_batch(x,y,'opensource')
        assert torch.equal(our_x,raw[:,:,:-1][mask])
        torch.testing.assert_close(our_y,ref_y,rtol=0,atol=0)
        torch.manual_seed(17)
        ref_model=ref_arch.MASTER(158,width,4,2,.5,.5,158,221,5)
        model=ours.make_model(width)
        model.load_state_dict(ref_model.state_dict())
        wrapper=reference.SequenceModel(1,1e-5,GPU=None,seed=None,train_stop_loss_thred=.95)
        wrapper.model=ref_model;wrapper.init_model()
        if wrapper.device.type!='cpu':raise RuntimeError('This equivalence protocol is CPU-only')
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        capture={}
        def hook(_,inputs,output):capture['prediction']=output.detach().clone()
        handle=ref_model.register_forward_hook(hook)
        gradients={}
        raw_gradients={}
        for n, parameter in ref_model.named_parameters():
            parameter.register_hook(lambda grad, key=n: raw_gradients.__setitem__(key, grad.detach().clone()))
        old_step=wrapper.train_optimizer.step
        def capture_step(*args,**kwargs):
            gradients.update({n:p.grad.detach().clone() for n,p in ref_model.named_parameters()})
            return old_step(*args,**kwargs)
        wrapper.train_optimizer.step=capture_step
        torch.manual_seed(123)
        reference_loss=wrapper.train_epoch([raw.unsqueeze(0)])
        torch.manual_seed(123)
        model.train();prediction=model(our_x)
        loss=((prediction-our_y)**2).mean()
        optimizer.zero_grad();loss.backward()
        raw_grad_diff=max(maxdiff(p.grad,raw_gradients[n]) for n,p in model.named_parameters())
        torch.nn.utils.clip_grad_value_(model.parameters(),3.)
        grad_diff=max(maxdiff(p.grad,gradients[n]) for n,p in model.named_parameters())
        optimizer.step()
        pred_diff=maxdiff(prediction.detach(),capture['prediction'])
        param_diff=max(maxdiff(p,dict(ref_model.named_parameters())[n]) for n,p in model.named_parameters())
        assert raw_grad_diff<=1e-6 and pred_diff<=1e-6 and grad_diff<=1e-6 and param_diff<=1e-7
        assert abs(loss.item()-reference_loss)<=1e-6
        handle.remove()
        results.append({'width':width,'date':str(date.date()),'raw_dtype':str(raw.dtype),
                        'stocks_before':len(y),'stocks_after':len(our_y),'label_max_diff':0.,
                        'prediction_max_diff':pred_diff,'loss_abs_diff':abs(loss.item()-reference_loss),
                        'raw_gradient_max_diff':raw_grad_diff,'clipped_gradient_max_diff':grad_diff,'updated_parameter_max_diff':param_diff})
        print(results[-1],flush=True)
    # Reference DataLoader construction consumes a torch random draw when iterated;
    # our direct date loop does not. This explains different seeded dropout trajectories.
    torch.manual_seed(123);before=torch.get_rng_state().clone()
    next(iter(DataLoader([torch.ones(1)])))
    dataloader_changes_rng=not torch.equal(before,torch.get_rng_state())
    # Robustness changes are deliberately tested, not mistaken for exact equivalence.
    tiny=torch.arange(10,dtype=torch.float32)
    _,original_tiny=reference.drop_extreme(tiny)
    _,ours_tiny=ours.training_batch(torch.zeros(10,8,221),tiny,'opensource')
    assert len(original_tiny)==0 and len(ours_tiny)==10
    # Real-batch inference equivalence is independent of dropout and checkpoint bookkeeping.
    ref_model.eval();model.eval()
    with torch.no_grad():torch.testing.assert_close(ref_model(x),model(x),rtol=0,atol=0)
    report={'passed':True,'cases':results,'tolerances':{'prediction':1e-6,'loss':1e-6,'gradient':1e-6,'parameter':1e-7},
        'scope':'9 one-step actual reference train_epoch comparisons, with identical initial weights, order, and dropout RNG',
        'documented_deviations':{
          'iterator_rng':{'reference_dataloader_advances_torch_rng':dataloader_changes_rng,
                          'effect':'Whole-run same-seed trajectories differ unless iterator RNG consumption is matched; not a loss/update formula difference.'},
          'tiny_batch':{'reference_retained':len(original_tiny),'ours_retained':len(ours_tiny)},
          'missing_labels':'Our wrapper removes nonfinite labels before extremes; original lightweight wrapper assumes prefiltered finite data.',
          'split_filtering':'Our calendar filtering/purge intentionally changes allowed training/validation prediction dates.',
          'checkpointing':'Original saves only on train threshold; ours also saves last and will retain validation best.'}}
    (ROOT/'reports/equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: reference preprocessing/forward/loss/clipped gradients/Adam updates',flush=True)


if __name__=='__main__':main()
