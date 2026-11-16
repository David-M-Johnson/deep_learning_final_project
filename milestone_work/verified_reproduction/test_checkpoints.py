"""Meaningful checkpoint selection/reload checks on real training updates."""
import copy
import json
from pathlib import Path
import tempfile

import torch
from checkpointing import CheckpointTracker
from train import make_model

ROOT=Path(__file__).resolve().parent


def main():
    (ROOT/'.tmp').mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(99)
    records=[]
    for width in [64,128,256]:
      with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
        model=make_model(width);opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        x=torch.randn(48,8,221);y=torch.randn(48)
        tracker=CheckpointTracker(tmp)
        expected={}
        for epoch,ic in [(1,.10),(2,.05)]:
            model.train();opt.zero_grad()
            loss=(model(x)-y).square().mean();loss.backward();opt.step()
            payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':{'width':width}}
            tracker.update(payload,{'IC':ic},threshold_reached=epoch==2)
            model.eval()
            with torch.no_grad():expected[epoch]=model(x).clone()
        for name,epoch in [('last.pt',2),('best_validation.pt',1),('threshold_selected.pt',2)]:
            state=torch.load(Path(tmp)/name,map_location='cpu',weights_only=False)
            assert state['epoch']==epoch
            other=make_model(width);other.load_state_dict(state['model']);other.eval()
            with torch.no_grad():torch.testing.assert_close(other(x),expected[epoch],rtol=0,atol=0)
        assert not list(Path(tmp).glob('*.tmp'))
        # A tie must not replace the earlier best checkpoint.
        tracker.update(payload,{'IC':.10},False)
        assert torch.load(Path(tmp)/'best_validation.pt',weights_only=False)['epoch']==1
        records.append({'width':width,'last_epoch':2,'best_epoch':1,'threshold_epoch':2,'exact_reload':True})
    (ROOT/'reports/checkpoint_tests.json').write_text(json.dumps({'passed':True,'cases':records},indent=2)+'\n')
    print('PASS: latest, validation best, threshold selection, tie rule, atomic saves and exact reload at all widths')


if __name__=='__main__':main()
