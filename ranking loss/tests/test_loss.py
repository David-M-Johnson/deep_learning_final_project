import math
import pytest
import torch
from ranking_experiment.loss import ranking_objective
from ranking_experiment.shared import MASTER


def test_correct_order_beats_reversed_order():
    y=torch.tensor([-1.,0.,1.])
    assert ranking_objective(y,y)[2] < ranking_objective(-y,y)[2]


def test_equal_scores_have_log_two_pair_loss():
    assert ranking_objective(torch.zeros(3),torch.tensor([-1.,0.,1.]))[2].item()==pytest.approx(math.log(2))


def test_pair_loss_is_shift_and_label_scale_invariant():
    p,y=torch.tensor([-.2,.3,.1]),torch.tensor([-1.,1.,0.])
    a=ranking_objective(p,y)[2]
    torch.testing.assert_close(a,ranking_objective(p+5,y*7+4)[2])


def test_ties_are_excluded():
    p=torch.tensor([0.,1.,2.],requires_grad=True)
    loss,mse,ranking=ranking_objective(p,torch.ones(3))
    assert ranking.item()==0
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_zero_weight_matches_mse_and_gradient_exactly():
    p=torch.tensor([-.4,.2,.6],requires_grad=True);y=torch.tensor([-1.,0.,1.])
    expected=(p-y).square().mean(); loss,_,_=ranking_objective(p,y,0)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss,p)[0],torch.autograd.grad(expected,p)[0],rtol=0,atol=0)


def test_ranking_gradient_corrects_reversed_pair():
    p=torch.tensor([1.,-1.],requires_grad=True)
    ranking=ranking_objective(p,torch.tensor([-1.,1.]))[2]
    ranking.backward()
    assert p.grad[0]>0 and p.grad[1]<0


@pytest.mark.parametrize('weight',[-1,float('nan'),float('inf')])
def test_bad_weight_rejected(weight):
    with pytest.raises(ValueError):ranking_objective(torch.ones(2),torch.ones(2),weight)


def test_dense_model_ranking_update_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    model=MASTER(158,64,4,2,0,0,158,221,5,'dense',10)
    x,y=torch.randn(20,8,221),torch.randn(20)
    before=model.layers[0].weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss,_,_=ranking_objective(model(x),y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before,model.layers[0].weight)
    path=tmp_path/'checkpoint.pt';torch.save(model.state_dict(),path)
    restored=MASTER(158,64,4,2,0,0,158,221,5,'dense',10)
    restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(model.eval()(x),restored.eval()(x),rtol=0,atol=0)


def test_nonfinite_predictions_rejected():
    with pytest.raises(ValueError):
        ranking_objective(torch.tensor([float('nan'),0.]),torch.ones(2))


def test_control_configs_change_only_ranking_weight():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'configs'
    ranking=json.loads((root/'dense_ranking.json').read_text())
    control=json.loads((root/'dense_mse.json').read_text())
    assert ranking.pop('ranking_weight')==.1
    assert control.pop('ranking_weight')==0
    assert ranking==control
