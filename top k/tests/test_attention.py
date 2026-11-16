import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from src.model import MASTER, SAttention

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'original_paper/MASTER'))
spec = importlib.util.spec_from_file_location('reference_master', ROOT.parent / 'original_paper/MASTER/master.py')
reference = importlib.util.module_from_spec(spec)
previous_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    spec.loader.exec_module(reference)
finally:
    sys.dont_write_bytecode = previous_bytecode


def make_model(mode='dense', k=10):
    return MASTER(158, 64, 4, 2, 0.0, 0.0, 158, 221, 5, attention_mode=mode, k=k)


def test_full_model_dense_and_all_neighbors_match_reference():
    torch.manual_seed(12)
    original = reference.MASTER(158, 64, 4, 2, 0.0, 0.0, 158, 221, 5).eval()
    x = torch.randn(60, 8, 221)
    expected = original(x)
    for mode, k in [('dense', 10), ('topk', 60), ('topk', 100)]:
        model = make_model(mode, k).eval()
        model.load_state_dict(original.state_dict(), strict=True)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize('k', [1, 10, 50])
def test_exact_neighbors_and_normalization(k):
    layer = SAttention(64, 2, 0, attention_mode='topk', k=k)
    logits = torch.randn(8, 60, 60)
    weights = layer.attention_weights(logits)
    assert torch.all((weights > 0).sum(-1) == k)
    torch.testing.assert_close(weights.sum(-1), torch.ones(8, 60))
    selected = logits.topk(k, dim=-1).indices
    torch.testing.assert_close(weights.gather(-1, selected), logits.gather(-1, selected).softmax(-1))


@pytest.mark.parametrize('k', [10, 50])
def test_backward_update_and_checkpoint(tmp_path, k):
    torch.manual_seed(2)
    model = make_model('topk', k)
    x, y = torch.randn(60, 8, 221), torch.randn(60)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.layers[0].weight.detach().clone()
    loss = (model(x) - y).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.layers[0].weight)
    path = tmp_path / 'weights.pt'
    torch.save(model.state_dict(), path)
    restored = make_model('topk', k)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model.eval()(x), restored.eval()(x), rtol=0, atol=0)


@pytest.mark.parametrize('k', [0, -1, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        SAttention(64, 2, 0, 'topk', k)


def test_supplied_checkpoint_loads_without_key_changes():
    model = MASTER(158, 256, 4, 2, 0, 0, 158, 221, 5, 'topk', 10)
    checkpoint = ROOT.parent / 'original_paper/MASTER/model/csi300_opensource_0.pkl'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(60, 8, 221))).all()
