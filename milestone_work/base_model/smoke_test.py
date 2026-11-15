"""Fast synthetic checks; never evidence of financial predictive quality."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from train import make_model, score_frame, training_batch, daily_groups


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = make_model()
    assert sum(p.numel() for p in model.parameters()) == 66177
    x = torch.randn(48, 8, 221)
    y = torch.randn(48)
    y[0] = float('nan')
    xx, yy = training_batch(x, y, 'opensource')
    assert len(yy) == 45 and torch.isfinite(yy).all()
    small_x, small_y = training_batch(x[:10], y[:10], 'opensource')
    assert len(small_y) == 9  # upstream's [0:-0] bug must not empty tiny batches
    old = next(model.parameters()).detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = ((model(xx) - yy)**2).mean()
    loss.backward(); opt.step()
    assert not torch.equal(old, next(model.parameters()))
    model.eval()
    with torch.no_grad(): expected = model(x)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'weights.pt'
        torch.save(model.state_dict(), path)
        other = make_model(); other.load_state_dict(torch.load(path, weights_only=True)); other.eval()
        with torch.no_grad(): torch.testing.assert_close(expected, other(x))
    f = pd.DataFrame({'datetime': ['2020-07-01']*4 + ['2020-07-02']*4,
                      'prediction': [1,2,3,4]*2, 'label': [1,2,3,4,4,3,2,1]})
    daily, metrics = score_frame(f)
    assert np.allclose(daily.IC, [1,-1]) and metrics['RankIC'] == 0
    f.loc[:, 'prediction'] = 1
    assert score_frame(f)[1]['undefined_dates'] == 2
    class Fixture:
        def get_index(self):
            return pd.MultiIndex.from_product([pd.bdate_range('2020-03-20', '2020-06-30'), ['A', 'B']], names=['datetime', 'instrument'])
    _, groups, info = daily_groups(Fixture(), 'valid')
    assert groups[0][0] == pd.Timestamp('2020-04-01')
    assert info['context_or_outside_rows_excluded'] > 0
    assert len(info['boundary_dates_removed']) == 5
    print('PASS: width/parameter count, forward/backward/update, checkpoint reload, label filtering, metrics')


if __name__ == '__main__':
    main()
