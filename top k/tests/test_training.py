import numpy as np
import pandas as pd
import torch
from src.train import training_labels, evaluate


def test_training_filter_preserves_order_and_normalizes():
    x=torch.arange(100).reshape(100,1,1).float()
    y=torch.arange(100).float()
    y[50]=float('nan')
    xx,yy=training_labels(x,y)
    # floor(.025*99)=2 removed from each tail.
    assert len(yy)==95
    assert torch.all(xx[1:,0,0] > xx[:-1,0,0])
    torch.testing.assert_close(yy.mean(),torch.tensor(0.),atol=1e-7,rtol=0)
    torch.testing.assert_close(yy.std(),torch.tensor(1.))


def test_evaluation_keeps_stocks_with_missing_labels():
    class Data:
        groups=[(pd.Timestamp("2020-04-01"),0,4)]
        def batch(self,g):
            return torch.arange(4).float().reshape(4,1,1),torch.tensor([0.,1.,float('nan'),3.])
    class Model(torch.nn.Module):
        def forward(self,x):
            assert len(x)==4
            return x[:,0,0]
    result=evaluate(Model(),Data(),torch.device('cpu'))
    assert np.isclose(result['IC'],1)
    assert np.isclose(result['RankIC'],1)
    assert result['IC_valid_dates']==1
