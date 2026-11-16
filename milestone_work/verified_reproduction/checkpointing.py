"""Atomic latest/best-validation/threshold checkpoints without changing optimization."""
import math
from pathlib import Path
import torch


def atomic_save(value,path):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


class CheckpointTracker:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.best_ic=-math.inf
        self.best_epoch=None
        self.best_metrics=None
        self.threshold_epoch=None
        self.threshold_metrics=None

    def update(self,payload,metrics,threshold_reached):
        epoch=int(payload['epoch'])
        atomic_save(payload,self.directory/'last.pt')
        ic=metrics.get('IC')
        if ic is not None and math.isfinite(ic) and ic>self.best_ic:
            self.best_ic=float(ic);self.best_epoch=epoch;self.best_metrics=dict(metrics)
            atomic_save(payload,self.directory/'best_validation.pt')
        if threshold_reached and self.threshold_epoch is None:
            self.threshold_epoch=epoch;self.threshold_metrics=dict(metrics)
            atomic_save(payload,self.directory/'threshold_selected.pt')

    def summary(self):
        return {'selection_metric':'validation IC','tie_rule':'earliest epoch (strict improvement)',
                'best_validation_epoch':self.best_epoch,'best_validation_metrics':self.best_metrics,
                'threshold_selected_epoch':self.threshold_epoch,'threshold_selected_metrics':self.threshold_metrics,
                'baseline_stopping_unchanged':True,
                'note':'Validation selection is supplemental and optimistic on validation; no test-set selection.'}
