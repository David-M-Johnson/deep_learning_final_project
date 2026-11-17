"""Project pairwise logistic loss; not an exact reproduction of a published loss."""
import math
import torch
from torch.nn import functional as F


def ranking_objective(prediction, target, weight=0.1):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("ranking weight must be finite and nonnegative")
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("Expected matching one-dimensional predictions and targets")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Loss inputs must be finite")
    if not len(target):
        raise ValueError("Empty target")
    mse = F.mse_loss(prediction, target)
    if weight == 0:
        ranking = prediction.sum() * 0
        return mse, mse, ranking
    pairs = torch.triu_indices(len(target), len(target), offset=1, device=target.device)
    differences = target[pairs[0]] - target[pairs[1]]
    keep = differences != 0
    score_diff = prediction[pairs[0]] - prediction[pairs[1]]
    ranking = (F.softplus(-differences[keep].sign() * score_diff[keep]).mean()
               if keep.any() else prediction.sum() * 0)
    return mse + weight * ranking, mse, ranking
