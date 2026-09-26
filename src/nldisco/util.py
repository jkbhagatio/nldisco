"""General utility functions for NLDisco."""

import random

import numpy as np
import torch as t


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for one independent SED replica."""

    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def vector_r2(y_pred: t.Tensor, y_true: t.Tensor) -> t.Tensor:
    """Calculate R² over the final dimension for matching tensors."""

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have matching shapes")
    residual = (y_true - y_pred).pow(2).sum(dim=-1)
    total = (y_true - y_true.mean(dim=-1, keepdim=True)).pow(2).sum(dim=-1)
    result = 1 - residual / total
    return t.where(t.isfinite(result), result, t.zeros_like(result))
