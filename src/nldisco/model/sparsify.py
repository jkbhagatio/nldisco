"""Sparse activation operators used by SED models."""

from typing import Dict, Literal

import torch as t
from torch import Tensor, nn


def batch_topk(acts: Tensor, average_top_k: int) -> Tensor:
    """Keep ``batch * average_top_k`` activations across a complete batch."""

    if acts.ndim < 2:
        raise ValueError("BatchTopK expects a batch dimension and feature dimension")
    if average_top_k < 1:
        raise ValueError("average_top_k must be positive")
    n_keep = min(acts.shape[0] * average_top_k, acts.numel())
    flat = acts.reshape(-1)
    values, indices = flat.topk(n_keep)
    sparse = t.zeros_like(flat)
    sparse.scatter_(0, indices, values)
    return sparse.view_as(acts)


def sample_topk(acts: Tensor, top_k: int) -> Tensor:
    """Keep TopK within each sample for batch-partition-invariant inference."""

    if acts.ndim < 2:
        raise ValueError("TopK expects a batch dimension and feature dimension")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    flat = acts.flatten(start_dim=1)
    n_keep = min(top_k, flat.shape[1])
    values, indices = flat.topk(n_keep, dim=1)
    sparse = t.zeros_like(flat)
    sparse.scatter_(1, indices, values)
    return sparse.view_as(acts)


class BatchTopK(nn.Module):
    """Use BatchTopK for training and an explicit batch-invariant inference policy."""

    def __init__(
        self,
        dsed_topk_map: Dict[int, int],
        *,
        inference_sparsity: Literal["training_threshold", "sample_topk"] = "training_threshold",
        threshold_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.dsed_topk_map = dict(sorted(dsed_topk_map.items()))
        self.inference_sparsity = inference_sparsity
        self.threshold_decay = threshold_decay
        self._calibrating = False
        self._calibration_step = 0
        if inference_sparsity not in {"training_threshold", "sample_topk"}:
            raise ValueError(f"Unknown inference sparsity policy: {inference_sparsity}")
        if not 0 <= threshold_decay < 1:
            raise ValueError("threshold_decay must be in [0, 1)")
        self._threshold_names = {}
        for d_sed in self.dsed_topk_map:
            name = f"inference_threshold_{d_sed}"
            self.register_buffer(name, t.tensor(float("nan")))
            self._threshold_names[d_sed] = name

    def inference_threshold(self, d_sed: int) -> Tensor:
        """Return the persisted fixed inference threshold for one SED level."""

        if d_sed not in self._threshold_names:
            raise KeyError(f"Unknown SED level: {d_sed}")
        return getattr(self, self._threshold_names[d_sed])

    def begin_threshold_calibration(self) -> None:
        """Reset thresholds before measuring final-model training cutoffs."""

        self._calibrating = True
        self._calibration_step = 0
        for d_sed in self.dsed_topk_map:
            self.inference_threshold(d_sed).fill_(float("nan"))

    def end_threshold_calibration(self) -> None:
        """Finish a nonempty threshold-calibration pass."""

        if self._calibration_step < 1:
            raise RuntimeError("Threshold calibration requires at least one batch")
        self._calibrating = False

    def _training_level(self, level_acts: Tensor, top_k: int, d_sed: int) -> Tensor:
        sparse = batch_topk(level_acts, top_k)
        flat = level_acts.reshape(-1)
        n_keep = min(level_acts.shape[0] * top_k, flat.numel())
        cutoff = flat.topk(n_keep).values[-1].detach()
        threshold = self.inference_threshold(d_sed)
        with t.no_grad():
            if t.isnan(threshold):
                threshold.copy_(cutoff)
            elif self._calibrating:
                threshold.add_((cutoff - threshold) / self._calibration_step)
            else:
                threshold.mul_(self.threshold_decay).add_(cutoff * (1 - self.threshold_decay))
        return sparse

    def _inference_level(self, level_acts: Tensor, top_k: int, d_sed: int) -> Tensor:
        if self.inference_sparsity == "sample_topk":
            return sample_topk(level_acts, top_k)
        threshold = self.inference_threshold(d_sed)
        if t.isnan(threshold):
            raise RuntimeError(
                "BatchTopK inference threshold is uncalibrated; train the SED first or set "
                "inference_sparsity='sample_topk' explicitly"
            )
        return t.where(level_acts >= threshold, level_acts, t.zeros_like(level_acts))

    def forward(self, acts: Tensor) -> Dict[int, Tensor]:
        if self._calibrating:
            self._calibration_step += 1
        sparse = {}
        for d_sed, top_k in self.dsed_topk_map.items():
            level_acts = acts[..., :d_sed]
            if self.training or self._calibrating:
                sparse[d_sed] = self._training_level(level_acts, top_k, d_sed)
            else:
                sparse[d_sed] = self._inference_level(level_acts, top_k, d_sed)
        return sparse
