"""Training, splitting, and evaluation for window-reconstructing SEDs."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as t
import wandb
from sklearn.metrics import r2_score
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nldisco.config import LossConfig, TrainConfig
from nldisco.data.window import WindowSample
from nldisco.evaluation import DiagnosticResult, EvaluationConfig, evaluate_diagnostics
from nldisco.model.sed import Sed
from nldisco.model.sparsify import batch_topk

ACTIVATION_TABLE_COLUMNS = [
    "window_idx",
    "anchor_idx",
    "window_start_idx",
    "source_time_idx",
    "feature_time_idx",
    "lag_from_anchor",
    "trial_code",
    "session_code",
    "replica_id",
    "latent_idx",
    "activation_value",
]


@dataclass
class TrainingResult:
    """Scalar training histories for one independently trained SED."""

    loss: Dict[int, float]
    weighted_reconstruction: Dict[int, float]
    l0_mean: Dict[int, float]
    dead_feature_fraction: Dict[int, float]


@dataclass
class EvaluationResult:
    """Full-window evaluation outputs for one SED replica."""

    reconstructions: Tensor  # [window, timebin, output_unit]
    targets: Tensor  # [window, timebin, output_unit]
    metrics_by_lag: pd.DataFrame
    weighted_reconstruction: float
    activation_table: pd.DataFrame
    evaluation_index: pd.DataFrame
    inference_sparsity: str
    diagnostics: Optional[DiagnosticResult] = None


def split_rows_by_trial_proportion(
    trial_ids: np.ndarray,
    session_ids: Optional[np.ndarray] = None,
    train_proportion: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return row masks after splitting unique (session, trial) identities."""

    if not 0 < train_proportion < 1:
        raise ValueError("train_proportion must be in (0, 1)")
    trial_ids = np.asarray(trial_ids, dtype=object)
    if session_ids is None:
        session_ids = np.zeros(len(trial_ids), dtype=object)
    else:
        session_ids = np.asarray(session_ids, dtype=object)
    if len(trial_ids) != len(session_ids):
        raise ValueError("trial_ids and session_ids must have the same length")
    valid = ~(pd.isna(trial_ids) | pd.isna(session_ids))
    keys = np.empty(len(trial_ids), dtype=object)
    keys[:] = list(zip(session_ids.tolist(), trial_ids.tolist()))
    ordered = np.asarray(pd.Series(keys[valid]).unique(), dtype=object)
    if len(ordered) < 2:
        raise ValueError("At least two valid trial identities are required for a split")
    if shuffle:
        np.random.default_rng(seed).shuffle(ordered)
    n_train = max(1, min(len(ordered) - 1, int(len(ordered) * train_proportion)))
    train_keys = set(ordered[:n_train].tolist())
    validation_keys = set(ordered[n_train:].tolist())
    train_rows = np.asarray([is_valid and key in train_keys for key, is_valid in zip(keys, valid)])
    validation_rows = np.asarray(
        [is_valid and key in validation_keys for key, is_valid in zip(keys, valid)]
    )
    return train_rows, validation_rows


def split_rows_by_session(
    session_ids: np.ndarray,
    train_sessions: List[object],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/validation row masks according to session membership."""

    if not train_sessions:
        raise ValueError("At least one training session is required")
    if shuffle or seed is not None:
        # Retained in the signature for call-site symmetry; row membership is deterministic.
        pass
    valid = ~pd.isna(session_ids)
    train_session_values = np.asarray(train_sessions, dtype=object)
    train_rows = valid & np.isin(session_ids, train_session_values)
    validation_rows = valid & ~np.isin(session_ids, train_session_values)
    if not train_rows.any() or not validation_rows.any():
        raise ValueError("Session split must leave at least one row in both partitions")
    return np.asarray(train_rows), np.asarray(validation_rows)


def elementwise_reconstruction_loss(
    target: Tensor,
    reconstruction: Tensor,
    loss_cfg: LossConfig,
) -> Tensor:
    """Return an unreduced reconstruction loss with shape [B, S, N]."""

    if target.shape != reconstruction.shape or target.ndim != 3:
        raise ValueError(
            "target and reconstruction must have matching [batch, timebin, neuron] shapes"
        )
    if loss_cfg.type == "mse":
        return (target - reconstruction).pow(2)
    if loss_cfg.type != "msle":
        raise ValueError(f"Unknown reconstruction loss: {loss_cfg.type}")
    if not math.isfinite(loss_cfg.tau) or loss_cfg.tau <= 0:
        raise ValueError("tau must be finite and positive")
    if (target < 0).any():
        raise ValueError("MSLE requires nonnegative targets")
    return (t.log1p(target) - loss_cfg.tau * t.log1p(reconstruction.clamp_min(0))).pow(2)


def weighted_reconstruction_loss(
    target: Tensor,
    reconstruction: Tensor,
    loss_cfg: LossConfig,
) -> Tuple[Tensor, Tensor]:
    """Return weighted full-window loss and the loss at every time bin."""

    if len(loss_cfg.timebin_weights) != target.shape[1]:
        raise ValueError("timebin_weights must contain exactly one value per target time bin")
    if any(not math.isfinite(weight) or weight <= 0 for weight in loss_cfg.timebin_weights):
        raise ValueError("Every timebin weight must be finite and positive")
    per_timebin = elementwise_reconstruction_loss(target, reconstruction, loss_cfg).mean(
        dim=(0, 2)
    )
    weights = t.as_tensor(
        loss_cfg.timebin_weights,
        dtype=per_timebin.dtype,
        device=per_timebin.device,
    )
    weighted = (per_timebin * weights).sum() / weights.sum()
    return weighted, per_timebin


def simple_cosine_lr_schedule(
    step: int,
    n_steps: int,
    initial_lr: float,
    min_lr: float,
) -> float:
    """Warm up, hold, and cosine-decay the learning rate."""

    n_warmup_steps = max(1, int(n_steps * 0.1))
    decay_start_step = int(n_steps * 0.5)
    if step < n_warmup_steps:
        return max(initial_lr * step / n_warmup_steps, min_lr)
    if step < decay_start_step:
        return initial_lr
    decay_steps = max(1, n_steps - decay_start_step)
    decay_position = (step - decay_start_step) / decay_steps
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * decay_position))


def _batch_values(batch: WindowSample, device: t.device, dtype: t.dtype) -> Tensor:
    return batch.values.to(device=device, dtype=dtype, non_blocking=True)


def _validate_population_shapes(model: Sed, inputs: Tensor, target: Tensor, batched: bool) -> None:
    """Reject incompatible populations before encoding or updating model parameters."""
    ndim = 3 if batched else 2
    if inputs.ndim != ndim or inputs.shape[-2:] != (model.cfg.seq_len, model.cfg.n_neurons):
        raise ValueError(
            f"inputs must have shape {'[batch, ' if batched else '['}"
            f"{model.cfg.seq_len}, {model.cfg.n_neurons}]"
        )
    if target.ndim != ndim or target.shape[-2:] != (
        model.cfg.seq_len, model.cfg.n_output_neurons
    ) or (batched and inputs.shape[0] != target.shape[0]):
        raise ValueError(
            f"targets must have shape {'[batch, ' if batched else '['}"
            f"{model.cfg.seq_len}, {model.cfg.n_output_neurons}] aligned with inputs"
        )


def _validate_target_domain(model: Sed, target: Tensor, loss_cfg: LossConfig) -> None:
    if not t.isfinite(target).all():
        raise ValueError("Targets must be finite")
    if (target < 0).any():
        if loss_cfg.type == "msle":
            raise ValueError("MSLE requires nonnegative targets")
        if model.cfg.decoder.output_activation != "none":
            raise ValueError("Signed targets require decoder.output_activation=none")


def _validate_loader_populations(model: Sed, loader: DataLoader, loss_cfg: LossConfig) -> None:
    """Validate dataset shapes before training, without advancing its batch sampler."""
    sample = loader.dataset[0]
    _validate_population_shapes(model, sample.values, sample.target, batched=False)
    # Window construction has already excluded invalid rows. Check all retained
    # targets up front when the dataset exposes its underlying recording.
    dataset = loader.dataset
    if hasattr(dataset, "targets") and hasattr(dataset, "allowed_rows"):
        for start in range(0, len(dataset.targets), 100_000):
            stop = start + 100_000
            rows = t.as_tensor(dataset.allowed_rows[start:stop], device=dataset.targets.device)
            _validate_target_domain(model, dataset.targets[start:stop][rows], loss_cfg)
    else:
        _validate_target_domain(model, sample.target, loss_cfg)


def _batch_populations(batch: WindowSample, model: Sed, loss_cfg: LossConfig) -> Tuple[Tensor, Tensor]:
    device = next(model.parameters()).device
    inputs = _batch_values(batch, device, model.cfg.dtype)
    target = batch.target.to(device=device, dtype=model.cfg.dtype, non_blocking=True)
    _validate_population_shapes(model, inputs, target, batched=True)
    _validate_target_domain(model, target, loss_cfg)
    if not t.isfinite(inputs).all():
        raise ValueError("Inputs must be finite")
    return inputs, target


def _dead_feature_reconstruction(
    model: Sed,
    acts: Tensor,
    dead_features: Tensor,
    max_dead_feature_fraction: float,
) -> Optional[Tensor]:
    n_dead = int(dead_features.sum().item())
    if n_dead == 0:
        return None
    broadcast_shape = [1] * (acts.ndim - 1) + [acts.shape[-1]]
    dead_acts = acts * dead_features.view(*broadcast_shape)
    average_top_k = max(1, min(n_dead, int(max_dead_feature_fraction * acts.shape[-1])))
    sparse_dead_acts = batch_topk(dead_acts, average_top_k)
    return model.decode(
        sparse_dead_acts,
        apply_output_activation=False,
        include_bias=False,
    )


def train_model(
    model: Sed,
    train_loader: DataLoader,
    loss_cfg: LossConfig,
    train_cfg: TrainConfig,
    *,
    optimizer: Optional[t.optim.Optimizer] = None,
    log_wandb: bool = False,
) -> TrainingResult:
    """Encode input windows and supervise every level with aligned target windows."""

    loss_cfg.validate_for(model.cfg)
    if len(train_loader) == 0:
        raise ValueError("train_loader contains no valid windows")
    _validate_loader_populations(model, train_loader, loss_cfg)
    resolved_optimizer = (
        t.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
        if optimizer is None
        else optimizer
    )
    device = next(model.parameters()).device
    n_steps = train_cfg.epochs * len(train_loader)
    initial_lr = resolved_optimizer.param_groups[0]["lr"]
    min_lr = initial_lr * 1e-2
    largest_level = model.cfg.n_features
    inactive_steps = t.zeros(largest_level, dtype=t.long, device=device)
    dead_features = t.zeros(largest_level, dtype=t.bool, device=device)
    histories: Dict[str, Dict[int, float]] = {
        "loss": {},
        "weighted_reconstruction": {},
        "l0_mean": {},
        "dead_feature_fraction": {},
    }

    progress = tqdm(total=n_steps, desc="SED training step")
    step = 0
    model.train()
    for _epoch in range(train_cfg.epochs):
        for batch in train_loader:
            if train_cfg.use_lr_schedule:
                resolved_optimizer.param_groups[0]["lr"] = simple_cosine_lr_schedule(
                    step, n_steps, initial_lr, min_lr
                )

            inputs, target = _batch_populations(batch, model, loss_cfg)
            resolved_optimizer.zero_grad()
            output = model(inputs, occurrence_mask=batch.occurrence_mask.to(device))
            reconstruction_loss = t.zeros((), dtype=t.float32, device=device)
            largest_weighted_loss = None
            for d_sed, reconstruction in output.reconstructions.items():
                level_loss, _ = weighted_reconstruction_loss(target, reconstruction, loss_cfg)
                reconstruction_loss = reconstruction_loss + (
                    level_loss.float() * loss_cfg.level_weight(d_sed)
                )
                if d_sed == largest_level:
                    largest_weighted_loss = level_loss

            total_loss = reconstruction_loss
            dead_reconstruction = _dead_feature_reconstruction(
                model,
                output.acts,
                dead_features,
                train_cfg.max_dead_feature_fraction,
            )
            if dead_reconstruction is not None and loss_cfg.dead_feature_loss_weight:
                residual = target - output.reconstructions[largest_level].detach()
                per_timebin_aux = (residual - dead_reconstruction).pow(2).mean(dim=(0, 2))
                weights = t.as_tensor(
                    loss_cfg.timebin_weights,
                    device=device,
                    dtype=per_timebin_aux.dtype,
                )
                auxiliary_loss = (per_timebin_aux * weights).sum() / weights.sum()
                total_loss = total_loss + loss_cfg.dead_feature_loss_weight * auxiliary_loss

            total_loss.backward()
            model.constrain_dictionary()
            resolved_optimizer.step()
            model.normalize_dictionary()

            largest_sparse = output.sparse_acts[largest_level]
            reduce_dims = tuple(range(largest_sparse.ndim - 1))
            active_features = largest_sparse.sum(dim=reduce_dims) > 0
            inactive_steps[active_features] = 0
            inactive_steps[~active_features] += 1
            dead_features = inactive_steps > train_cfg.dead_feature_window

            if step % train_cfg.log_frequency == 0 or step + 1 == n_steps:
                l0 = (largest_sparse > 0).flatten(start_dim=1).sum(dim=1).float().mean()
                dead_fraction = dead_features.float().mean()
                histories["loss"][step] = float(total_loss.detach())
                assert largest_weighted_loss is not None
                histories["weighted_reconstruction"][step] = float(largest_weighted_loss.detach())
                histories["l0_mean"][step] = float(l0)
                histories["dead_feature_fraction"][step] = float(dead_fraction)
                progress.set_postfix(
                    loss=f"{float(total_loss.detach()):.5f}",
                    l0=f"{float(l0):.2f}",
                    dead=f"{float(dead_fraction):.3f}",
                )
                if log_wandb:
                    wandb.log(  # pyright: ignore[reportAttributeAccessIssue]
                        {
                            "train/loss": float(total_loss.detach()),
                            "train/reconstruction/window_weighted": float(
                                largest_weighted_loss.detach()
                            ),
                            "train/l0_mean": float(l0),
                            "train/dead_feature_fraction": float(dead_fraction),
                            "train/step": step,
                        }
                    )
            step += 1
            progress.update(1)
    progress.close()
    if model.cfg.inference_sparsity == "training_threshold":
        was_training = model.training
        model.eval()
        model.sparsifier.begin_threshold_calibration()
        with t.no_grad():
            for batch in train_loader:
                inputs = _batch_values(batch, device, model.cfg.dtype)
                model(inputs, occurrence_mask=batch.occurrence_mask.to(device))
        model.sparsifier.end_threshold_calibration()
        model.train(was_training)
    return TrainingResult(**histories)


def _activation_rows(
    sparse_acts: Tensor,
    batch: WindowSample,
    replica_id: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sparse_cpu = sparse_acts.detach().float().cpu()
    if sparse_cpu.ndim == 3:
        sparse_cpu = sparse_cpu * batch.occurrence_mask.cpu().unsqueeze(-1)
    if sparse_cpu.ndim == 2:
        batch_indices, feature_indices = t.where(sparse_cpu > 0)
        for batch_idx, feature_idx in zip(batch_indices.tolist(), feature_indices.tolist()):
            rows.append(
                {
                    "window_idx": int(batch.window_index[batch_idx]),
                    "anchor_idx": int(batch.anchor_index[batch_idx]),
                    "window_start_idx": int(batch.source_indices[batch_idx, 0]),
                    "source_time_idx": int(batch.anchor_index[batch_idx]),
                    "feature_time_idx": -1,
                    "lag_from_anchor": 0,
                    "trial_code": int(batch.trial_code[batch_idx]),
                    "session_code": int(batch.session_code[batch_idx]),
                    "replica_id": replica_id,
                    "latent_idx": feature_idx,
                    "activation_value": float(sparse_cpu[batch_idx, feature_idx]),
                }
            )
    else:
        batch_indices, time_indices, feature_indices = t.where(sparse_cpu > 0)
        seq_len = sparse_cpu.shape[1]
        for batch_idx, time_idx, feature_idx in zip(
            batch_indices.tolist(), time_indices.tolist(), feature_indices.tolist()
        ):
            rows.append(
                {
                    "window_idx": int(batch.window_index[batch_idx]),
                    "anchor_idx": int(batch.anchor_index[batch_idx]),
                    "window_start_idx": int(batch.source_indices[batch_idx, 0]),
                    "source_time_idx": int(batch.source_indices[batch_idx, time_idx]),
                    "feature_time_idx": time_idx,
                    "lag_from_anchor": time_idx - seq_len + 1,
                    "trial_code": int(batch.trial_code[batch_idx]),
                    "session_code": int(batch.session_code[batch_idx]),
                    "replica_id": replica_id,
                    "latent_idx": feature_idx,
                    "activation_value": float(sparse_cpu[batch_idx, time_idx, feature_idx]),
                }
            )
    return rows


def _evaluation_index_rows(
    model: Sed,
    batch: WindowSample,
    replica_id: int,
) -> List[Dict[str, int]]:
    """List source rows on which an activation could have been selected."""

    if model.code_layout == "global":
        source_indices = batch.anchor_index.reshape(-1)
    else:
        required_left, required_right = model.temporal_occurrence_support()
        stop = batch.source_indices.shape[1] - required_right
        source_indices = batch.source_indices[:, required_left:stop]
        occurrence_mask = batch.occurrence_mask[:, required_left:stop]
        source_indices = source_indices[occurrence_mask]
    return [
        {"source_time_idx": int(source_idx), "replica_id": replica_id}
        for source_idx in source_indices
    ]


def evaluate_model(
    model: Sed,
    evaluation_loader: DataLoader,
    loss_cfg: LossConfig,
    *,
    replica_id: int = 0,
    log_wandb: bool = False,
    diagnostics: Optional[EvaluationConfig] = None,
    spectral_loader: Optional[DataLoader] = None,
) -> EvaluationResult:
    """Encode input windows and score their aligned targets at every output lag."""

    loss_cfg.validate_for(model.cfg)
    if len(evaluation_loader) == 0:
        raise ValueError("evaluation_loader contains no valid windows")
    _validate_loader_populations(model, evaluation_loader, loss_cfg)
    largest_level = model.cfg.n_features
    targets = []
    reconstructions = []
    activation_rows: List[Dict[str, object]] = []
    evaluation_index_rows: List[Dict[str, int]] = []
    seen_occurrence_sources = set()
    model.eval()
    with t.no_grad():
        for batch in tqdm(evaluation_loader, desc="SED evaluation batch"):
            batch_index_rows = _evaluation_index_rows(model, batch, replica_id)
            if model.code_layout == "temporal":
                batch_source_list = [row["source_time_idx"] for row in batch_index_rows]
                batch_sources = set(batch_source_list)
                repeated_across_batches = seen_occurrence_sources.intersection(batch_sources)
                if len(batch_source_list) != len(batch_sources) or repeated_across_batches:
                    raise ValueError(
                        "Shift-equivariant evaluation requires each physical occurrence once; "
                        "use validation windows whose valid occurrence supports do not overlap"
                    )
                seen_occurrence_sources.update(batch_sources)
            inputs, target = _batch_populations(batch, model, loss_cfg)
            output = model(inputs)
            reconstruction = output.reconstructions[largest_level]
            targets.append(target.float().cpu())
            reconstructions.append(reconstruction.float().cpu())
            activation_rows.extend(
                _activation_rows(output.sparse_acts[largest_level], batch, replica_id)
            )
            evaluation_index_rows.extend(batch_index_rows)

    target_tensor = t.cat(targets)
    reconstruction_tensor = t.cat(reconstructions)
    weighted_loss, per_timebin_loss = weighted_reconstruction_loss(
        target_tensor,
        reconstruction_tensor,
        loss_cfg,
    )
    metric_rows = []
    seq_len = target_tensor.shape[1]
    for time_idx in range(seq_len):
        target_at_lag = target_tensor[:, time_idx]
        reconstruction_at_lag = reconstruction_tensor[:, time_idx]
        cosine = F.cosine_similarity(reconstruction_at_lag, target_at_lag, dim=-1).mean()
        try:
            r2 = r2_score(
                target_at_lag.numpy(),
                reconstruction_at_lag.numpy(),
                multioutput="variance_weighted",
            )
        except ValueError:
            r2 = float("nan")
        metric_rows.append(
            {
                "lag": time_idx - seq_len + 1,
                "timebin_index": time_idx,
                "reconstruction_loss": float(per_timebin_loss[time_idx]),
                "cosine_similarity": float(cosine),
                "r2": float(r2),
            }
        )

    metrics_by_lag = pd.DataFrame(metric_rows)
    if log_wandb:
        values = {
            f"evaluation/reconstruction/lag_{row['lag']}": row["reconstruction_loss"]
            for row in metric_rows
        }
        values["evaluation/reconstruction/window_weighted"] = float(weighted_loss)
        wandb.log(values)  # pyright: ignore[reportAttributeAccessIssue]
    evaluation_index = pd.DataFrame(evaluation_index_rows).drop_duplicates(ignore_index=True)
    return EvaluationResult(
        reconstructions=reconstruction_tensor,
        targets=target_tensor,
        metrics_by_lag=metrics_by_lag,
        weighted_reconstruction=float(weighted_loss),
        activation_table=pd.DataFrame(activation_rows, columns=ACTIVATION_TABLE_COLUMNS),
        evaluation_index=evaluation_index,
        inference_sparsity=model.cfg.inference_sparsity,
        diagnostics=evaluate_diagnostics(
            model, evaluation_loader, diagnostics, list(loss_cfg.timebin_weights),
            spectral_loader=spectral_loader,
        ) if diagnostics is not None else None,
    )
