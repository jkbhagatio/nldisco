"""One reproducible training trial, shared by every execution backend."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data.preprocessing import NormalizationStats, fit_normalizer
from nldisco.data.window import SpikeWindowDataset
from nldisco.evaluation import (
    AblationConfig,
    EvaluationConfig,
    SpectralConfig,
    save_diagnostics,
    validate_diagnostics,
)
from nldisco.model import Sed, build_sed
from nldisco.train import evaluate_model, split_rows_by_session, split_rows_by_trial_proportion
from nldisco.train import train_model as fit_model


@beartype
def diagnostic_config(cfg: DictConfig) -> EvaluationConfig:
    """Resolve optional diagnostic settings, including a shared recording bin size."""
    values = OmegaConf.to_container(cfg.evaluation) if "evaluation" in cfg else {}
    result = EvaluationConfig(
        ablation=AblationConfig(**values.get("ablation", {})),
        spectral=SpectralConfig(**values.get("spectral", {})),
    )
    if result.spectral.bin_size is None:
        result.spectral.bin_size = cfg.data.expected_bin_size
    elif result.spectral.enabled and cfg.data.expected_bin_size is not None and not np.isclose(
        result.spectral.bin_size, cfg.data.expected_bin_size
    ):
        raise ValueError("Spectral bin_size disagrees with data.expected_bin_size")
    return result


@beartype
def spectral_validation_loader(loader: DataLoader) -> DataLoader:
    """Dense windows on the same validation rows; every window uses complete context.

    Occurrences may recur in overlapping windows here: only one fixed reconstruction
    lag is exported. This loader is exclusively for spectral diagnostics.
    """
    source = loader.dataset
    dataset = SpikeWindowDataset(
        source.spike_counts, source.seq_len, stride=1, source_indices=source.source_indices,
        targets=source.targets,
        trial_ids=source.trial_codes, session_ids=source.session_codes,
        timestamps=source.timestamps, expected_bin_size=source.expected_bin_size,
        allowed_rows=source.allowed_rows, valid_rows=source.valid_rows,
    )
    return DataLoader(dataset, batch_size=loader.batch_size, shuffle=False)


@beartype
def model_and_loss(
    cfg: DictConfig, n_units: int, n_output_units: Optional[int] = None
) -> tuple[SedConfig, LossConfig]:
    """Translate the serializable schema into the library's runtime configs."""
    values = OmegaConf.to_container(cfg.model, resolve=True)
    dtype = values.pop("dtype")
    if dtype not in {"float32", "float64", "bfloat16"}:
        raise ValueError("model.dtype must be float32, float64, or bfloat16")
    model = SedConfig(
        n_neurons=n_units,
        n_output_neurons=n_output_units,
        encoder=EncoderConfig(**values.pop("encoder")),
        decoder=DecoderConfig(**values.pop("decoder")),
        dtype=getattr(torch, dtype),
        **values,
    )
    loss_values = OmegaConf.to_container(cfg.loss, resolve=True)
    loss_values["timebin_weights"] = loss_values["timebin_weights"] or [1.0] * model.seq_len
    loss = LossConfig(**loss_values)
    loss.validate_for(model)
    if _target_normalization(cfg) == "zscore" and (
        loss.type != "mse" or model.decoder.output_activation != "none"
    ):
        raise ValueError(
            "Target zscore requires loss.type=mse and model.decoder.output_activation=none"
        )
    return model, loss


def _target_normalization(cfg: DictConfig) -> str:
    method = cfg.data.get("target_normalization")
    if method not in {None, "none", "zscore", "minmax"}:
        raise ValueError("data.target_normalization must be none, zscore, minmax, or null")
    if cfg.data.get("target_path"):
        return "none" if method is None else method
    if method is not None and method != cfg.data.normalization:
        raise ValueError("Separate target normalization requires data.target_path")
    return cfg.data.normalization


@beartype
def prepare_data(
    cfg: DictConfig, model: SedConfig, *, return_target_stats: bool = False
) -> Union[
    tuple[DataLoader, DataLoader, Optional[NormalizationStats]],
    tuple[DataLoader, DataLoader, Optional[NormalizationStats], Optional[NormalizationStats]],
]:
    """Split rows first, fit training statistics, then construct safe windows.

    Invalid rows retain their original positions but are excluded from both
    partitions. Metadata arrays must use NumPy's non-object dtypes. The default
    return preserves the original three-tuple with input normalization statistics;
    ``return_target_stats=True`` appends the target normalization statistics.
    """
    counts = np.load(cfg.data.path, mmap_mode="r", allow_pickle=False)
    if counts.ndim != 2 or min(counts.shape) < 1 or counts.dtype.kind not in "fiu":
        raise ValueError("data.path must contain numeric [timebin, unit] activity")
    if counts.shape[1] != model.n_neurons:
        raise ValueError("Input activity width must match model.n_neurons")
    n_rows = len(counts)
    target_path = cfg.data.get("target_path")
    targets = counts if target_path is None else np.load(
        target_path, mmap_mode="r", allow_pickle=False
    )
    if (
        targets.ndim != 2 or min(targets.shape) < 1 or targets.dtype.kind not in "fiu"
        or len(targets) != n_rows
    ):
        raise ValueError("data.target_path must contain numeric [timebin, unit] activity aligned to inputs")
    if targets.shape[1] != model.n_output_neurons:
        raise ValueError("Target activity width must match model.n_output_neurons")
    target_method = _target_normalization(cfg)

    def metadata(name):
        path = cfg.data.get(name)
        values = None if path is None else np.load(path, allow_pickle=False)
        if values is not None and values.shape != (n_rows,):
            raise ValueError(f"data.{name} must have shape [timebin]")
        return values

    valid = np.ones(n_rows, dtype=bool)
    for name in ("valid_rows", "target_valid_rows"):
        rows = metadata(name)
        if rows is not None:
            if rows.dtype != np.bool_:
                raise ValueError(f"data.{name} must contain booleans")
            valid &= rows

    shared = {}
    for name in ("trial_ids", "session_ids", "timestamps"):
        source, target = metadata(name), metadata(f"target_{name}")
        for values in (source, target):
            if values is not None:
                if name == "timestamps":
                    if values.dtype.kind not in "fiu":
                        raise ValueError(f"data.{name} must contain numeric timestamps")
                    valid &= np.isfinite(values)
                else:
                    valid &= ~pd.isna(values)
        if source is not None and target is not None:
            present = ~pd.isna(source) & ~pd.isna(target)
            aligned = source[present] == target[present]
            if not np.all(aligned):
                raise ValueError(f"data.target_{name} must align with input {name}")
        shared[name] = source if source is not None else target
    trials, sessions, timestamps = (shared[k] for k in ("trial_ids", "session_ids", "timestamps"))
    if cfg.data.split == "chronological":
        cut = int(n_rows * cfg.data.train_fraction)
        train_rows = np.arange(n_rows) < cut
        val_rows = ~train_rows
    elif cfg.data.split == "trial":
        if trials is None:
            raise ValueError("A trial split requires data.trial_ids")
        train_rows, val_rows = split_rows_by_trial_proportion(
            trials, sessions, cfg.data.train_fraction, seed=cfg.data.split_seed
        )
    else:
        if sessions is None:
            raise ValueError("A session split requires data.session_ids")
        train_rows, val_rows = split_rows_by_session(
            sessions.astype(str), list(cfg.data.train_sessions)
        )
    train_rows &= valid
    val_rows &= valid
    if not train_rows.any() or not val_rows.any():
        raise ValueError("The split must leave valid training and validation rows")

    def preprocess(data, method, *, is_target):
        stats = None if method == "none" else fit_normalizer(
            np.asarray(data[train_rows]), method
        )
        # CPU storage is shared between train/validation views; GPU copies are per batch.
        activity = np.zeros(
            data.shape, dtype=np.float64 if cfg.model.dtype == "float64" else np.float32
        )
        for start in range(0, n_rows, 100_000):
            stop = min(start + 100_000, n_rows)
            mask = valid[start:stop]
            if not mask.any():
                continue
            block = np.asarray(data[start:stop][mask])
            if not np.isfinite(block).all():
                raise ValueError("Valid activity rows must be finite")
            values = block if stats is None else stats.transform(block)
            if is_target and np.any(values < 0):
                if cfg.loss.type == "msle":
                    raise ValueError("MSLE requires nonnegative targets; use MSE for signed targets")
                if model.decoder.output_activation != "none":
                    raise ValueError("Signed targets require model.decoder.output_activation=none")
            activity[start:stop][mask] = values
        if not np.isfinite(activity).all():
            raise ValueError("Activity overflows the configured dtype after preprocessing")
        return torch.from_numpy(activity), stats

    tensor, stats = preprocess(counts, cfg.data.normalization, is_target=target_path is None)
    target_tensor, target_stats = (tensor, stats) if target_path is None else preprocess(
        targets, target_method, is_target=True
    )
    support = model.temporal_occurrence_support() if model.encoder.shift_equivariant else None
    stride = cfg.data.stride or (
        model.usable_occurrence_positions if support is not None else model.seq_len
    )
    loaders = []
    for mask, shuffle in ((train_rows, True), (val_rows, False)):
        dataset = SpikeWindowDataset(
            tensor,
            model.seq_len,
            stride,
            targets=target_tensor,
            trial_ids=trials,
            session_ids=sessions,
            timestamps=timestamps,
            expected_bin_size=cfg.data.expected_bin_size,
            occurrence_support=support,
            allowed_rows=mask,
        )
        if not len(dataset):
            raise ValueError("The split/window settings leave a partition with no valid windows")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=cfg.training.batch_size,
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(cfg.seed),
            )
        )
    if return_target_stats:
        return loaders[0], loaders[1], stats, target_stats
    return loaders[0], loaders[1], stats


@beartype
@dataclass(frozen=True)
class LoadedCheckpoint:
    """An inference model with its independent input/target preprocessing states."""

    model: Sed
    config: DictConfig
    input_normalization: Optional[NormalizationStats]
    target_normalization: Optional[NormalizationStats]


def _normalization_state(stats: Optional[NormalizationStats]) -> Optional[dict[str, Any]]:
    if stats is None:
        return None
    return {
        "method": stats.method,
        "offset": torch.from_numpy(stats.offset),
        "scale": torch.from_numpy(stats.scale),
    }


def _restore_normalization(
    state: Optional[dict[str, Any]], method: str, n_units: int
) -> Optional[NormalizationStats]:
    if state is None:
        if method != "none":
            raise ValueError(f"Checkpoint is missing {method} normalization statistics")
        return None
    offset, scale = (np.asarray(state[key], dtype=np.float64) for key in ("offset", "scale"))
    if (
        str(state["method"]) != method or method not in {"zscore", "minmax"}
        or offset.shape != (n_units,) or scale.shape != (n_units,)
        or not np.isfinite(offset).all() or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Checkpoint normalization statistics do not match the model and config")
    return NormalizationStats(method=method, offset=offset, scale=scale)


@beartype
def load_checkpoint(path: Union[str, Path], device: str = "cpu") -> LoadedCheckpoint:
    """Reload an inference checkpoint without requiring input or target data files.

    New checkpoints embed both normalization states and dimensions. Older
    autoencoder checkpoints use ``n_units`` and, if needed, the adjacent
    ``normalization.npz`` file. The restored model is in evaluation mode.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(checkpoint["config"])
    n_input = checkpoint.get("n_input_units", checkpoint.get("n_units"))
    n_output = checkpoint.get("n_output_units", n_input)
    model_cfg, _ = model_and_loss(cfg, n_input, n_output)
    if "normalization" in checkpoint:
        states = checkpoint["normalization"]
        input_state, target_state = states["input"], states["target"]
    else:
        input_state = None
        if cfg.data.normalization != "none":
            with np.load(path.with_name("normalization.npz"), allow_pickle=False) as saved:
                input_state = {key: saved[key] for key in ("method", "offset", "scale")}
        target_state = input_state
    input_stats = _restore_normalization(input_state, cfg.data.normalization, n_input)
    target_stats = _restore_normalization(target_state, _target_normalization(cfg), n_output)
    model = build_sed(model_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return LoadedCheckpoint(model, cfg, input_stats, target_stats)


@beartype
def run_trial(config: dict[str, Any], output_dir: str, device: str) -> dict[str, Any]:
    """Train and evaluate once, saving config, calibrated weights and metrics.

    The caller owns any W&B run lifecycle. Output directories must be new;
    failed runs retain a status file and their resolved configuration.
    """
    cfg = OmegaConf.create(config)
    if cfg.env_file:
        load_dotenv(cfg.env_file, override=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    try:
        torch.set_num_threads(cfg.execution.threads_per_worker)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device {device}, but CUDA is unavailable")
        shape = np.load(cfg.data.path, mmap_mode="r", allow_pickle=False).shape
        if len(shape) != 2:
            raise ValueError("Activity must have shape [timebin, unit]")
        target_shape = shape if not cfg.data.get("target_path") else np.load(
            cfg.data.target_path, mmap_mode="r", allow_pickle=False
        ).shape
        if len(target_shape) != 2 or target_shape[0] != shape[0]:
            raise ValueError("Target activity must have shape [timebin, unit] aligned to inputs")
        model_cfg, loss_cfg = model_and_loss(cfg, shape[1], target_shape[1])
        diagnostics = diagnostic_config(cfg)
        validate_diagnostics(diagnostics, model_cfg)
        train_loader, validation_loader, stats, target_stats = prepare_data(
            cfg, model_cfg, return_target_stats=True
        )
        for name, normalizer in (("normalization", stats), ("target_normalization", target_stats)):
            if normalizer is not None:
                np.savez(
                    output / f"{name}.npz",
                    method=normalizer.method,
                    offset=normalizer.offset,
                    scale=normalizer.scale,
                )
        model = build_sed(model_cfg).to(device)
        training = fit_model(
            model,
            train_loader,
            loss_cfg,
            TrainConfig(**OmegaConf.to_container(cfg.training)),
            log_wandb=cfg.wandb.enabled,
        )
        spectral_loader = spectral_validation_loader(validation_loader) if diagnostics.spectral.enabled else None
        evaluation = evaluate_model(
            model, validation_loader, loss_cfg, replica_id=cfg.seed,
            diagnostics=diagnostics if diagnostics.ablation.enabled or diagnostics.spectral.enabled else None,
            spectral_loader=spectral_loader,
        )
        metric = evaluation.weighted_reconstruction
        if not np.isfinite(metric):
            raise ValueError("Validation reconstruction loss is not finite")
        metrics = {
            "validation/loss": metric,
            "train/final_loss": list(training.loss.values())[-1],
            "train/windows": len(train_loader.dataset),
            "validation/windows": len(validation_loader.dataset),
        }
        if evaluation.diagnostics is not None:
            summaries = save_diagnostics(evaluation.diagnostics, output / "diagnostics")
            metrics.update({f"validation/{key}": value for key, value in summaries.items()})
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "config": config,
                "n_units": shape[1],
                "n_input_units": shape[1],
                "n_output_units": target_shape[1],
                "normalization": {
                    "input": _normalization_state(stats),
                    "target": _normalization_state(target_stats),
                },
            },
            output / "model.pt",
        )
        (output / "history.json").write_text(json.dumps(asdict(training), indent=2))
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        evaluation.metrics_by_lag.to_csv(output / "metrics_by_lag.csv", index=False)
        if cfg.wandb.enabled:
            import wandb

            wandb.log(metrics)
            if evaluation.diagnostics is not None:
                diagnostic = evaluation.diagnostics
                if diagnostic.ablation is not None:
                    wandb.log({"evaluation/latent_ablation": wandb.Table(dataframe=diagnostic.ablation)})
                if diagnostic.spectral is not None:
                    wandb.log({"evaluation/spectral_metrics": wandb.Table(dataframe=diagnostic.spectral.metrics)})
        result = {"status": "completed", "output_dir": str(output), "device": device, **metrics}
    except Exception as error:
        (output / "status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
        )
        raise
    (output / "status.json").write_text(json.dumps(result, indent=2))
    return result
