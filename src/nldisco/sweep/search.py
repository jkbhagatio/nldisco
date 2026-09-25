"""Validated parameter spaces for local and W&B search."""

import itertools
import math
import random
from dataclasses import is_dataclass
from typing import Any

from beartype import beartype
from omegaconf import DictConfig, OmegaConf

from nldisco.config import TrainConfig
from nldisco.evaluation import validate_diagnostics
from nldisco.sweep.config import RunConfig
from nldisco.sweep.training import diagnostic_config, model_and_loss

METRIC = "validation/loss"


@beartype
def validate_config(cfg: DictConfig) -> None:
    """Fail before scheduling work for unsupported modes or resource settings."""
    choices = {
        "mode": {"train", "sweep"},
        "data.normalization": {"none", "zscore", "minmax"},
        "data.split": {"chronological", "trial", "session"},
        "execution.backend": {"local", "slurm"},
        "search.backend": {"local", "wandb"},
        "search.method": {"grid", "random", "bayes"},
        "wandb.mode": {"online", "offline", "disabled"},
    }
    for key, allowed in choices.items():
        if OmegaConf.select(cfg, key) not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)}")
    target_normalization = cfg.data.get("target_normalization")
    if target_normalization not in {None, "none", "zscore", "minmax"}:
        raise ValueError("data.target_normalization must be none, zscore, minmax, or null")
    if not cfg.data.get("target_path") and any(
        cfg.data.get(key) is not None
        for key in ("target_valid_rows", "target_trial_ids", "target_session_ids", "target_timestamps")
    ):
        raise ValueError("Target row metadata requires data.target_path")
    if not 0 < cfg.data.train_fraction < 1:
        raise ValueError("data.train_fraction must be between 0 and 1")
    if cfg.data.stride is not None and cfg.data.stride < 1:
        raise ValueError("data.stride must be positive")
    if (cfg.data.timestamps or cfg.data.get("target_timestamps")) and (
        cfg.data.expected_bin_size is None or cfg.data.expected_bin_size <= 0
    ):
        raise ValueError("Timestamps require a positive data.expected_bin_size")
    if not 0 <= cfg.seed < 2**32 or not 0 <= cfg.data.split_seed < 2**32:
        raise ValueError("Training and split seeds must be in [0, 2**32)")
    execution = cfg.execution
    if execution.max_parallel < 1 or execution.threads_per_worker < 1:
        raise ValueError("Worker concurrency and thread counts must be positive")
    if not execution.devices:
        raise ValueError("execution.devices must contain at least one device")
    for device in execution.devices:
        if (
            device != "cpu"
            and device != "cuda"
            and not (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("Devices must be cpu, cuda, or cuda:<index>")
    if execution.backend == "local":
        slots = [
            execution.devices[i % len(execution.devices)] for i in range(execution.max_parallel)
        ]
        gpu_slots = ["cuda:0" if d == "cuda" else d for d in slots if d != "cpu"]
        if len(gpu_slots) != len(set(gpu_slots)):
            raise ValueError("Each concurrent GPU worker needs a distinct device")
    else:
        slurm = execution.slurm
        if min(slurm.timeout_min, slurm.cpus_per_task, slurm.mem_gb) <= 0:
            raise ValueError("Slurm time, CPUs and memory must be positive")
        if slurm.gpus_per_node not in {0, 1}:
            raise ValueError("Each Slurm worker supports zero or one GPU")
        if any(d != "cpu" for d in execution.devices) != bool(slurm.gpus_per_node):
            raise ValueError("Slurm GPU resources must match execution.devices")
        if len(execution.devices) != 1 or execution.devices[0] not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("Slurm devices must be [cpu] or [cuda]; Slurm assigns physical GPUs")
    if cfg.mode == "sweep":
        if cfg.search.max_runs < 1:
            raise ValueError("search.max_runs must be positive")
        if cfg.search.backend == "local" and cfg.search.method == "bayes":
            raise ValueError("Bayesian search requires search.backend=wandb")
        if cfg.search.backend == "wandb" and cfg.wandb.mode != "online":
            raise ValueError("W&B sweeps require wandb.mode=online")
        if not cfg.search.parameters:
            raise ValueError("A sweep requires search.parameters (include seed to sweep replicas)")
        validate_parameters(cfg)
    model, _ = model_and_loss(cfg, 1)
    validate_diagnostics(diagnostic_config(cfg), model)
    TrainConfig(**OmegaConf.to_container(cfg.training))


@beartype
def apply_parameters(config: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted paths strictly; dictionary-valued parameters replace maps."""
    cfg = OmegaConf.merge(OmegaConf.structured(RunConfig), config)
    for key, value in parameters.items():
        if key != "seed" and key.split(".")[0] not in {"model", "training", "loss"}:
            raise ValueError(f"Unsupported sweep parameter: {key}")
        current = OmegaConf.select(cfg, key, default="__absent__")
        if current == "__absent__":
            raise ValueError(f"Unknown sweep parameter: {key}")
        if key in {"model.dsed_topk_map", "loss.dsed_loss_weight_map"} and isinstance(value, dict):
            # W&B assignments arrive through JSON, which stringifies integer keys.
            value = {int(k): v for k, v in value.items()}
        elif isinstance(current, DictConfig) and is_dataclass(OmegaConf.get_type(current)):
            value = OmegaConf.merge(current, value)
        OmegaConf.update(cfg, key, value, merge=False)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


@beartype
def validate_parameters(cfg: DictConfig) -> None:
    """Validate the common discrete/uniform search vocabulary before submission."""
    base = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    for key, spec in base["search"]["parameters"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"{key}: expected a parameter specification mapping")
        if set(spec) == {"values"}:
            if not isinstance(spec["values"], list) or not spec["values"]:
                raise ValueError(f"{key}: values must be a nonempty list")
            samples = spec["values"]
        elif set(spec) == {"value"}:
            samples = [spec["value"]]
        elif set(spec) == {"distribution", "min", "max"}:
            kind, low, high = spec["distribution"], spec["min"], spec["max"]
            if cfg.search.method == "grid":
                raise ValueError("Grid search requires discrete values")
            if kind not in {"uniform", "log_uniform_values", "int_uniform"}:
                raise ValueError(f"Unsupported distribution: {kind}")
            if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (low, high)):
                raise ValueError("Distribution bounds must be finite numbers")
            if low >= high or (kind == "log_uniform_values" and low <= 0):
                raise ValueError("Distribution bounds must increase (and be positive for log)")
            if kind == "int_uniform" and any(type(v) is not int for v in (low, high)):
                raise ValueError("int_uniform requires integer bounds")
            samples = [low, high]
        else:
            raise ValueError(f"Invalid parameter specification for {key}")
        for value in samples:
            apply_parameters(base, {key: value})


@beartype
def local_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate a capped deterministic grid or seeded random trials."""
    cfg = OmegaConf.create(config)
    validate_config(cfg)
    parameters = config["search"]["parameters"]
    keys = list(parameters)
    rng = random.Random(cfg.search.seed)

    def sample(spec):
        if "values" in spec:
            return rng.choice(spec["values"])
        if "value" in spec:
            return spec["value"]
        low, high = spec["min"], spec["max"]
        if spec["distribution"] == "int_uniform":
            return rng.randint(low, high)
        if spec["distribution"] == "log_uniform_values":
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)

    if cfg.search.method == "grid":
        combinations = itertools.islice(
            itertools.product(
                *[
                    spec["values"] if "values" in spec else [spec["value"]]
                    for spec in parameters.values()
                ]
            ),
            cfg.search.max_runs,
        )
    else:
        combinations = (
            tuple(sample(spec) for spec in parameters.values()) for _ in range(cfg.search.max_runs)
        )
    trials = [apply_parameters(config, dict(zip(keys, values))) for values in combinations]
    for trial in trials:
        validate_config(OmegaConf.create(trial))
    return trials


@beartype
def wandb_sweep_config(config: dict[str, Any]) -> dict[str, Any]:
    """Use a server-side cap in addition to per-agent quotas."""
    return {
        "method": config["search"]["method"],
        "metric": {"name": METRIC, "goal": "minimize"},
        "parameters": config["search"]["parameters"],
        "run_cap": config["search"]["max_runs"],
    }
