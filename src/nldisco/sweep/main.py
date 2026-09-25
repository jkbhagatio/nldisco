"""Hydra CLI and reusable launch function."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
from beartype import beartype
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from nldisco.sweep.config import DATA_PATH_FIELDS, RunConfig, register_configs
from nldisco.sweep.execution import execute
from nldisco.sweep.search import local_trials, validate_config, wandb_sweep_config

register_configs()
logger = logging.getLogger(__name__)


@beartype
def launch(cfg: DictConfig) -> dict[str, Any]:
    """Resolve shared config, reserve an output directory and dispatch work.

    Relative paths are interpreted relative to the original invocation directory,
    including when Hydra changes working directory. The .env file is read before
    resolving environment interpolations; existing environment values win.
    """
    cfg = OmegaConf.merge(OmegaConf.structured(RunConfig), cfg)
    if cfg.env_file:
        load_dotenv(to_absolute_path(cfg.env_file), override=False)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    for key in DATA_PATH_FIELDS:
        if config["data"][key]:
            config["data"][key] = to_absolute_path(config["data"][key])
    if config["env_file"]:
        config["env_file"] = to_absolute_path(config["env_file"])
    config["output_dir"] = to_absolute_path(config["output_dir"])
    cfg = OmegaConf.create(config)
    validate_config(cfg)
    # Validate input paths before starting a remote sweep or creating Slurm jobs.
    for key in DATA_PATH_FIELDS:
        if config["data"][key] and not Path(config["data"][key]).is_file():
            raise FileNotFoundError(config["data"][key])
    trials = None
    if cfg.mode == "train":
        trials = [config]
    elif cfg.search.backend == "local":
        trials = local_trials(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = Path(config["output_dir"]) / f"{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    logger.info("Outputs: %s", output)
    if trials is not None:
        tasks = [
            {"kind": "trial", "config": trial, "output_dir": str(output / f"run-{i:04d}")}
            for i, trial in enumerate(trials)
        ]
    else:
        import wandb

        sweep_spec = wandb_sweep_config(config)
        sweep_id = cfg.wandb.sweep_id or wandb.sweep(
            sweep_spec,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
        )
        (output / "sweep.json").write_text(
            json.dumps(
                {
                    "sweep_id": sweep_id,
                    "specification": sweep_spec,
                    "existing_sweep": cfg.wandb.sweep_id is not None,
                },
                indent=2,
            )
        )
        n_agents = min(cfg.execution.max_parallel, cfg.search.max_runs)
        quotient, remainder = divmod(cfg.search.max_runs, n_agents)
        tasks = [
            {
                "kind": "agent",
                "config": config,
                "sweep_id": sweep_id,
                "count": quotient + (i < remainder),
                "output_dir": str(output / f"agent-{i:04d}"),
            }
            for i in range(n_agents)
        ]
    return execute(tasks, config, str(output))


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def train_main(cfg: DictConfig) -> None:
    """Train once, with Hydra YAML composition and CLI overrides."""
    launch(cfg)


@hydra.main(version_base="1.3", config_path="conf", config_name="sweep")
def sweep_main(cfg: DictConfig) -> None:
    """Run a local or W&B search using the shared training configuration."""
    launch(cfg)
