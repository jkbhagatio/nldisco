"""Synthetic training integration and isolated orchestration contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from nldisco.model import build_sed
from nldisco.sweep.config import RunConfig, register_configs
from nldisco.sweep.execution import _execute_task, execute
from nldisco.sweep.main import launch
from nldisco.sweep.search import apply_parameters, local_trials, validate_config
from nldisco.sweep.training import model_and_loss, prepare_data, run_trial


@pytest.fixture
def config(tmp_path):
    counts = np.random.default_rng(3).poisson(2, (80, 3)).astype(np.float32)
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    cfg = OmegaConf.structured(RunConfig)
    cfg.data.path = str(path)
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.env_file = None
    cfg.model.seq_len = 4
    cfg.model.dsed_topk_map = {8: 2}
    cfg.training.epochs = 1
    cfg.training.batch_size = 8
    cfg.loss.type = "mse"
    return cfg


def plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def test_hydra_yaml_and_cli_override():
    register_configs()
    with initialize_config_module(version_base="1.3", config_module="nldisco.sweep.conf"):
        cfg = compose(config_name="train", overrides=["training.learning_rate=0.012", "seed=13"])
        assert cfg.training.learning_rate == 0.012
        assert cfg.seed == 13
        with pytest.raises(Exception, match="learning_raet"):
            compose(config_name="train", overrides=["training.learning_raet=0.1"])
        assert compose(config_name="wandb").search.backend == "wandb"
        assert compose(config_name="slurm").execution.backend == "slurm"


@pytest.mark.parametrize("normalization", ["none", "zscore", "minmax"])
def test_synthetic_trial_and_reload(config, tmp_path, normalization):
    config.data.normalization = normalization
    config.model.decoder.output_activation = "none"
    result = run_trial(plain(config), str(tmp_path / "trial"), "cpu")
    assert result["status"] == "completed"
    assert np.isfinite(result["validation/loss"])
    checkpoint = torch.load(tmp_path / "trial/model.pt", weights_only=True)
    saved = OmegaConf.create(checkpoint["config"])
    model_cfg, _ = model_and_loss(saved, checkpoint["n_units"])
    model = build_sed(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    if normalization != "none":
        stats = np.load(tmp_path / "trial/normalization.npz")
        original = np.load(config.data.path)[:64]
        expected = original.mean(0) if normalization == "zscore" else original.min(0)
        np.testing.assert_allclose(stats["offset"], expected)
    with pytest.raises(FileExistsError):
        run_trial(plain(config), str(tmp_path / "trial"), "cpu")


def test_split_normalization_and_invalid_boundaries(config, tmp_path):
    counts = np.tile(np.arange(80, dtype=np.float32)[:, None], (1, 3))
    counts[20] = np.nan
    valid = np.ones(80, dtype=bool)
    valid[20] = False
    np.save(config.data.path, counts)
    np.save(tmp_path / "valid.npy", valid)
    config.data.valid_rows = str(tmp_path / "valid.npy")
    config.data.normalization = "zscore"
    config.model.decoder.output_activation = "none"
    model, _ = model_and_loss(config, 3)
    train, val, stats = prepare_data(config, model)
    np.testing.assert_allclose(stats.offset, counts[:64][valid[:64]].mean(0), rtol=1e-6)
    train_sources = set(torch.cat([sample.source_indices for sample in train.dataset]).tolist())
    val_sources = set(torch.cat([sample.source_indices for sample in val.dataset]).tolist())
    assert not train_sources.intersection(val_sources)
    assert 20 not in train_sources
    assert min(val_sources) >= 64
    assert val.dataset.spike_counts[64:].mean() > 1


@pytest.mark.parametrize("split", ["trial", "session"])
def test_group_splits(config, tmp_path, split):
    groups = np.repeat(np.arange(4), 20)
    np.save(tmp_path / "groups.npy", groups)
    config.data.trial_ids = config.data.session_ids = str(tmp_path / "groups.npy")
    config.data.split = split
    config.data.train_sessions = ["0", "1"]
    model, _ = model_and_loss(config, 3)
    train, val, _ = prepare_data(config, model)
    train_groups = {int(groups[s.source_indices[0]]) for s in train.dataset}
    val_groups = {int(groups[s.source_indices[0]]) for s in val.dataset}
    assert train_groups and val_groups and not train_groups.intersection(val_groups)


def test_temporal_trial(config, tmp_path):
    config.model.encoder.type = "TransformerWindow"
    config.model.encoder.shift_equivariant = True
    config.model.encoder.d_model = 8
    config.model.encoder.n_heads = 2
    config.model.encoder.n_layers = 1
    config.model.encoder.d_feedforward = 16
    config.model.encoder.attention_radius = 1
    config.model.decoder.temporal_kernel_len = 2
    result = run_trial(plain(config), str(tmp_path / "temporal"), "cpu")
    assert result["status"] == "completed"


def test_grid_cap_map_replacement_and_seed(config):
    config.mode = "sweep"
    config.search.parameters = {
        "seed": {"values": [0, 1]},
        "training.learning_rate": {"values": [0.001, 0.01]},
    }
    config.search.max_runs = 3
    trials = local_trials(plain(config))
    assert [(t["seed"], t["training"]["learning_rate"]) for t in trials] == [
        (0, 0.001),
        (0, 0.01),
        (1, 0.001),
    ]
    replaced = apply_parameters(plain(config), {"model.dsed_topk_map": {16: 3}})
    assert replaced["model"]["dsed_topk_map"] == {16: 3}
    assert apply_parameters(plain(config), {"model.dsed_topk_map": {"16": 3}}) == replaced
    with pytest.raises(ValueError, match="Unknown"):
        apply_parameters(plain(config), {"training.typo": 1})
    with pytest.raises(ConfigKeyError):
        apply_parameters(plain(config), {"model.encoder": {"typo": True}})


def test_random_reproducible_bounds(config):
    config.mode = "sweep"
    config.search.method = "random"
    config.search.parameters = {
        "seed": {"distribution": "int_uniform", "min": 1, "max": 9},
        "training.learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
    }
    a = local_trials(plain(config))
    assert a == local_trials(plain(config))
    assert len(a) == config.search.max_runs
    assert all(1 <= x["seed"] <= 9 and 1e-4 <= x["training"]["learning_rate"] <= 1e-2 for x in a)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"execution.max_parallel": 2, "execution.devices": ["cuda:0"]}, "distinct"),
        ({"execution.backend": "slurm", "execution.devices": ["cuda"]}, "resources"),
        ({"data.normalization": "zscore"}, "zscore requires"),
        ({"mode": "sweep", "search.backend": "wandb", "wandb.mode": "offline"}, "online"),
        ({"data.stride": 0}, "positive"),
    ],
)
def test_invalid_config(config, overrides, match):
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match=match):
        validate_config(config)


def test_failed_trial_records_error(config, tmp_path):
    config.data.train_fraction = 0.999
    with pytest.raises(ValueError, match="no valid windows"):
        run_trial(plain(config), str(tmp_path / "failed"), "cpu")
    assert json.loads((tmp_path / "failed/status.json").read_text())["status"] == "failed"


def test_env_precedence_and_no_secret_serialization(config, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"WANDB_API_KEY=secret-from-file\nNLDISCO_DATA={config.data.path}\n")
    monkeypatch.setenv("WANDB_API_KEY", "secret-from-shell")
    monkeypatch.delenv("NLDISCO_DATA", raising=False)
    config.env_file = str(env)
    config.data.path = "${oc.env:NLDISCO_DATA}"
    result = launch(config)
    assert result["status"] == "completed"
    assert os.environ["WANDB_API_KEY"] == "secret-from-shell"
    for path in Path(config.output_dir).rglob("*.yaml"):
        assert "secret-from" not in path.read_text()


def test_local_training_sweep(config):
    config.mode = "sweep"
    config.search.parameters = {"seed": {"values": [3, 4]}}
    result = launch(config)
    assert len(result["results"]) == 2
    assert all(r["status"] == "completed" for r in result["results"])


def test_worker_failure_propagates_to_launcher(config):
    config.data.train_fraction = 0.999
    with pytest.raises(RuntimeError, match="workers failed"):
        launch(config)
    summary = next(Path(config.output_dir).rglob("summary.json"))
    assert json.loads(summary.read_text())["status"] == "failed"


def test_submitit_resources_and_no_wait(config, tmp_path, monkeypatch):
    fake = MagicMock()
    fake.map_array.return_value = [
        SimpleNamespace(job_id="123_0"),
        SimpleNamespace(job_id="123_1"),
    ]
    factory = MagicMock(return_value=fake)
    monkeypatch.setattr("submitit.AutoExecutor", factory)
    config.execution.backend = "slurm"
    config.execution.devices = ["cuda"]
    config.execution.slurm.gpus_per_node = 1
    config.execution.slurm.partition = "test"
    config.execution.max_parallel = 2
    config.execution.wait = False
    tasks = [{"kind": "trial"}, {"kind": "trial"}]
    result = execute(tasks, plain(config), str(tmp_path))
    assert result["job_ids"] == ["123_0", "123_1"]
    assert factory.call_args.kwargs["cluster"] == "slurm"
    params = fake.update_parameters.call_args.kwargs
    assert params["slurm_array_parallelism"] == 2
    assert params["gpus_per_node"] == 1
    assert params["slurm_partition"] == "test"
    assert all(t["device"] == "cuda:0" for t in tasks)


def test_wandb_sweep_agent_quotas(config, monkeypatch):
    config.mode = "sweep"
    config.search.backend = "wandb"
    config.search.max_runs = 5
    config.execution.max_parallel = 3
    config.search.parameters = {"seed": {"values": [0, 1, 2, 3, 4]}}
    create = MagicMock(return_value="abc123")
    dispatch = MagicMock(return_value={"status": "submitted"})
    monkeypatch.setattr("wandb.sweep", create)
    monkeypatch.setattr("nldisco.sweep.main.execute", dispatch)
    launch(config)
    assert create.call_args.args[0]["run_cap"] == 5
    tasks = dispatch.call_args.args[0]
    assert [t["count"] for t in tasks] == [2, 2, 1]
    assert all(t["sweep_id"] == "abc123" for t in tasks)
    config.wandb.sweep_id = "existing"
    launch(config)
    assert create.call_count == 1


def test_wandb_agent_runs_real_training(config, tmp_path, monkeypatch):
    fake_run = MagicMock()
    fake_run.id = "mock-run"
    fake_run.config = {"training.learning_rate": 0.002, "seed": 17}
    fake_run.__enter__.return_value = fake_run
    monkeypatch.setattr("wandb.init", MagicMock(return_value=fake_run))
    log = MagicMock()
    monkeypatch.setattr("wandb.log", log)
    monkeypatch.setattr("wandb.agent", lambda sweep_id, function, **kwargs: function())
    result = _execute_task(
        {
            "kind": "agent",
            "config": plain(config),
            "sweep_id": "mock",
            "count": 1,
            "device": "cpu",
            "output_dir": str(tmp_path / "agent"),
        }
    )
    assert result["status"] == "completed"
    saved = OmegaConf.load(tmp_path / "agent/mock-run/config.yaml")
    assert saved.seed == 17 and saved.training.learning_rate == 0.002
    assert any("validation/loss" in c.args[0] for c in log.call_args_list)


def test_wandb_init_failure_is_not_reported_as_success(config, tmp_path, monkeypatch):
    monkeypatch.setattr("wandb.init", MagicMock(side_effect=RuntimeError("failed init")))

    def swallow_callback_error(sweep_id, function, **kwargs):
        try:
            function()
        except RuntimeError:
            pass

    monkeypatch.setattr("wandb.agent", swallow_callback_error)
    result = _execute_task(
        {
            "kind": "agent",
            "config": plain(config),
            "sweep_id": "mock",
            "count": 1,
            "device": "cpu",
            "output_dir": str(tmp_path / "failed-agent"),
        }
    )
    assert result["status"] == "failed"


def test_external_yaml_cli_and_parallel_workers(config, tmp_path):
    config.mode = "sweep"
    config.search.parameters = {"seed": {"values": [1, 2]}}
    config.execution.max_parallel = 2
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text("defaults:\n  - nldisco_schema\n  - _self_\n" + OmegaConf.to_yaml(config))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nldisco.sweep",
            "--config-dir",
            str(tmp_path),
            "--config-name",
            "custom",
            "training.learning_rate=0.002",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    checkpoints = list(Path(config.output_dir).rglob("model.pt"))
    assert len(checkpoints) == 2
    assert all(
        torch.load(p, weights_only=True)["config"]["training"]["learning_rate"] == 0.002
        for p in checkpoints
    )
