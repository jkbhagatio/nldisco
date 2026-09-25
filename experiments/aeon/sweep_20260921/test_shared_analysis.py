"""Small exact checks for full-recording descriptive statistics."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "aeon_shared_analysis", Path(__file__).with_name("shared_analysis.py")
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_selectivity_is_not_precision_and_inactive_is_undefined():
    active = np.array([True, False, True, False, False, False])
    condition = np.array([True, True, False, False, False, False])
    valid = np.ones(6, dtype=bool)
    result = m.binary_scores(active, condition, valid)
    assert result["selectivity"] == 2 / 3
    assert result["precision"] == 0.5
    assert np.isnan(m.binary_scores(active & False, condition, valid)["selectivity"])


def test_support_and_bouts_respect_gaps():
    assert np.array_equal(
        m.complete_support(np.array([True, True, False, True, True]), 2),
        [False, True, False, False, True],
    )
    assert np.array_equal(
        m.bout_starts(np.ones(4, dtype=bool), np.array([19, 20, 24, 25])), [0, 2]
    )
    assert np.array_equal(
        m.separated_bout_starts(np.ones(4, dtype=bool), np.array([19, 20, 24, 25])), [0, 2]
    )
    assert np.array_equal(
        m.centered_support(np.array([True, True, True, False, True]), 1),
        [False, True, False, False, False],
    )


def test_auc_ties_and_sparse_screen():
    a = np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    condition = np.array([False, True, False, True])
    assert m.continuous_auc(a[:, 0], condition) == 1.0
    assert m.continuous_auc(np.zeros(4), condition) == 0.5
    c = m.Condition("x", "x", condition, np.ones(4, dtype=bool), "test")
    rows = m.screen(a, [c]).set_index("latent")
    assert rows.loc[0, "selectivity"] == 1.0
    assert rows.loc[1, "selectivity"] == 0.5


def test_full_analysis_artifacts(tmp_path):
    """Exercise the actual candidate/figure/notebook path on known selective signals."""
    n = 4000
    rng = np.random.default_rng(1)
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    behavior = pd.DataFrame(
        {
            field: rng.normal(size=n)
            for field in (
                "x",
                "y",
                "area",
                "speed",
                "acceleration",
                "angular_velocity",
                "patch_distance",
                "radial_velocity",
                "heading",
                "wheel_speed",
                "wheel_acceleration",
            )
        }
    )
    behavior["speed"] = np.abs(behavior.speed)
    behavior["patch_distance"] = np.abs(behavior.patch_distance)
    behavior["camera_valid"] = True
    behavior["wheel_valid"] = True
    behavior.to_parquet(data / "behavior.parquet")
    pd.DataFrame({"bin_index": np.arange(n), "maintenance": np.arange(n) < 1000}).to_parquet(
        data / "environment_context.parquet"
    )
    endpoints = np.arange(19, n)
    x = behavior.x.to_numpy()[endpoints]
    activations = np.stack(
        [np.maximum(x, 0), np.maximum(-x, 0), np.maximum(x - 1, 0)], axis=1
    ).astype(np.float32)
    np.save(run / "activations.npy", activations)
    np.save(run / "endpoint_bins.npy", endpoints)
    np.save(run / "decoder_kernels.npy", rng.normal(size=(3, 4, 20)))
    np.save(data / "counts.npy", rng.poisson(0.1, size=(n, 4)).astype(np.uint16))
    np.save(data / "neural_valid.npy", np.ones(n, dtype=bool))
    m.analyze(data, run)
    result = pd.read_csv(run / "analysis/candidate_audit.csv")
    assert result.native_continuous_auroc.max() > 0.99
    assert "active_maintenance_fraction" in result
    assert "no_maintenance_native_auroc" in result
    assert (run / "analysis/analysis.ipynb").exists()
    assert list((run / "analysis").glob("latent*_context.png"))
