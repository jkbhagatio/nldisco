"""Tests for reproducible synthetic position-feature selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.synthetic.scripts.feature_selection import (
    SelectionConfig,
    best_features,
    compute_tuning_curves,
    ground_truth_curves,
    score_tuning_curves,
)


def test_compute_tuning_curves_includes_inactive_eligible_rows() -> None:
    positions = np.array([0.1, 0.1, 0.9, 0.9])
    evaluation_index = pd.DataFrame({"source_time_idx": np.arange(4)})
    activations = pd.DataFrame(
        {
            "source_time_idx": [0, 2],
            "latent_idx": [0, 1],
            "activation_value": [2.0, 4.0],
        }
    )
    config = SelectionConfig(n_position_bins=2)

    curves, centers = compute_tuning_curves(
        activations, evaluation_index, positions, n_features=2, config=config
    )

    np.testing.assert_allclose(centers, [0.25, 0.75])
    np.testing.assert_allclose(curves, [[1.0, 0.0], [0.0, 2.0]])


def test_exact_asymmetric_templates_select_all_four_cells() -> None:
    config = SelectionConfig()
    centers = np.linspace(0.0125, 0.9875, config.n_position_bins)
    from experiments.synthetic.scripts.simulate import place_rates

    rates = place_rates(centers)
    expected = rates / rates.max(axis=1, keepdims=True)
    np.testing.assert_allclose(ground_truth_curves(centers, config), expected)
    scores = score_tuning_curves(expected, centers, config)
    selected = best_features(scores)

    for unit in range(4):
        assert selected[f"cell_{unit}"] == {"latent_idx": unit, "hit": True, "score": 1.0}


def test_best_passing_feature_precedes_higher_scoring_failure() -> None:
    scores = pd.DataFrame({"latent_idx": [0, 1]})
    for unit in range(4):
        scores[f"cell_{unit}_score"] = [0.99, 0.8]
        scores[f"cell_{unit}_hit"] = [False, unit < 3]
    selected = best_features(scores)
    for unit in range(3):
        assert selected[f"cell_{unit}"]["latent_idx"] == 1
    assert selected["cell_3"] == {"latent_idx": None, "score": None, "hit": False}


def test_field_support_uses_asymmetric_generative_widths() -> None:
    from experiments.synthetic.scripts.feature_selection import field_support

    config = SelectionConfig()
    support = field_support(np.array([0.09, 0.11, 0.39, 0.41, 0.49, 0.51]), config)
    np.testing.assert_array_equal(support[0], [False, True, True, False, False, False])
    np.testing.assert_array_equal(support[1], [False, False, True, True, True, False])


def test_pure_noise_coactivity_cannot_pass_biology_gate() -> None:
    from experiments.synthetic.scripts.feature_selection import biological_traceback

    config = SelectionConfig(min_active_samples=1, min_active_per_field=1)
    positions = np.array([0.2, 0.4], dtype=np.float32)
    zscores = np.zeros((2, 100), dtype=np.float32)
    zscores[:, 10] = 4
    decoder = np.zeros((1, 100), dtype=np.float32)
    decoder[:, 10] = 1
    table = pd.DataFrame(
        {"source_time_idx": [0, 1], "latent_idx": [0, 0], "activation_value": [1.0, 3.0]}
    )
    scores = pd.DataFrame({f"cell_{unit}_hit": [True] for unit in range(4)})
    assessed, traces = biological_traceback(table, zscores, positions, decoder, scores, config)
    assert not assessed[[f"cell_{unit}_hit" for unit in range(4)]].to_numpy().any()
    np.testing.assert_array_equal(traces["conditional_zscore"][0], zscores[0])


def test_aggregate_does_not_fill_empty_target_with_failures(tmp_path: Path) -> None:
    from experiments.synthetic.scripts.aggregate_results import aggregate

    run = tmp_path / "runs" / "seed_00"
    run.mkdir(parents=True)
    config = SelectionConfig()
    centers = np.linspace(0.0125, 0.9875, config.n_position_bins)
    curves = np.zeros((1, config.n_position_bins))
    scores = score_tuning_curves(curves, centers, config)
    scores.to_csv(run / "feature_scores.csv", index=False)
    np.savez(
        run / "tuning_curves.npz",
        position_centers=centers,
        normalized_tuning_curves=curves,
        ground_truth_curves=ground_truth_curves(centers, config),
    )
    np.savez(
        run / "traceback.npz",
        conditional_zscore=np.zeros((1, 100)),
        activation_weighted_zscore=np.zeros((1, 100)),
        decoder_weights=np.zeros((1, 100)),
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "selection_config": config.to_dict(),
                "architecture": "vanilla_per_time_bin",
                "weighted_discovery_reconstruction": 0.1,
                "model_config": {},
                "data_usage": "full",
                "decoder_loading_definition": "unit-L2",
                "traceback_definition": "conditional mean",
            }
        )
    )
    output = tmp_path / "summary"
    aggregate(run.parent, output, n_runs=1, n_selected=1)
    with np.load(output / "final_features.npz") as artifact:
        assert artifact["cell_3_curves"].shape == (0, 40)
        assert artifact["cell_0_conditional_zscore"].shape == (0, 100)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["hit_counts"] == {f"cell_{unit}": 0 for unit in range(4)}
    assert summary["n_selected_per_target"] == {f"cell_{unit}": 0 for unit in range(4)}
    assert summary["all_four_joint_hit_count"] == 0


def test_traceback_is_unweighted_mean_at_unique_active_sources() -> None:
    from experiments.synthetic.scripts.feature_selection import biological_traceback

    table = pd.DataFrame(
        {
            "source_time_idx": [0, 0, 1],
            "latent_idx": [0, 0, 0],
            "activation_value": [1.0, 2.0, 6.0],
        }
    )
    zscores = np.zeros((2, 100), dtype=np.float32)
    zscores[:, 0] = [1, 3]
    scores = pd.DataFrame({f"cell_{unit}_hit": [True] for unit in range(4)})
    _, traces = biological_traceback(
        table,
        zscores,
        np.array([0.2, 0.2], dtype=np.float32),
        np.ones((1, 100), dtype=np.float32),
        scores,
        SelectionConfig(),
    )
    assert traces["active_counts"][0] == 2
    assert traces["conditional_zscore"][0, 0] == 2.0
    assert traces["activation_weighted_zscore"][0, 0] == 2.5


def test_biological_identity_makes_four_assignments_distinct() -> None:
    from experiments.synthetic.scripts.feature_selection import biological_traceback

    config = SelectionConfig(min_active_samples=1, min_active_per_field=1)
    table = pd.DataFrame(
        {"source_time_idx": np.arange(4), "latent_idx": np.arange(4), "activation_value": 1.0}
    )
    zscores = np.eye(4, 100, dtype=np.float32) * 2
    decoder = np.eye(4, 100, dtype=np.float32) * 3
    scores = pd.DataFrame({f"cell_{unit}_hit": [True] * 4 for unit in range(4)})
    assessed, traces = biological_traceback(
        table, zscores, np.array(config.cell_centers), decoder, scores, config
    )
    np.testing.assert_array_equal(
        assessed[[f"cell_{unit}_hit" for unit in range(4)]].to_numpy(), np.eye(4, dtype=bool)
    )
    np.testing.assert_allclose(np.linalg.norm(traces["decoder_weights"], axis=1), 1)


def test_zero_and_negative_activations_do_not_count_as_active() -> None:
    from experiments.synthetic.scripts.feature_selection import biological_traceback

    table = pd.DataFrame(
        {"source_time_idx": [0, 1, 2], "latent_idx": [0, 0, 0], "activation_value": [0, -1, 2]}
    )
    zscores = np.ones((3, 100), dtype=np.float32)
    zscores[2, 0] = 3
    scores = pd.DataFrame({f"cell_{unit}_hit": [True] for unit in range(4)})
    _, traces = biological_traceback(
        table, zscores, np.full(3, 0.2), np.ones((1, 100)), scores, SelectionConfig()
    )
    assert traces["active_counts"][0] == 1
    assert traces["conditional_zscore"][0, 0] == 3
