"""Scientific selection, plotted values, and uncertainty in synthetic paper results."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.synthetic.scripts import paper
from experiments.synthetic.scripts.paper import choose_temporal


def test_temporal_selection_never_replaces_passing_feature_with_higher_failed_score() -> None:
    records = [
        {"summary": {"seed": seed, "selected_directions": {
            "right": {"passes": passes, "selection_score": score}
        }}}
        for seed, passes, score in [(0, False, 10), (1, True, 1), (2, True, 1)]
    ]
    results = {"temporal": {"FlatWindow": records}}
    assert choose_temporal(results, "FlatWindow", "right") is records[1]
    records[1]["summary"]["selected_directions"]["right"]["passes"] = False
    records[2]["summary"]["selected_directions"]["right"]["passes"] = False
    assert choose_temporal(results, "FlatWindow", "right") is None
    assert choose_temporal(results, "FlatWindow", "right", allow_diagnostic=True) is records[0]
    for record in records:
        record["summary"]["selected_directions"]["right"]["candidate_available"] = False
    assert choose_temporal(results, "FlatWindow", "right", allow_diagnostic=True) is None


def test_main_and_supplementary_panels_preserve_spatial_curves(monkeypatch) -> None:
    centers = np.array([0.2, 0.4, 0.6, 0.8])
    positions = np.linspace(0.0125, 0.9875, 40)
    curves = np.exp(-0.5 * ((positions[None] - centers[:, None]) / 0.08) ** 2)
    data = {"position_centers": positions, "ground_truth_curves": curves}
    data.update({f"cell_{cell}_curves": np.repeat(curve[None], 5, axis=0)
                 for cell, curve in enumerate(curves)})
    matryoshka_curves = np.repeat(curves[None, 2:], 5, axis=0)
    results = {
        "parameters": {"centers_m": centers}, "vanilla": data,
        "positions": positions, "timestamps": np.linspace(0, 600, 40),
        "empirical_rates": curves, "empirical_edges": np.linspace(0, 1, 41),
        "temporal": {"FlatWindow": [], "TransformerWindow": []},
        "matryoshka": {"position_centers": positions, "selected_curves": matryoshka_curves,
                       "ground_truth_curves": curves},
    }
    monkeypatch.setattr(paper, "_save", lambda fig, name: None)
    main = paper.main_figure(results)
    for cell in range(2):
        mean, truth = main.axes[2].lines[2 * cell:2 * cell + 2]
        np.testing.assert_allclose(mean.get_ydata(), curves[cell])
        np.testing.assert_allclose(truth.get_ydata(), curves[cell])
    np.testing.assert_allclose(main.axes[3].lines[0].get_ydata(), curves[2])
    np.testing.assert_allclose(main.axes[3].lines[2].get_ydata(), curves[3])
    spatial = paper.spatial_figure(results)
    assert len(spatial.axes) == 4
    for cell, ax in enumerate(spatial.axes):
        np.testing.assert_allclose(ax.lines[0].get_ydata(), curves[cell])
    plt.close(main)
    plt.close(spatial)


def test_main_panel_uses_conditioned_examples_without_direction_filter(monkeypatch) -> None:
    calls = []
    original = paper._trajectories

    def capture(ax, record, direction, ramp_conditioned=False):
        calls.append((direction, ramp_conditioned))
        return original(ax, record, direction, ramp_conditioned)

    monkeypatch.setattr(paper, "_trajectories", capture)
    test_main_and_supplementary_panels_preserve_spatial_curves(monkeypatch)
    assert calls == [("right", True)]


def test_main_pair_refuses_incomplete_five_run_average() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="five qualifying"):
        paper._paired_spatial(ax, {"vanilla": {
            "position_centers": np.linspace(0, 1, 40), "cell_0_curves": np.zeros((4, 40))
        }})
    plt.close(fig)


def _matryoshka_summaries():
    return [{"seed": seed, "levels": {"128": {"selected": {
        "cell_2": {"hit": seed != 4, "latent_idx": 3, "score": 0.8},
        "cell_3": {"hit": seed != 4, "latent_idx": 7, "score": 0.9},
    }}}} for seed in range(7)]


def test_matryoshka_extension_retains_original_pairs_and_breaks_ties_by_seed() -> None:
    summaries = _matryoshka_summaries()
    assert [s["seed"] for s in paper.select_matryoshka_runs(summaries)] == [0, 1, 2, 3, 5]
    summaries[6]["levels"]["128"]["selected"]["cell_2"]["score"] = 0.95
    assert [s["seed"] for s in paper.select_matryoshka_runs(summaries)] == [0, 1, 2, 3, 6]
    summaries[6]["levels"]["128"]["selected"]["cell_3"]["latent_idx"] = 3
    assert paper.select_matryoshka_runs(summaries)[-1]["seed"] == 5
    summaries[5]["levels"]["128"]["selected"]["cell_2"]["hit"] = False
    with pytest.raises(ValueError, match="Neither new"):
        paper.select_matryoshka_runs(summaries)


def test_matryoshka_extension_rejects_missing_or_duplicate_seeds() -> None:
    summaries = _matryoshka_summaries()
    with pytest.raises(ValueError, match="seven unique"):
        paper.select_matryoshka_runs(summaries[:-1])
    with pytest.raises(ValueError, match="seven unique"):
        paper.select_matryoshka_runs(summaries + [summaries[0]])
    summaries[0]["levels"]["128"]["selected"]["cell_2"]["hit"] = False
    with pytest.raises(ValueError, match="original qualifying"):
        paper.select_matryoshka_runs(summaries)


def test_matryoshka_panel_uses_sample_sd_over_the_five_paired_runs() -> None:
    positions = np.linspace(0, 1, 3)
    curves = np.array([[[0.05 * seed, 1, 0.2], [0.1, 1, 0.05 * seed]] for seed in range(5)])
    fig, ax = plt.subplots()
    captured = []
    original = ax.fill_between

    def capture(x, lower, upper, **kwargs):
        captured.append((lower, upper))
        return original(x, lower, upper, **kwargs)

    ax.fill_between = capture
    results = {"matryoshka": {"position_centers": positions, "selected_curves": curves,
                              "ground_truth_curves": np.ones((4, 3))}}
    paper._overlapping_spatial(ax, results)
    for index in range(2):
        mean, sd = curves[:, index].mean(0), curves[:, index].std(0, ddof=1)
        np.testing.assert_allclose(ax.lines[2 * index].get_ydata(), mean)
        np.testing.assert_allclose(captured[index][0], np.clip(mean - sd, 0, 1))
        np.testing.assert_allclose(captured[index][1], np.clip(mean + sd, 0, 1))
    results["matryoshka"]["selected_curves"] = curves[:4]
    with pytest.raises(ValueError, match="five qualifying"):
        paper._overlapping_spatial(ax, results)
    plt.close(fig)
