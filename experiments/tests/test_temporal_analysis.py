"""Scientific guards for temporal feature selection and example plotting."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.synthetic.scripts import temporal_analysis as module


def test_direction_means_include_inactive_zeros_and_match_position():
    metadata = pd.DataFrame(
        {
            "position": [0.1] * 40 + [0.8] * 40,
            "direction": [-1] * 10 + [1] * 30 + [-1] * 30 + [1] * 10,
            "clean": True,
        }
    )
    acts = np.zeros((80, 1))
    acts[:40] = 2  # Position-only feature: direction imbalance must cancel.
    metrics = module.matched_direction_metrics(acts, metadata)
    assert abs(metrics.right_selectivity.iloc[0]) < 1e-8
    assert metrics.right_mean_activation.iloc[0] == 1
    assert metrics.matched_position_bins.iloc[0] == 2


def test_top_windows_deoverlap_without_direction_filter():
    starts = np.array([0, 2, 10, 20])
    selected = module.nonoverlapping_top_indices(np.array([3, 4, 2, 1]), starts, 5, 3)
    np.testing.assert_array_equal(selected, [1, 2, 3])


def test_reversals_and_small_displacement_are_excluded():
    positions = np.array([0.0, 0.1, 0.2, 0.1, 0.1, 0.11])
    metadata = module.window_motion(positions, np.array([0, 1, 3]), 3)
    np.testing.assert_array_equal(metadata.clean, [True, False, False])


def test_decoder_motion_uses_known_unit_centers():
    decoder = np.ones((6, 100, 2)) * 0.01
    decoder[:2, 0, 0] = 1
    decoder[-2:, 1, 0] = 1
    decoder[:2, 1, 1] = 1
    decoder[-2:, 0, 1] = 1
    metrics = module.decoder_metrics(decoder, np.array([0.2, 0.3, 0.8, 0.9]))
    assert metrics.decoder_center_displacement.iloc[0] > 0.09
    assert metrics.decoder_center_displacement.iloc[1] < -0.09
    assert (metrics.place_energy_enrichment > 100).all()


def test_passing_candidate_precedes_higher_scoring_failure():
    metrics = pd.DataFrame(
        {
            "latent_idx": [0, 1],
            "left_active_windows": [30, 30],
            "right_active_windows": [30, 30],
            "matched_position_bins": [3, 3],
            "right_selectivity": [0.24, 0.3],
            "decoder_center_displacement": [0.6, 0.02],
            "place_energy_enrichment": [100, 3],
            "pair_min_unit_energy_fraction": [0.2, 0.2],
            "pair_known_place_energy_fraction": [0.8, 0.8],
        }
    )
    selected = module.select_features(metrics)
    assert selected["right"] == 1
    assert metrics.right_selection_score.iloc[0] > metrics.right_selection_score.iloc[1]
    assert metrics.right_passes.tolist() == [False, True]


def test_ramp_crossing_excludes_unrelated_motion_and_reports_jitter():
    positions = np.array([.19, .25, .24, .40, .60, .70, .80, .90])
    metadata = module.window_motion(positions, np.array([0, 4]), 4)
    np.testing.assert_array_equal(metadata.first_pair_crossing, [True, False])
    np.testing.assert_array_equal(metadata.strict_first_pair_transition, [False, False])
    np.testing.assert_array_equal(metadata.purposeful_first_pair_transition, [True, False])
    assert not metadata.ramp_clean.any()


def test_first_pair_gate_rejects_other_pair_decoder():
    decoder = np.ones((6, 100, 1)) * .01
    decoder[:2, 2, 0] = 1
    decoder[-2:, 3, 0] = 1
    metrics = module.decoder_metrics(decoder, np.array([.2, .4, .6, .8]))
    assert metrics.global_decoder_center_displacement.iloc[0] > .19
    assert abs(metrics.decoder_center_displacement.iloc[0]) < 1e-8
    assert metrics.pair_known_place_energy_fraction.iloc[0] < .01


def test_position_matching_honors_ramp_eligibility():
    metadata = pd.DataFrame({
        "position": [.25]*40 + [.65]*40,
        "direction": [-1]*20+[1]*20+[-1]*20+[1]*20,
        "clean": True, "ramp_clean": [True]*40+[False]*40,
    })
    acts = np.ones((80, 1))
    acts[60:] = 100
    ramp = module.matched_direction_metrics(acts, metadata, bins=40, eligible_column="ramp_clean")
    full = module.matched_direction_metrics(acts, metadata)
    assert abs(ramp.right_selectivity.iloc[0]) < 1e-8
    assert full.right_selectivity.iloc[0] > .9


def test_missing_candidate_diagnostics_are_explicit(monkeypatch):
    scripts = Path(module.__file__).resolve().parent
    monkeypatch.syspath_prepend(str(scripts))
    finalize_spec = importlib.util.spec_from_file_location("finalize_temporal", scripts / "finalize_temporal.py")
    finalize = importlib.util.module_from_spec(finalize_spec)
    finalize_spec.loader.exec_module(finalize)
    assert finalize.safe_fraction(np.array([])) is None
    assert finalize.safe_fraction(np.array([True, False])) == .5
    assert finalize.safe_correlation(np.ones(3), np.arange(3)) is None
    assert finalize.safe_correlation(np.arange(3), np.arange(3)) == 1
