"""Fast checks for activity-only temporal traceback and plot support."""

import numpy as np
import pandas as pd

from experiments.aeon.sweep_20260921.paper_results import (
    WHEEL_CM_PER_TURN,
    WHEEL_CUTOFF,
    WHEEL_CUTOFF_CM_S,
    binned_probability,
    center_edges,
    roc_operating_point,
    score,
    wheel_location_controls,
    wheel_tuning,
)




def test_histogram_includes_last_edge_and_masks_low_support():
    values = np.r_[np.zeros(120), np.ones(120), np.full(20, 2.)]
    active = np.r_[np.ones(60), np.zeros(60), np.ones(140)].astype(bool)
    result = binned_probability(values, active, np.ones(260, dtype=bool), np.array([0., 1., 2.]))
    np.testing.assert_array_equal(result["total"], [120, 140])
    np.testing.assert_allclose(result["probability"], [.5, 1.])
    result = binned_probability(values, active, np.ones(260, dtype=bool), np.array([0., 1., 1.5, 2.]))
    assert np.isnan(result["probability"][-1])


def test_score_keeps_native_zeros_and_positive_polarity():
    result = score(np.array([0., 0., 1., 2.]), np.array([True, True, False, False]), np.ones(4, dtype=bool))
    assert result["auroc"] == 0.
    assert result["selectivity"] == 0.


def test_requested_speed_centers_partition_nonnegative_support():
    centers = np.array([0., .002, .005, .01, .02, .03, .1, .5, 1., 2.5])
    edges = center_edges(centers, 2.772827)
    np.testing.assert_allclose(edges, [0, .001, .0035, .0075, .015, .025, .065, .3, .75, 1.75, 3.25])
    assert len(edges) == len(centers) + 1
    assert center_edges(centers, 4.)[-1] > 4.


def test_low_support_wheel_point_is_computable_without_fabrication():
    result = binned_probability(np.array([2., 2.5]), np.array([True, False]),
                               np.ones(2, dtype=bool), np.array([1.75, 3.25]), minimum_count=1)
    assert result["total"][0] == 2
    assert result["probability"][0] == .5


def test_revised_wheel_bins_and_cm_conversion_keep_tail_out_of_plot_only():
    values = np.array([0., .1, .175, .25, .375, .5, 1., 1.74, 1.75, 2.5])
    active = np.ones(len(values), dtype=bool)
    valid = active.copy()
    tuning = wheel_tuning(values, active, valid)
    np.testing.assert_allclose(tuning["centers"], [0, .002, .005, .01, .02, .03, .1, .25, .5, 1])
    np.testing.assert_allclose(tuning["edges"], [0, .001, .0035, .0075, .015, .025, .065, .175, .375, .75, 1.75])
    np.testing.assert_allclose(tuning["centers_cm_s"], tuning["centers"] * 8 * np.pi)
    np.testing.assert_allclose(tuning["edges_cm_s"], tuning["edges"] * 8 * np.pi)
    np.testing.assert_allclose(.03 * WHEEL_CM_PER_TURN, .7539822368615503)
    assert tuning["total"][7] == 2
    assert tuning["omitted_upper_tail_windows"] == 2
    assert tuning["total"].sum() == 8
    assert valid.all()  # Plot eligibility must not mutate score eligibility.


def test_exact_cm_cutoff_is_not_the_previous_rounded_turns_cutoff():
    assert WHEEL_CUTOFF_CM_S == .75
    assert WHEEL_CUTOFF < .03
    speeds_turns_s = np.array([.749, .75, .751]) / WHEEL_CM_PER_TURN
    np.testing.assert_array_equal(speeds_turns_s > WHEEL_CUTOFF, [False, False, True])
    assert .0299072265625 > WHEEL_CUTOFF  # A measured speed below the old .03turn/s cutoff.


def test_roc_marker_is_attainable_and_distinct_from_native_activity():
    values = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, 0.])
    labels = np.array([True, True, False, True, False, False, False, False, False, False])
    marker = roc_operating_point(values, labels, target=.18)
    predicted = values >= marker["threshold"]
    assert marker["fpr"] == predicted[~labels].mean()
    assert marker["tpr"] == predicted[labels].mean()
    assert marker["threshold"] > 0
    np.testing.assert_allclose(marker["fpr"], 1/7)
    assert marker["tpr"] == 1.


def test_position_standardization_removes_pure_location_confound(tmp_path):
    endpoints = np.arange(19, 219)
    values = np.r_[np.ones(100), np.zeros(100)]
    labels = np.r_[np.ones(80), np.zeros(20), np.ones(20), np.zeros(80)].astype(bool)
    behavior = pd.DataFrame(dict(camera_analysis_valid=np.ones(219, dtype=bool),
                                 patch_distance=np.full(219, 50.),
                                 x=np.r_[np.full(119, 891.), np.full(100, 941.)],
                                 y=np.full(219, 851.5)))
    wheel_location_controls(behavior, endpoints, values, labels, np.ones(200, dtype=bool), tmp_path)
    results = pd.read_csv(tmp_path / "wheel_location_controls.csv").set_index("comparison")
    np.testing.assert_allclose(results.loc["within100px_patch", "selectivity"], .8)
    adjusted = results.loc["within100px_patch_xy25px_standardized"]
    np.testing.assert_allclose(adjusted[["selectivity", "auroc", "positive_coverage"]], [.5, .5, 1.])
