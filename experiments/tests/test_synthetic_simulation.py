"""Design and reproducibility checks for the controlled ramp simulation."""

import hashlib

import numpy as np
import pytest

from experiments.synthetic.scripts.simulate import (
    CENTERS_M,
    LEFT_WIDTHS_M,
    PEAK_RATES_HZ,
    RIGHT_WIDTHS_M,
    first_pair_derivative,
    generate_dataset,
    place_rates,
    ramp_summary,
    reflect,
    simulate_motion,
)


@pytest.mark.parametrize(
    "position,velocity,expected",
    [(-0.2, -0.3, (0.2, 0.3)), (1.2, 0.3, (0.8, -0.3)), (3.2, 2.0, (0.8, -2.0))],
)
def test_reflection_preserves_speed(position, velocity, expected) -> None:
    np.testing.assert_allclose(reflect(position, velocity), expected)


def test_signed_motion_matches_undamped_ou_innovations() -> None:
    positions, velocities = simulate_motion(42)
    proposed_velocity = velocities[:-1] * 0.99 + 0.04 * np.random.default_rng(42).standard_normal(
        len(positions) - 1
    )
    proposed_position = positions[:-1] + proposed_velocity * 0.01
    reflected = (proposed_position < 0.0) | (proposed_position > 1.0)
    assert reflected.sum() > 10
    assert np.all((positions >= 0.0) & (positions <= 1.0))
    assert np.any(velocities < 0.0) and np.any(velocities > 0.0)
    np.testing.assert_allclose(
        velocities[1:], np.where(reflected, -proposed_velocity, proposed_velocity)
    )
    assert np.histogram(positions, bins=np.linspace(0, 1, 21))[0].min() > 0


def test_known_split_gaussian_fields_and_baseline() -> None:
    np.testing.assert_array_equal(CENTERS_M, [0.2, 0.4, 0.6, 0.8])
    rates = place_rates(CENTERS_M)
    np.testing.assert_allclose(np.diag(rates), PEAK_RATES_HZ + 0.1)
    np.testing.assert_allclose(
        np.diag(place_rates(CENTERS_M - LEFT_WIDTHS_M)), 0.1 + PEAK_RATES_HZ * np.exp(-0.5)
    )
    np.testing.assert_allclose(
        np.diag(place_rates(CENTERS_M + RIGHT_WIDTHS_M)), 0.1 + PEAK_RATES_HZ * np.exp(-0.5)
    )
    left = np.diag(place_rates(CENTERS_M - 0.05))
    right = np.diag(place_rates(CENTERS_M + 0.05))
    assert right[0] > left[0] and left[1] > right[1]
    np.testing.assert_allclose(left[2:], right[2:])
    assert PEAK_RATES_HZ[1] > PEAK_RATES_HZ[0]


def test_expected_pair_has_substantial_smooth_ramp_and_turn() -> None:
    positions = np.linspace(0.2, 0.37, 17001)
    summed_rates = place_rates(positions)[:2].sum(axis=0)
    assert np.all(np.diff(summed_rates) > 0)
    assert summed_rates[-1] - summed_rates[0] > 15.0
    derivative = first_pair_derivative(positions)
    assert derivative.min() > 40.0
    np.testing.assert_allclose(
        derivative[1:-1], np.gradient(summed_rates, positions)[1:-1], atol=2e-6
    )
    ramp = ramp_summary()
    assert 0.38 < ramp["peak_position_m"] < 0.4
    np.testing.assert_allclose(
        first_pair_derivative(np.array([ramp["peak_position_m"]])), 0, atol=1e-8
    )
    assert np.all(first_pair_derivative(np.linspace(0.4, 0.6, 100)) < 0)
    # Both split fields have matching zero left/right derivatives at their peaks.
    for center in CENTERS_M[:2]:
        epsilon = 1e-8
        adjacent = place_rates(np.array([center - epsilon, center, center + epsilon]))
        assert np.max(np.abs(np.diff(adjacent, axis=1))) < 1e-5


@pytest.mark.parametrize("positions", [np.zeros((2, 3)), np.array([np.nan])])
def test_invalid_rate_positions_are_rejected(positions) -> None:
    with pytest.raises(ValueError):
        place_rates(positions)


def test_dataset_is_deterministic_and_records_hashes(tmp_path) -> None:
    first = generate_dataset(tmp_path / "first")
    second = generate_dataset(tmp_path / "second")
    assert first == second
    for name, metadata in first["files"].items():
        contents = (tmp_path / "first" / name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == metadata["sha256"]
        assert contents == (tmp_path / "second" / name).read_bytes()
    spikes = np.load(tmp_path / "first" / "spike_matrix.npy")
    assert spikes.shape == (100, 60000) and spikes.dtype == np.bool_
    assert first["place_cell_ids"] == [0, 1, 2, 3]
    assert first["directional_modulation"] is False
    assert first["rate_dependencies"] == ["position"]
    assert first["empirical_rate_maps"]["gaussian_smoothing_sigma_bins"] == 0.5
    assert "controlled_revised_design" in first["realization_status"]
    assert first["analysis_scope"]["samples"] == [0, 60000]
    assert first["analysis_scope"]["held_out_split"] is False
    positions = np.load(tmp_path / "first" / "positions.npy")
    np.testing.assert_array_equal(
        np.load(tmp_path / "first" / "generative_place_rates.npy"), place_rates(positions)
    )
    exact_maps = np.load(tmp_path / "first" / "theoretical_rate_maps.npy")
    np.testing.assert_array_equal(
        np.load(tmp_path / "first" / "theoretical_first_pair_sum.npy"), exact_maps[:2].sum(axis=0)
    )
    expected = np.r_[place_rates(positions).mean(axis=1), first["noise_rates_hz"]]
    observed = spikes.sum(axis=1) / 600.0
    # Five Poisson standard errors conservatively bound Bernoulli count deviations.
    assert np.all(np.abs(observed - expected) < 5.0 * np.sqrt(expected / 600.0))
    occupancy = np.load(tmp_path / "first" / "empirical_rate_occupancy_s.npy")
    empirical = np.load(tmp_path / "first" / "empirical_rate_maps_unsmoothed.npy")
    np.testing.assert_allclose(occupancy.sum(), 600.0)
    np.testing.assert_allclose((empirical * occupancy).sum(axis=1), spikes[:4].sum(axis=1))
    with pytest.raises(FileExistsError):
        generate_dataset(tmp_path / "first")
