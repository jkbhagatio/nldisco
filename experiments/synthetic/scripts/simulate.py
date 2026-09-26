"""Generate a controlled asymmetric-ramp revision of the synthetic experiment.

These explicitly seeded position-only fields are a revised design, not the
original Theodoni/paper field parameters or historical realization.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from jaxtyping import Float
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

CENTERS_M = np.array([0.2, 0.4, 0.6, 0.8])
LEFT_WIDTHS_M = np.array([0.05, 0.10, 0.07, 0.07])
RIGHT_WIDTHS_M = np.array([0.10, 0.05, 0.07, 0.07])
PEAK_RATES_HZ = np.array([10.0, 28.0, 20.0, 20.0])
BASELINE_RATE_HZ = 0.1
DT_S = 0.01
N_SAMPLES = 60_000
EMPIRICAL_SMOOTHING_SIGMA_BINS = 0.5


def reflect(position: float, velocity: float) -> tuple[float, float]:
    """Reflect a tentative position at the unit-track ends without damping."""
    while position < 0.0 or position > 1.0:
        position = -position if position < 0.0 else 2.0 - position
        velocity = -velocity
    return position, velocity


def simulate_motion(
    seed: int, n_samples: int = N_SAMPLES
) -> tuple[Float[np.ndarray, "time"], Float[np.ndarray, "time"]]:  # noqa: F821
    """Euler--Maruyama signed OU velocity with specular endpoint reflection."""
    rng = np.random.default_rng(seed)
    positions = np.zeros(n_samples)
    velocities = np.zeros(n_samples)
    positions[0] = 0.5
    innovations = rng.standard_normal(n_samples - 1)
    for index, innovation in enumerate(innovations, start=1):
        velocity = velocities[index - 1] * (1.0 - DT_S) + 0.4 * np.sqrt(DT_S) * innovation
        positions[index], velocities[index] = reflect(
            positions[index - 1] + velocity * DT_S, velocity
        )
    return positions, velocities


def place_rates(positions: Float[np.ndarray, "time"]) -> Float[np.ndarray, "4 time"]:  # noqa: F821
    """Evaluate position-only split-Gaussian fields, including baseline, in Hz."""
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be a finite one-dimensional array")
    displacement = positions[None, :] - CENTERS_M[:, None]
    widths = np.where(displacement < 0, LEFT_WIDTHS_M[:, None], RIGHT_WIDTHS_M[:, None])
    return BASELINE_RATE_HZ + PEAK_RATES_HZ[:, None] * np.exp(-0.5 * (displacement / widths) ** 2)


def first_pair_derivative(positions: Float[np.ndarray, "time"]) -> Float[np.ndarray, "time"]:  # noqa: F821
    """Return the analytic derivative of the first pair's summed rate, Hz/m."""
    rates = place_rates(positions)[:2] - BASELINE_RATE_HZ
    displacement = positions[None, :] - CENTERS_M[:2, None]
    widths = np.where(displacement < 0, LEFT_WIDTHS_M[:2, None], RIGHT_WIDTHS_M[:2, None])
    return np.sum(-displacement * rates / widths**2, axis=0)


def ramp_summary() -> dict:
    """Quantify the predetermined ramp using exact rates, without fitted features."""
    start, verified_end = 0.2, 0.37
    peak = brentq(lambda x: first_pair_derivative(np.array([x]))[0], verified_end, 0.4)
    start_rate, end_rate, peak_rate = place_rates(np.array([start, verified_end, peak]))[:2].sum(
        axis=0
    )
    return {
        "unit_ids": [0, 1],
        "definition": "sum of exact expected rates including both baselines",
        "selection_basis": "fixed before training using analytic rate derivative only",
        "monotonic_ramp_interval_m": [start, peak],
        "verified_positive_slope_interval_m": [start, verified_end],
        "minimum_slope_on_17001_point_verified_grid_hz_per_m": float(
            first_pair_derivative(np.linspace(start, verified_end, 17001)).min()
        ),
        "start_rate_hz": float(start_rate),
        "rate_at_0_37_m_hz": float(end_rate),
        "peak_position_m": float(peak),
        "peak_rate_hz": float(peak_rate),
        "increase_hz": float(peak_rate - start_rate),
        "relative_increase": float((peak_rate - start_rate) / start_rate),
    }


def generate_dataset(
    output_dir: Path,
    parameter_seed: int = 0,
    motion_seed: int = 42,
    place_spike_seed: int = 123,
    noise_spike_seed: int = 456,
) -> dict:
    """Save a new fully specified realization; refuse to overwrite any directory."""
    if output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve existing datasets")
    positions, velocities = simulate_motion(motion_seed)
    rates = place_rates(positions)
    noise_rates = np.random.default_rng(parameter_seed).uniform(1.0, 5.0, 96)
    place_spikes = np.random.default_rng(place_spike_seed).random(rates.shape) < rates * DT_S
    noise_spikes = np.random.default_rng(noise_spike_seed).random((96, N_SAMPLES)) < (
        noise_rates[:, None] * DT_S
    )
    map_positions = np.linspace(0.0, 1.0, 501)
    empirical_edges = np.linspace(positions.min(), positions.max(), 21)
    occupancy_s = np.histogram(positions, empirical_edges)[0] * DT_S
    empirical_counts = np.stack(
        [np.histogram(positions[spikes], empirical_edges)[0] for spikes in place_spikes]
    )
    empirical_rates = empirical_counts / occupancy_s[None, :]
    arrays = {
        "spike_matrix": np.concatenate([place_spikes, noise_spikes], axis=0),
        "positions": positions,
        "velocities": velocities,
        "timestamps": np.arange(N_SAMPLES) * DT_S,
        "theoretical_rate_positions": map_positions,
        "theoretical_rate_maps": place_rates(map_positions),
        "theoretical_first_pair_sum": place_rates(map_positions)[:2].sum(axis=0),
        "theoretical_first_pair_derivative": first_pair_derivative(map_positions),
        "generative_place_rates": rates,
        "empirical_rate_bin_edges": empirical_edges,
        "empirical_rate_positions": (empirical_edges[:-1] + empirical_edges[1:]) / 2.0,
        "empirical_rate_occupancy_s": occupancy_s,
        "empirical_rate_maps_unsmoothed": empirical_rates,
        "empirical_rate_maps": gaussian_filter1d(
            empirical_rates, sigma=EMPIRICAL_SMOOTHING_SIGMA_BINS, axis=1, mode="nearest"
        ),
    }
    parameters = {
        "schema_version": 2,
        "design": "controlled_asymmetric_ramp_four_place_cells_96_position_independent_noise_neurons",
        "realization_status": "controlled_revised_design_not_original_Theodoni_parameters_or_realization",
        "duration_s": 600.0,
        "dt_s": DT_S,
        "sampling_rate_hz": 100.0,
        "n_samples": N_SAMPLES,
        "n_units": 100,
        "spike_matrix_axes": ["unit", "time"],
        "place_cell_ids": list(range(4)),
        "noise_cell_ids": list(range(4, 100)),
        "centers_m": CENTERS_M.tolist(),
        "left_widths_m": LEFT_WIDTHS_M.tolist(),
        "right_widths_m": RIGHT_WIDTHS_M.tolist(),
        "peak_rates_hz": PEAK_RATES_HZ.tolist(),
        "baseline_rate_hz": BASELINE_RATE_HZ,
        "rate_equation": "baseline + peak_amplitude * exp(-0.5 * ((position-center)/width)**2); width=left_width if position<center else right_width",
        "directional_modulation": False,
        "rate_dependencies": ["position"],
        "first_pair_ramp": ramp_summary(),
        "peak_rate_convention": "Gaussian amplitude above baseline; absolute peaks are amplitude+0.1 Hz",
        "noise_rates_hz": noise_rates.tolist(),
        "noise_rate_distribution": "independent Uniform(1,5) Hz draws, constant over time and position",
        "spike_distribution": "independent Bernoulli(rate_hz * dt_s) at each unit and sample",
        "parameter_seed": parameter_seed,
        "motion_seed": motion_seed,
        "place_spike_seed": place_spike_seed,
        "noise_spike_seed": noise_spike_seed,
        "rng": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__,
        "motion": {
            "process": "signed Ornstein-Uhlenbeck velocity",
            "integrator": "Euler-Maruyama velocity; position uses new velocity",
            "theta_per_s": 1.0,
            "mu_m_per_s": 0.0,
            "sigma_m_per_s_to_three_halves": 0.4,
            "initial_velocity_m_per_s": 0.0,
            "initial_position_m": 0.5,
            "track_length_m": 1.0,
            "boundary_condition": "specular reflection of position and velocity; no damping",
            "saved_velocity": "signed OU state after endpoint reflection, m/s",
        },
        "analysis_scope": {
            "kind": "full_recording_descriptive_feature_discovery",
            "samples": [0, N_SAMPLES],
            "held_out_split": False,
            "interval_convention": "zero-based half-open",
            "interpretation": "training, feature inspection, and descriptive scores use the full recording; no held-out generalization claim",
        },
        "theoretical_rate_maps": {
            "unit_ids": list(range(4)),
            "axes": ["place_unit", "position"],
            "units": "Hz",
            "positions_file": "theoretical_rate_positions.npy",
            "definition": "exact split-Gaussian generative rate including baseline, not empirical spike tuning",
        },
        "empirical_rate_maps": {
            "unit_ids": list(range(4)),
            "axes": ["place_unit", "position_bin"],
            "units": "Hz",
            "data_scope": "all 60000 samples; descriptive illustration of simulated input",
            "n_bins": 20,
            "range": "observed position minimum to maximum",
            "definition": "spike counts divided by occupancy seconds, then spatial Gaussian smoothing",
            "gaussian_smoothing_sigma_bins": EMPIRICAL_SMOOTHING_SIGMA_BINS,
            "smoothing_rationale": "modest 0.5-bin (approximately 0.025 m) smoothing preserves the narrower fields; unsmoothed and exact maps are retained",
            "gaussian_smoothing_mode": "nearest",
            "historical_correction": "np.histogram used for both occupancy and spike counts, consistently including the exact maximum sample in the final bin",
        },
        "provenance": {
            "design_sources": [
                "paper/iclr_paper/results/synthetic_results.tex",
                "paper/iclr_paper/appendix/additional_results/additional_results_main.tex",
            ],
            "implementation_reference": "2e15908f0e48a3b06d197884c4f11c2e34ca1853:notebooks/simulation_fourPlacecellsAndNoise.ipynb",
            "implementation_details": "Motion and noise generation retain the earlier signed-velocity design; place-field centers, asymmetric widths, and amplitudes are an explicitly controlled revision, not original paper/Theodoni parameters.",
            "seed_provenance": "motion42/place123/noise456 occur in historical notebook; noise-rate parameter seed0 newly specified",
            "figure_reference": "paper/iclr_paper/figures/synthetic_results.png",
            "figure_heatmap_caveat": "Historical notebook includes empirical occupancy-normalized smoothed heatmaps and separate theoretical curves; exact historical plotting realization is unavailable.",
        },
    }
    output_dir.mkdir(parents=True)
    parameters["files"] = {}
    for name, array in arrays.items():
        path = output_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        parameters["files"][path.name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    parameters["simulator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    return parameters


def main() -> None:
    """Parse independent seeds and generate the controlled revised-design dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parameter-seed", type=int, default=0)
    parser.add_argument("--motion-seed", type=int, default=42)
    parser.add_argument("--place-spike-seed", type=int, default=123)
    parser.add_argument("--noise-spike-seed", type=int, default=456)
    args = parser.parse_args()
    generate_dataset(**vars(args))


if __name__ == "__main__":
    main()
