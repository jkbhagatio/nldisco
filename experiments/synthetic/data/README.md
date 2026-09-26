# Synthetic data used by the paper

`matryoshka_overlap/` is the shared dataset for the five-panel Figure 3 and current
vanilla, temporal, and retained Matryoshka results. Its name reflects its origin.
`paper_simulation/` is the earlier ramp-only source recording, retained solely to
regenerate the overlap variant deterministically. The notebook does not use that
earlier population for its plots.

Both contain four place cells plus 96 noise units, 600 seconds at 100 Hz.
The paper spike file SHA-256 is
`ce5bd9573966e96cd36a44a68cc4e84836c95f749a59260e87398aaab06db09f`.

| Cell (plot index) | Array row | Center (m) | Left/right widths (m) | Amplitude above baseline (Hz) |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0.2 | 0.05 / 0.10 | 10 |
| 2 | 1 | 0.4 | 0.10 / 0.05 | 28 |
| 3, broad | 2 | 0.7 | 0.14 / 0.14 | 20 |
| 4, narrow | 3 | 0.7 | 0.06 / 0.06 | 20 |

Rates are `0.1 + amplitude * exp(-0.5*((x-center)/width)**2)`, using the left
width below the center and right width otherwise. They depend only on position,
not velocity or direction. Gaussian support is not strictly nested. The first
pair's expected sum rises from 13.9894 Hz at 0.2 m to 29.7076 Hz at 0.38854 m,
then falls. This is an explicitly controlled design revision, not the original
Theodoni parameters or realization.

## Saved arrays

| File | Shape / meaning |
| --- | --- |
| `spike_matrix.npy` | bool [100, 60000], units × samples |
| `positions.npy`, `timestamps.npy`, `velocities.npy` | float64 [60000], m / s / m·s⁻¹ |
| `generative_place_rates.npy` | float64 [4, 60000], exact expected Hz at observed positions |
| `theoretical_rate_positions.npy` | float64 [501], uniform 0–1 m grid |
| `theoretical_rate_maps.npy` | float64 [4, 501], exact generative Hz |
| `empirical_rate_bin_edges.npy` | float64 [21], equal bins over observed position range |
| `empirical_rate_occupancy_s.npy` | float64 [20], occupancy |
| `empirical_rate_maps_unsmoothed.npy` | float64 [4, 20], spikes / occupancy |
| `empirical_rate_maps.npy` | float64 [4, 20], smoothed empirical Hz |
| `parameters.json` | seeds, source metadata, changed fields, array hashes |

Heatmaps use Gaussian smoothing σ=0.5 spatial bins, mode nearest. Both occupancy
and spike histograms include the maximum-position sample. Exact theoretical
rates are separate from empirical rates. The source directory additionally saves
empirical bin centers and analytic first-pair sum/derivative arrays.

## Generation and provenance

Signed OU velocity has θ=1/s, mean 0, volatility 0.4 m/s^(3/2), initial velocity 0,
and initial position 0.5 m. Euler–Maruyama updates velocity; position uses the new
velocity. Track endpoints reflect position overshoot and velocity without damping.
Motion seed is 42. Noise units have constant independent Uniform(1,5)-Hz rates
(parameter seed 0). Spikes are Bernoulli(rate × 0.01 s), with place seed 123 and
noise seed 456; all generators are NumPy PCG64.

The overlap variant reuses the source trajectory, first two spike rows, and all
96 noise spike rows **exactly**. Only rows 2/3 change, reusing the same four-by-60000
place-spike uniform draws with updated rates. The source had centers 0.6/0.8 m
and widths 0.07 m for those two rows.

The overlap metadata preserves the original source metadata and historical
exploratory description. `scripts/dataset_metadata.py` normalizes both schemas,
ensuring current fields and file hashes override the source values and verifying
the actual spike fingerprint. Original training configs are not retrospectively
rewritten when a result is promoted into the paper.

To regenerate from an empty destination, first run `scripts/simulate.py` with
seeds 0/42/123/456 into `data/paper_simulation/`, then call
`scripts.matryoshka_overlap.generate_variant()`. See the parent README.
Neither generator overwrites existing data. All training and descriptive feature
selection use samples [0,60000); there is no held-out split.
