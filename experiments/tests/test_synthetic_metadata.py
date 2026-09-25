"""Metadata normalization must never substitute an earlier spike realization."""

import hashlib
import json

import pytest

from experiments.synthetic.scripts.dataset_metadata import load_parameters


def test_overlap_metadata_overrides_source_fields_and_hash(tmp_path):
    spikes = tmp_path / "spike_matrix.npy"
    spikes.write_bytes(b"new realization")
    digest = hashlib.sha256(spikes.read_bytes()).hexdigest()
    metadata = {
        "source_parameters": {"centers_m": [0.2, 0.4, 0.6, 0.8], "n_units": 100,
                              "peak_rates_hz": [1, 2, 3, 4], "baseline_rate_hz": 0},
        "centers_m": [0.2, 0.4, 0.7, 0.7], "peak_amplitudes_hz": [10, 28, 20, 20],
        "baseline_hz": 0.1, "files": {"spike_matrix.npy": digest},
    }
    (tmp_path / "parameters.json").write_text(json.dumps(metadata))
    actual = load_parameters(tmp_path)
    assert actual["centers_m"] == [0.2, 0.4, 0.7, 0.7]
    assert actual["peak_rates_hz"] == [10, 28, 20, 20]
    assert actual["baseline_rate_hz"] == 0.1
    assert actual["n_units"] == 100
    assert actual["files"]["spike_matrix.npy"]["sha256"] == digest
    spikes.write_bytes(b"different realization")
    with pytest.raises(ValueError, match="fingerprint"):
        load_parameters(tmp_path)
