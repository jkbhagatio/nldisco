"""Read either saved synthetic dataset schema without altering its realization."""

import hashlib
import json
from pathlib import Path

from typeguard import typechecked


@typechecked
def load_parameters(directory: Path) -> dict:
    """Normalize metadata and verify the exact spike file used by every model."""
    raw = json.loads((directory / "parameters.json").read_text())
    parameters = {**raw.get("source_parameters", {}), **raw}
    parameters.pop("source_parameters", None)
    if "peak_amplitudes_hz" in raw:
        parameters["peak_rates_hz"] = raw["peak_amplitudes_hz"]
        parameters["baseline_rate_hz"] = raw["baseline_hz"]
    parameters["files"] = {
        name: {"sha256": value} if isinstance(value, str) else value
        for name, value in raw["files"].items()
    }
    actual = hashlib.sha256((directory / "spike_matrix.npy").read_bytes()).hexdigest()
    if actual != parameters["files"]["spike_matrix.npy"]["sha256"]:
        raise ValueError(f"Spike data do not match the saved fingerprint: {directory}")
    return parameters
