"""Canonical Window model names and compatibility with retained run artifacts."""

from copy import deepcopy

from nldisco.config import canonical_encoder_type

ARCHITECTURES = ("FlatWindow", "TransformerWindow")
PAPER_LABELS = {"FlatWindow": "FW-SED", "TransformerWindow": "TW-SED"}
# Historical on-disk names are provenance, not the current public API.
RUN_DIRECTORIES = {"FlatWindow": "flat_window", "TransformerWindow": "temporal_transformer"}


def canonical_run_config(config: dict) -> dict:
    """Normalize names for resume comparisons without changing saved metadata."""
    result = deepcopy(config)
    result["architecture"] = canonical_encoder_type(result["architecture"])
    encoder = result["sed_config"]["encoder"]
    encoder["type"] = canonical_encoder_type(encoder["type"])
    return result
