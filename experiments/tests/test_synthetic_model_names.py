"""Configuration normalization for resuming synthetic experiments."""

from experiments.synthetic.scripts.model_names import canonical_run_config


def test_resume_normalizes_names_only_and_preserves_original():
    legacy = {"architecture": "flat", "sed_config": {"encoder": {"type": "flat"}}, "epochs": 40}
    canonical = {"architecture": "FlatWindow", "sed_config": {"encoder": {"type": "FlatWindow"}}, "epochs": 40}
    assert canonical_run_config(legacy) == canonical
    assert legacy["architecture"] == "flat"
    assert legacy["sed_config"]["encoder"]["type"] == "flat"
    canonical["epochs"] = 41
    assert canonical_run_config(legacy) != canonical_run_config(canonical)

