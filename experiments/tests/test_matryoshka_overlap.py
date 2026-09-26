"""Fixed nested-field design and same-run recovery contracts."""

import numpy as np
import pytest

from experiments.synthetic.scripts.matryoshka_overlap import (
    generate_variant,
    joint_recovery,
    rates,
)
from experiments.synthetic.scripts.simulate import place_rates


def test_overlap_variant_preserves_first_fields_and_nests_rate_profiles() -> None:
    grid = np.linspace(0, 1, 1001)
    maps = rates(grid)
    np.testing.assert_array_equal(maps[:2], place_rates(grid)[:2])
    np.testing.assert_allclose(grid[maps[2:].argmax(axis=1)], [0.7, 0.7])
    assert np.all(maps[2] >= maps[3])
    assert maps[2, 560] > maps[3, 560]
    np.testing.assert_allclose(maps[2:, 700], [20.1, 20.1])


def test_joint_recovery_requires_two_distinct_passing_features() -> None:
    selected = {"cell_2": {"hit": True, "latent_idx": 3},
                "cell_3": {"hit": True, "latent_idx": 7}}
    assert joint_recovery(selected)
    selected["cell_3"]["latent_idx"] = 3
    assert not joint_recovery(selected)
    selected["cell_3"] = {"hit": False, "latent_idx": None}
    assert not joint_recovery(selected)


def test_variant_refuses_existing_directory(tmp_path) -> None:
    with pytest.raises(FileExistsError, match="Refusing"):
        generate_variant(destination=tmp_path)
