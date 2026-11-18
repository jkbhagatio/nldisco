"""Helpers shared by the Aeon-window runners of change 3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from beartype import beartype

LATENTS = 192
STEPS = 5000
BATCH_SIZE = 1024


@beartype
def write_json(path: Path, value: dict) -> None:
    """Write strict JSON (no NaN)."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@beartype
def open_representation(output: Path, n_windows: int, width: int = LATENTS) -> np.ndarray:
    """Create `representation.npy`: one row per window, aligned with valid_starts."""
    return np.lib.format.open_memmap(output / "representation.npy", mode="w+",
                                     dtype=np.float32, shape=(n_windows, width))


@beartype
def new_output(path: Path) -> Path:
    """Create a run directory, refusing to reuse a nonempty one."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@beartype
def cap_gpu_memory(gigabytes: float) -> None:
    """Cap this process's CUDA allocator on a shared GPU."""
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(gigabytes * 2**30 / total, 1.), 0)
