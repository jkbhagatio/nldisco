"""Prepare the approved shared endpoint index and private sweep logging gate."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wandb
from beartype import beartype
from wandb_gql import gql

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "experiments/churchland/data/nitschke_20090812"
OUTPUT = Path("/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1")


@beartype
def sha256(path: Path) -> str:
    """Hash a file without materializing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@beartype
def main() -> None:
    """Create only new sweep outputs; verify privacy before setting the run gate."""
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite the existing sweep manifest")
    with np.load(DATA / "metadata.npz") as archive:
        trials = archive["trial_id"]
    counts = np.load(DATA / "counts.npy", mmap_mode="r")
    if len(trials) != len(counts):
        raise AssertionError("Neural rows and trial identities differ")
    bounds = np.r_[0, np.flatnonzero(trials[1:] != trials[:-1]) + 1, len(trials)]
    anchors = np.concatenate([
        np.arange(start + 9, stop, dtype=np.int64)
        for start, stop in zip(bounds[:-1], bounds[1:]) if stop - start >= 10
    ])
    source = anchors[:, None] + np.arange(-9, 1, dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors):
        raise AssertionError("Duplicate endpoints")
    if not np.all(trials[source] == trials[anchors, None]):
        raise AssertionError("A window crosses trial boundaries")
    if len(anchors) != 130507:
        raise AssertionError(f"Unexpected endpoint count: {len(anchors)}")
    (OUTPUT / "shared").mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / "shared/index.npz", anchor_index=anchors,
             trial_id=trials[anchors], source_indices=source)
    api = wandb.Api(timeout=15)
    viewer = api.viewer
    attrs = viewer if isinstance(viewer, dict) else viewer._attrs
    if attrs.get("username") != "jkbhagatio" or attrs.get("email") != "jkbhagatio@gmail.com":
        raise RuntimeError("Configured W&B identity differs from the approved personal account")
    variables = {"entity": "jkbhagatio", "name": "NLDisco-paper"}
    query = gql("""query($entity:String!, $name:String!) {
        project(name:$name, entityName:$entity) { name access }
    }""")
    project = api.client.execute(query, variable_values=variables)["project"]
    if project is None:
        mutation = gql("""mutation($entity:String!, $name:String!) {
            upsertModel(input:{name:$name,entityName:$entity,access:"PRIVATE",
                description:"NLDisco paper experiments; neural-only representation sweeps"}) {
                model { name access }
            }
        }""")
        api.client.execute(mutation, variable_values=variables)
    project = api.client.execute(query, variable_values=variables)["project"]
    if str(project["access"]).upper() != "PRIVATE":
        raise RuntimeError(f"W&B project privacy is not PRIVATE: {project['access']}")
    manifest = {
        "sweep_id": OUTPUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_ready",
        "wandb_privacy_verified": True,
        "wandb_entity": "jkbhagatio", "wandb_project": "NLDisco-paper",
        "wandb_project_access": project["access"],
        "wandb_url": "https://wandb.ai/jkbhagatio/NLDisco-paper",
        "data_dir": str(DATA), "output_dir": str(OUTPUT),
        "source_root": str(ROOT), "source_script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(DATA / name) for name in ("counts.npy", "metadata.npz")},
        "index_sha256": sha256(OUTPUT / "shared/index.npz"),
        "neural_rows": len(counts), "neural_units": counts.shape[1],
        "n_trials": len(bounds) - 1, "n_windows": len(anchors),
        "window_bins": 10, "bin_seconds": .05, "stride": 1,
        "alignment": "One endpoint t from neural bins t-9..t; first9 trial bins excluded",
        "seed": 0, "epochs": 10,
        "behavior_policy": "No behavioral feature analysis or decoder evaluation in this sweep",
        "checkpoint_policy": "Save final trained models; no periodic recovery checkpoints",
        "gpu_memory_ceiling_bytes": 40_000_000_000,
        "batch_policy": "Neural-only throughput pilots; then fixed per method across configurations",
        "nldisco": [dict(k=k, D=d, levels=levels) for k, d, levels in itertools.product((16,32,48),(128,256),(2,3))],
        "cebra": [dict(D=d, temperature=temp, lag=lag) for d,temp,lag in itertools.product((32,64,128),(.3,1.),(1,5))],
        "langevinflow": [dict(hidden=h, learning_rate=lr, ramp_epochs=ramp) for h,lr,ramp in itertools.product((32,64,128),(.001,.003),(5,500))],
        "gpu_plan": {"nldisco": "local authorized H100 physical GPU1",
                     "cebra": "2 Slurm A100 workers", "langevinflow": "2 Slurm A100 workers"},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "windows": len(anchors),
                      "wandb": manifest["wandb_url"], "privacy": project["access"]}))


if __name__ == "__main__":
    main()
