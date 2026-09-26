"""Regenerate source activity traceback for all four Churchland features.

Rows follow the manuscript's shared feature order and NLDisco selection rule:
illustrative latents for main-text features, both selectors for supplementary
features. Each map averages activity times the encoder gradient of the latent's
endpoint pre-activation over every eligible development window with native
positive latent activation. The right column shows temporal decoder weights.

Usage: uv run --no-sync python -m experiments.churchland.sweeps.s10_endpoint_v1.attribution_traceback
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from beartype import beartype  # noqa: E402

from experiments.attribution import mean_encoder_attribution  # noqa: E402
from nldisco.model import build_sed  # noqa: E402
from experiments.churchland.paper_results import (  # noqa: E402
    TITLES,
    selected_rows,
)

from . import comparison_features as shared  # noqa: E402
from . import feature_decoding as decoder  # noqa: E402
from . import nice_cutoffs_revision as revision  # noqa: E402
from .nldisco import configuration  # noqa: E402
from .publication_style import COLORMAP  # noqa: E402

OUTPUT = revision.OUTPUT
FIGURES = decoder.ROOT / "experiments/churchland/outputs/paper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024


def _batches(inputs: np.ndarray, source: np.ndarray):
    for start in range(0, len(source), BATCH):
        rows = source[start : start + BATCH, None] + np.arange(-9, 1)
        yield torch.from_numpy(np.ascontiguousarray(inputs[rows]))


@beartype
def main() -> None:
    """Compute active-window attribution maps and render the six selected latents."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    spec = json.loads((OUTPUT / "frozen_spec.json").read_text())
    record = spec["representations"]["nldisco"]
    folder = Path(record["path"]).parent
    ckpt = torch.load(folder / "final_model.pt", map_location="cpu", weights_only=True)
    model = build_sed(configuration(191, 192, 16, 2))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE).eval()
    with np.load(folder / "preprocessing.npz") as pre:
        counts = np.load(decoder.DATA / "counts.npy").astype(np.float32)
        inputs = ((counts - pre["mean"]) / pre["scale"]).astype(np.float32)
    weights = ckpt["state_dict"]["decoder.weight"].numpy()
    activations = decoder.load_representation(record, shared.SWEEP)
    with np.load(shared.SWEEP / "shared/index.npz") as data:
        anchors = data["anchor_index"]
    features = revision.features()
    selections = [row for row in selected_rows(spec["selections"])
                  if row["method"] == "nldisco"]

    # The rebuilt model must reproduce the frozen endpoint activations on real windows.
    probe = np.arange(BATCH)
    with torch.inference_mode():
        batch = next(_batches(inputs, anchors[probe])).to(DEVICE)
        occurrence = torch.zeros(len(probe), 10, dtype=torch.bool, device=DEVICE)
        occurrence[:, -1] = True
        reproduced = model(batch, occurrence_mask=occurrence).sparse_acts[192][:, -1].cpu().numpy()
    np.testing.assert_allclose(reproduced, activations[probe], rtol=1e-3, atol=1e-3)

    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(selections), 2, figsize=(7.2, 9.4), layout="constrained")
    exported, provenance = {}, []
    for i, r in enumerate(selections):
        name, criterion = r["feature"], r["criterion"]
        f = features[name]
        latent = int(r["latent_id"])
        active = activations[f.rows, latent] > 0
        source = anchors[f.rows][active]
        if len(f.rows) != r["n_bins"] or int(f.labels.sum()) != r["positive_bins"]:
            raise AssertionError(f"Development population changed for {name}")
        # Use the native activation export for averaging, rather than counts in
        # manuscript-facing selection summaries. Record the actual endpoints below.
        attribution = mean_encoder_attribution(model, _batches(inputs, source), latent).T
        if attribution.shape != (191, 10) or not np.isfinite(attribution).all():
            raise AssertionError(f"Invalid attribution map for {name}, latent {latent}")
        print(f"{name} ({criterion}): latent {latent}, {len(source):,} active development windows", flush=True)
        key = f"{name}_{criterion}"
        exported[f"{key}_attribution"] = attribution
        exported[f"{key}_decoder_weights"] = weights[latent]
        exported[f"{key}_active_windows"] = np.asarray(len(source))
        exported[f"{key}_endpoint_indices"] = source
        exported[f"{key}_latent_id"] = np.asarray(latent)
        # Preserve the original numeric keys used by readers of the two main maps.
        if criterion == "illustrative":
            exported[f"{name}_attribution"] = attribution
            exported[f"{name}_active_windows"] = np.asarray(len(source))
        selector = {"illustrative": "illustrative", "sel": "best Sel", "auroc": "best AUROC"}[criterion]
        limits = []
        for j, (array, title, label) in enumerate([
                (attribution, "Encoder attribution", "Mean attribution"),
                (weights[latent], "Temporal decoder weights", "Decoder weight")]):
            limit = float(np.quantile(abs(array), .99))
            limits.append(limit)
            im = axes[i, j].imshow(array, aspect="auto", origin="lower", cmap=COLORMAP,
                                   vmin=-limit, vmax=limit, extent=[-.475, .025, -.5, 190.5])
            axes[i, j].set_title(f"{TITLES[name]}: {latent} ({selector})\n{title}", fontsize=7)
            axes[i, j].set(xlabel="Time relative to endpoint (s)", ylabel="Recorded channel")
            fig.colorbar(im, ax=axes[i, j], shrink=.8).set_label(label)
        provenance.append(dict(feature=name, criterion=criterion, latent_id=latent,
                               active_windows=len(source), absolute_color_limits=limits))
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"traceback_attribution.{extension}", dpi=180)
    plt.close(fig)
    np.savez_compressed(OUTPUT / "traceback_attribution.npz", lags_s=np.arange(-9, 1) * .05,
                        **exported)
    (OUTPUT / "traceback_attribution.json").write_text(json.dumps(dict(
        checkpoint=str(folder / "final_model.pt"), preprocessing=str(folder / "preprocessing.npz"),
        partition="behaviorally valid development windows for each feature",
        activation_rule="native z > 0; includes condition-positive and condition-negative windows",
        attribution="input times gradient of endpoint pre-activation; equal weight per active window",
        device=DEVICE, batch_size=BATCH, rows=provenance,
    ), indent=2) + "\n")
    for extension in ("pdf", "png"):
        (FIGURES / f"churchland_traceback.{extension}").write_bytes(
            (OUTPUT / f"traceback_attribution.{extension}").read_bytes())


if __name__ == "__main__":
    main()
