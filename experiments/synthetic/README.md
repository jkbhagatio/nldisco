# Synthetic results — sections 4.1 and 6.4.1

[analysis.ipynb](analysis.ipynb) is the report notebook for Figure 3 and **all eight
synthetic appendix figures S4–S11**, including the PCA comparison. It regenerates
plots from saved numeric artifacts without retraining or modifying the paper.

```bash
uv run --no-sync python -m experiments.synthetic.reproduce
```

Outputs go to `outputs/reproduced/`. The notebook shows the same stages separately.

| Figure | Artifact stem |
| --- | --- |
| 3 | `figure3` |
| S4: place-field design | `supplement_rate_design` |
| S5: FW-SED/TW-SED comparison | `supplement_temporal_comparison` |
| S6: right/left temporal features | `supplement_temporal_features` |
| S7: clean crossings | `supplement_ramp_conditioned` |
| S8: spatial source attribution | `supplement_unit_traceback` |
| S9: temporal source attribution | `supplement_temporal_coactivity` |
| S10: spatial PCA results | `supplement_pca_spatial` |
| S11: directional PCA results | `supplement_pca_direction` |

## Retained inputs

`data/matryoshka_overlap/` is the common 600-s, 100-unit recording used by all models.
`data/paper_simulation/` is its deterministic generation input. See [data/README.md](data/README.md).
`outputs/runs/` retains all ten vanilla, five FW-SED, five TW-SED and seven
Matryoshka fits: even unillustrated runs are needed for the reported recovery counts.
Training configs, checkpoints, activations, window indices and selection records remain.

`outputs/paper/` retains the frozen numerical source-attribution arrays and selected
feature summaries. The active source activity figures use **gradient times input**,
not the superseded coactivity displays. Spatial attributions average equally within
position bins and across selected runs; temporal attributions average over all
native-active windows. The notebook optionally recomputes these from checkpoints
with `recompute=True`. Its default renders the saved attribution arrays, checking
temporal seed/latent identities.

`standalone_baselines_20260916/pca/` retains the PCA fit, analysis code and numerical
artifacts behind S10/S11. The renderer checks source hashes and window alignment.
The NMF comparison, former Matryoshka-prefix display, separate forward/reverse
previews, and old coactivity plots/archives are removed. Dated model/data paths retain
artifact identity; they are not separate report workflows.

## Fixed training and selection

Vanilla: seeds 0–9, 128 latents, BatchTopK 4, one 10-ms sample, full-recording
per-unit z-scoring, MSLE (tau 1), bfloat16, batch 1024, learning rate 0.005,
20 epochs. Recovery requires fixed shape/position/support criteria and the expected
source unit being strongest in positive decoder loading and conditional coactivity.
All ten seeds are counted; at most five passing runs are displayed per cell.
The notebook refuses to silently render fewer than five runs in main panel c.

Matryoshka: original seeds 0–4 plus new seeds 5/6, nested prefixes 16/32/64/128,
TopK 4 at each prefix, equal reconstruction weights, otherwise the same per-bin
training settings. Joint recovery requires distinct qualifying latents in one run
at one prefix. Joint hits are 0/7, 0/7, 1/7, and 5/7 respectively. The extension
policy in `outputs/runs/matryoshka_overlap/extension_plan.json` retains the original
four joint hits (0/1/2/3), adding the strongest new joint hit by minimum pair shape
score, ties by lower seed. Seed 5 passes for both cells; seed 6 passes only for the
broad cell at prefix 128. The five-run band describes selected successful runs,
not uncertainty over all seven models. Prefix recovery counts include all seven runs.
This does not establish a
unique Matryoshka benefit or consistent broad-to-narrow ordering.

Window models: five seeds per architecture, 128 latents, BatchTopK 4, raw 50-ms counts,
MSE, float32, batch 256, learning rate 0.005, 40 epochs. Motion-only selection
chooses 100 raw samples (1 s support, 0.99 s timestamp span), or twenty tokens.
Transformer: two causal layers, four heads, width 128, feedforward width 256,
last-token readout. Both architectures reconstruct the complete window.

Fixed temporal gates require signed ramp-matched DI ≥0.25, signed first-pair
decoder-center displacement ≥0.01 m, pair/noise positive energy enrichment ≥2,
at least 20 active eligible windows and two matched position bins, each pair
unit contributing ≥10% of pair energy, and the pair ≥50% of known-place energy.
Passing features rank by signed DI × signed displacement × log(1+enrichment).
The strongest passing feature across all ten temporal runs is used per direction.
Global DI, unrestricted top windows, and 100 circular-shift maximum-score references
are reported separately. These are descriptive, not independent significance tests.

Fresh vanilla hit counts for cells 1–4 are 9/10, 6/10, 8/10, and 8/10; three runs
recover all four. FlatWindow forward/reverse hits are 2/5 and 1/5; transformer
hits are 3/5 in each direction. Main e uses transformer seed 0, latent 20; the
reverse example uses transformer seed 2, latent 66. Both have 16/16 desired-direction
crossing-conditioned examples, versus 15/16 and 12/16 unrestricted examples.
No older-dataset fallback was needed.

Exact fresh recovery counts, selected seed/latent identities, and diagnostics are
generated in [outputs/paper/results_summary.json](outputs/paper/results_summary.json)
and adjacent CSVs. [outputs/paper/artifact_manifest.json](outputs/paper/artifact_manifest.json)
records SHA-256 hashes. Dataset fingerprints are checked before plotting.
Fresh vanilla and temporal runs use the overlap recording; retained Matryoshka
configs keep their original historical “exploratory” wording without rewriting provenance.


## Retrain from scratch

Preserve the existing canonical run directories before rerunning these commands.
Vanilla refuses existing seed directories; temporal training can resume a matching
checkpoint, so **use empty output directories for fresh models**. Available GPUs
must be selected locally; the commands below assume GPU 1 is available.

```console
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.synthetic.scripts.train_vanilla \
  --seeds 0 1 2 3 4 5 6 7 8 9
uv run --no-sync python -m experiments.synthetic.scripts.aggregate_results

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python experiments/synthetic/scripts/train_window.py \
  --architecture FlatWindow --seeds 0 1 2 3 4
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python experiments/synthetic/scripts/train_window.py \
  --architecture TransformerWindow --seeds 0 1 2 3 4
uv run --no-sync python experiments/synthetic/scripts/finalize_temporal.py --architecture FlatWindow
uv run --no-sync python experiments/synthetic/scripts/finalize_temporal.py --architecture TransformerWindow
```

Training scripts accept `--output-dir`; the vanilla aggregator accepts matching
`--runs-dir` / `--output-dir`. Finalization and the paper notebook use the canonical
directories. The retained Matryoshka runs need not be trained again; to recreate
them in an empty run directory, use:

```console
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.synthetic.scripts.matryoshka_overlap \
  --seeds 0 1 2 3 4 5 6 --epochs 20 --device cuda
```

To regenerate input arrays, first run `scripts/simulate.py` into an absent
`data/paper_simulation/`, with parameter/motion/place/noise seeds 0/42/123/456.
Then call `matryoshka_overlap.generate_variant()` with an absent
`data/matryoshka_overlap/`. Both generators refuse to overwrite existing data.
The existing source recording is retained to make this derivation explicit.


Run scientific checks with `uv run --no-sync pytest experiments` from the repository root.
