# Paper experiments

These workflows reproduce the **current `paper/iclr_paper/full_paper.tex` results**.
Only experiment code, data, fitted artifacts, and checks supporting those results are retained.

| Notebook | Paper outputs | Generated files |
| --- | --- | --- |
| [synthetic/analysis.ipynb](synthetic/analysis.ipynb) | Figure 3; Figures S4–S11; direction-index and recovery summaries | `synthetic/outputs/reproduced/` |
| [churchland/analysis.ipynb](churchland/analysis.ipynb) | Figure 4; Tables 1/S2; Figures S12–S15 | `churchland/outputs/paper/` |
| [aeon/analysis.ipynb](aeon/analysis.ipynb) | Figure 5; Table S3; Figures S16–S20 | `aeon/outputs/paper/` |

Open a notebook and run all cells in the project environment. The default workflow
replots numerical results and checks association/decoder scores; it does not retrain
representations. See each dataset's README for inputs and upstream training scripts.
Synthetic inputs are local. Churchland and Aeon require the existing frozen Ceph
artifacts; raw data/checkpoints on Ceph have not been moved or deleted.

```bash
uv run --no-sync jupyter lab experiments/synthetic/analysis.ipynb
# Or execute a notebook in place, including its rendered outputs:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg uv run --no-sync jupyter nbconvert \
  --to notebook --execute --inplace experiments/synthetic/analysis.ipynb \
  --ExecutePreprocessor.timeout=600
```

All notebook output stays in `experiments/`. **Nothing installs figures or edits
manuscript files.** Regeneration uses measured results, even when they differ from
the current draft. Churchland's `verification.json` and `churchland_appendix_metrics.json`
record measured scores alongside the frozen manuscript values. Aeon's
`wheel_tick_comparison.json` records the temporal excerpts regenerated from native
activations; these differ from the current manuscript tick geometry. The paper is unchanged.

Figures 1/2 and S3 are conceptual/assay illustrations. Figure S1's methods illustration
has its existing notebook under `paper/iclr_paper/figures/`; it is outside these dataset
workflows. The retained Allen-data methods illustration, Figure S2, is also excluded.

## Shared helpers and tests

`attribution.py` implements gradient-times-input source traceback; `decoder_metrics.py`
implements validation-selected thresholds and test balanced accuracy. `plot_style.py`
contains plot-label typography. `evaluate_decoder_balanced_accuracy.py` reproduces the
current Churchland decoding and the original Aeon wheel decoder from saved fits.
The newer `aeon/fit_appendix_decoders.py` fits all Aeon single/full-space decoders.

```bash
uv run --no-sync pytest experiments
```

This includes `experiments/tests/` and dataset-specific scientific checks, excluding
the vendored LangevinFlow environment. Its model test skips when Lightning is absent
from the project environment; run it separately using the documented overlay in
[churchland/environments/README.md](churchland/environments/README.md).

`cleanup_manifest.json` lists deleted files and the reason for removal. Historical
reports, abandoned drafts, retired PCA/NMF comparisons, and incomplete benchmark runs
are removed. Saved models, original selection records, trial/window identities, and
fits still used by the paper are retained, including older-revision decoder files
that remain inputs to current results. Dated input paths preserve artifact identity;
there is one maintained report notebook per dataset.
