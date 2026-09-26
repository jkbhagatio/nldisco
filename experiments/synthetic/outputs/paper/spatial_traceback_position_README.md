# Spatial source activity traceback versus position (Figure S8)

Regenerate with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg uv run --no-sync python -m experiments.synthetic.scripts.attribution_traceback --spatial-only
```

The five selected vanilla SED runs per field, latent IDs, original training standardization, and unique positive-activation source bins are unchanged. Gradient-times-input attribution is evaluated for those bins, then averaged within 20 equal position bins on [0, 1] m (the last bin includes 1 m). Each run contributes equally to the displayed mean at positions where it has active bins; missing positions remain NaN, never zero-filled. Shading for the four place cells shows sample standard deviation across contributing runs. Curves are unsmoothed and all 100 neurons are plotted. The fourth place cell is dotted because its center matches the third cell.

`appendix_unit_attribution.npz` retains the original per-run, per-unit pooled attribution arrays and adds position edges, centers, per-run position attribution arrays, and active-bin counts. Position means weighted by those counts recover the original pooled attributions. Selected seeds and latent IDs remain in this numerical artifact.

