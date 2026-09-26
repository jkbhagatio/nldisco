# Examples and tutorials

[Paired-population transcoding](../../examples/paired_transcoder.py) is a complete
walkthrough using generated data: normalize two populations, train, evaluate
reconstruction and diagnostics, then predict targets from inputs alone.

```console
uv run python examples/paired_transcoder.py --layout flat
```

Other layouts are `single_bin`, `transformer`, and `shift_equivariant`.
Use `--epochs 1` for a shorter run. No external data files are required.

For a paper analysis, see the [synthetic place-cell experiment](../../experiments/synthetic/README.md).
New general tutorials should use generated or small example data and run independently
of paper artifacts. Follow the [canonical model names](../model_names.md).
