# LangevinFlow environment

`LangevinFlow_CCN/` is the unmodified MIT-licensed published source from
<https://github.com/KingJamesSong/LangevinFlow_CCN>, pinned at
`68dcbfaa78b2371851842333ae933eea1f961832`. Keep its license and source intact.
The windowed paper adapter lives in `../sweeps/s10_endpoint_v1/langevinflow.py`.

`langevinflow-venv/` isolates Lightning 1.6.0 and compatibility dependencies while
inheriting the project environment through `churchland_project.pth`.
`langevinflow-requirements.txt` records the overlay pins. Recreate/check it with:

```bash
uv run --no-sync python experiments/churchland/environments/setup_langevinflow.py --check-only
# Omit --check-only to recreate the overlay when needed.
```

Run the endpoint/dynamics/checkpoint test in this environment:

```bash
uv run --no-project --python experiments/churchland/environments/langevinflow-venv/bin/python   python -m pytest -q experiments/churchland/sweeps/s10_endpoint_v1/test_langevinflow.py
```

Use the same interpreter for LangevinFlow training. The default report notebook
needs neither Lightning nor retraining: it reads exported latents and saved timings.
The published state components `[q, p, h]` are concatenated into the reported latent
space; a 64-dimensional state therefore exposes 192 coordinates. The adapter's
window and posterior-mean export checks remain beside the implementation.
