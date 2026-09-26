# Common-H100 training-time comparison

`training_time_h100.sh` requests one H100, four CPU cores, and 12 GB of RAM in
`gpu_branco`. `training_time_benchmark.py` runs five fresh fits per method, using
seeds 0–4 and rotating method order. All 15 fits run sequentially on the same
allocated GPU, identified by its UUID after Slurm's device remapping.

The benchmark reuses the selected 192-dimensional Churchland models and the
130,507 complete ten-bin windows. NLDisco and LangevinFlow receive ten exhaustive
epochs (1,280 updates); CEBRA-Time receives ten reference-draw epoch-equivalents
(1,275 updates of 1,024 reference draws with replacement). Method-specific
preprocessing, objectives, and configurations are retained.

The measured interval is CUDA-synchronized training-loop wall time. Initial
loading/preprocessing, model and optimizer construction, checkpoint saving, and
latent export are excluded. NLDisco threshold calibration is timed separately.
Training starts in fresh processes without discarded warm-up updates. No W&B
traffic occurs. GPU occupancy is sampled every two seconds, and a run with an
observed competing process cannot produce the final report.

## Outputs

Slurm job **3649999 completed successfully** in 19 minutes on 22 September 2026.
It was submitted with a `--partition=gpu_lowp` override and started without
preemption on physical H100 GPU 0 of `gpu-sr675-34`. Its outputs are under
`outputs/training_time_h100_3649999/`, with the job log at
`outputs/training-time-3649999.log`.

| Method | Median training time (s) | Interquartile interval (s) |
| --- | ---: | ---: |
| NLDisco | 43.42 | 43.36–44.10 |
| CEBRA-Time | 8.18 | 6.51–8.23 |
| LangevinFlow | 132.77 | 132.22–132.89 |

NLDisco's separately measured median calibration time was 3.61 seconds.
The paper includes the comparison in Section 6.4.2 and Figure S15.
`verification.json` records the final integrity checks and the minor plotting
and caption updates after the batch job finished; measured values were unchanged.
The title is "LVM training time comparison". The caption reports the recording,
examples per epoch, window definition, 192-D representations, parameter counts,
and timing protocol. `parameter_audit_<method>.json` records named parameter
counts and hashes of all five strictly reloaded checkpoints per method. Total
and trainable counts are identical: NLDisco 614,859; CEBRA-Time 270,784;
LangevinFlow 449,199 (buffers excluded).

Each run retains its configuration, history, final model, timing, and software
metadata. `manifest.json` records input/source hashes and run order;
`results.json` and `results.csv` retain all 15 observations; `statistics.json`
contains the medians and quartiles. `training_times.pdf` and
`training_times.png` contain the final boxplots with every observation overlaid.

Regenerate the current figure without retraining through `analysis.ipynb` or
`uv run --no-sync python -m experiments.churchland.paper_results`. The report uses
the GPU UUID in the saved measurements and writes under `outputs/paper/timing/`.

`training_time_h100.sh` remains the optional Slurm entry point for a fresh benchmark.
It writes only experiment artifacts; it no longer copies figures/captions into the
paper or invokes LaTeX. Failed/incomplete preliminary jobs are removed because they
contribute no observations to the reported fifteen-fit experiment.
