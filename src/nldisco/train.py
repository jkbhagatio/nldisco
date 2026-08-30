"""SED model setup and training."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch as t
import wandb
from einops import asnumpy, einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import r2_score
from torch import Tensor, bfloat16, nn
from torch.nn import functional as F
from tqdm import tqdm

from nldisco import plot as mp
from nldisco.util import vec_r2

# <s> SED class config

@dataclass
class SedConfig:
    """Config class to set some params for the batch-topk MSAE."""
    n_input: int  # number of inputs to the MSAE
    dsed_topk_map: Dict[int, int]  # {d_sed: topk} pairing for the MSAE levels
    dsed_loss_x_map: Dict[int, int]  # {d_sed: loss_x} pairing for the MSAE levels
    seq_len: int = 1  # number of time bins in an input sequence
    n_instances: int = 2  # number of model instances to optimize in parallel
    dtype: t.dtype = bfloat16  # data type for the model and spike data


class Sed(nn.Module):
    """SED model for learning sparse representations of binned spike counts."""
    # Shapes of weights and biases for the encoder and decoder in the single-layer SED.
    W_enc: Float[Tensor, "inst d_sed (in_sed seq_len)"]
    W_dec: Float[Tensor, "inst in_sed d_sed"]
    b_enc: Float[Tensor, "inst d_sed"]
    b_dec: Float[Tensor, "inst in_sed"]

    def __init__(self, cfg: SedConfig):
        """Initializes model parameters."""
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.n_input * cfg.seq_len  # expand input dim for sequences
        d_levels = cfg.dsed_topk_map.keys()
        d_sed = max(d_levels)
        dtype = cfg.dtype

        # Tied weights initialization to reduce dead neurons (https://arxiv.org/pdf/2406.04093).
        self.W_enc = t.empty((cfg.n_instances, d_sed, in_dim), dtype=dtype)
        self.W_enc = nn.init.kaiming_normal_(self.W_enc, mode="fan_in", nonlinearity="relu")
        self.W_dec = rearrange(
            self.W_enc[..., :cfg.n_input], "inst d_sed in_sed -> inst in_sed d_sed"
        ).clone()
        self.W_enc, self.W_dec = nn.Parameter(self.W_enc), nn.Parameter(self.W_dec)

        self.b_enc = nn.Parameter(t.zeros((cfg.n_instances, d_sed), dtype=dtype))
        self.b_dec = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_input), dtype=dtype))

    def forward(self, x: Float[Tensor, "batch inst seq in_sed"]) -> (
        Tuple[
            List[Float[Tensor, "batch inst d_level"]],  # reconstructions for each level
            List[Float[Tensor, "batch inst d_level"]],  # topk activations per level
            Float[Tensor, "batch inst d_sed"]  # activations for all neurons
        ]
    ):
        """Computes loss as a function of SED feature sparsity and spike_count reconstructions."""
        # Compute encoder activations.
        batch_sz = x.shape[0]
        x = rearrange(x, "batch inst seq in_sed -> batch inst (seq in_sed)")
        acts_enc = einsum(x, self.W_enc, "batch inst in_dim, inst d_sed in_dim -> batch inst d_sed")
        acts_enc += self.b_enc
        acts_enc = F.relu(acts_enc)

        d_levels = sorted(self.cfg.dsed_topk_map.keys())
        recon_levels = []
        topk_acts_levels = []
        for d_l in d_levels:
            # Attempt reconstruction separately for each level in the group.
            level_acts = acts_enc[..., :d_l]
            batch_topk = batch_sz * self.cfg.n_instances * self.cfg.dsed_topk_map[d_l]
            feat_keep_vals, feat_keep_idxs = level_acts.ravel().topk(batch_topk)
            topk_acts = level_acts.ravel().zero_().scatter_(
                0, feat_keep_idxs, feat_keep_vals
            ).view_as(level_acts)
            topk_acts_levels.append(topk_acts)
            # Compute reconstructed input.
            W_dec_slice = self.W_dec[..., :d_l]
            x_recon = einsum(
                topk_acts, W_dec_slice, "batch inst d_l, inst in_sed d_l -> batch inst in_sed"
            )
            x_recon += self.b_dec
            x_recon = F.relu(x_recon)  # ensure reconstructed spikes are non-negative
            recon_levels.append(x_recon)

        return recon_levels, topk_acts_levels, acts_enc

    @t.no_grad()
    def norm_decoder(self) -> None:
        """Weight norm (l2) for the hidden dimension via projected gradients."""
        self.W_dec.data /= self.W_dec.norm(dim=1, p=2, keepdim=True)
        # Update grad to keep weights normalized for optimizer step.
        W_dec_grad_dot = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True)
        W_dec_grad_proj = W_dec_grad_dot * self.W_dec
        self.W_dec.grad -= W_dec_grad_proj  # subtract grad proj to ensure weights stay normed

# </s>

# <s> Loss and optimization functions

# <ss> Loss functions for reconstruction.

def mse(
    x: Float[Tensor, "batch inst in_sed"],  # input
    x_recon: Float[Tensor, "batch inst in_sed"],  # reconstruction
    **kwargs: Optional[Dict],  # catch additional parameters (for interchangeable call with `msle`)
) -> Float[Tensor, "batch inst"]:
    """Computes the mean squared error loss between true input and reconstruction."""
    return reduce((x - x_recon).pow(2), "batch inst in_sed -> batch inst", "mean")


def msle(
    x: Float[Tensor, "batch inst in_sed"],  # input
    x_recon: Float[Tensor, "batch inst in_sed"],  # reconstruction
    tau: int = 1,  # relative overestimation/underestimation penalty (1 for symmetric)
) -> Float[Tensor, "batch inst"]:
    """Computes the mean squared log error loss between true input and reconstruction."""
    return reduce(
        (tau * t.log(x_recon + 1) - t.log(x + 1)).pow(2), "batch inst in_sed -> batch inst", "mean"
    )


def res_recon_loss(
    x: Float[Tensor, "batch inst in_sed"],  # input
    x_recon: Float[Tensor, "batch inst in_sed"],  # reconstruction
    sed: Sed,  # SED model
    acts_enc: Float[Tensor, "batch inst d_sed"],  # activations
    dead_features: Bool[Tensor, "inst d_sed"],  # mask of dead neurons
    loss_fn: callable,  # loss function to use for the auxiliary loss
    max_revive_frac: float = 0.1,  # max fraction of dead neurons used for residual reconstruction
    **loss_fn_kwargs: Optional[Dict],  # kwargs for the loss function
) -> Float[Tensor, "batch inst"]:
    """Computes an auxiliary loss for dead neurons that perform a residual reconstruction."""
    res = x - x_recon  # the residual to try to reconstruct.

    n_dead = dead_features.sum().item()
    if n_dead:  # if dead neurons, try to reconstruct the residual from topk dead
        topk_aux = min(n_dead, int(max_revive_frac * dead_features.shape[-1]))
        acts_dead = acts_enc * repeat(
            dead_features, "inst d_sed -> batch inst d_sed", batch=x.shape[0]
        )
        feat_keep_vals, feat_keep_idxs = acts_dead.ravel().topk(topk_aux)
        topk_dead_acts = acts_dead.ravel().zero_().scatter_(
            0, feat_keep_idxs, feat_keep_vals
        ).view_as(acts_dead)
        res_recon = einsum(
            topk_dead_acts,
            sed.W_dec,
            "batch inst d_sed, inst in_sed d_sed -> batch inst in_sed"
        )
        return loss_fn(res, res_recon, **loss_fn_kwargs)

    return reduce(t.zeros_like(x, device=x.device), "batch inst in_sed -> batch inst", "mean")

# </ss>

def simple_cosine_lr_sched(step: int, n_steps: int, initial_lr: float, min_lr: float):
    """Learning rate schedule with warmup, decay and cosyne cycle."""
    n_warmup_steps = int(n_steps * 0.1)
    decay_start_step = int(n_steps * 0.5)
    n_decay_steps = n_steps - decay_start_step
    n_cycle_steps = int(n_decay_steps * 0.2)

    # Warmup phase
    if step < n_warmup_steps:
        return max(initial_lr * (step / n_warmup_steps), min_lr)

    # Decay phase: cosine decay with cycles
    if step >= decay_start_step:
        decay_steps = n_steps - decay_start_step
        decay_position = (step - decay_start_step) / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_position))
        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay
        cycle_position = ((step - decay_start_step) % n_cycle_steps) / n_cycle_steps
        cycle_factor = 0.5 * (1 + math.cos(2 * math.pi * cycle_position))
        cycle_amplitude = 0.1 * (initial_lr - min_lr)

        return decayed_lr + cycle_amplitude * cycle_factor

    # Constant phase: between warmup and decay start
    return initial_lr


def optimize(
    spk_cts: Int[Tensor, "n_examples n_units"],
    sed: Sed,
    loss_fn: callable,
    optimizer: t.optim.Optimizer,
    use_lr_sched: bool,
    dead_neuron_window: int,  # min consec steps a feature didn't fire for it to be considered dead
    n_steps: int,
    log_freq: int,
    batch_sz: int = 1024,
    log_wandb: bool = False,
    plot_l0: bool = False,
    **loss_fn_kwargs: Optional[Dict],
):
    """Optimizes the autoencoder."""
    device=spk_cts.device
    l0_history = []  # history of l0 mean and std for each step
    data_log = {
        "frac_active": {},
        "loss": {},
        "l0": {}
    }
    n_examples, _n_units = spk_cts.shape
    n_inst = sed.cfg.n_instances
    seq_len = sed.cfg.seq_len
    valid_starts = n_examples - seq_len + 1  # valid start indices for sequences
    d_sed = max(sed.cfg.dsed_topk_map.keys())  # max number of features in the SED
    n_steps_features_inactive = t.zeros((n_inst, d_sed), dtype=int, device=device)
    dead_features = t.zeros((n_inst, d_sed), dtype=bool, device=device)

    lr = optimizer.param_groups[0]["lr"]
    if use_lr_sched:
        min_lr = lr * 1e-2

    pbar = tqdm(range(n_steps), desc="SED batch training step")
    for step in pbar:

        if use_lr_sched:
            optimizer.param_groups[0]["lr"] = (
                simple_cosine_lr_sched(step, n_steps, lr, min_lr)
            )

        # Get batch of spike counts to feed into SED.
        start_idxs = t.randint(0, valid_starts, (batch_sz, sed.cfg.n_instances))
        seq_idxs = start_idxs.unsqueeze(-1) + t.arange(seq_len)  # broadcast seq idxs to new dim
        spike_count_seqs = spk_cts[seq_idxs]  # [batch_sz, n_instances, seq_len, n_units]

        # Forward pass -- get reconstruction loss for each level:
        # take loss between reconstructions and last timebin (sequence) of true spike counts
        optimizer.zero_grad()
        recon_levels, topk_acts_levels, acts_enc = sed(spike_count_seqs)  # forward
        recon_loss = t.zeros((batch_sz, sed.cfg.n_instances), device=device)
        loss_xs = list(dict(sorted(sed.cfg.dsed_loss_x_map.items())).values())  # sorted by d_sed
        for level_idx, recon_level in enumerate(recon_levels):
            recon_loss += (
                loss_fn(spike_count_seqs[..., -1, :], recon_level, **loss_fn_kwargs)
                * loss_xs[level_idx]
            )
        recon_loss = reduce(recon_loss, "batch inst -> ", "mean")

        # Save these gradients before computing gradients for dead neurons for aux loss
        recon_loss.backward(retain_graph=True)
        grad_buffer = {
            name: p.grad.clone() for name, p in sed.named_parameters() if p.grad is not None
        }

        # Get auxiliary loss for dead neurons
        if dead_features.any():
            optimizer.zero_grad()
            aux_loss = res_recon_loss(
                x=spike_count_seqs[..., -1, :],
                x_recon=recon_levels[-1],
                sed=sed,
                acts_enc=acts_enc,
                dead_features=dead_features,
                loss_fn=loss_fn,
                **loss_fn_kwargs
            )
            aux_loss = reduce(aux_loss, "batch inst -> ", "mean")

            # Apply aux loss grads to dead neurons only (mask out active neurons)
            aux_loss.backward()
            for name, p in sed.named_parameters():
                if p.grad is not None:
                    if "W_enc" in name:
                        p.grad *= dead_features.unsqueeze(2)  # broadcast to input dim
                    elif "W_dec" in name:
                        p.grad *= dead_features.unsqueeze(1)  # broadcast to output dim
                    elif "b_enc" in name:  # don't need for 'b_dec': acts as global offset
                        p.grad *= dead_features

                    p.grad += grad_buffer.get(name, 0)  # add in gradients from recon_loss backward

        sed.norm_decoder()  # normalize decoder weights
        optimizer.step()

        feat_active = reduce(
            topk_acts_levels[-1], "batch inst d_sed -> inst d_sed", "sum"
        ) > 0
        n_steps_features_inactive[feat_active] = 0
        n_steps_features_inactive[~feat_active] += 1
        dead_features = n_steps_features_inactive > dead_neuron_window

        # Display progress bar, and append new values for plotting.
        if step % log_freq == 0 or (step + 1 == n_steps):
            # import ipdb; ipdb.set_trace()
            l0 = reduce(topk_acts_levels[-1] > 0, "batch inst d_sed -> batch inst", "sum").float()
            l0_mean, l0_std = l0.mean().item(), l0.std().item()
            frac_dead = dead_features.float().mean().item()
            pbar.set_postfix(loss=f"{recon_loss.item():.5f},  {l0_mean=}, {l0_std=}, {frac_dead=}")
            data_log["l0"][step] = {"mean": l0_mean, "std": l0_std}
            data_log["loss"][step] = recon_loss.item()

            if log_wandb:
                wandb.log(
                    {"loss": recon_loss.item(), "l0_mean": l0_mean, "l0_std": l0_std, "step": step}
                )

                if dead_features.any():
                    wandb.log({"frac_dead": frac_dead, "step": step})

                if plot_l0:
                    alpha = 0.3 + (0.7 * step / n_steps)  # alpha from 0.3 to 1.0
                    l0_history.append(
                        {"step": step, "mean": l0_mean, "std": l0_std, "alpha": alpha}
                    )
                    l0_fig = mp.plot_l0_stats(l0_history)
                    wandb.log({"l0_std_vs_mean": l0_fig, "step": step})

    return data_log


def eval_model(
    spk_cts: Int[Tensor, "n_examples n_units"],
    sed: Sed,
    batch_sz: int = 1024,
    log_wandb: bool = False,
) -> Tuple[
    Figure,
    # 4d topk acts info (instance, example, feature, act_val)
    Float[Tensor, "(n_recon_examples n_inst max_topk) 4"],
    Float[Tensor, "n_recon_examples n_inst n_units"],  # reconstructions
    Float[np.ndarray, "n_units n_inst"],  # R² per unit
    Float[Tensor, "n_recon_examples n_inst"],  # R² per example
    Float[Tensor, "n_units n_inst"],  # Cosine similarity per unit
    Float[Tensor, "n_recon_examples n_inst"],  # Cosine similarity per example
]:
    """Evaluates the model after training, and generates plots/metrics.

    Plots/Metrics:
    1. L0 boxplot (per example)
    2. Latent activity-density histogram
    3a. Cosine-Similarity boxplot of reconstructions vs. true over all units (per example)
    3b. Cosine-Similarity boxplot of reconstructions vs. true over all examples (per unit)
    4a. R² boxplot of reconstructions vs. true over all units (per example)
    4b. R² boxplot of reconstructions vs. true over all examples (per unit)

    """
    device = spk_cts.device
    n_inst = sed.cfg.n_instances
    n_units = spk_cts.shape[1]
    n_examples = spk_cts.shape[0]
    valid_starts = n_examples - sed.cfg.seq_len + 1
    d_sed = max(sed.cfg.dsed_topk_map.keys())
    n_steps = valid_starts // batch_sz  # total number of examples
    n_recon_examples = n_steps * batch_sz

    # <ss> Run examples through model and compute metrics.

    # Create placeholders to store metrics.
    l0 = t.zeros((n_recon_examples, n_inst), dtype=t.float32, device=device)
    latent_activity_count = t.zeros((n_inst, d_sed), dtype=t.float32, device=device)
    recon_spk_cts = t.empty((n_recon_examples, n_inst, n_units), dtype=sed.cfg.dtype, device=device)
    r2_per_example = t.empty((n_recon_examples, n_inst), dtype=sed.cfg.dtype, device=device)
    cos_sim_per_example = t.empty((n_recon_examples, n_inst), dtype=sed.cfg.dtype, device=device)
    topk_acts_4d = []  # stores (inst_idx, ex_idx, feat_idx, act_val) for each topk act

    progress_bar = tqdm(range(n_steps), desc="SED batch evaluation step")
    with t.no_grad():
        for step in progress_bar:  # loop over all examples
            # Get start index for each seq in batch, and then get the full seq indices.
            start_idxs = t.arange(step * batch_sz, (step + 1) * batch_sz)
            seq_idxs = repeat(start_idxs, "batch -> batch inst", inst=n_inst)
            seq_idxs = seq_idxs.unsqueeze(-1) + t.arange(sed.cfg.seq_len)  # broadcast to seq dim
            spike_count_seqs = spk_cts[seq_idxs]  # [batch, inst, seq, unit]
            # Forward pass through SED.
            recon_levels, topk_acts_levels, _acts_raw = sed(spike_count_seqs)
            nonzero_mask = (topk_acts_levels[-1] > 0)
            cur_l0 = reduce(nonzero_mask.float(), "batch inst sed_feat -> batch inst", "sum")
            latent_activity_count += reduce(
                nonzero_mask.float(), "batch inst sed_feat -> inst sed_feat", "sum"
            )
            # Store results.
            l0[start_idxs] = cur_l0
            recon_spk_cts[start_idxs] = recon_levels[-1]
            # Calculate metrics for examples.
            r2_per_example[start_idxs] = vec_r2(recon_levels[-1], spk_cts[start_idxs])
            cos_sim_per_example[start_idxs] = (
                t.cosine_similarity(recon_levels[-1], spk_cts[start_idxs].unsqueeze(1), dim=-1)
            )
            # Get top-k features and acts for examples in batch
            # (just need last level of msae -- it's a superset of the other levels)
            topk_acts = topk_acts_levels[-1][nonzero_mask]
            batch_ex_idxs, inst_idxs, feat_idxs = t.where(nonzero_mask)
            global_ex_idxs = batch_ex_idxs + start_idxs[0]
            cur_topk_acts_4d = t.stack(
                [global_ex_idxs.float(), inst_idxs.float(), feat_idxs.float(), topk_acts.float()],
                dim=1
            )
            topk_acts_4d.append(cur_topk_acts_4d)

    topk_acts_4d = t.cat(topk_acts_4d, dim=0)

    r2_per_example[~t.isfinite(r2_per_example)] = 0.0  # div by 0 cases

    # Calculate metrics for units.
    cos_sim_per_unit = t.empty((n_units, n_inst))
    r2_per_unit = np.empty((n_units, n_inst))

    spk_cts_np = asnumpy(spk_cts.float())
    recon_spk_cts_np = asnumpy(recon_spk_cts.float())

    for unit in range(n_units):
        cos_sim_per_unit[unit] = t.cosine_similarity(
            recon_spk_cts[..., unit], spk_cts[:n_recon_examples, unit].unsqueeze(-1), dim=0
        )
        for inst in range(n_inst):
            r2_per_unit[unit, inst] = r2_score(
                spk_cts_np[:n_recon_examples, unit], recon_spk_cts_np[:, inst, unit]
        )

    # </ss>

    # <ss> Create plots.

    fig, ((ax_l0, ax_density), (ax_r2, ax_cos)) = plt.subplots(
        2, 2, figsize=(12, 10), constrained_layout=True
    )
    sns.set_theme(style="whitegrid")

    # <sss> L0 boxplot.

    l0_data = [asnumpy(l0[:, i]) for i in range(n_inst)]
    mp.box_strip_plot(
        ax=ax_l0,
        data=l0_data,
        show_legend=True
    )
    # Update width of box plot and size and alpha of strip plot.
    # for box in ax_l0.artists:
    #     box.set_width(0.4)  # Change boxplot width
    # for collection in ax_l0.collections:
    #     if isinstance(collection, plt.matplotlib.collections.PathCollection):
    #         collection.set_sizes([4])
    #         collection.set_alpha(0.4)
    # Prettify axes.
    ax_l0.set_xlabel("")
    ax_l0.set_ylabel("")
    ax_l0.set_xticks(range(n_inst))
    ax_l0.set_xticklabels([f"SED {i}" for i in range(n_inst)])
    ax_l0.set_yticks(np.arange(0, l0.max().item() + 1, 10))
    ax_l0.set_title("L0 of SED features")

    # </sss>

    # <sss> Latent density histogram.

    latent_activity_frac = latent_activity_count / n_recon_examples
    for i in range(n_inst):
        data = asnumpy(latent_activity_frac[i])
        ax_density.hist(
            data,
            bins=max(1, d_sed // 5),
            alpha=0.6,
            weights=np.ones_like(data) / len(data),
            label=f"SED {i}",
        )
    ax_density.set_title("Latent activity density")
    ax_density.set_xlabel("Fraction of time active")
    ax_density.set_ylabel("Fraction of latents")
    ax_density.set_xticks(np.arange(0, 1.05, 0.05))
    ax_density.set_xlim(-0.025, 1.025)
    plt.setp(ax_density.get_xticklabels(), rotation=-40, ha="left")
    ax_density.legend()

    # </sss>

    # <sss> Format and plot R² and Cosine Similarity data.

    cos_sim_per_example = asnumpy(cos_sim_per_example.float())
    r2_per_example = asnumpy(r2_per_example.float())
    cos_sim_per_unit = asnumpy(cos_sim_per_unit.float())

    model_names = [f"SED {i}" for i in range(n_inst)]
    dfs = []

    dfs.append(
        pd.DataFrame(cos_sim_per_example, columns=model_names)
        .melt(var_name="SED", value_name="Value")
        .assign(Type="Examples", Metric="Cosine Similarity")
    )

    dfs.append(
        pd.DataFrame(cos_sim_per_unit, columns=model_names)
        .melt(var_name="SED", value_name="Value")
        .assign(Type="Units", Metric="Cosine Similarity")
    )

    dfs.append(
        pd.DataFrame(r2_per_example, columns=model_names)
        .melt(var_name="SED", value_name="Value")
        .assign(Type="Examples", Metric="R²")
    )

    dfs.append(
        pd.DataFrame(r2_per_unit, columns=model_names)
        .melt(var_name="SED", value_name="Value")
        .assign(Type="Units", Metric="R²")
    )

    df = pd.concat(dfs, ignore_index=True)

    cos_sim_df = df[df["Metric"] == "Cosine Similarity"]
    r2_df = df[df["Metric"] == "R²"]

    mp.box_strip_plot(
        ax=ax_r2,
        data=r2_df,
        x="Type",
        y="Value",
        hue="SED",
        show_legend=False
    )
    # Prettify axes.
    ax_r2.set_xlabel("")
    ax_r2.set_ylabel("")
    ax_r2.set_ylim(-1.0, 1.0)
    ax_r2.set_yticks(np.arange(-1.0, 1.1, 0.1))
    ax_r2.set_title("R² of SED reconstructions")

    mp.box_strip_plot(
        ax=ax_cos,
        data=cos_sim_df,
        x="Type",
        y="Value",
        hue="SED",
        show_legend=False
    )
    # Prettify axes.
    ax_cos.set_xlabel("")
    ax_cos.set_ylabel("")
    ax_cos.set_ylim(0.1, 1.0)
    ax_cos.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax_cos.set_title("Cosine Similarity of true and reconstructed spike counts")

    # </sss>

    # </ss>

    # <ss> Log to wandb.

    if log_wandb:

        # Log metrics figure
        wandb.log({"combined_metrics_plot": wandb.Image(fig)})
        plt.close(fig)

        # Log metrics values
        wandb.log({
            "r2_per_example_mean": np.mean(r2_per_example),
            "r2_per_unit_mean": np.mean(r2_per_unit),
            "cos_per_example_mean": np.mean(cos_sim_per_example),
            "cos_per_unit_mean": np.mean(cos_sim_per_unit),
        })

    return fig, topk_acts_4d, recon_spk_cts, r2_per_unit, r2_per_example, cos_sim_per_unit, cos_sim_per_example

    # </ss>

# </s>
