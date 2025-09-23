"""MSAE model set up and training."""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch as t
import wandb
from einops import asnumpy, einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int, Bool
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from torch import bfloat16, nn, Tensor
from torch.nn import functional as F
from tqdm import tqdm

from mini import plot as mp
from mini.util import vec_r2

# <s> SAE class config

@dataclass
class SaeConfig:
    """Config class to set some params for the batch-topk MSAE."""
    n_input_ae: int  # number of inputs to the MSAE
    dsae_topk_map: Dict[int, int]  # {d_sae: topk} pairing for the MSAE levels
    dsae_loss_x_map: Dict[int, int]  # {d_sae: loss_x} pairing for the MSAE levels
    seq_len: int = 1  # number of time bins in an input sequence
    n_instances: int = 2  # number of model instances to optimize in parallel
    dtype: t.dtype = bfloat16  # data type for the model and spike data


class Sae(nn.Module):
    """SAE model for learning sparse representations of binned spike counts."""
    # Shapes of weights and biases for the encoder and decoder in the single-layer SAE.
    W_enc: Float[Tensor, "inst d_sae (in_sae seq_len)"]
    W_dec: Float[Tensor, "inst in_sae d_sae"]
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst in_sae"]

    def __init__(self, cfg: SaeConfig):
        """Initializes model parameters."""
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.n_input_ae * cfg.seq_len  # expand input dim for sequences
        d_levels = cfg.dsae_topk_map.keys()
        d_sae = max(d_levels)
        dtype = cfg.dtype

        # Tied weights initialization to reduce dead neurons (https://arxiv.org/pdf/2406.04093).
        self.W_enc = t.empty((cfg.n_instances, d_sae, in_dim), dtype=dtype)
        self.W_enc = nn.init.kaiming_normal_(self.W_enc, mode="fan_in", nonlinearity="relu")
        self.W_dec = rearrange(
            self.W_enc[..., :cfg.n_input_ae], "inst d_sae in_sae -> inst in_sae d_sae"
        ).clone()
        self.W_enc, self.W_dec = nn.Parameter(self.W_enc), nn.Parameter(self.W_dec)
        
        self.b_enc = nn.Parameter(t.zeros((cfg.n_instances, d_sae), dtype=dtype))
        self.b_dec = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_input_ae), dtype=dtype))

    def forward(self, x: Float[Tensor, "batch inst seq in_sae"]) -> (
        Tuple[
            List[Float[Tensor, "batch inst d_level"]],  # reconstructions for each level
            List[Float[Tensor, "batch inst d_level"]],  # topk activations per level
            Float[Tensor, "batch inst d_sae"]  # activations for all neurons
        ]
    ):
        """Computes loss as a function of SAE feature sparsity and spike_count reconstructions."""
        # Compute encoder activations.
        batch_sz = x.shape[0]
        x = rearrange(x, "batch inst seq in_sae -> batch inst (seq in_sae)")
        acts_enc = einsum(x, self.W_enc, "batch inst in_dim, inst d_sae in_dim -> batch inst d_sae")
        acts_enc += self.b_enc
        acts_enc = F.relu(acts_enc)
        
        d_levels = sorted(self.cfg.dsae_topk_map.keys())
        recon_levels = []
        topk_acts_levels = []
        for d_l in d_levels:
            # Attempt reconstruction separately for each level in the group.
            level_acts = acts_enc[..., :d_l]
            batch_topk = batch_sz * self.cfg.n_instances * self.cfg.dsae_topk_map[d_l]
            feat_keep_vals, feat_keep_idxs = level_acts.ravel().topk(batch_topk)
            topk_acts = level_acts.ravel().zero_().scatter_(
                0, feat_keep_idxs, feat_keep_vals
            ).view_as(level_acts)
            topk_acts_levels.append(topk_acts)
            # Compute reconstructed input.
            W_dec_slice = self.W_dec[..., :d_l]
            x_recon = einsum(
                topk_acts, W_dec_slice, "batch inst d_l, inst in_sae d_l -> batch inst in_sae"
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
    x: Float[Tensor, "batch inst in_sae"],  # input
    x_recon: Float[Tensor, "batch inst in_sae"],  # reconstruction
    **kwargs: Optional[Dict],  # catch additional parameters (for interchangeable call with `msle`)
) -> Float[Tensor, "batch inst"]:
    """Computes the mean squared error loss between true input and reconstruction."""
    return reduce((x - x_recon).pow(2), "batch inst in_sae -> batch inst", "mean")


def msle(
    x: Float[Tensor, "batch inst in_sae"],  # input
    x_recon: Float[Tensor, "batch inst in_sae"],  # reconstruction
    tau: int = 1,  # relative overestimation/underestimation penalty (1 for symmetric)
) -> Float[Tensor, "batch inst"]:
    """Computes the mean squared log error loss between true input and reconstruction."""
    return reduce(
        (tau * t.log(x_recon + 1) - t.log(x + 1)).pow(2), "batch inst in_sae -> batch inst", "mean"
    )


def res_recon_loss(
    x: Float[Tensor, "batch inst in_sae"],  # input
    x_recon: Float[Tensor, "batch inst in_sae"],  # reconstruction
    sae: Sae,  # SAE model
    acts_enc: Float[Tensor, "batch inst d_sae"],  # activations
    dead_features: Bool[Tensor, "inst d_sae"],  # mask of dead neurons
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
            dead_features, "inst d_sae -> batch inst d_sae", batch=x.shape[0]
        )
        feat_keep_vals, feat_keep_idxs = acts_dead.ravel().topk(topk_aux)
        topk_dead_acts = acts_dead.ravel().zero_().scatter_(
            0, feat_keep_idxs, feat_keep_vals
        ).view_as(acts_dead)
        res_recon = einsum(
            topk_dead_acts,
            sae.W_dec,
            "batch inst d_sae, inst in_sae d_sae -> batch inst in_sae"
        )
        return loss_fn(res, res_recon, **loss_fn_kwargs)

    return reduce(t.zeros_like(x, device=x.device), "batch inst in_sae -> batch inst", "mean")

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
    sae: Sae,
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
    n_inst = sae.cfg.n_instances
    seq_len = sae.cfg.seq_len
    valid_starts = n_examples - seq_len + 1  # valid start indices for sequences
    d_sae = max(sae.cfg.dsae_topk_map.keys())  # max number of features in the SAE
    n_steps_features_inactive = t.zeros((n_inst, d_sae), dtype=int, device=device)
    dead_features = t.zeros((n_inst, d_sae), dtype=bool, device=device)

    lr = optimizer.param_groups[0]["lr"]
    if use_lr_sched:
        min_lr = lr * 1e-2

    pbar = tqdm(range(n_steps), desc="SAE batch training step")
    for step in pbar:

        if use_lr_sched:
            optimizer.param_groups[0]["lr"] = (
                simple_cosine_lr_sched(step, n_steps, lr, min_lr)
            )

        # Get batch of spike counts to feed into SAE.
        start_idxs = t.randint(0, valid_starts, (batch_sz, sae.cfg.n_instances))
        seq_idxs = start_idxs.unsqueeze(-1) + t.arange(seq_len)  # broadcast seq idxs to new dim
        spike_count_seqs = spk_cts[seq_idxs]  # [batch_sz, n_instances, seq_len, n_units]
        
        # Forward pass -- get reconstruction loss for each level:
        # take loss between reconstructions and last timebin (sequence) of true spike counts
        optimizer.zero_grad()
        recon_levels, topk_acts_levels, acts_enc = sae(spike_count_seqs)  # forward
        recon_loss = t.zeros((batch_sz, sae.cfg.n_instances), device=device)
        loss_xs = list(dict(sorted(sae.cfg.dsae_loss_x_map.items())).values())  # sorted by d_sae
        for l in range(len(recon_levels)):
            recon_loss += (
                loss_fn(spike_count_seqs[..., -1, :], recon_levels[l], **loss_fn_kwargs)
                * loss_xs[l]
            )
        recon_loss = reduce(recon_loss, "batch inst -> ", "mean")
        
        # Save these gradients before computing gradients for dead neurons for aux loss
        recon_loss.backward(retain_graph=True)
        grad_buffer = {
            name: p.grad.clone() for name, p in sae.named_parameters() if p.grad is not None
        }
        
        # Get auxiliary loss for dead neurons
        if dead_features.any():
            optimizer.zero_grad()
            aux_loss = res_recon_loss(
                x=spike_count_seqs[..., -1, :],
                x_recon=recon_levels[-1],
                sae=sae, 
                acts_enc=acts_enc,
                dead_features=dead_features,
                loss_fn=loss_fn,
                **loss_fn_kwargs
            )
            aux_loss = reduce(aux_loss, "batch inst -> ", "mean")
        
            # Apply aux loss grads to dead neurons only (mask out active neurons)
            aux_loss.backward()
            for name, p in sae.named_parameters():
                if p.grad is not None:
                    if "W_enc" in name:
                        p.grad *= dead_features.unsqueeze(2)  # broadcast to input dim
                    elif "W_dec" in name:
                        p.grad *= dead_features.unsqueeze(1)  # broadcast to output dim
                    elif "b_enc" in name:  # don't need for 'b_dec': acts as global offset
                        p.grad *= dead_features
                
                    p.grad += grad_buffer.get(name, 0)  # add in gradients from recon_loss backward
            
            else:  # restore recon_loss gradients
                for name, p in sae.named_parameters():
                    if p.grad is not None and name in grad_buffer:
                        p.grad = grad_buffer[name]
        
        sae.norm_decoder()  # normalize decoder weights
        optimizer.step()

        feat_active = reduce(
            topk_acts_levels[-1], "batch inst d_sae -> inst d_sae", "sum"
        ) > 0
        n_steps_features_inactive[feat_active] = 0
        n_steps_features_inactive[~feat_active] += 1
        dead_features = n_steps_features_inactive > dead_neuron_window

        # Display progress bar, and append new values for plotting.
        if step % log_freq == 0 or (step + 1 == n_steps):
            # import ipdb; ipdb.set_trace()
            l0 = reduce(topk_acts_levels[-1] > 0, "batch inst d_sae -> batch inst", "sum").float()
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
    sae: Sae,
    batch_sz: int = 1024,
    log_wandb: bool = False
) -> Tuple[
    # 4d topk acts info (instance, example, feature, act_val)
    Float[Tensor, "(n_recon_examples n_inst max_topk) 4"],
    Float[Tensor, "n_recon_examples n_inst n_units"],  # reconstructions
    Float[Tensor, "n_units n_inst"],  # R² per unit
    Float[Tensor, "n_recon_examples n_inst"],  # R² per example
    Float[Tensor, "n_units n_inst"],  # Cosine similarity per unit
    Float[Tensor, "n_recon_examples n_inst"],  # Cosine similarity per example
]:
    """Evaluates the model after training, and generates plots/metrics.
    
    Plots/Metrics:
    1. L0 boxplot (per example)
    2a. Cosine-Similarity boxplot of reconstructions vs. true over all neurons (per example)
    2b. Cosine-Similarity boxplot of reconstructions vs. true over all examples (per neuron)
    3b. R² boxplot of reconstructions vs. true over all neurons (per example)
    3a. R² boxplot of reconstructions vs. true over all examples (per neuron)

    """
    device = spk_cts.device
    n_inst = sae.cfg.n_instances
    n_units = spk_cts.shape[1]
    n_examples = spk_cts.shape[0]
    valid_starts = n_examples - sae.cfg.seq_len + 1
    n_steps = valid_starts // batch_sz  # total number of examples
    n_recon_examples = n_steps * batch_sz
    
    # <ss> Run examples through model and compute metrics.

    # Create placeholders to store metrics.
    l0 = t.zeros((n_recon_examples, n_inst), dtype=t.float32, device=device)
    recon_spk_cts = t.empty((n_recon_examples, n_inst, n_units), dtype=sae.cfg.dtype, device=device)
    r2_per_example = t.empty((n_recon_examples, n_inst), dtype=sae.cfg.dtype, device=device)
    cos_sim_per_example = t.empty((n_recon_examples, n_inst), dtype=sae.cfg.dtype, device=device)
    topk_acts_4d = []  # stores (inst_idx, ex_idx, feat_idx, act_val) for each topk act
    latent_storage_created = False
    Z_all = None  # will become [n_recon_examples, n_inst, latent_dim]

    progress_bar = tqdm(range(n_steps), desc="SAE batch evaluation step")
    with t.no_grad():
        for step in progress_bar:  # loop over all examples
            # Get start index for each seq in batch, and then get the full seq indices.
            start_idxs = t.arange(step * batch_sz, (step + 1) * batch_sz)
            seq_idxs = repeat(start_idxs, "batch -> batch inst", inst=n_inst)
            seq_idxs = seq_idxs.unsqueeze(-1) + t.arange(sae.cfg.seq_len)  # broadcast to seq dim
            spike_count_seqs = spk_cts[seq_idxs]  # [batch, inst, seq, unit]
            # Forward pass through SAE.
            recon_levels, topk_acts_levels, _acts_raw = sae(spike_count_seqs)

            # ---------- EXTRACT LATENTS ----------
            # Heuristic: prefer _acts_raw if present, else fall back to topk_acts_levels[-1].
            acts = _acts_raw if (_acts_raw is not None) else topk_acts_levels[-1]

            # Inspect and unify shape possibilities:
            # common shapes you may see:
            #  - (batch, inst, seq, feat)
            #  - (levels, batch, inst, seq, feat)  <- pick last level
            #  - (batch, inst, feat)  <- already collapsed across seq
            # We'll handle these programmatically:
            if isinstance(acts, (list, tuple)):
                acts = acts[-1]   # pick last level if list/tuple

            # ensure tensor on CPU/GPU is okay (we keep on device for storing)
            # acts: a torch tensor
            if acts.ndim == 5:
                # shape: (levels, batch, inst, seq, feat) -> take final level
                acts = acts[-1]            # now (batch, inst, seq, feat)
            if acts.ndim == 4:
                # shape: (batch, inst, seq, feat) -> select central timepoint
                seq_len = acts.shape[2]
                center = seq_len // 2
                acts_center = acts[:, :, center, :]  # (batch, inst, feat)
            elif acts.ndim == 3:
                # shape: (batch, inst, feat) -> already central/pooled
                acts_center = acts
            else:
                raise RuntimeError(f"Unexpected activation tensor shape: {acts.shape}")

            # On first step create storage
            if not latent_storage_created:
                _, _, feat_dim = acts_center.shape
                latent_dim = feat_dim
                Z_all = t.empty((n_recon_examples, n_inst, latent_dim), device=device, dtype=acts_center.dtype)
                latent_storage_created = True

            # place into global array at correct global example indices
            # start_idxs are global example indices for this batch
            global_idxs = start_idxs.cpu().numpy()  # numpy ints
            # acts_center shape: (batch, inst, feat)
            # store per global index: Z_all[global_idx] = acts_center[batch_idx]
            for b_idx, gidx in enumerate(global_idxs):
                Z_all[gidx] = acts_center[b_idx]  # broadcast to [inst, feat]
            
            nonzero_mask = (topk_acts_levels[-1] > 0)
            cur_l0 = reduce(nonzero_mask.float(), "batch inst sae_feat -> batch inst", "sum")
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
    
    # move to CPU & numpy
    Z_np = Z_all.detach().to(t.float32).cpu().numpy()   # shape: (n_recon_examples, n_inst, latent_dim)

    # if you want per-time latent for a single instance (inst_idx)
    #inst_idx = 0
    #Z_time_inst0 = Z_np[:, inst_idx, :]   # shape (T, latent_dim)

    # if model has only one instance (n_inst==1), squeeze:
    #Z_time = Z_np.squeeze(1)   # (T, latent_dim)

    # Save to disk for later decoding/plotting
    np.save("sae_latents_Z.npy", Z_np)

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

    fig, (ax_l0, ax_r2, ax_cos) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
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
    ax_l0.set_xticklabels([f"SAE {i}" for i in range(n_inst)])
    ax_l0.set_yticks(np.arange(0, l0.max().item() + 1, 10))
    ax_l0.set_title("L0 of SAE features")
    
    # </sss>

    # <sss> Format and plot R² and Cosine Similarity data.
    
    cos_sim_per_example = asnumpy(cos_sim_per_example.float())
    r2_per_example = asnumpy(r2_per_example.float())
    cos_sim_per_unit = asnumpy(cos_sim_per_unit.float())

    model_names = [f"SAE {i}" for i in range(2)]
    dfs = []

    dfs.append(
        pd.DataFrame(cos_sim_per_example, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Examples", Metric="Cosine Similarity")
    )

    dfs.append(
        pd.DataFrame(cos_sim_per_unit, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Units", Metric="Cosine Similarity")
    )

    dfs.append(
        pd.DataFrame(r2_per_example, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Examples", Metric="R²")
    )

    dfs.append(
        pd.DataFrame(r2_per_unit, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
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
        hue="SAE",
        show_legend=False
    )
    # Prettify axes.
    ax_r2.set_xlabel("")
    ax_r2.set_ylabel("")
    ax_r2.set_ylim(-1.0, 1.0)
    ax_r2.set_yticks(np.arange(-1.0, 1.1, 0.1))
    ax_r2.set_title("R² of SAE reconstructions")

    mp.box_strip_plot(
        ax=ax_cos,
        data=cos_sim_df,
        x="Type",
        y="Value",
        hue="SAE",
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

    return topk_acts_4d, recon_spk_cts, r2_per_unit, r2_per_example, cos_sim_per_unit, cos_sim_per_example

    # </ss>

# </s>
