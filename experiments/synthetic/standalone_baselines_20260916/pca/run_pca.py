"""Standalone descriptive windowed PCA; never writes into paper inputs/outputs."""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, jaxtyped
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.synthetic.scripts.temporal_analysis import window_motion


@jaxtyped(typechecker=beartype)
def tuning(scores: Float[np.ndarray, "n k"], position: Float[np.ndarray, "n"]) -> np.ndarray:
    """Average all continuous scores in fixed 40 spatial bins, including negatives."""
    bins = np.clip((position * 40).astype(int), 0, 39)
    counts = np.bincount(bins, minlength=40)
    values = np.zeros((40, scores.shape[1]))
    np.add.at(values, bins, scores)
    return (values / counts[:, None]).T


@jaxtyped(typechecker=beartype)
def correlations(a: Float[np.ndarray, "a n"], b: Float[np.ndarray, "b n"]) -> np.ndarray:
    """Pearson correlations between rows; zero for any constant row."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return a @ b.T / np.maximum(np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :], 1e-30)


@jaxtyped(typechecker=beartype)
def auc(scores: Float[np.ndarray, "n k"], right: np.ndarray) -> np.ndarray:
    """Mann–Whitney AUROC with half credit for ties, rightward is positive."""
    nr, nl = int(right.sum()), int((~right).sum())
    ranks = rankdata(scores, axis=0)
    return (ranks[right].sum(axis=0) - nr * (nr + 1) / 2) / (nr * nl)


@beartype
def matched_auc(scores: np.ndarray, metadata: pd.DataFrame, bins: int, eligible: str) -> tuple:
    """Within-position-bin AUROC, weighted by smaller direction count (>=10)."""
    posbin = np.clip((metadata.position.to_numpy() * bins).astype(int), 0, bins - 1)
    records, weights, values = [], [], []
    for b in range(bins):
        mask = metadata[eligible].to_numpy() & (posbin == b) & (metadata.direction.to_numpy() != 0)
        right = metadata.direction.to_numpy()[mask] == 1
        nr, nl = int(right.sum()), int((~right).sum())
        if min(nr, nl) >= 10:
            values.append(auc(scores[mask], right))
            weights.append(min(nr, nl))
            records.append(dict(bin=b, right=nr, left=nl, weight=min(nr, nl)))
    return np.average(values, weights=weights, axis=0), records


@beartype
def top_disjoint(scores: np.ndarray, starts: np.ndarray, eligible: np.ndarray, count: int = 16) -> np.ndarray:
    """Select score-ranked nonoverlapping windows; no positivity/direction gate."""
    selected = []
    for i in np.argsort(-scores, kind="stable"):
        if eligible[i] and all(abs(int(starts[i]) - int(starts[j])) >= 100 for j in selected):
            selected.append(int(i))
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=int)


@beartype
def width(curve: np.ndarray) -> float:
    """Span of samples above halfway from minimum to maximum; display diagnostic."""
    active = np.flatnonzero(curve >= (curve.max() + curve.min()) / 2)
    return float((active[-1] - active[0] + 1) * .025)


@beartype
def savefig(fig: plt.Figure, name: str) -> None:
    """Save labeled PNG/PDF artifacts exclusively in this analysis directory."""
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


@beartype
def main() -> None:
    """Fit exact PCA once and export all-candidate descriptive diagnostics."""
    begin = time.perf_counter()
    data = ROOT / "experiments/synthetic/data/matryoshka_overlap"
    maskpath = ROOT / "experiments/synthetic/outputs/runs/temporal_transformer/analysis_windows.csv"
    files = [data / x for x in ("spike_matrix.npy", "positions.npy", "generative_place_rates.npy", "parameters.json")]
    files += [maskpath]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert hashes[str(files[0].relative_to(ROOT))] == "ce5bd9573966e96cd36a44a68cc4e84836c95f749a59260e87398aaab06db09f"
    spikes = np.load(data / "spike_matrix.npy").T.astype(np.float64)
    position = np.load(data / "positions.npy")
    rates = np.load(data / "generative_place_rates.npy").T
    counts = reduce(spikes, "(t b) n -> t n", "sum", b=5)
    windows = rearrange(np.lib.stride_tricks.sliding_window_view(counts, 20, axis=0), "w n t -> w t n")
    x = rearrange(windows, "w t n -> w (t n)").copy()
    assert x.shape == (11981, 2000)
    starts = np.arange(len(x)) * 5
    metadata = pd.read_csv(maskpath)
    calculated = window_motion(position, starts, 100)
    for col in calculated:
        np.testing.assert_allclose(metadata[col].to_numpy(), calculated[col].to_numpy(), atol=1e-12)
    midpoint = (position[starts + 49] + position[starts + 50]) / 2
    rate_window = np.lib.stride_tricks.sliding_window_view(rates, 100, axis=0)[::5].mean(axis=-1)
    fit_start = time.perf_counter()
    model = PCA(n_components=128, svd_solver="full", whiten=False)
    scores = model.fit_transform(x)
    fit_seconds = time.perf_counter() - fit_start
    components = model.components_.reshape(128, 20, 100)
    np.savez_compressed(OUT / "pca_fit.npz", mean=model.mean_, components=model.components_,
                        explained_variance=model.explained_variance_, explained_variance_ratio=model.explained_variance_ratio_,
                        singular_values=model.singular_values_, scores=scores, starts=starts)
    print(f"Full SVD done in {fit_seconds:.2f}s; variance={model.explained_variance_ratio_.sum():.6f}", flush=True)
    projected = (x - model.mean_) @ model.components_.T
    reconstructed = scores @ model.components_ + model.mean_
    residual = (x - reconstructed).reshape(-1, 20, 100)
    checks = dict(orthogonality_max_abs=float(np.abs(model.components_ @ model.components_.T - np.eye(128)).max()),
                  projection_max_abs=float(np.abs(scores - projected).max()),
                  reconstruction_mse=float(np.square(residual).mean()),
                  centered_total_variance=float(np.var(x, axis=0, ddof=1).sum()),
                  saved_mask_alignment=True, input_shape=list(x.shape))
    reloaded = np.load(OUT / "pca_fit.npz")
    np.testing.assert_allclose((x[:31] - reloaded["mean"]) @ reloaded["components"].T, reloaded["scores"][:31], atol=1e-10)
    np.testing.assert_allclose(np.square(residual).sum(), np.square(x-model.mean_).sum() - np.square(scores).sum(), rtol=1e-12)
    assert checks["orthogonality_max_abs"] < 1e-10 and checks["projection_max_abs"] < 1e-10
    checks["reload_projection_verified"] = True
    unit_energy = np.square(components).sum(axis=1)
    temporal_energy = np.square(components).sum(axis=2)
    curves = tuning(scores, midpoint)
    centers = (np.arange(40) + .5) / 40
    c = np.array([.2,.4,.7,.7]); left = np.array([.05,.10,.14,.06]); right = np.array([.10,.05,.14,.06])
    sigma = np.where(centers[None, :] < c[:,None], left[:,None], right[:,None])
    exact = .1 + np.array([10,28,20,20])[:,None] * np.exp(-.5 * ((centers[None,:] - c[:,None]) / sigma)**2)
    corr = correlations(exact, curves)
    start_corr = correlations(exact, tuning(scores, position[starts]))
    end_corr = correlations(exact, tuning(scores, position[starts+99]))
    window_corr = correlations(rate_window.T, scores.T)
    window_rate_tuning_corr = correlations(tuning(rate_window, midpoint), curves)
    target_ids, selected = linear_sum_assignment(-np.abs(corr))
    rows = []
    for cell in range(4):
        for pc in range(128):
            sign = 1 if corr[cell,pc] >= 0 else -1
            oriented = sign * curves[pc]
            rows.append(dict(cell=cell+1,pc=pc+1,orientation=sign,shape_pearson=float(abs(corr[cell,pc])),
                             raw_shape_pearson=float(corr[cell,pc]), start_shape_pearson=float(sign*start_corr[cell,pc]),
                             end_shape_pearson=float(sign*end_corr[cell,pc]),
                             window_mean_rate_score_pearson=float(sign*window_corr[cell,pc]),
                             window_mean_rate_tuning_pearson=float(sign*window_rate_tuning_corr[cell,pc]),
                             peak_m=float(centers[oriented.argmax()]),half_range_width_m=width(oriented),
                             target_width_m=width(exact[cell]),source_energy=float(unit_energy[pc,cell]),
                             dominant_unit=int(unit_energy[pc].argmax())+1,
                             source_fraction_of_place_energy=float(unit_energy[pc,cell]/unit_energy[pc,:4].sum()),
                             source_time_centroid_s=float(np.dot(np.arange(20)*.05+.02,components[pc,:,cell]**2)/unit_energy[pc,cell]),
                             unique_shape_selected=bool(selected[cell]==pc)))
    spatial = pd.DataFrame(rows)
    spatial.to_csv(OUT / "spatial_all_candidates.csv",index=False)
    chosen = spatial[spatial.unique_shape_selected].copy()
    chosen.to_csv(OUT / "spatial_selected.csv",index=False)
    identity_best = []
    for cell in range(1,5):
        subset = spatial[(spatial.cell==cell)&(spatial.dominant_unit==cell)]
        identity_best.append(subset.sort_values("shape_pearson",ascending=False).head(1))
    identity = pd.concat(identity_best)
    identity.to_csv(OUT / "spatial_best_source_dominant.csv",index=False)
    reconstruction = pd.DataFrame(dict(unit=np.arange(100)+1, mse=np.square(residual).mean(axis=(0,1)),
        variance=np.var(windows,axis=0).mean(axis=0), captured_loading_energy=unit_energy.sum(axis=0)))
    reconstruction["r2"] = 1 - reconstruction.mse/reconstruction.variance
    reconstruction.to_csv(OUT / "reconstruction_per_unit.csv",index=False)
    clean_auc, clean_bins = matched_auc(scores, metadata,40,"ramp_clean")
    global_auc, global_bins = matched_auc(scores,metadata,10,"clean")
    pair = unit_energy[:,:2].sum(axis=1); known=unit_energy[:,:4].sum(axis=1)
    enrichment = (pair/2) / (unit_energy[:,4:].sum(axis=1)/96)
    pair_min = unit_energy[:,:2].min(axis=1)/pair
    structure = (pair/known>=.5)&(pair_min>=.1)&(enrichment>=2)
    direction = pd.DataFrame(dict(pc=np.arange(128)+1, right_clean_auc=clean_auc,
        best_oriented_clean_auc=np.maximum(clean_auc,1-clean_auc),right_global_auc=global_auc,
        pair_energy=pair, pair_fraction_of_place_energy=pair/known, pair_min_unit_fraction=pair_min,
        pair_noise_energy_enrichment=enrichment, passes_pair_structure=structure))
    direction.to_csv(OUT / "direction_all_candidates.csv",index=False)
    best_pc = int(np.argmax(np.maximum(clean_auc,1-clean_auc)))
    structural_pc = int(np.argmax(np.where(structure,np.maximum(clean_auc,1-clean_auc),-1)))
    examples = {}; example_rows=[]
    for label, desired in (("forward",1),("backward",-1)):
        orient = desired * (1 if clean_auc[best_pc]>=.5 else -1)
        for pool,mask in (("unrestricted",np.ones(len(x),dtype=bool)),("clean_crossing",metadata.ramp_clean.to_numpy())):
            idx = top_disjoint(orient*scores[:,best_pc],starts,mask)
            trajectories=position[starts[idx,None]+np.arange(100)]
            examples[f"{label}_{pool}_indices"]=idx
            examples[f"{label}_{pool}_trajectories"]=trajectories
            example_rows.append(dict(direction=label,pool=pool,pc=best_pc+1,orientation=orient,n=len(idx),
              preferred=int((metadata.direction.to_numpy()[idx]==desired).sum()),
              clean_crossings=int(metadata.ramp_clean.to_numpy()[idx].sum()),
              mean_position=float(metadata.position.to_numpy()[idx].mean())))
    pd.DataFrame(example_rows).to_csv(OUT / "top_trajectories_summary.csv",index=False)
    np.savez_compressed(OUT / "diagnostics.npz",shape_correlation=corr,spatial_tuning=curves,exact_fields=exact,
       position_centers=centers,unit_energy=unit_energy,temporal_energy=temporal_energy,midpoint=midpoint,
       clean_mask=metadata.ramp_clean.to_numpy(),direction=metadata.direction.to_numpy(),**examples)
    # Full target-by-candidate matrix avoids hiding inconvenient candidates.
    fig,ax=plt.subplots(figsize=(15,3))
    im=ax.imshow(abs(corr),aspect="auto",vmin=0,vmax=1,cmap="viridis",extent=(.5,128.5,4.5,.5))
    ax.set(xlabel="Principal component (one-based)",ylabel="Target place cell",yticks=[1,2,3,4],title="All 128 native PCs: absolute spatial tuning Pearson correlation")
    fig.colorbar(im,ax=ax,label="|Pearson r| (sign chosen per target)")
    savefig(fig,"spatial_all_candidate_matrix")
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for cell,ax in enumerate(axes.flat):
        pc=selected[cell];sgn=np.sign(corr[cell,pc]); y=sgn*curves[pc]
        # z scaling affects presentation only; it does not alter scores or fit.
        ax.plot(centers,(y-y.mean())/y.std(),label=f"{'+' if sgn>0 else '-'}PC{pc+1}")
        target=exact[cell];ax.plot(centers,(target-target.mean())/target.std(),"--",label=f"Exact field {cell+1}")
        ax.set(xlabel="Midpoint position (m)",ylabel="Spatial-curve z units",title=f"Cell {cell+1}: r={abs(corr[cell,pc]):.3f}; source energy={unit_energy[pc,cell]:.1%}")
        ax.legend()
    fig.suptitle("Unique shape-optimal assignment; signed tuning, display-only curve standardization")
    fig.tight_layout();savefig(fig,"spatial_four_targets")
    plotted=list(dict.fromkeys(list(selected)+[best_pc,structural_pc]))
    fig,axes=plt.subplots(len(plotted),3,figsize=(14,3*len(plotted)),squeeze=False)
    for pc,row in zip(plotted,axes):
        vmax=abs(components[pc]).max()
        row[0].imshow(components[pc].T,aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax,extent=(-.005,.995,100.5,.5))
        row[0].set(title=f"PC{pc+1}: native signed weights",xlabel="Window time (s)",ylabel="Unit (1–100)")
        for cell in range(4):row[1].plot(np.arange(20)*.05+.02,components[pc,:,cell],label=f"Cell {cell+1}")
        row[1].set(title="Four place-cell weights",xlabel="Window time (s)",ylabel="Loading");row[1].legend(fontsize=8)
        row[2].bar(np.arange(100)+1,unit_energy[pc]);row[2].set(title="Unit squared loading energy",xlabel="Unit",ylabel="Fraction of total component energy")
    fig.tight_layout();savefig(fig,"component_loading_maps")
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for r,label in enumerate(("forward","backward")):
        for col,pool in enumerate(("unrestricted","clean_crossing")):
            traj=examples[f"{label}_{pool}_trajectories"];ax=axes[r,col]
            ax.plot(np.arange(100)*.01,traj.T,color=".65",lw=.7);ax.plot(np.arange(100)*.01,traj.mean(axis=0),color="black",lw=2)
            ax.axhline(.2,ls=":",color="C0");ax.axhline(.4,ls=":",color="C1")
            rec=example_rows[r*2+col];ax.set(xlabel="Window time (s)",ylabel="Position (m)",ylim=(0,1),
             title=f"{label}: {rec['orientation']:+d}PC{best_pc+1}, {pool}\n{rec['preferred']}/{rec['n']} preferred; {rec['clean_crossings']} clean")
    fig.suptitle("Top 16 nonoverlapping windows: opposite poles of ONE native PC")
    fig.tight_layout();savefig(fig,"direction_top_trajectories")
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    score=scores[:,best_pc];sign=1 if clean_auc[best_pc]>=.5 else -1
    bins=np.clip((metadata.position.to_numpy()*40).astype(int),0,39)
    for d,name in ((1,"Right"),(-1,"Left")):
        means=[np.mean(sign*score[(bins==b)&(metadata.direction.to_numpy()==d)]) for b in range(40)]
        axes[0].plot(centers,means,label=name)
    axes[0].set(title=f"PC{best_pc+1}: scores over all windows",xlabel="Mean window position (m)",ylabel="Oriented mean score");axes[0].legend()
    energy=[np.mean(score[bins==b]**2)/np.mean(score**2) for b in range(40)]
    axes[1].plot(centers,energy);axes[1].axvspan(.2,.4,alpha=.15);axes[1].set(title="Score energy vs spatial location",xlabel="Mean window position (m)",ylabel="E[score²|position] / E[score²]")
    axes[2].scatter(direction.right_global_auc,direction.right_clean_auc,c=pair,cmap="viridis",s=18)
    axes[2].scatter(global_auc[best_pc],clean_auc[best_pc],marker="*",c="red",s=120)
    axes[2].set(xlabel="Global within-bin rightward AUROC",ylabel="Clean-crossing within-bin rightward AUROC",title="All 128 PCs; color=pair energy")
    fig.tight_layout();savefig(fig,"direction_localization")
    # Prefix sensitivity uses this same fitted ordered basis, with no extra fits.
    prefix=[]
    for k in (16,32,64,128):
        _,assign=linear_sum_assignment(-abs(corr[:,:k]));prefix.append(dict(k=k,pcs=(assign+1).tolist(),
          shape_r=abs(corr[np.arange(4),assign]).tolist(),best_direction_auc=float(np.maximum(clean_auc[:k],1-clean_auc[:k]).max())))
    summary=dict(method="PCA",n_components=128,solver="full SVD",dtype="float64",center_columns=True,scale_columns=False,
       flatten_order="time then unit",reference="raw midpoint: mean positions at starts+49 and starts+50",
       fit_seconds=fit_seconds,total_seconds=time.perf_counter()-begin,explained_variance_fraction=float(model.explained_variance_ratio_.sum()),
       hashes=hashes,software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
       numerical_checks=checks,selected_spatial=chosen.to_dict("records"),best_source_dominant=identity.to_dict("records"),
       clean_position_bins=clean_bins,global_position_bins=global_bins,selected_direction_pc=best_pc+1,
       best_pair_structural_pc=structural_pc+1,direction_distinct_components=1,
       selected_direction=direction.iloc[best_pc].to_dict(),best_pair_structural=direction.iloc[structural_pc].to_dict(),
       top_examples=example_rows,prefix_sensitivity=prefix)
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
