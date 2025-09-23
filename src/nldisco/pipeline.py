"""Mini pipeline functions."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from typing import List, Dict, Optional, Tuple, Union

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import ipywidgets as widgets
from IPython.display import clear_output

def analyze_discrete_variable(
    acts_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    variable: str,
    min_activation_frac: float
) -> List[Dict]:
    """Analyzes a discrete variable to find latent selectivity."""
    results = []
    unique_values = metadata_binned[variable].dropna().unique()

    for value in unique_values:
        event_idxs = np.where(metadata_binned[variable] == value)[0]
        if len(event_idxs) == 0:
            continue

        event_acts_df = acts_df[acts_df["example_idx"].isin(event_idxs)]
        if len(event_acts_df) == 0:
            continue

        in_df = event_acts_df.groupby(["instance_idx", "latent_idx"]).agg(
            activation_count=("activation_value", "count")
        ).reset_index()
        n_in = len(event_idxs)
        in_df["activation_frac_during"] = in_df["activation_count"] / n_in

        promising = in_df[in_df["activation_frac_during"] >= min_activation_frac]
        if promising.empty:
            continue

        out_mask = ~acts_df["example_idx"].isin(event_idxs)
        out_acts_df = acts_df[out_mask].merge(
            promising[["instance_idx", "latent_idx"]],
            on=["instance_idx", "latent_idx"], how="inner"
        )

        if not out_acts_df.empty:
            out_df = out_acts_df.groupby(["instance_idx", "latent_idx"]).agg(
                activation_count=("activation_value", "count")
            ).reset_index()
            n_out = len(metadata_binned) - n_in
            out_df["activation_frac_outside"] = out_df["activation_count"] / n_out
            merged = promising.merge(out_df, on=["instance_idx", "latent_idx"], how="left")
            merged["activation_frac_outside"] = merged["activation_frac_outside"].fillna(0.0)
        else:
            merged = promising.copy()
            merged["activation_frac_outside"] = 0.0

        merged["selectivity_score"] = (
            merged["activation_frac_during"] /
            (merged["activation_frac_during"] + merged["activation_frac_outside"] + 1e-9)
        )

        for _, row in merged.iterrows():
            results.append({
                'variable': variable, 'variable_type': 'discrete', 'value': value,
                'instance_idx': row['instance_idx'], 'latent_idx': row['latent_idx'],
                'activation_frac_during': row['activation_frac_during'],
                'activation_frac_outside': row['activation_frac_outside'],
                'selectivity_score': row['selectivity_score']
            })

    return results


def analyze_continuous_variable(
    acts_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    variable: str,
    n_bins,  # int OR sequence of edges
    min_activation_frac: float
) -> List[Dict]:
    """Analyzes a continuous variable by binning it and then using the discrete analysis method."""
    is_edges = isinstance(n_bins, (list, tuple, np.ndarray))

    if is_edges:
        try:
            n_bins_eff = len(n_bins) - 1
        except TypeError:
            n_bins_eff = n_bins
        print(f"    Binning '{variable}' into predefined bins ({n_bins_eff} bins)...")
    else:
        print(f"    Binning '{variable}' into {n_bins} bins...")

    binned_col_name = f"{variable}_binned"

    data_to_bin = metadata_binned[variable].dropna()
    if data_to_bin.empty:
        return []

    if is_edges:
        edges = np.asarray(n_bins)
        metadata_binned[binned_col_name] = pd.cut(
            data_to_bin, bins=edges, include_lowest=True
        )
    elif variable == 'movement_angle':
        bins = np.linspace(-180, 180, int(n_bins) + 1)
        metadata_binned[binned_col_name] = pd.cut(
            data_to_bin, bins=bins, include_lowest=True
        )
    else:
        metadata_binned[binned_col_name] = pd.qcut(
            data_to_bin, q=int(n_bins), labels=None, duplicates='drop'
        )

    results = analyze_discrete_variable(acts_df, metadata_binned, binned_col_name, min_activation_frac)

    for res in results:
        res['variable'] = variable
        res['variable_type'] = 'continuous'

    return results


def map_latents_to_metadata(
    acts_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    discrete_vars: List[str] = None,
    continuous_vars: List[str] = None,
    min_activation_frac: float = 0.1,
    n_bins_continuous: List = None,  # accepts ints or edge lists per variable
    top_n_mappings: int = 3
) -> pd.DataFrame:  
    """Automatically maps SAE latents to metadata variables, returning a ranked DataFrame of associations."""
    if discrete_vars is None: discrete_vars = []
    if continuous_vars is None: continuous_vars = []

    # Default or broadcast [single] to all continuous variables
    if n_bins_continuous is None:
        n_bins_continuous = [10] * len(continuous_vars)
    elif isinstance(n_bins_continuous, list) and len(n_bins_continuous) == 1 and len(continuous_vars) > 1:
        n_bins_continuous = n_bins_continuous * len(continuous_vars)

    if len(n_bins_continuous) != len(continuous_vars):
        raise ValueError("n_bins_continuous must match length of continuous_vars (or be a single-element list to broadcast).")
    
    all_results = []
    print("Starting automated latent-to-metadata mapping...")

    for variable in discrete_vars + continuous_vars:
        if variable not in metadata_binned.columns:
            raise ValueError(f"Variable '{variable}' not found in metadata_binned")
        print(f"\nAnalyzing variable: {variable}...")
        
        if variable in discrete_vars:
            results = analyze_discrete_variable(acts_df, metadata_binned, variable, min_activation_frac)
            all_results.extend(results)
        elif variable in continuous_vars:
            idx = continuous_vars.index(variable)
            n_bins = n_bins_continuous[idx]  # may be int OR list of edges
            results = analyze_continuous_variable(
                acts_df, metadata_binned, variable,
                n_bins=n_bins, min_activation_frac=min_activation_frac
            )
            all_results.extend(results)
        print(f"    Found {len(results)} potential associations")
    
    if not all_results:
        print("\nNo associations found meeting the minimum activation fraction!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    results_df['value'] = results_df['value'].astype(str)

    print(f"\nSelecting top {top_n_mappings} mappings for each variable/value/instance combination...")
    ranked_df = (
        results_df.sort_values('selectivity_score', ascending=False)
        .groupby(['variable', 'value', 'instance_idx'])
        .head(top_n_mappings)
    )
    
    sort_order = ['variable_type', 'variable', 'value', 'instance_idx', 'selectivity_score']
    ascending_order = [True, True, True, True, False]
    
    final_df = ranked_df.sort_values(by=sort_order, ascending=ascending_order).reset_index(drop=True)
    
    discrete_count = len(final_df[final_df['variable_type'] == 'discrete'])
    continuous_count = len(final_df[final_df['variable_type'] == 'continuous'])
    
    print(f"\nFound {discrete_count} top discrete associations")
    print(f"Found {continuous_count} top continuous associations")
    print(f"Total: {len(final_df)} associations returned in single DataFrame")
    
    return final_df

def plot_latent_tuning(
    acts_df: pd.DataFrame,
    spikes_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    variable: str,
    instance_idx: int,
    latent_idx: int,
    return_data: bool = False,
):
    """Visualizes SAE latent tuning to metadata variables."""
    # Z-scores across units (biological neurons)
    unit_means = spikes_df.mean(axis=0)
    unit_stds  = spikes_df.std(axis=0)
    spk_z_scores_df = spikes_df.sub(unit_means, axis=1).div(unit_stds, axis=1)
    spk_z_scores_df = spk_z_scores_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Latent activations
    latent_acts = acts_df[
        (acts_df['instance_idx'] == instance_idx) &
        (acts_df['latent_idx']   == latent_idx)
    ]

    # Find top/bottom co-active units
    if len(latent_acts) > 0:
        latent_active_indices = latent_acts['example_idx'].values
        unit_mean_zscores = spk_z_scores_df.iloc[latent_active_indices].mean(axis=0)
        top_unit    = unit_mean_zscores.idxmax()
        bottom_unit = unit_mean_zscores.idxmin()
        top_zscore    = unit_mean_zscores[top_unit]
        bottom_zscore = unit_mean_zscores[bottom_unit]
        print(f"Top co-active unit: {top_unit} (z-score: {top_zscore:.3f})")
        print(f"Bottom co-active unit: {bottom_unit} (z-score: {bottom_zscore:.3f})")
    else:
        print("No activations found for this instance/latent.")
        return None if return_data else None

    # Fill missing activations with 0 for inactive examples
    all_examples = pd.DataFrame({'example_idx': range(len(metadata_binned))})
    all_examples = (
        all_examples.merge(
            latent_acts[['example_idx', 'activation_value']],
            on='example_idx',
            how='left'
        ).fillna(0)
    )

    # Slices
    metadata_slice     = metadata_binned[[variable]].iloc[all_examples['example_idx']]
    top_unit_slice     = spk_z_scores_df[[top_unit]].iloc[all_examples['example_idx']]
    bottom_unit_slice  = spk_z_scores_df[[bottom_unit]].iloc[all_examples['example_idx']]

    # Plotting dataframe
    data_df = metadata_slice.reset_index(drop=True)
    data_df['activation_value']   = all_examples['activation_value'].reset_index(drop=True)
    data_df['top_unit_zscore']    = top_unit_slice[top_unit].reset_index(drop=True)
    data_df['bottom_unit_zscore'] = bottom_unit_slice[bottom_unit].reset_index(drop=True)

    # Interval/categorical?
    try:
        is_interval_data = pd.api.types.is_interval_dtype(data_df[variable].cat.categories)
    except AttributeError:
        is_interval_data = False

    # Summary stats
    stats_df = (
        data_df.groupby(variable).agg({
            'activation_value'  : ['mean', 'sem'],
            'top_unit_zscore'   : ['mean', 'sem'],
            'bottom_unit_zscore': ['mean', 'sem'],
        }).reset_index()
    )
    stats_df.columns = [
        variable, 'latent_mean', 'latent_sem',
        'top_unit_mean', 'top_unit_sem',
        'bottom_unit_mean', 'bottom_unit_sem'
    ]

    # Selectivity score (rate proportions)
    if not latent_acts.empty:
        condition_masks = {cond: metadata_binned[variable] == cond for cond in stats_df[variable]}
        active_example_set = set(latent_acts['example_idx'])
        rate_props = []
        for _, row in stats_df.iterrows():
            cond = row[variable]
            mask = condition_masks[cond]
            cond_idxs = np.where(mask)[0]
            noncond_idxs = np.where(~mask)[0]

            cond_act = len(active_example_set.intersection(cond_idxs))
            noncond_act = len(active_example_set.intersection(noncond_idxs))

            frac_during  = cond_act / len(cond_idxs)     if len(cond_idxs)     > 0 else 0
            frac_outside = noncond_act / len(noncond_idxs) if len(noncond_idxs) > 0 else 0
            sel = frac_during / (frac_during + frac_outside) if (frac_during + frac_outside) > 0 else 0
            rate_props.append(sel)
        stats_df['selectivity_score'] = rate_props
    else:
        stats_df['selectivity_score'] = 0

    # Per-unit z-score stats for bottom bar chart
    if len(latent_acts) > 0:
        zscore_stats = spk_z_scores_df.iloc[latent_active_indices].agg(['mean', 'sem']).T
        zscore_stats.columns = ['mean_zscore', 'sem_zscore']
        zscore_stats = zscore_stats.reset_index()
        zscore_stats.columns = ['unit', 'mean_zscore', 'sem_zscore']
    else:
        zscore_stats = pd.DataFrame({'unit': spk_z_scores_df.columns, 'mean_zscore': 0, 'sem_zscore': 0})

    # Prepare optional payload
    if return_data:
        x_labels = stats_df[variable].astype(str).tolist()
        payload = {
            "variable": variable,
            "instance_idx": instance_idx,
            "latent_idx": latent_idx,
            "is_interval_data": bool(is_interval_data),
            "x": x_labels,
            "latent_mean": stats_df['latent_mean'].tolist(),
            "latent_sem": stats_df['latent_sem'].tolist(),
            "top_unit_mean": stats_df['top_unit_mean'].tolist(),
            "top_unit_sem": stats_df['top_unit_sem'].tolist(),
            "bottom_unit_mean": stats_df['bottom_unit_mean'].tolist(),
            "bottom_unit_sem": stats_df['bottom_unit_sem'].tolist(),
            "selectivity_score": stats_df['selectivity_score'].tolist(),
            "zbar_unit": zscore_stats['unit'].tolist(),
            "zbar_mean": zscore_stats['mean_zscore'].tolist(),
            "zbar_sem": zscore_stats['sem_zscore'].tolist(),
        }

    # === Plotting ===
    if 'angle' in variable:  # Polar plots
        stats_df['theta'] = stats_df[variable].apply(lambda x: x.mid if isinstance(x, pd.Interval) else x)
        stats_df = stats_df.sort_values('theta')
        plot_df = pd.concat([stats_df, stats_df.head(1)], ignore_index=True)

        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}, {"type": "polar"}],
                   [{"type": "xy", "colspan": 4}, None, None, None]],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            subplot_titles=[
                "Latent Activation",
                "Top Unit Z-score",
                "Bottom Unit Z-score",
                "Selectivity Score",
                "Mean Z-scores when Latent Active"
            ]
        )

        # Latent activation
        fig.add_trace(go.Scatterpolar(r=plot_df['latent_mean'] + plot_df['latent_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatterpolar(r=plot_df['latent_mean'] - plot_df['latent_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), fill='tonext', fillcolor='rgba(220,20,60,0.2)', name='Latent ±SEM'), row=1, col=1)
        fig.add_trace(go.Scatterpolar(r=plot_df['latent_mean'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='crimson', width=3), name='Latent Activation'), row=1, col=1)

        # Top / bottom units
        fig.add_trace(go.Scatterpolar(r=plot_df['top_unit_mean'] + plot_df['top_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=plot_df['top_unit_mean'] - plot_df['top_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), fill='tonext', fillcolor='rgba(0,0,139,0.2)', name='Top Unit ±SEM'), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=plot_df['top_unit_mean'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='darkblue', width=3), name='Top Unit Z-score'), row=1, col=2)

        fig.add_trace(go.Scatterpolar(r=plot_df['bottom_unit_mean'] + plot_df['bottom_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatterpolar(r=plot_df['bottom_unit_mean'] - plot_df['bottom_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), fill='tonext', fillcolor='rgba(255,165,0,0.2)', name='Bottom Unit ±SEM'), row=1, col=3)
        fig.add_trace(go.Scatterpolar(r=plot_df['bottom_unit_mean'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='orange', width=3), name='Bottom Unit Z-score'), row=1, col=3)

        # Selectivity + bottom bar
        fig.add_trace(go.Scatterpolar(r=plot_df['selectivity_score'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='green', width=3), name='Selectivity Score'), row=1, col=4)
        fig.add_trace(go.Bar(x=zscore_stats['unit'], y=zscore_stats['mean_zscore'], error_y=dict(type='data', array=zscore_stats['sem_zscore']), marker_color='purple', marker_line_width=0, opacity=0.7, name='Mean Z-score'), row=2, col=1)

        # Ticks/labels
        if 'movement_angle' not in variable:
            tick_labels = [f"{int(round(t))}°" for t in stats_df['theta']]
        else:
            tick_labels = [f"{int(interval.mid)}°" if hasattr(interval, 'mid') else str(interval) for interval in stats_df[variable]]

        fig.update_layout(
            title=f"Instance {instance_idx} latent {latent_idx}  |  Top Unit {top_unit}  |  Bottom Unit {bottom_unit}",
            showlegend=True, height=800, width=1600, margin=dict(t=80, b=60, l=50, r=50)
        )
        rotation = 195 if 'movement_angle' in variable else 0
        tickvals = stats_df['theta'].tolist() if 'movement_angle' in variable else [(angle % 360) for angle in stats_df['theta'].tolist()]
        for i in range(1, 5):
            polar_key = f'polar{i if i > 1 else ""}'
            fig.update_layout(**{
                polar_key: dict(
                    angularaxis=dict(direction="counterclockwise", rotation=rotation, tickvals=tickvals, ticktext=tick_labels, tickfont=dict(size=10)),
                    radialaxis=dict(range=[0, None])
                )
            })
        fig.update_xaxes(title_text="unit", row=2, col=1)
        fig.update_yaxes(title_text="Mean Z-score", row=2, col=1)

    else:  # Linear (categorical or binned-continuous)
        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{}, {}, {}, {}], [{"colspan": 4}, None, None, None]],
            subplot_titles=[
                "Latent Activation", "Top Unit Z-score",
                "Bottom Unit Z-score", "Selectivity Score",
                "Mean Z-scores when Latent Active"
            ],
            horizontal_spacing=0.1, vertical_spacing=0.3
        )

        # x labels
        if is_interval_data:
            x_axis_labels = stats_df[variable].apply(lambda x: str(x))
            stats_df = stats_df.sort_values(by=variable)
        else:
            x_axis_labels = stats_df[variable].astype(str)

        if is_interval_data:
            # Lines with SEM ribbons
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['latent_mean'] + stats_df['latent_sem'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['latent_mean'] - stats_df['latent_sem'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(220,20,60,0.2)', name='Latent ±SEM'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['latent_mean'], mode='lines+markers', line_color='crimson', name='Latent Activation'), row=1, col=1)

            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['top_unit_mean'] + stats_df['top_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['top_unit_mean'] - stats_df['top_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(0,0,139,0.2)', name='Top Unit ±SEM'), row=1, col=2)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['top_unit_mean'], mode='lines+markers', line_color='darkblue', name='Top Unit Z-score'), row=1, col=2)

            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['bottom_unit_mean'] + stats_df['bottom_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=3)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['bottom_unit_mean'] - stats_df['bottom_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(255,165,0,0.2)', name='Bottom Unit ±SEM'), row=1, col=3)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['bottom_unit_mean'], mode='lines+markers', line_color='orange', name='Bottom Unit Z-score'), row=1, col=3)

            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['selectivity_score'], mode='lines+markers', line=dict(color='green', width=3), name='Selectivity Score'), row=1, col=4)
        else:
            # Bars + error bars
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['latent_mean'], error_y=dict(type='data', array=stats_df['latent_sem']), marker_color='crimson', marker_line_width=0, opacity=0.7, name='Latent Activation'), row=1, col=1)
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['top_unit_mean'], error_y=dict(type='data', array=stats_df['top_unit_sem']), marker_color='darkblue', marker_line_width=0, opacity=0.7, name='Top Unit Z-score'), row=1, col=2)
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['bottom_unit_mean'], error_y=dict(type='data', array=stats_df['bottom_unit_sem']), marker_color='orange', marker_line_width=0, opacity=0.7, name='Bottom Unit Z-score'), row=1, col=3)
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['selectivity_score'], marker_color='green', marker_line_width=0, opacity=0.7, name='Selectivity Score'), row=1, col=4)

        # Bottom bar: per-unit mean z-scores
        fig.add_trace(go.Bar(x=zscore_stats['unit'], y=zscore_stats['mean_zscore'], error_y=dict(type='data', array=zscore_stats['sem_zscore']), marker_color='purple', marker_line_width=0, opacity=0.7, name='Mean Z-score'), row=2, col=1)

        fig.update_layout(
            title=f"Instance {instance_idx} latent {latent_idx}  |  Top Unit {top_unit}  |  Bottom Unit {bottom_unit}",
            height=800, width=1600, showlegend=True, margin=dict(t=80, b=60, l=50, r=50)
        )
        # Axes
        for c in [1,2,3,4]:
            fig.update_xaxes(title_text=variable, tickangle=45, row=1, col=c)
        fig.update_xaxes(title_text="unit", row=2, col=1)

        fig.update_yaxes(title_text="Latent Activation", color="crimson",  range=[0, None], row=1, col=1)
        fig.update_yaxes(title_text="Top Unit Z-score", color="darkblue",               row=1, col=2)
        fig.update_yaxes(title_text="Bottom Unit Z-score", color="orange",              row=1, col=3)
        fig.update_yaxes(title_text="Selectivity Score", color="green", rangemode='tozero', row=1, col=4)
        fig.update_yaxes(title_text="Mean Z-score", row=2, col=1)
        fig.update_yaxes(rangemode='tozero', row=1, col=1)

    fig.show()
    if return_data:
        return payload


def build_feature_finding_dashboard(
    latent_metadata_mapping: pd.DataFrame,
    acts_df: pd.DataFrame,
    spikes_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    save_path: Optional[str] = None,   # optional CSV export
):
    """Interactive dashboard for exploring latent-to-metadata associations and finding interpretable latents."""
    from pathlib import Path

    def _bvar_name(var_name: str, var_type: str) -> str:
        return f"{var_name}_binned" if var_type == "continuous" else var_name

    # Mode selector
    mode_radio = widgets.RadioButtons(
        options=[('Preset (from latent metadata mapping table)', 'preset'),
                 ('Manual selection', 'manual')],
        value='preset', description=''
    )

    # Ensure 'bvar'
    if 'bvar' not in latent_metadata_mapping.columns:
        latent_metadata_mapping['bvar'] = latent_metadata_mapping.apply(
            lambda r: _bvar_name(r['variable'], r['variable_type']), axis=1
        )

    # Presets
    preset_entries = []
    for _, r in latent_metadata_mapping.iterrows():
        bvar = r['bvar']
        if bvar not in metadata_binned.columns:
            continue
        parts = [
            f"Inst:{int(r.instance_idx)}",
            f"latent:{int(r.latent_idx)}",
            f"Var:{bvar}",
            f"Val:{r['value']}",
            f"FracDuring:{float(r.activation_frac_during):.3f}",
            f"FracOutside:{float(r.activation_frac_outside):.3f}",
            f"SelectivityScore:{float(r.selectivity_score):.3f}",
        ]
        label = " | ".join(parts)
        preset_entries.append((label, (int(r.instance_idx), int(r.latent_idx), bvar)))
    if not preset_entries:
        preset_entries = [("— no results available —", None)]

    preset_dropdown = widgets.Dropdown(
        options=preset_entries,
        description='Select Result:',
        layout=widgets.Layout(width='80%')
    )
    preset_box = widgets.VBox([preset_dropdown])

    # Manual selection
    instance_dropdown = widgets.Dropdown(
        options=sorted(pd.unique(acts_df['instance_idx']).tolist()),
        description='Instance:'
    )
    variable_dropdown = widgets.Dropdown(options=[], description='Variable:')
    latent_dropdown   = widgets.Dropdown(description='latent:', options=[])

    def _refresh_variable_options(*_):
        inst = instance_dropdown.value
        if inst is None:
            variable_dropdown.options = []
            variable_dropdown.value = None
            return
        mask = (latent_metadata_mapping['instance_idx'] == inst)
        vars_for_inst = sorted(latent_metadata_mapping.loc[mask, 'bvar'].unique())
        vars_for_inst = [v for v in vars_for_inst if v in metadata_binned.columns]
        manual_var_options = [(v, v) for v in vars_for_inst]
        prev = variable_dropdown.value
        variable_dropdown.options = manual_var_options
        valid = [v for _, v in manual_var_options]
        variable_dropdown.value = prev if prev in valid else (valid[0] if valid else None)

    def _refresh_latent_options(*_):
        inst = instance_dropdown.value
        sel_var = variable_dropdown.value
        if (inst is None) or (sel_var is None):
            latent_dropdown.options = []
            latent_dropdown.value = None
            latent_dropdown.disabled = True
            return
        mask = (latent_metadata_mapping['instance_idx'] == inst) & (latent_metadata_mapping['bvar'] == sel_var)
        latents = sorted(latent_metadata_mapping.loc[mask, 'latent_idx'].unique())
        prev_latent = latent_dropdown.value
        latent_dropdown.options = latents
        latent_dropdown.value = prev_latent if prev_latent in latents else (latents[0] if latents else None)
        latent_dropdown.disabled = (len(latents) == 0)

    instance_dropdown.observe(_refresh_variable_options, names='value')
    instance_dropdown.observe(_refresh_latent_options, names='value')
    variable_dropdown.observe(_refresh_latent_options, names='value')
    _refresh_variable_options()
    _refresh_latent_options()

    manual_box = widgets.VBox([instance_dropdown, variable_dropdown, latent_dropdown])
    manual_box.layout.display = 'none'

    # Buttons/output
    generate_btn = widgets.Button(description='Generate Plot', button_style='info')
    clear_btn    = widgets.Button(description='Clear', button_style='warning')

    save_btn = None
    if save_path:
        save_btn = widgets.Button(description='Save plot data', button_style='success', disabled=True)
        button_box = widgets.HBox([generate_btn, save_btn, clear_btn])
        save_dir = Path(save_path)
        print(f"[Save ready] Data will be saved under: {save_dir.resolve()}")
    else:
        button_box = widgets.HBox([generate_btn, clear_btn])

    out = widgets.Output()
    last_payload = {"data": None, "inst": None, "latent": None, "var": None}

    def _on_mode_change(change):
        if change['new'] == 'preset':
            preset_box.layout.display = ''
            manual_box.layout.display = 'none'
        else:
            preset_box.layout.display = 'none'
            manual_box.layout.display = ''
            _refresh_variable_options()
            _refresh_latent_options()
    mode_radio.observe(_on_mode_change, names='value')

    def _on_generate(_):
        with out:
            clear_output()
            if mode_radio.value == 'preset':
                sel = preset_dropdown.value
                if sel is None:
                    print("No result selected.")
                    last_payload.update({"data": None})
                    if save_btn is not None:
                        save_btn.disabled = True
                    return
                inst, latent, var = sel
            else:
                inst = instance_dropdown.value
                var  = variable_dropdown.value
                latent = latent_dropdown.value
                if (inst is None) or (var is None) or (latent is None):
                    print("No matching (instance, variable, latent) for the current selection.")
                    last_payload.update({"data": None})
                    if save_btn is not None:
                        save_btn.disabled = True
                    return

            data = plot_latent_tuning(
                acts_df=acts_df,
                spikes_df=spikes_df,
                metadata_binned=metadata_binned,
                variable=var,
                instance_idx=int(inst),
                latent_idx=int(latent),
                return_data=True,
            )
            last_payload.update({"data": data, "inst": int(inst), "latent": int(latent), "var": var})
            if save_btn is not None:
                save_btn.disabled = (data is None)

    def _on_save(_):
        with out:
            if last_payload["data"] is None:
                print("No plot data available to save. Generate a plot first.")
                return
            data  = last_payload["data"]
            inst  = last_payload["inst"]
            latent = last_payload["latent"]
            var   = last_payload["var"]

            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[Error] Could not create save directory '{save_dir}': {e}")
                return

            df_main = pd.DataFrame({
                "x": data["x"],
                "latent_mean": data["latent_mean"],
                "latent_sem": data["latent_sem"],
                "top_unit_mean": data["top_unit_mean"],
                "top_unit_sem": data["top_unit_sem"],
                "bottom_unit_mean": data["bottom_unit_mean"],
                "bottom_unit_sem": data["bottom_unit_sem"],
                "selectivity_score": data["selectivity_score"],
            })
            df_zbar = pd.DataFrame({
                "unit": data["zbar_unit"],
                "mean_zscore": data["zbar_mean"],
                "sem_zscore": data["zbar_sem"],
            })

            safe_var = str(var).replace("/", "_")
            base = f"plotdata_inst{inst}_latent{latent}_{safe_var}"
            f_main = (save_dir / f"{base}.csv")
            f_zbar = (save_dir / f"{base}_zbars.csv")

            try:
                df_main.to_csv(f_main, index=False)
                df_zbar.to_csv(f_zbar, index=False)
            except Exception as e:
                print(f"[Error] Failed to save files: {e}")
                return

            print(f"[Saved] Main: {f_main.exists()} | Z-bars: {f_zbar.exists()}")

    def _on_clear(_):
        out.clear_output()

    generate_btn.on_click(_on_generate)
    if save_btn is not None:
        save_btn.on_click(_on_save)
    clear_btn.on_click(_on_clear)

    ui = widgets.VBox([
        widgets.HTML("<h2>SAE Feature Finding Dashboard</h2>"),
        mode_radio,
        preset_box,
        manual_box,
        button_box,
        out
    ])
    return ui

def load_selectivity_score_df(main_data: Union[Path, pd.DataFrame, Dict]) -> pd.DataFrame:
    """Return DataFrame with ['x','selectivity_score'] and attach a __source_name for labeling."""
    if isinstance(main_data, Path):
        df = pd.read_csv(main_data)
        df.attrs["__source_name"] = os.path.basename(main_data)
    elif isinstance(main_data, pd.DataFrame):
        df = main_data.copy()
    elif isinstance(main_data, dict):
        df = pd.DataFrame({
            "x": main_data.get("x", []),
            "selectivity_score": main_data.get("selectivity_score", [])
        })
        var   = main_data.get("variable", "var")
        inst  = main_data.get("instance_idx", "inst")
        latent = main_data.get("latent_idx", "latent")
        df.attrs["__source_name"] = f"inst{inst}_latent{latent}_{var}"
    else:
        raise TypeError("Input must be a CSV path, DataFrame, or payload dict.")
    if not {"x", "selectivity_score"}.issubset(df.columns):
        raise ValueError('Data must include "x" and "selectivity_score".')
    return df

def plot_selectivity_score_from_saved(
    *main_datas: Union[Path, pd.DataFrame, Dict],
    labels: Optional[List[str]] = None,
    y_max: float = 0.7,
    title: str = "Selectivity Score"
):
    """Overlay selectivity score curves/bars from multiple saved datasets."""
    if len(main_datas) == 1 and isinstance(main_datas[0], (list, tuple)):
        main_datas = tuple(main_datas[0])
    if not main_datas:
        raise ValueError("Provide at least one dataset.")

    dfs = [load_selectivity_score_df(md) for md in main_datas]
    base_x = dfs[0]["x"].astype(str).tolist()
    for i, df in enumerate(dfs[1:], 1):
        if df["x"].astype(str).tolist() != base_x:
            raise ValueError(f"Bins mismatch between dataset 0 and {i}.")
    cats = list(base_x)
    for df in dfs:
        df["x"] = pd.Categorical(df["x"], categories=cats, ordered=True)

    interval_like = all(re.match(r'^[\(\[].*,.*[\)\]]$', s) for s in base_x)

    if labels is None:
        labels = [df.attrs.get("__source_name", f"series{i}") for i, df in enumerate(dfs)]
    if len(labels) != len(dfs):
        raise ValueError("labels length must match number of datasets.")

    fig = go.Figure()
    x_vals = [str(x) for x in cats]

    if interval_like:
        for df, lab in zip(dfs, labels):
            fig.add_trace(go.Scatter(x=x_vals, y=df["selectivity_score"], mode="lines+markers", name=lab))
    else:
        for idx, (df, lab) in enumerate(zip(dfs, labels)):
            fig.add_trace(go.Bar(
                x=x_vals, y=df["selectivity_score"], name=lab, offsetgroup=str(idx),
                hovertemplate=f"{lab}<br>%{{x}}<br>Selectivity=%{{y:.3f}}<extra></extra>", opacity=0.85
            ))
        fig.update_layout(barmode="group")

    fig.update_layout(
        title=title, height=450, width=600, template="plotly_white", showlegend=True,
        margin=dict(t=60, r=30, b=60, l=60)
    )
    fig.update_xaxes(title_text="Bin")
    fig.update_yaxes(title_text="Selectivity Score", range=[0, y_max])
    return fig

def load_zbar_df(zbar_data: Union[Path, pd.DataFrame, Dict]) -> pd.DataFrame:
    """Return DataFrame with ['unit','mean_zscore','sem_zscore'] and attach a __source_name for labeling."""
    if isinstance(zbar_data, Path):
        df = pd.read_csv(zbar_data)
        df.attrs["__source_name"] = os.path.basename(zbar_data)
    elif isinstance(zbar_data, pd.DataFrame):
        df = zbar_data.copy()
    elif isinstance(zbar_data, dict):
        df = pd.DataFrame({
            "unit": zbar_data.get("zbar_unit", []),
            "mean_zscore": zbar_data.get("zbar_mean", []),
            "sem_zscore": zbar_data.get("zbar_sem", []),
        })
        var   = zbar_data.get("variable", "var")
        inst  = zbar_data.get("instance_idx", "inst")
        latent = zbar_data.get("latent_idx", "latent")
        df.attrs["__source_name"] = f"inst{inst}_latent{latent}_{var}_zbars"
    else:
        raise TypeError("Input must be a *_zbars.csv Path, DataFrame, or payload dict.")

    return df


def plot_neuron_zscores_from_saved(
    *zbar_datas: Union[Path, pd.DataFrame, Dict],
    labels: Optional[List[str]] = None,
    title: str = "Mean Z-scores when Latent Active",
    show_sem: bool = True,   # 🔑 new flag
):
    """Overlay neuron mean z-scores (±SEM) from multiple saved datasets, like the dashboard plot."""
    if len(zbar_datas) == 1 and isinstance(zbar_datas[0], (list, tuple)):
        zbar_datas = tuple(zbar_datas[0])
    if not zbar_datas:
        raise ValueError("Provide at least one zbar dataset.")

    dfs = [load_zbar_df(zb) for zb in zbar_datas]

    # Labels
    if labels is None:
        labels = [df.attrs.get("__source_name", f"series{i}") for i, df in enumerate(dfs)]
    if len(labels) != len(dfs):
        raise ValueError("labels length must match number of datasets.")

    fig = go.Figure()
    for idx, (df, lab) in enumerate(zip(dfs, labels)):
        fig.add_trace(go.Bar(
            x=df["unit"], 
            y=df["mean_zscore"],
            error_y=(dict(type="data", array=df["sem_zscore"]) if show_sem else None),
            name=lab, opacity=0.8, offsetgroup=str(idx)
        ))

    fig.update_layout(
        title=title, height=450, width=900, template="plotly_white",
        barmode="group", showlegend=True,
        margin=dict(t=60, r=30, b=60, l=60)
    )
    fig.update_xaxes(title_text="Unit", tickangle=45)
    fig.update_yaxes(title_text="Mean Z-score", zeroline=True, zerolinewidth=1)

    return fig

