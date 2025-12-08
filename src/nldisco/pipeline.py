"""Mini pipeline functions."""

import numpy as np
import pandas as pd

from typing import List, Dict, Optional, Tuple

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
    n_bins: int,
    min_activation_frac: float
) -> List[Dict]:
    """Analyzes a continuous variable by binning it and then using the discrete analysis method."""
    print(f"    Binning '{variable}' into {n_bins} bins...")
    binned_col_name = f"{variable}_binned"

    data_to_bin = metadata_binned[variable].dropna()
    if data_to_bin.empty:
        return []

    if variable == 'movement_angle':
        bins = np.linspace(-180, 180, n_bins + 1)
        labels = [f"({bins[i]:.0f}, {bins[i+1]:.0f}]" for i in range(n_bins)]
        metadata_binned[binned_col_name] = pd.cut(data_to_bin, bins=bins, labels=labels, include_lowest=True)
    else:
        metadata_binned[binned_col_name] = pd.qcut(data_to_bin, q=n_bins, labels=None, duplicates='drop')

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
    n_bins_continuous: List[int] = None,
    top_n_mappings: int = 3
) -> pd.DataFrame:  
    """Automatically maps SAE latents to metadata variables, returning a ranked DataFrame of associations."""
    if discrete_vars is None: discrete_vars = []
    if continuous_vars is None: continuous_vars = []
    if n_bins_continuous is None: n_bins_continuous = [10] * len(continuous_vars)
    if len(n_bins_continuous) != len(continuous_vars):
        raise ValueError("n_bins_continuous must match length of continuous_vars")
    
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
            n_bins = n_bins_continuous[idx]
            results = analyze_continuous_variable(acts_df, metadata_binned, variable, n_bins=n_bins, min_activation_frac=min_activation_frac)
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


"""Visualisation functions for all latent associations"""

def plot_latent_tuning(
    acts_df: pd.DataFrame,
    spikes_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    variable: str,
    instance_idx: int,
    latent_idx: int
):
    """Visualizes SAE latent tuning to metadata variables."""
    # Calculate z-scores for spike counts across units (biological neurons)
    # Calculate mean and standard deviation for each unit (column)
    unit_means = spikes_df.mean(axis=0)
    unit_stds = spikes_df.std(axis=0)
    # Calculate z-scores
    # Handle cases where standard deviation is zero to avoid division by zero
    spk_z_scores_df = spikes_df.sub(unit_means, axis=1).div(unit_stds, axis=1)
    spk_z_scores_df = spk_z_scores_df.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN for clarity
    # Set z-score to 0 where standard deviation was 0 (and thus z-score would be NaN)
    spk_z_scores_df = spk_z_scores_df.fillna(0.0)

    # Get latent activations
    latent_acts = acts_df[(acts_df['instance_idx'] == instance_idx) & (acts_df['latent_idx'] == latent_idx)]
    
    # Find top/bottom co-active units
    if len(latent_acts) > 0:
        latent_active_indices = latent_acts['example_idx'].values
        unit_mean_zscores = spk_z_scores_df.iloc[latent_active_indices].mean(axis=0)
        top_unit = unit_mean_zscores.idxmax()
        bottom_unit = unit_mean_zscores.idxmin()
        top_zscore = unit_mean_zscores[top_unit]
        bottom_zscore = unit_mean_zscores[bottom_unit]
        print(f"Top co-active unit: {top_unit} (z-score: {top_zscore:.3f})")
        print(f"Bottom co-active unit: {bottom_unit} (z-score: {bottom_zscore:.3f})")
    else:
        print("No activations found for this instance/latent.")
        return
    
    # Create complete dataset with zeros for inactive latents
    all_examples = pd.DataFrame({'example_idx': range(len(metadata_binned))})
    all_examples = all_examples.merge(latent_acts[['example_idx', 'activation_value']], on='example_idx', how='left').fillna(0)
    
    # Get metadata and z-scores for all examples
    metadata_slice = metadata_binned[[variable]].iloc[all_examples['example_idx']]
    top_unit_slice = spk_z_scores_df[[top_unit]].iloc[all_examples['example_idx']]
    bottom_unit_slice = spk_z_scores_df[[bottom_unit]].iloc[all_examples['example_idx']]
    
    # Create plotting dataframe
    data_df = metadata_slice.reset_index(drop=True)
    data_df['activation_value'] = all_examples['activation_value'].reset_index(drop=True)
    data_df['top_unit_zscore'] = top_unit_slice[top_unit].reset_index(drop=True)
    data_df['bottom_unit_zscore'] = bottom_unit_slice[bottom_unit].reset_index(drop=True)
    
    # Check if data is interval type
    try:
        is_interval_data = pd.api.types.is_interval_dtype(data_df[variable].cat.categories)
    except AttributeError:
        is_interval_data = False
    
    # Calculate summary statistics
    stats_df = data_df.groupby(variable).agg({
        'activation_value': ['mean', 'sem'],
        'top_unit_zscore': ['mean', 'sem'],
        'bottom_unit_zscore': ['mean', 'sem']
    }).reset_index()
    
    # Flatten column names
    stats_df.columns = [variable, 'latent_mean', 'latent_sem', 'top_unit_mean', 'top_unit_sem', 'bottom_unit_mean', 'bottom_unit_sem']
    
    # Calculate rate proportions
    if not latent_acts.empty:
        condition_masks = {condition: metadata_binned[variable] == condition for condition in stats_df[variable]}
        active_example_set = set(latent_acts['example_idx'])
        
        rate_props = []
        for _, row in stats_df.iterrows():
            condition = row[variable]
            condition_mask = condition_masks[condition]
            
            condition_example_idxs = np.where(condition_mask)[0]
            condition_activations = len(active_example_set.intersection(condition_example_idxs))
            activation_frac_during = condition_activations / len(condition_example_idxs) if len(condition_example_idxs) > 0 else 0
            
            non_condition_example_idxs = np.where(~condition_mask)[0]
            non_condition_activations = len(active_example_set.intersection(non_condition_example_idxs))
            activation_frac_outside = non_condition_activations / len(non_condition_example_idxs) if len(non_condition_example_idxs) > 0 else 0
            
            selectivity_score = activation_frac_during / (activation_frac_during + activation_frac_outside) if (activation_frac_during + activation_frac_outside) > 0 else 0
            rate_props.append(selectivity_score)
        
        stats_df['selectivity_score'] = rate_props
    else:
        stats_df['selectivity_score'] = 0
    
    # Calculate z-score stats for bar plot
    if len(latent_acts) > 0:
        zscore_stats = spk_z_scores_df.iloc[latent_active_indices].agg(['mean', 'sem']).T
        zscore_stats.columns = ['mean_zscore', 'sem_zscore']
        zscore_stats = zscore_stats.reset_index()
        zscore_stats.columns = ['unit', 'mean_zscore', 'sem_zscore']
    else:
        zscore_stats = pd.DataFrame({'unit': spk_z_scores_df.columns, 'mean_zscore': 0, 'sem_zscore': 0})
    
    # Create plots based on variable type
    if 'angle' in variable:  # Polar plot
        stats_df['theta'] = stats_df[variable].apply(lambda x: x.mid if isinstance(x, pd.Interval) else x)
        stats_df = stats_df.sort_values('theta')
        plot_df = pd.concat([stats_df, stats_df.head(1)], ignore_index=True)
        
        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}, {"type": "polar"}],
                   [{"type": "xy", "colspan": 4}, None, None, None]],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            subplot_titles=["latent Activation", "Top unit Z-score", "Bottom unit Z-score", "Selectivity Score", "Mean Z-scores when latent Active"]
        )
        
        # latent activation polar plot
        fig.add_trace(go.Scatterpolar(r=plot_df['latent_mean'] + plot_df['latent_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatterpolar(r=plot_df['latent_mean'] - plot_df['latent_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), fill='tonext', fillcolor='rgba(220,20,60,0.2)', name='latent ±SEM'), row=1, col=1)
        fig.add_trace(go.Scatterpolar(r=plot_df['latent_mean'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='crimson', width=3), name='latent Activation'), row=1, col=1)
        
        # Top unit polar plot
        fig.add_trace(go.Scatterpolar(r=plot_df['top_unit_mean'] + plot_df['top_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=plot_df['top_unit_mean'] - plot_df['top_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), fill='tonext', fillcolor='rgba(0,0,139,0.2)', name='Top unit ±SEM'), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=plot_df['top_unit_mean'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='darkblue', width=3), name='Top unit Z-score'), row=1, col=2)
        
        # Bottom unit polar plot
        fig.add_trace(go.Scatterpolar(r=plot_df['bottom_unit_mean'] + plot_df['bottom_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatterpolar(r=plot_df['bottom_unit_mean'] - plot_df['bottom_unit_sem'], theta=plot_df['theta'], mode='lines', line=dict(width=0), fill='tonext', fillcolor='rgba(255,165,0,0.2)', name='Bottom unit ±SEM'), row=1, col=3)
        fig.add_trace(go.Scatterpolar(r=plot_df['bottom_unit_mean'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='orange', width=3), name='Bottom unit Z-score'), row=1, col=3)
        
        # Selectivity score polar plot
        fig.add_trace(go.Scatterpolar(r=plot_df['selectivity_score'], theta=plot_df['theta'], mode='lines+markers', line=dict(color='green', width=3), name='Selectivity Score'), row=1, col=4)
        
        # Z-score bar plot
        fig.add_trace(go.Bar(x=zscore_stats['unit'], y=zscore_stats['mean_zscore'], error_y=dict(type='data', array=zscore_stats['sem_zscore']), marker_color='purple', marker_line_width=0, opacity=0.7, name='Mean Z-score'), row=2, col=1)
        
        # Create tick labels
        if 'movement_angle' not in variable:
            tick_labels = [f"{int(round(theta))}°" for theta in stats_df['theta']]
        else:
            tick_labels = [f"{int(interval.mid)}°" if hasattr(interval, 'mid') else str(interval) for interval in stats_df[variable]]    
        
        fig.update_layout(title=f"Instance {instance_idx} latent {latent_idx} & Top unit {top_unit} & Bottom unit {bottom_unit}", showlegend=True, height=800, width=1600, margin=dict(t=80, b=60, l=50, r=50))
        
        # Update polar plots
        rotation = 195 if 'movement_angle' in variable else 0
        tickvals = stats_df['theta'].tolist() if 'movement_angle' in variable else [(angle % 360) for angle in stats_df['theta'].tolist()]
        for i in range(1, 5):
            polar_key = f'polar{i if i > 1 else ""}'
            fig.update_layout(**{polar_key: dict(angularaxis=dict(direction="counterclockwise", rotation=rotation, tickvals=tickvals, ticktext=tick_labels, tickfont=dict(size=10)), radialaxis=dict(range=[0, None]))})

        fig.update_xaxes(title_text="unit", row=2, col=1)
        fig.update_yaxes(title_text="Mean Z-score", row=2, col=1)
            
    else:  # Linear plot
        fig = make_subplots(rows=2, cols=4, specs=[[{}, {}, {}, {}], [{"colspan": 4}, None, None, None]], subplot_titles=["latent Activation", "Top unit Z-score", "Bottom unit Z-score", "Selectivity Score", "Mean Z-scores when latent Active"], horizontal_spacing=0.1, vertical_spacing=0.3)
        
        # Setup x-axis labels
        if is_interval_data:
            x_axis_labels = stats_df[variable].apply(lambda x: str(x))
            stats_df = stats_df.sort_values(by=variable)
        else:
            x_axis_labels = stats_df[variable].astype(str)
        
        # Plot based on data type
        if is_interval_data:
            # Line plots for continuous data
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['latent_mean'] + stats_df['latent_sem'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['latent_mean'] - stats_df['latent_sem'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(220,20,60,0.2)', name='latent ±SEM'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['latent_mean'], mode='lines+markers', line_color='crimson', name='latent Activation'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['top_unit_mean'] + stats_df['top_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['top_unit_mean'] - stats_df['top_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(0,0,139,0.2)', name='Top unit ±SEM'), row=1, col=2)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['top_unit_mean'], mode='lines+markers', line_color='darkblue', name='Top unit Z-score'), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['bottom_unit_mean'] + stats_df['bottom_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=3)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['bottom_unit_mean'] - stats_df['bottom_unit_sem'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(255,165,0,0.2)', name='Bottom unit ±SEM'), row=1, col=3)
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['bottom_unit_mean'], mode='lines+markers', line_color='orange', name='Bottom unit Z-score'), row=1, col=3)
            
            fig.add_trace(go.Scatter(x=x_axis_labels, y=stats_df['selectivity_score'], mode='lines+markers', line=dict(color='green', width=3), name='Selectivity Score'), row=1, col=4)
        else:
            # Bar plots for categorical data
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['latent_mean'], error_y=dict(type='data', array=stats_df['latent_sem']), marker_color='crimson', marker_line_width=0, opacity=0.7, name='latent Activation'), row=1, col=1)
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['top_unit_mean'], error_y=dict(type='data', array=stats_df['top_unit_sem']), marker_color='darkblue', marker_line_width=0, opacity=0.7, name='Top unit Z-score'), row=1, col=2)
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['bottom_unit_mean'], error_y=dict(type='data', array=stats_df['bottom_unit_sem']), marker_color='orange', marker_line_width=0, opacity=0.7, name='Bottom unit Z-score'), row=1, col=3)
            fig.add_trace(go.Bar(x=x_axis_labels, y=stats_df['selectivity_score'], marker_color='green', marker_line_width=0, opacity=0.7, name='Selectivity Score'), row=1, col=4)
        
        # Z-score bar plot
        fig.add_trace(go.Bar(x=zscore_stats['unit'], y=zscore_stats['mean_zscore'], error_y=dict(type='data', array=zscore_stats['sem_zscore']), marker_color='purple', marker_line_width=0, opacity=0.7, name='Mean Z-score'), row=2, col=1)
        
        fig.update_layout(title=f"Instance {instance_idx} latent {latent_idx} & Top unit {top_unit} & Bottom unit {bottom_unit}", height=800, width=1600, showlegend=True, margin=dict(t=80, b=60, l=50, r=50))
        
        # Update axes
        fig.update_xaxes(title_text=variable, tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text=variable, tickangle=45, row=1, col=2)
        fig.update_xaxes(title_text=variable, tickangle=45, row=1, col=3)
        fig.update_xaxes(title_text=variable, tickangle=45, row=1, col=4)
        fig.update_xaxes(title_text="unit", row=2, col=1)
        
        fig.update_yaxes(title_text="latent Activation", color="crimson", range=[0, None], row=1, col=1)
        fig.update_yaxes(title_text="Top unit Z-score", color="darkblue", row=1, col=2)
        fig.update_yaxes(title_text="Bottom unit Z-score", color="orange", row=1, col=3)
        fig.update_yaxes(title_text="Selectivity Score", range=[0, None], color="green", row=1, col=4)
        fig.update_yaxes(title_text="Mean Z-score", row=2, col=1)
        
        fig.update_yaxes(rangemode='tozero', row=1, col=1)
        fig.update_yaxes(rangemode='tozero', row=1, col=4)
    
    fig.show()

def build_feature_finding_dashboard(
    latent_metadata_mapping: pd.DataFrame,
    acts_df: pd.DataFrame,
    spikes_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
):
    """Builds an interactive dashboard for exploring latent-to-metadata associations and finding interpretable latents (features)."""

    def _bvar_name(var_name: str, var_type: str) -> str:
        return f"{var_name}_binned" if var_type == "continuous" else var_name

    # Mode selector
    mode_radio = widgets.RadioButtons(
        options=[('Preset (from latent metadata mapping table)', 'preset'),
                 ('Manual selection', 'manual')],
        value='preset', description=''
    )

    # Ensure latent_metadata_mapping has 'bvar'
    if 'bvar' not in latent_metadata_mapping.columns:
        latent_metadata_mapping['bvar'] = latent_metadata_mapping.apply(
            lambda r: _bvar_name(r['variable'], r['variable_type']), axis=1
        )

    # Preset dropdown
    preset_entries = []
    for _, r in latent_metadata_mapping.iterrows():
        bvar = r['bvar']
        if bvar not in metadata_binned.columns:
            continue  # skip vars you haven't binned

        parts = [
            f"Inst:{int(r.instance_idx)}",
            f"latent:{int(r.latent_idx)}",
            f"Var:{bvar}",
            f"Val:{r['value']}",
            f"FracDuring:{float(r.activation_frac_during):.3f}",
            f"FracOutside:{float(r.activation_frac_outside):.3f}",
            f"SelectivityScore:{float(r.selectivity_score):.3f}"
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
    latent_dropdown = widgets.Dropdown(description='latent:', options=[])

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
        valid_vals = [v for _, v in manual_var_options]
        variable_dropdown.value = prev if prev in valid_vals else (valid_vals[0] if valid_vals else None)

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
    manual_box.layout.display = 'none'  # start hidden

    # Buttons + output
    generate_btn = widgets.Button(description='Generate Plot', button_style='info')
    clear_btn    = widgets.Button(description='Clear', button_style='warning')
    button_box   = widgets.HBox([generate_btn, clear_btn])
    out = widgets.Output()

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
                sel: Optional[Tuple[int, int, str]] = preset_dropdown.value
                if sel is None:
                    print("No result selected.")
                    return
                inst, latent, var = sel
            else:
                inst = instance_dropdown.value
                var  = variable_dropdown.value
                latent = latent_dropdown.value
                if (inst is None) or (var is None) or (latent is None):
                    print("No matching (instance, variable, latent) for the current selection.")
                    return

            plot_latent_tuning(
                acts_df=acts_df,
                spikes_df=spikes_df,
                metadata_binned=metadata_binned,
                variable=var,
                instance_idx=int(inst),
                latent_idx=int(latent),
            )

    def _on_clear(_):
        out.clear_output()

    generate_btn.on_click(_on_generate)
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
