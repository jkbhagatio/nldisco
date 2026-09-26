"""Select one frozen LangevinFlow configuration using four development features.

The label and scoring contract is supplied by comparison_features. Preflight is
available independently and never selects a configuration from incomplete labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Optional

import numpy as np
from beartype import beartype

SWEEP = Path('/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1')
FEATURE_NAMES = ('recent_braking','early_leftward','fast_target_specific','late_target_specific')


@beartype
def select_configuration(best_by_config: dict) -> Optional[dict]:
    """Require four eligible dimensions before ranking a configuration equally."""
    complete = []
    for config_id, features in best_by_config.items():
        if any(not features.get(name) for name in FEATURE_NAMES):
            continue
        if any(not features[name]['eligible'] for name in FEATURE_NAMES):
            continue
        complete.append(dict(config_id=config_id,
                             mean_auroc=float(np.mean([features[name]['auroc'] for name in FEATURE_NAMES])),
                             features={name:features[name] for name in FEATURE_NAMES}))
    return sorted(complete,key=lambda row:(-row['mean_auroc'],row['config_id']))[0] if complete else None


@beartype
def sha256(path: Path) -> str:
    """Hash a saved artifact without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@beartype
def exports(sweep: Path = SWEEP) -> list[dict]:
    """Validate every frozen export and return exact paths and provenance."""
    expected = {f'h{h}_lr{str(lr).replace(".","p")}_ramp{r}'
                for h, lr, r in itertools.product((32,64,128),(.001,.003),(5,500))}
    folders = sorted((sweep/'langevinflow').glob('h*'))
    if {folder.name for folder in folders} != expected:
        raise ValueError('Expected exactly the approved twelve LangevinFlow configurations')
    shared_path = sweep/'shared/index.npz'
    shared_hash = sha256(shared_path)
    with np.load(shared_path) as index:
        anchor = index['anchor_index']
        source = index['source_indices']
    records = []
    for folder in folders:
        config = json.loads((folder/'config.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        with np.load(folder/'index.npz') as index:
            np.testing.assert_array_equal(index['anchor_index'],anchor)
            np.testing.assert_array_equal(index['source_indices'],source)
        values_path = folder/'activations.npy'
        values = np.load(values_path,mmap_mode='r')
        width = config['hidden_size']
        if values.shape != (len(anchor),3*width) or values.dtype != np.float32:
            raise ValueError(f'Unexpected activation shape or dtype: {folder.name}')
        if not summary['complete'] or summary['reload_max_abs'] != 0:
            raise ValueError(f'Incomplete or failed reload: {folder.name}')
        if config['index_sha256'] != shared_hash:
            raise ValueError(f'Shared index provenance changed: {folder.name}')
        for key,file in (('adapter_sha256','runner_source.py'),('source_sha256','upstream_models.py')):
            if sha256(folder/file) != config[key]:
                raise ValueError(f'Source provenance changed: {folder.name}/{file}')
        records.append(dict(config_id=folder.name,activation_path=str(values_path),
                            activation_shape=list(values.shape),activation_dtype=str(values.dtype),
                            activation_sha256=sha256(values_path),
                            config_sha256=sha256(folder/'config.json'),
                            shared_index_sha256=shared_hash,adapter_sha256=config['adapter_sha256'],
                            source_sha256=config['source_sha256'],
                            coordinates=dict(q=[0,width],p=[width,2*width],h=[2*width,3*width])))
    return records


@beartype
def preflight(output: Path, sweep: Path = SWEEP) -> None:
    """Save reproducible input checks while feature 181 remains unresolved."""
    records = exports(sweep)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(method='langevinflow',status='pending_feature_181_definition',
                  selected_config=None,ranking=None,pending_features=[181],
                  expected_features=[31,132,181,4],configs=records,
                  selection_rule='One configuration maximizing equal-weight mean of four per-feature best eligible oriented continuous AUROCs',
                  dense_activity_rule='Strict > threshold matching illustrative NLDisco z>0 development activity rate on identical valid rows; no tie splitting',
                  eligibility='Varying, TPR>=.05, at least20 active positive trials; no Sel>=.65 gate',
                  prohibited_fallback='Do not select or finalize a three-feature configuration',
                  development_only=True,test_labels_evaluated=False,
                  source_sha256=sha256(Path(__file__)))
    with (output/'preflight.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    with (output/'preflight_source.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    print(json.dumps(dict(status=report['status'],configs=len(records),selected_config=None)),flush=True)


@beartype
def evaluate(output: Path, sweep: Path = SWEEP) -> None:
    """Score only after all four shared definitions are explicitly available."""
    if __package__:
        from . import comparison_features as contract
    else:
        import comparison_features as contract
    features = contract.load_development_features()
    missing = sorted(set(contract.EXPECTED_FEATURES)-set(features))
    if missing:
        raise ValueError(f'Configuration selection blocked; missing verified labels: {missing}')
    if tuple(contract.EXPECTED_FEATURES) != FEATURE_NAMES:
        raise ValueError('Shared feature contract does not match the four frozen features')
    records = exports(sweep)
    destination = output/'evaluation'
    destination.mkdir(parents=True,exist_ok=False)
    all_rows, winners = [], {}
    for record in records:
        config_id = record['config_id']
        values = np.load(record['activation_path'],mmap_mode='r')
        winners[config_id] = {}
        for name in FEATURE_NAMES:
            scores = contract.score_dense(values,features[name])
            winners[config_id][name] = contract.best_eligible(scores)
            all_rows.extend(dict(config_id=config_id,feature=name,**row) for row in scores)
        print(json.dumps(dict(config_id=config_id,features_scored=4)),flush=True)
    choice = select_configuration(winners)
    summary = dict(method='langevinflow',selected=choice,all_config_feature_winners=winners,
                   selection_rule='Equal-weight mean of four best eligible oriented continuous AUROCs; one configuration only',
                   feature_definitions={name:features[name].definition for name in FEATURE_NAMES},
                   reference_rates={name:features[name].reference_rate for name in FEATURE_NAMES},
                   comparator_provenance=records,source_sha256=sha256(Path(__file__)),
                   contract_sha256=sha256(Path(contract.__file__)),
                   development_only=True,test_labels_evaluated=False,
                   decoder_requirement='Any later single-latent decoder must use exactly the selected configuration and these same selected dimensions and orientations')
    with (destination/'summary.json').open('x') as stream:json.dump(summary,stream,indent=2)
    with (destination/'all_latent_metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    for name,feature in features.items():
        np.savez(destination/f'{name}_development.npz',rows=feature.rows,
                 labels=feature.labels,trials=feature.trials)
    for name,path in (('source.py',Path(__file__)),('comparison_features.py',Path(contract.__file__))):
        with (destination/name).open('xb') as stream:stream.write(path.read_bytes())


@beartype
def sensitivity(output: Path, sweep: Path = SWEEP) -> None:
    """Limit the fixed selected model to 32 random candidates, without model search."""
    destination = output/'evaluation'
    report = json.loads((destination/'summary.json').read_text())
    selected = report['selected']
    config_id = selected['config_id']
    metrics = {name:[] for name in FEATURE_NAMES}
    with (destination/'all_latent_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['config_id'] == config_id:
                metrics[row['feature']].append(dict(latent_id=int(row['latent_id']),
                    auroc=float(row['auroc']),sel=float(row['sel']),eligible=row['eligible']=='True'))
    dimensions = len(metrics[FEATURE_NAMES[0]])
    rng = np.random.default_rng(0)
    draws = []
    for draw in range(100):
        candidates = set(rng.choice(dimensions,size=32,replace=False).tolist())
        winners = {}
        for name,rows in metrics.items():
            eligible = [row for row in rows if row['eligible'] and row['latent_id'] in candidates]
            winners[name] = max(eligible,key=lambda row:(row['auroc'],-row['latent_id'])) if eligible else None
        complete = all(winners.values())
        draws.append(dict(draw=draw,candidate_ids=sorted(candidates),complete=complete,
                          mean_auroc=float(np.mean([row['auroc'] for row in winners.values()])) if complete else None,
                          winners=winners))
    complete_values = [row['mean_auroc'] for row in draws if row['complete']]
    result = dict(config_id=config_id,candidates=32,draws=100,seed=0,
                  rule='Fixed selected model; same sampled 32 coordinates for all four features; unchanged eligibility, AUROC orientation and thresholds; no model reselection',
                  full_candidate_count=dimensions,full_candidate_mean_auroc=selected['mean_auroc'],
                  complete_draws=len(complete_values),incomplete_draws=100-len(complete_values),
                  complete_draw_mean_auroc_mean=float(np.mean(complete_values)) if complete_values else None,
                  complete_draw_mean_auroc_quantiles=np.quantile(complete_values,[0,.025,.5,.975,1]).tolist() if complete_values else [],
                  per_feature={name:dict(eligible_draws=sum(row['winners'][name] is not None for row in draws),
                    auroc_quantiles=np.quantile([row['winners'][name]['auroc'] for row in draws if row['winners'][name]],
                                               [0,.025,.5,.975,1]).tolist()) for name in FEATURE_NAMES},
                  samples=draws,source_sha256=sha256(Path(__file__)))
    model_folder = sweep/'langevinflow'/config_id
    result['saved_training_config'] = json.loads((model_folder/'config.json').read_text())
    result['saved_training_summary'] = json.loads((model_folder/'summary.json').read_text())
    with (destination/'candidate_sensitivity.json').open('x') as stream:json.dump(result,stream,indent=2)
    with (destination/'sensitivity_source.py').open('xb') as stream:stream.write(Path(__file__).read_bytes())
    print(json.dumps({key:value for key,value in result.items() if key not in ('samples','saved_training_config','saved_training_summary')}),flush=True)


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=SWEEP/'feature_comparison/langevinflow')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--sensitivity',action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.output)
    elif args.sensitivity:
        sensitivity(args.output)
    else:
        evaluate(args.output)


if __name__ == '__main__':
    main()
