"""Independently audit task coverage, paired timing arithmetic and saved state."""
import argparse
import json
from pathlib import Path
import numpy as np
from aggregation import backbone_statistics, write_csv, close
from portable_dc import CompiledCalibrator

OPERATIONS = ['reference_decode', 'compiled_decode', 'reference_decode_and_queries',
              'direct_queries_from_rows', 'direct_queries_from_summaries']
VERSION = 'compiled-dc16-queries-v1-20261102'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--primary-runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {t['id']:t for t in json.loads(args.manifest.read_text())}
    rows = []; complete = []; errors = []; maximum_difference = 0.
    for path in sorted(args.runs.glob('*/results.json')):
        try:
            result = json.loads(path.read_text()); task = result['task']; identity = task['id']
            if identity not in expected or path.parent.name != identity or result['version'] != VERSION:
                raise ValueError('Compiled task identity/version mismatch')
            for key in ['backbone', 'dataset', 'horizon', 'seed']:
                if task[key] != expected[identity][key]:
                    raise ValueError('Compiled task differs from manifest')
            if result['samples'] != 100 or result['thresholds_per_coordinate'] != 3 or result['repeats'] != 3:
                raise ValueError('Incomplete output/timing protocol')
            if not result['exact_reconstructed_values'] or not result['exact_event_probabilities'] or not result['brier_matches_existing_control']:
                raise ValueError('Compiled output failed exact numerical parity')
            maximum_difference = max(maximum_difference, result['maximum_value_difference'])
            control = json.loads((args.primary_runs/identity/'controls.json').read_text())
            if result['coordinates'] != control['retained_test_coordinates'] or result['complete_test_coordinates'] != control['full_test_coordinates_seen']:
                raise ValueError('Compiled benchmark population differs from its actual cache')
            close(result['brier'], control['methods']['Budget50000']['events']['brier'], 'Compiled Brier differs from original control')
            medians = {}
            if set(result['times']) != set(OPERATIONS):
                raise ValueError('Missing paired timing operation')
            for operation in OPERATIONS:
                times = np.asarray(result['times'][operation])
                if times.shape != (3,) or not np.isfinite(times).all() or np.any(times <= 0):
                    raise ValueError('Invalid timing samples')
                medians[operation] = float(np.median(times))
                close(medians[operation], result['median_seconds'][operation], 'Median timing mismatch')
            model_path = path.parent/'compiled.npz'; compiled = CompiledCalibrator.load(model_path)
            if model_path.stat().st_size != result['compiled_npz_bytes'] or compiled.horizon_length != task['horizon'] or compiled.n_horizon_bins != 1:
                raise ValueError('Saved deployment-state identity/size mismatch')
            if compiled.quantiles.shape[1] != 100 or not 1 <= len(compiled.quantiles) <= 16:
                raise ValueError('Unexpected compiled template dimensions')
            if compiled.state_to_template.shape != (16,) or np.any(compiled.state_to_template < 0) or np.any(compiled.state_to_template >= len(compiled.quantiles)):
                raise ValueError('Invalid state/template mapping')
            if not np.isfinite(compiled.quantiles).all() or np.any(np.diff(compiled.quantiles, axis=1) < 0):
                raise ValueError('Invalid ordered quantile templates')
            if result['compilation_seconds'] < 0 or not np.isfinite(result['compilation_seconds']):
                raise ValueError('Invalid compilation time')
            row = {k:task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
            row.update(method='CompiledDC16', coordinates=result['coordinates'],
                compile_ms=1000*result['compilation_seconds'], original_kib=result['original_pickle_bytes']/1024,
                compiled_kib=result['compiled_npz_bytes']/1024,
                state_size_ratio=result['original_pickle_bytes']/result['compiled_npz_bytes'],
                decode_speedup=medians['reference_decode']/medians['compiled_decode'],
                direct_rows_query_speedup=medians['reference_decode_and_queries']/medians['direct_queries_from_rows'],
                cached_summary_query_speedup=medians['reference_decode_and_queries']/medians['direct_queries_from_summaries'])
            row.update({k+'_us_per_coordinate':1e6*v/result['coordinates'] for k,v in medians.items()})
            rows.append(row); complete.append(identity)
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    missing = sorted(set(expected)-set(complete)); args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'raw_task_metrics.csv', rows)
    keys = [key for key in rows[0] if key not in ['id', 'backbone', 'dataset', 'horizon', 'seed', 'method', 'coordinates']] if rows else []
    write_csv(args.output/'backbone_statistics.csv', backbone_statistics(rows, keys))
    report = dict(expected_tasks=len(expected), audited_tasks=len(complete), errors=errors,
        missing_tasks=missing, all_tasks_complete=not missing and not errors,
        maximum_value_difference=maximum_difference,
        evaluation='runtime/parity on shared retained coordinate caches, not replacement full-test quality scores',
        timing='median of three randomized paired repetitions; ratios formed per task before seed-aware aggregation')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'missing_tasks'}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
