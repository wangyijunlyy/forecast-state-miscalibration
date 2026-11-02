"""Check the portable deployment against actual completed server calibrators."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from run_task import fit_dc, atomic
from portable_dc import EmpiricalCalibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    by_backbone = {}
    for path in sorted((args.root/'runs').glob('*/results.json')):
        result = json.loads(path.read_text()); backbone = result['task']['backbone']
        excluded = result['validation']['models']['DC16']['excluded_sources']
        if backbone not in by_backbone or excluded > by_backbone[backbone][0]:
            by_backbone[backbone] = (excluded, path.parent)
    records = []
    for backbone, (_, directory) in sorted(by_backbone.items()):
        task = json.loads((directory/'task.json').read_text())
        with np.load(directory/'calibration.npz') as data:
            rows, y, h = (data[k] for k in ['rows', 'targets', 'horizon'])
        with (directory/'calibrators.pkl').open('rb') as handle:
            stored = pickle.load(handle)['DC16']
        with np.load(directory/'test_reservoir.npz') as data:
            test = data['rows'][:512]; test_h = data['horizon'][:512]
        for bins in (1, 4):
            portable = EmpiricalCalibrator.fit_rows(rows, y, h, task['horizon'], bins)
            reference = stored if bins == 1 else fit_dc(rows, y, h, task['horizon'], n_horizon=bins)[0]
            np.testing.assert_array_equal(portable.mean_edges, reference.mean_edges)
            np.testing.assert_array_equal(portable.scale_edges, reference.scale_edges)
            np.testing.assert_array_equal(portable.global_pool, reference.global_residuals)
            if portable.local_pools.keys() != reference.bin_residuals.keys():
                raise ValueError('Local residual-state identities differ')
            for key in portable.local_pools:
                np.testing.assert_array_equal(portable.local_pools[key], reference.bin_residuals[key])
            if bins == 1:
                # The original DC16 pools horizon positions, so the row layout
                # can use H=C=1 without changing its original transform.
                old_values = reference.transform(test[:, :, None, None])[:, :, 0, 0]
                new_values = portable.transform_rows(test, test_h)
                np.testing.assert_allclose(new_values, old_values, atol=1e-12, rtol=1e-12)
                checked = len(test)
            else:
                # Replicate each real row along the full horizon, then select
                # its retained true position from the original transform.
                checked = min(16, len(test))
                x = np.broadcast_to(test[:checked, :, None, None],
                                    (checked, 100, task['horizon'], 1)).copy()
                transformed = reference.transform(x)
                old_values = np.stack([transformed[i, :, test_h[i], 0] for i in range(checked)])
                new_values = portable.transform_rows(test[:checked], test_h[:checked])
                np.testing.assert_allclose(new_values, old_values, atol=1e-12, rtol=1e-12)
            output = args.root/'portable_calibrators'/directory.name/f'ValDC{16*bins}.npz'
            output.parent.mkdir(parents=True, exist_ok=True); portable.save(output)
            np.testing.assert_array_equal(portable.transform_rows(test, test_h),
                                          EmpiricalCalibrator.load(output).transform_rows(test, test_h))
            records.append(dict(task=directory.name, backbone=backbone, nominal_states=16*bins,
                                fitting_coordinates=len(y), excluded_sources=portable.excluded_sources,
                                exact_pool_parity=True, exact_native_transform_parity=bool(np.array_equal(new_values, old_values)),
                                native_transform_within_float64_tolerance=True,
                                native_transform_max_absolute_difference=float(np.abs(new_values-old_values).max()),
                                native_transform_coordinates_checked=checked,
                                export_roundtrip_exact=True))
    atomic(args.root/'portable_dc_parity.json', dict(audited_backbones=len(by_backbone), records=records))
    print(json.dumps(dict(audited_backbones=len(by_backbone), checked_calibrators=len(records), records=records), indent=2))


if __name__ == '__main__':
    main()
