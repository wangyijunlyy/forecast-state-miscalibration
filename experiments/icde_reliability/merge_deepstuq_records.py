"""Assemble a paired seven-backbone analysis view without duplicating raw caches."""
import json
from pathlib import Path


def link(source, destination):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError('Analysis view already points to a different record: '+str(destination))
    elif destination.exists():
        raise ValueError('Refusing to replace an existing analysis record: '+str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_records(records, manifest, training, output):
    extension = records/'deepstuq_backbone'
    tasks = json.loads(manifest.read_text())
    extra = json.loads((extension/'manifest.json').read_text())
    if len(tasks) != 237 or len(extra) != 12:
        raise ValueError('Expected the original237 tasks and all12 DeepSTUQ tasks')
    correction = records/'nsdiff_h192'
    corrected = {}
    if (correction/'manifest.json').exists():
        corrected = {t['id']: t for t in json.loads((correction/'manifest.json').read_text())}
        selected = {t['id']: t for t in tasks}
        if len(corrected) != 9 or not set(corrected) <= set(selected):
            raise ValueError('The corrected NsDiff manifest must replace exactly nine selected tasks')
        for identity, task in corrected.items():
            if selected[identity] != task or task['horizon'] != 192:
                raise ValueError('Corrected NsDiff task identity/configuration mismatch')
    output.mkdir(parents=True, exist_ok=True)
    groups = [(records, [t for t in tasks if t['id'] not in corrected], False),
              (correction, [t for t in tasks if t['id'] in corrected], False),
              (extension, extra, True)]
    for source_root, task_list, is_extra in groups:
        gaussian = source_root/('gaussian/runs' if (source_root/'gaussian/runs').exists() else 'gaussian_runs')
        compiled = source_root/('compiled_deployment/runs' if (source_root/'compiled_deployment/runs').exists() else 'compiled/runs')
        for task in task_list:
            identity = task['id']
            for source, directory in [(source_root/'runs', 'runs'), (gaussian, 'gaussian/runs'),
                                      (compiled, 'compiled/runs'), (source_root/'state_controls/runs', 'state_controls/runs')]:
                if not (source/identity/'results.json').exists():
                    raise ValueError('Incomplete seven-backbone input: '+str(source/identity))
                link(source/identity, output/directory/identity)
            destination = output/'training_runs'/identity
            if is_extra:
                comparison = json.loads((source_root/'runs'/identity/'training_partition_comparison.json').read_text())
                methods = comparison['methods']
                normalized = {new:methods[old] for old,new in [('Native','Native'),('TrainDC16','DC16'),('TrainDC64','DC64')]}
                destination.mkdir(parents=True, exist_ok=True)
                (destination/'results.json').write_text(json.dumps(dict(task=task,full_test_split=comparison['full_test_split'],
                    test_windows=comparison['test_windows'],test_batches=comparison['test_batches'],
                    common_normalized=normalized),indent=2)+'\n')
            else:
                source_training = correction/'train_calibration/runs' if identity in corrected else training
                link(source_training/identity, destination)
    link(records/'deepstuq', output/'deepstuq')
    combined = tasks+extra
    if len({t['id'] for t in combined}) != 249:
        raise ValueError('Duplicate backbone task identity')
    (output/'manifest.json').write_text(json.dumps(combined,indent=2)+'\n')
    if (records/'granularity/manifest.json').exists():
        granularity = output/'granularity'
        for task in combined:
            source = correction/'granularity' if task['id'] in corrected else records/'granularity'
            if not (source/'runs'/task['id']/'results.json').exists():
                raise ValueError('Incomplete current granularity input: '+task['id'])
            link(source/'runs'/task['id'], granularity/'runs'/task['id'])
        (granularity/'manifest.json').write_text(json.dumps(combined,indent=2)+'\n')
    return output, output/'manifest.json', output/'training_runs'
