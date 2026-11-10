"""Fixed scope, protocol, and immutable legacy-record reuse for mean anchoring."""
from collections import Counter, defaultdict

LEGACY_VERSION = 'fsdc36-mean-anchor-diffusionts-v1-20261103'
VERSION = 'fsdc36-mean-anchor-v1-20261103'
METHODS = ['Native', 'DC36', 'DC36MeanPreserving', 'DC36ControlledMean']
FULL_COUNTS = dict(K2VAE=108, PPM=21, TMDM=27, DiffusionTS=27,
                   TimeDiff=27, NsDiff=27, DeepSTUQ=12)


def version_for(task):
    return LEGACY_VERSION if task['backbone'] == 'DiffusionTS' else VERSION


def validate_tasks(tasks, reference=None):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(len({t['id'] for t in tasks}) == len(tasks), 'Duplicate task IDs')
    counts = dict(Counter(t['backbone'] for t in tasks))
    require(counts in [FULL_COUNTS, {'DiffusionTS': 27}], 'Incomplete campaign registry')
    configurations = defaultdict(list)
    for task in tasks:
        configurations[task['backbone'], task['dataset'], task['horizon']].append(task['seed'])
    require(all(sorted(seeds) == [1, 2, 3] for seeds in configurations.values()),
            'Each configuration requires exactly seeds 1, 2, 3')
    if reference is not None:
        expected = {t['id']:t for t in reference
                    if len(tasks)==249 or t['backbone']=='DiffusionTS'}
        require({t['id']:t for t in tasks} == expected, 'Canonical task identities changed')
    return dict(tasks=len(tasks), configurations=len(configurations), backbones=counts)


def source_proof_matches(proof, tasks, fits):
    return (proof.get('all_passed') is True and proof.get('tasks') == len(tasks)
            and proof.get('manifest') == tasks and proof.get('fits') == fits)


def validate_record(record, task):
    fit = record['fit']
    conditions = [
        record['task'] == task, fit['task'] == task,
        record['version'] == fit['version'] == version_for(task),
        fit['pseudo_count'] == fit['min_bias_count'] == 128,
        fit['summary_bins'] == 6, fit['horizon_groups'] == 1, fit['output_members'] == 100,
        set(record['methods']) == set(METHODS), record['full_test_split'],
        record['baseline_parity'], not record['neural_training_performed'],
        not fit['neural_training_performed'], fit['alpha_frozen_before_test'],
        fit['selection']['bias_fit_split'] == 'train',
        fit['selection']['selection_split'] == 'validation',
        not fit['selection']['test_targets_used'], 0 <= fit['selection']['strength'] <= 1,
        record['test_seed'] == task['seed'] + 30000,
    ]
    if not all(conditions):
        raise ValueError('Result identity or fixed mean-anchor protocol mismatch: ' + task['id'])
