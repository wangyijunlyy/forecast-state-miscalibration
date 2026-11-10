"""FSPC on the existing partition/budget axes, without changing its mean rule."""
import numpy as np
from granularity import fit_partition
from mean_anchor import fit_bias, select_strength

PARTITIONS=[(b,1) for b in [1,2,3,4,6,8,12,16]]+[(6,g) for g in [2,4,8]]
VARIANTS=[(b,g,50000) for b,g in PARTITIONS]+[(6,1,n) for n in [2000,10000]]
VERSION='fspc-granularity-v1-20261103'


def fit_variant(train,train_targets,train_h,validation,validation_targets,validation_h,
                horizon_length,bins,horizon_groups,budget,ordering_seed):
    if budget<1:raise ValueError('A positive validation-coordinate budget is required')
    take=np.random.default_rng(ordering_seed).permutation(len(validation_targets))[:budget]
    empirical,meta=fit_partition(validation[take],validation_targets[take],validation_h[take],
        horizon_length,bins,horizon_groups)
    cal=empirical.compile()
    bias,counts=fit_bias(cal,train,train_targets,train_h)
    strength,selection=select_strength(cal,validation[take],validation_targets[take],validation_h[take],bias)
    meta.update(training_coordinates=len(train_targets),training_state_counts=counts.tolist(),
        state_bias=bias.tolist(),strength=strength,selection=selection,requested_budget=budget,
        template_centered=True,training_budget='fixed existing training reservoir',
        output_members=100,pseudo_count=128,min_bias_count=128,
        bias_fit_split='train',selection_split='validation',test_targets_used=False)
    return cal,bias,strength,meta
