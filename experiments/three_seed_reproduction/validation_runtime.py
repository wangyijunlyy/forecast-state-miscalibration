"""Memory-bounded validation with gradients restricted to guidance inputs."""
from contextlib import contextmanager
DTS_VALIDATION_VERSION='single_window_input_grad_only_v2'
LARGE_DATASETS={'Electricity','SolarEnergy','Traffic'}


def validation_batch_size(backbone,dataset,training_batch):
    # DiffusionTS's original repeat/reshape code only retains window identity
    # for one conditioning window. Keep all100 draws for that window.
    return 1 if backbone=='DiffusionTS' or dataset in LARGE_DATASETS else training_batch


@contextmanager
def input_gradient_only(model):
    # Langevin guidance needs d(loss)/d(input), not network parameter gradients.
    # Original _train() can otherwise consume residual guidance gradients.
    parameters=tuple(model.parameters());flags=[p.requires_grad for p in parameters]
    try:
        for p in parameters:p.grad=None;p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(parameters,flags):p.grad=None;p.requires_grad_(flag)
