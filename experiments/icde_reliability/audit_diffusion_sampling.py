"""Inspect the real native interface without mutating its source repository."""
from functools import wraps
import inspect
import numpy as np


def find_native_model(interface):
    seen=set()
    def visit(value,depth=0):
        if id(value) in seen or depth>8:return None
        seen.add(id(value))
        if hasattr(value,'val_loader') and hasattr(value,'model'):return value
        if inspect.isfunction(value):
            for cell in value.__closure__ or []:
                result=visit(cell.cell_contents,depth+1)
                if result is not None:return result
        return None
    result=visit(interface['predict'])
    if result is None:raise ValueError('Native forecasting model is not reachable from prediction closure')
    return result


def dataset_stride(dataset):
    if type(dataset).__name__=='MultivariateFast':return int(dataset.total_len)
    if type(dataset).__name__=='MultiStepTimeFeatureSet':return 1
    raise ValueError('Unsupported chronological origin mapping: '+type(dataset).__name__)


def disable_clamp(model):
    native=model.model;original=native.fast_sample_infill
    @wraps(original)
    def sample(*args,**kwargs):
        kwargs['clip_denoised']=False
        return original(*args,**kwargs)
    native.fast_sample_infill=sample


def audit(model,rows,targets):
    dataset=model.val_loader.dataset;source=inspect.getsource(type(model.model).fast_sample_infill)
    prediction=inspect.getsource(type(model.model).model_predictions)
    sampler=inspect.getsource(type(model)._sample_diffusionts)
    return dict(scaler=type(model.scaler).__name__,invtrans_loss=bool(model.invtrans_loss),
        validation_dataset=type(dataset).__name__,validation_origin_stride=dataset_stride(dataset),
        validation_windows=len(dataset),training_scaler=model.scaler_type,
        sampler_default_clipping=inspect.signature(type(model.model).fast_sample_infill).parameters['clip_denoised'].default,
        sampler_passes_clipping_explicitly='clip_denoised=' in sampler,
        model_prediction_clamp_present='min=-1.' in prediction and 'max=1.' in prediction,
        calibration_native_range=[float(rows.min()),float(rows.max())],
        calibration_target_range=[float(targets.min()),float(targets.max())],
        calibration_target_outside_unit_interval_fraction=float(np.mean(abs(targets)>1)),
        calibration_collapsed_fraction=float(np.mean(rows.std(1,ddof=1)<=1.01e-6)),
        source_excerpt=dict(sampler=source,model_predictions=prediction),
        conclusion='Sampling clamp is inconsistent with unbounded standardized target support; evaluate no-clamp sampling separately')
