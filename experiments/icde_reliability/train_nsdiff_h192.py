"""Native NsDiff training with an equivalent attention overflow precision fallback."""
import argparse,json,sys
from pathlib import Path
import copy
import math
import numpy as np


def validation_metrics_class(base):
    class ValidationCandidateMetrics(base):
        def __init__(self):
            super().__init__();self.invalid_coordinates=0;self.seen_coordinates=0
        def update(self,x,y):
            assert np.isfinite(y).all(), 'Nonfinite validation targets'
            self.seen_coordinates+=y.size
            invalid=~np.isfinite(x).all(axis=1)
            if invalid.any():
                self.invalid_coordinates+=int(invalid.sum());return
            super().update(x,y)
        def result(self):
            if not self.invalid_coordinates:return super().result()
            return dict(CRPS=float('inf'),validation_candidate_finite=False,
                invalid_prediction_coordinates=self.invalid_coordinates,
                validation_coordinates_seen=self.seen_coordinates)
    return ValidationCandidateMetrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    task=json.loads(a.task.read_text());assert task['backbone']=='NsDiff' and task['horizon']==192 and task.get('replaces_task')
    native_code=Path('/home/wyj/three_seed_reproduction_20261031/code');sys.path.insert(0,str(native_code))
    import train_task as native
    native.CommonMetrics=validation_metrics_class(native.CommonMetrics)
    from tmdm_numerics import enable_stable_tmdm_attention,TMDM_NUMERICS_VERSION
    import os
    os.chdir(task['repo']);sys.path.insert(0,task['repo'])
    import src.experiments.NsDiff as module
    original=module.NsDiffForecast;instances=[];out=Path(task['run_root'])
    prefix=json.loads((out/'training_history.json').read_text()) if a.resume else []
    class PrecisionNsDiffForecast(original):
        def _init_model(self):
            super()._init_model();enable_stable_tmdm_attention(self.cond_pred_model);instances.append(self)
        def _check_run_exist(self,seed):
            exists=super()._check_run_exist(seed)
            if a.resume:
                if not exists:raise ValueError('A complete native epoch checkpoint is required for resume')
                self._resume_run(seed)
                if self.current_epoch!=len(prefix):raise ValueError('Saved history and native epoch boundary differ')
            return exists
    module.NsDiffForecast=PrecisionNsDiffForecast
    original_atomic=native.atomic
    def atomic(path,value):
        if Path(path)==out/'training_history.json':
            value=copy.deepcopy(prefix+value if a.resume else value)
            for row in value:
                if row['validation'].get('CRPS') is not None and not math.isfinite(row['validation']['CRPS']):
                    row['validation'].update(CRPS=None,selection_score='positive_infinity')
        original_atomic(path,value)
    native.atomic=atomic
    sys.argv=['train_task.py','--task',str(a.task)];native.main()
    if instances:
        record=json.loads((out/'training.json').read_text());model=instances[-1]
        record['metadata'].update(numerical_protocol=TMDM_NUMERICS_VERSION,precision_scope='conditional de-stationary attention only when native float32 exponent/scores overflow',attention_overflow_fallbacks=sum(getattr(m,'_overflow_fallbacks',0) for m in model.cond_pred_model.modules()),architecture_loss_optimizer_and_hyperparameters_unchanged=True)
        history=json.loads((out/'training_history.json').read_text())
        record['metadata'].update(invalid_validation_candidates=[r['epoch'] for r in history if r['validation'].get('CRPS') is None],
            validation_selection='minimum finite native empirical CRPS; invalid candidates count as no improvement')
        if a.resume:
            record.update(resumed_from_epoch=len(prefix),resume_protocol='full native network/optimizer/early-stopping state at a finite epoch boundary; original per-epoch random seed policy')
            record['seconds']+=sum(row['seconds'] for row in prefix)
        original_atomic(out/'training.json',record)
    print('NSDIFF_PRECISION_TRAINING_COMPLETE',task['id'],flush=True)


if __name__=='__main__':main()
