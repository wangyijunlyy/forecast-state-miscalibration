"""Original backbone training, isolated outputs, validation-only selection."""
import argparse,importlib,importlib.util,json,os,re,sys,time
from dataclasses import fields
from contextlib import nullcontext
from validation_runtime import input_gradient_only,validation_batch_size,DTS_VALIDATION_VERSION
from tmdm_numerics import enable_stable_tmdm_attention,TMDM_NUMERICS_VERSION
from pathlib import Path
import numpy as np
import torch
from core import CommonMetrics,atomic


def k2(t,out):
    os.chdir(t['train_repo']);sys.path.insert(0,t['train_repo'])
    spec=importlib.util.spec_from_file_location('native_k2_cli',str(Path(t['train_repo'])/'run.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    sys.argv=['run.py','--config',t['training_config'],'--seed_everything',str(t['seed']),
      '--data.data_manager.init_args.path',t['data_root'],'--data.data_manager.init_args.split_val','true',
      '--data.data_manager.init_args.dataset',t['dataset'],'--data.data_manager.init_args.context_length',str(t['context']),
      '--data.data_manager.init_args.prediction_length',str(t['horizon']),'--trainer.default_root_dir',str(out/'training'),
      '--trainer.max_epochs',str(t['max_epochs']),'--trainer.num_sanity_val_steps','0',
      '--data.batch_size',str(t['training_batch_size']),'--data.test_batch_size','4','--data.num_workers','2']
    cli=module.ProbTSCli(datamodule_class=module.ProbTSDataModule,model_class=module.ProbTSForecastModule,
                        save_config_kwargs={'overwrite':True},run=False)
    cli.init_exp();cli.set_fit_mode();cli.trainer.fit(model=cli.model,datamodule=cli.datamodule)
    ck=cli.checkpoint_callback.best_model_path
    if not ck:raise RuntimeError('No validation-selected K2VAE checkpoint')
    return ck,dict(epochs_completed=cli.trainer.current_epoch,selection='minimum validation CRPS',training_batch_size=t['training_batch_size'])


def ppm(t,out):
    os.chdir(t['repo']);sys.path.insert(0,t['repo'])
    spec=importlib.util.spec_from_file_location('original_ppm_eval',str(Path(t['repo'])/'scripts/evaluate_ppm_posthoc_baselines.py'))
    util=importlib.util.module_from_spec(spec);sys.modules[spec.name]=util;spec.loader.exec_module(util)
    from types import SimpleNamespace
    a=SimpleNamespace(**t['model_args']);a.num_samples=100;util.add_model_defaults(a)
    a.model='MLP';a.seed=t['seed'];a.K=100;a.tau=.1;a.var=.3;a.nu=5.;a.label_len=0
    a.use_gpu=True;a.gpu=0;a.use_multi_gpu=False;a.num_workers=0;a.batch_size=t['training_batch_size']
    a.test_batch_size=t['training_test_batch_size'];a.train_epochs=t['max_epochs'];a.patience=5;a.learning_rate=t['learning_rate']
    a.checkpoints=str(out/'training');a.pretrain_checkpoints=str(out/'unused_pretrain');a.setting=t['id'];a.lradj='type1';a.use_amp=False
    util.set_seed(t['seed']);exp=util.Exp_Main(a)
    # Original train() calls vali(test) only for logging. Skip those calls so
    # test targets cannot enter training, and RNG is not consumed by test draws.
    original_vali=exp.vali
    def vali(data,loader,criterion):
        if getattr(data,'set_type',None)==2:return float('nan')
        return original_vali(data,loader,criterion)
    exp.vali=vali;exp.train(t['id'])
    ck=Path(a.checkpoints)/t['id']/'checkpoint.pth'
    return str(ck),dict(selection='minimum original validation loss',args=vars(a),training_batch_size=a.batch_size,
                        seed_policy='fresh independent training seed; original script seed2021 replaced with requested1,2,3')


def diffusion(t,out):
    os.chdir(t['repo']);sys.path.insert(0,t['repo'])
    names={'NsDiff':('NsDiff','NsDiffForecast'),'TMDM':('TMDM','TMDMForecast'),
           'DiffusionTS':('DiffusionTS','DiffusionTSForecast'),'TimeDiff':('TimeDiff','TimeDiffForecast')}
    module,name=names[t['backbone']];cls=getattr(importlib.import_module('src.experiments.'+module),name)
    allowed={f.name for f in fields(cls)};a={k:v for k,v in t['training_args'].items() if k in allowed}
    a.update(device='cuda:0',num_worker=0,save_dir=str(out/'training'),data_path=t.get('resolved_data_root',str(Path(t['repo'])/'data')))
    model=cls(**a);model._init_metrics=lambda:None;model._setup_run(t['seed']);model._check_run_exist(t['seed'])
    if t['backbone']=='TMDM':enable_stable_tmdm_attention(model.cond_pred_model)
    if t['backbone']=='DiffusionTS' or t['dataset'] in ('Electricity','SolarEnergy','Traffic'):
        from torch.utils.data import DataLoader
        model.val_loader=DataLoader(model.val_loader.dataset,batch_size=validation_batch_size(t['backbone'],t['dataset'],t['training_batch_size']),shuffle=False,drop_last=False,num_workers=0)
        if t['backbone']=='NsDiff':model.diffusion_config.testing.minisample=10
    from torch_timeseries.utils.reproduce import reproducible
    def evaluate(loader):
        acc=CommonMetrics();model.model.eval()
        batch_trajectories=t['backbone']=='TimeDiff' and t['dataset'] in ('Electricity','SolarEnergy','Traffic')
        old_tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        if batch_trajectories:
            sys.path.insert(0,'/home/wyj/distribution_binned_diffusion/code')
            from timediff_batching import sample_timediff_batched
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        guidance_context=input_gradient_only(model.model) if t['backbone']=='DiffusionTS' else nullcontext()
        with guidance_context,torch.no_grad():
            for x,y,ox,oy,xm,ym in loader:
                x=x.to(model.device).float();y=y.to(model.device).float();xm=xm.to(model.device).float();ym=ym.to(model.device).float()
                if batch_trajectories:
                    pred=sample_timediff_batched(model.model,x,xm,y,ym,100,20).permute(0,2,3,1);true=y
                else:pred,true=model._process_val_batch(x,y,xm,ym)
                if model.invtrans_loss:pred=model.scaler.inverse_transform(pred);true=oy
                acc.update(pred.detach().cpu().numpy().transpose(0,3,1,2),true.detach().cpu().numpy())
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old_tf32
        return acc.result()
    history=[]
    while model.current_epoch<model.epochs and not model.early_stopper.early_stop:
        reproducible(t['seed']+model.current_epoch);begin=time.monotonic();loss=model._train()
        val=evaluate(model.val_loader);model.current_epoch+=1
        weights=model.model
        if t['backbone'] in ('NsDiff','TMDM'):weights={'model':model.model,'cond_pred_model':model.cond_pred_model}
        if t['backbone']=='NsDiff':weights['cond_pred_model_g']=model.cond_pred_model_g
        model.early_stopper(val['CRPS'],model=weights);model._save_run_check_point(t['seed'])
        row=dict(epoch=model.current_epoch,train_loss=float(np.mean(loss)),validation=val,seconds=time.monotonic()-begin)
        history.append(row);atomic(out/'training_history.json',history);print('EPOCH',t['id'],row,flush=True)
    ck=Path(model.best_checkpoint_filepath)
    if not ck.exists():raise RuntimeError('No validation checkpoint')
    return str(ck),dict(checkpoint_dir=str(ck.parent),epochs_completed=model.current_epoch,selection='minimum synchronous validation empirical CRPS',
        validation_metric_implementation='synchronous exact empirical CRPS',
        validation_protocol=DTS_VALIDATION_VERSION if t['backbone']=='DiffusionTS' else 'native_validation',
        validation_window_batch_size=model.val_loader.batch_size,validation_forecast_samples=100,
        numerical_protocol=TMDM_NUMERICS_VERSION if t['backbone']=='TMDM' else 'native',
        attention_overflow_fallbacks=sum(getattr(m,'_overflow_fallbacks',0) for m in model.cond_pred_model.modules()) if t['backbone']=='TMDM' else 0,
        training_batch_size=t['training_batch_size'],validation_sampling=('original_forward_independent_trajectory_chunks20_tf32_disabled' if t['backbone']=='TimeDiff' and t['dataset'] in ('Electricity','SolarEnergy','Traffic') else 'original_native_forward'),shared_original_pretrain=t.get('shared_original_pretrain',[]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args();t=json.loads(a.task.read_text());out=Path(t['run_root']);out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists():
        old=json.loads((out/'training.json').read_text())
        if old.get('training_complete') and Path(old['checkpoint']).is_file():return
    torch.set_num_threads(2);begin=time.monotonic()
    ck,meta=({'K2VAE':k2,'PPM':ppm}.get(t['backbone'],diffusion))(t,out)
    if not Path(ck).is_file():raise FileNotFoundError(ck)
    atomic(out/'training.json',dict(task_id=t['id'],seed=t['seed'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],
        checkpoint=ck,training_complete=True,seconds=time.monotonic()-begin,metadata=meta))
    print('TRAIN_COMPLETE',t['id'],ck,flush=True)
if __name__=='__main__':main()
