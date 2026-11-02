"""Full native forecast interfaces; no historical stochastic prediction caches."""
import importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from validation_runtime import DTS_VALIDATION_VERSION
K2='/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def chronological(loader,batch_size):
    return DataLoader(loader.dataset,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0)


def prepare(task,out):
    training=json.loads((Path(task['run_root'])/'training.json').read_text());t=dict(task,checkpoint=training['checkpoint'])
    assert training['training_complete'] and training['seed']==t['seed']
    if t['backbone']=='DiffusionTS':assert training.get('metadata',{}).get('validation_protocol')==DTS_VALIDATION_VERSION
    sys.path.insert(0,K2)
    if t['backbone']=='K2VAE':
        util=load('native_k2_evaluation',Path(K2)/'scripts/evaluate_k2vae_posthoc_baselines.py');util.set_seed(t['seed'])
        manager=util.DataManager(dataset=t['dataset'],path=t['data_root'],context_length=t['context'],prediction_length=t['horizon'],split_val=True,scaler='standard')
        data=util.ProbTSDataModule(data_manager=manager,batch_size=4,test_batch_size=4,num_workers=2)
        m=util.load_module(Path(t['checkpoint']),data,torch.device('cuda:0'))
        model=m.forecaster.backbone if hasattr(m.forecaster,'backbone') else m.forecaster
        def predict(batch,phase,step):
            bd=util.normalized_batch(m,batch,torch.device('cuda:0'))
            with torch.no_grad():x=model.forecast(bd,100)[:,:,:bd.future_target_cdf.shape[1]]
            return x.cpu().numpy(),bd.future_target_cdf.cpu().numpy(),util.tensor_to_denorm_numpy(m,bd.future_target_cdf)
        def score_native(acc,x,y,origin):
            values=util._denormalize(m,x,torch.device('cuda:0'),torch.float32)
            acc.update(util.evaluate_samples_like_k2vae(origin,values,quantiles_num=20),len(y))
        return dict(train=data.train_dataloader(),val=data.val_dataloader(),test=data.test_dataloader(),predict=predict,
                    native_factory=util.MetricAverager,native_update=score_native,
                    denormalize=lambda x:util._denormalize(m,x,torch.device('cuda:0'),torch.float32),fit_limit=100,checkpoint=t['checkpoint'])
    if t['backbone']=='PPM':
        sys.path.insert(0,t['repo']);os.chdir(t['repo'])
        util=load('native_ppm_evaluation',Path(t['repo'])/'scripts/evaluate_ppm_posthoc_baselines.py');util.set_seed(t['seed'])
        args=SimpleNamespace(**t['model_args']);args.num_samples=100;util.add_model_defaults(args)
        args.batch_size=t['fit_batch_size'];args.test_batch_size=1;args.num_workers=0;args.use_gpu=True;args.gpu=0;args.use_multi_gpu=False
        exp=util.Exp_Main(args);exp.model.load_state_dict(torch.load(t['checkpoint'],map_location=exp.device));exp.model.eval()
        _,train=util.data_provider(args,'train');_,val=util.data_provider(args,'val');_,test=util.data_provider(args,'test')
        val=chronological(val,1 if t['dataset'] in ('ECL','traffic') else 16)
        def predict(batch,phase,step):
            x,y,*_=batch
            with torch.no_grad():samples=util.batch_to_samples(exp.model,x.float().to(exp.device),100,args).cpu().numpy()
            truth=util.batch_target(y.float().to(exp.device),args).cpu().numpy();return samples,truth,truth
        return dict(train=train,val=val,test=test,predict=predict,native_factory=util.PPMMetricAccumulator,
                    native_update=lambda a,x,y,o:a.update(x,y),denormalize=lambda x:x,fit_limit=100,checkpoint=t['checkpoint'])
    sys.path.insert(0,'/home/wyj/distribution_binned_diffusion/code')
    util=load('original_diffusion_binned',Path('/home/wyj/distribution_binned_diffusion/code/run_task.py'))
    t['checkpoint']=str(Path(t['checkpoint']).parent);t['model_args']=dict(t['training_args'],batch_size=1)
    t.update(seed=t['seed'],test_cache_root='',fit_windows=100,num_samples=100)
    # All TimeDiff tasks use original-forward batched independent sample draws;
    # numerical equivalence was verified in prior native interface experiments.
    if t['backbone']=='TimeDiff':t['native_sampling_execution']='batched_original_forward_tf32_disabled'
    model,train,test,pred,_=util.prepare(t,out)
    if t['backbone']=='TMDM':
        from tmdm_numerics import enable_stable_tmdm_attention
        enable_stable_tmdm_attention(model.cond_pred_model)
    if t['backbone']=='NsDiff' and task.get('replaces_task'):
        from tmdm_numerics import enable_stable_tmdm_attention
        enable_stable_tmdm_attention(model.cond_pred_model)
    val=chronological(model.val_loader,1)
    def predict(batch,phase,step):
        x,y,origin,hit=pred(batch,phase,step);assert not hit
        return x,y,origin.cpu().numpy() if hasattr(origin,'cpu') else np.asarray(origin)
    from core import CommonMetrics
    def native_update(acc,x,y,origin):
        if model.invtrans_loss:
            x=model.scaler.inverse_transform(torch.as_tensor(x)).cpu().numpy();y=origin
        acc.update(x,y)
    return dict(train=train,val=val,test=test,predict=predict,native_factory=CommonMetrics,native_update=native_update,
                denormalize=lambda x:model.scaler.inverse_transform(torch.as_tensor(x)).cpu().numpy() if model.invtrans_loss else x,
                fit_limit=100,checkpoint=str(Path(t['checkpoint'])))
