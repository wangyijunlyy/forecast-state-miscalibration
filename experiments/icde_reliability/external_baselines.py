"""Paper-specific external calibrators; frozen native forecast input only."""
import copy
import time
import numpy as np
import torch
from torch import nn
from scipy.special import ndtri
from gaussian_models import fit_gaussian, predict as gaussian_predict
from qian_pit import SoftCRPSPIT

VERSION='icde-external-calibration-v2-20261105'
METHODS=['EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016']
LEVELS=(np.arange(100)+.5)/100
FEATURE_LEVELS=np.arange(1,10)/10

def forecast_features(rows,with_moments=False):
    q=np.quantile(np.asarray(rows,dtype=float),FEATURE_LEVELS,axis=1).T
    if with_moments:q=np.column_stack([rows.mean(1),rows.std(1,ddof=1),q])
    return q

def pinball(pred,y,tau):
    residual=y-pred
    return torch.maximum(tau*residual,(tau-1)*residual).mean()

class DenseQuantileNet(nn.Module):
    """Two 20-unit PReLU hidden layers with dense inter-layer skip links."""
    def __init__(self):
        super().__init__();self.first=nn.Linear(10,20);self.a1=nn.PReLU()
        self.second=nn.Linear(30,20);self.a2=nn.PReLU();self.output=nn.Linear(50,1)
    def forward(self,features,tau):
        x=torch.cat([features,tau],dim=1);h1=self.a1(self.first(x))
        h2=self.a2(self.second(torch.cat([x,h1],dim=1)))
        return self.output(torch.cat([x,h1,h2],dim=1))

class DistributionRecalibration:
    @classmethod
    def fit(cls,rows,y,seed,device='cuda',epochs=200):
        obj=cls();torch.manual_seed(seed);np.random.seed(seed)
        features=forecast_features(rows);obj.feature_mean=features.mean(0)
        obj.feature_scale=features.std(0).clip(1e-6)
        obj.target_mean=float(np.mean(y));obj.target_scale=max(float(np.std(y)),1e-6)
        f=torch.as_tensor((features-obj.feature_mean)/obj.feature_scale,dtype=torch.float32,device=device)
        yy=torch.as_tensor((y-obj.target_mean)/obj.target_scale,dtype=torch.float32,device=device)[:,None]
        model=DenseQuantileNet().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        history=[];model.train();n=len(y)
        for epoch in range(epochs):
            order=torch.randperm(n,device=device);total=0.
            for start in range(0,n,1024):
                ix=order[start:start+1024];tau=torch.rand((len(ix),1),device=device)
                loss=pinball(model(f[ix],tau),yy[ix],tau)
                if not torch.isfinite(loss):raise ValueError('Nonfinite distribution-recalibration objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                total+=float(loss.detach())*len(ix)
            if epoch%10==0 or epoch==epochs-1:
                history.append(dict(epoch=epoch+1,pinball=total/n))
                print('DISTRIBUTION_RECALIBRATION_EPOCH',epoch+1,round(total/n,6),flush=True)
        obj.state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        obj.metadata=dict(method='Kuleshov and Deshpande ICML2022 Algorithm 2',
            calibration_coordinates=n,input_quantiles=FEATURE_LEVELS.tolist(),hidden_units=[20,20],
            nonlinearity='PReLU',connections='dense skip connections',objective='uniform-tau pinball loss',
            optimizer='Adam',learning_rate=.001,epochs=epochs,batch_size=1024,
            hyperparameters='prespecified; no test or held-out scores select settings',
            normalization='per-input standardization and affine target standardization fitted on calibration data',
            output='100 midpoint quantiles, monotonically rearranged',history=history,
            architecture_source='paper Section 6.1',optimizer_source='benchmark adaptation; paper does not specify optimizer/epochs')
        obj._model=model.eval();obj._device=device;obj.raw_crossings=0;obj.raw_comparisons=0
        return obj

    def to(self,device):
        if getattr(self,'_model',None) is None:
            self._model=DenseQuantileNet();self._model.load_state_dict(self.state)
        self._model=self._model.to(device).eval();self._device=device;return self

    def predict(self,rows):
        if getattr(self,'_model',None) is None:self.to('cuda' if torch.cuda.is_available() else 'cpu')
        features=(forecast_features(rows)-self.feature_mean)/self.feature_scale;output=[]
        with torch.no_grad():
            for start in range(0,len(rows),512):
                f=torch.as_tensor(features[start:start+512],dtype=torch.float32,device=self._device)
                n=len(f);tau=torch.as_tensor(LEVELS,dtype=torch.float32,device=self._device).repeat(n)[:,None]
                values=self._model(f.repeat_interleave(100,dim=0),tau).reshape(n,100).cpu().numpy().astype(float)
                self.raw_crossings+=int(np.sum(np.diff(values,axis=1)<0));self.raw_comparisons+=n*99
                output.append(np.sort(values,axis=1)*self.target_scale+self.target_mean)
        return np.concatenate(output)

    def __getstate__(self):
        d=dict(self.__dict__);d.pop('_model',None);d.pop('_device',None);return d

class QRF:
    @classmethod
    def fit(cls,rows,y,seed,trees=1000):
        from ragged_qrf import RaggedQuantileForest
        obj=cls();features=forecast_features(rows,with_moments=True)
        obj.model=RaggedQuantileForest(n_estimators=trees,min_samples_leaf=5,
            max_features=max(1,features.shape[1]//3),bootstrap=True,
            n_jobs=2,random_state=seed)
        obj.model.fit(features,y)
        obj.metadata=dict(method='QRF ensemble postprocessing (Taillardat et al. MWR2016)',
            algorithm='Meinshausen2006 conditional distribution; all leaf observations, inverse leaf-size weighting',
            package='scikit-learn RF with ragged full-leaf adapter',reference_package='quantile-forest==1.4.2',calibration_coordinates=len(y),trees=trees,min_samples_leaf=5,
            max_features=max(1,features.shape[1]//3),max_samples_leaf=None,bootstrap=True,
            features=['native_mean','native_sample_sd']+['quantile_{:.1f}'.format(q) for q in FEATURE_LEVELS],
            auxiliary_weather_predictors_used=False,hyperparameters='prespecified; no test selection',
            weighted_leaves=True,aggregate_leaves_first=True,interpolation='linear',
            leaf_storage='ragged; all bootstrap observations retained; prediction parity checked against quantile-forest1.4.2')
        return obj
    def predict(self,rows):
        features=forecast_features(rows,with_moments=True)
        from concurrent.futures import ThreadPoolExecutor
        # Query chunks are independent; the Cython quantile kernel releases the GIL.
        # Use one tree-apply thread within each of four parallel query chunks.
        self.model.n_jobs=1
        def calculate(start):
            return self.model.predict(features[start:start+2048],quantiles=LEVELS.tolist(),
                weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True,interpolation='linear')
        with ThreadPoolExecutor(max_workers=4) as pool:
            return np.concatenate(list(pool.map(calculate,range(0,len(features),2048))))

class EMOS:
    @classmethod
    def fit(cls,rows,y,existing=None):
        obj=cls()
        if existing is None:
            obj.parameters=fit_gaussian(rows.mean(1),rows.std(1,ddof=1).clip(1e-6),y,affine=True)
        else:obj.parameters=copy.deepcopy(existing)
        if not obj.parameters['optimizer_success']:raise ValueError('EMOS CRPS optimizer failure')
        obj.metadata=dict(method='exchangeable Gaussian EMOS/NGR',
            mean='a+b*native_mean',variance='c+d*native_sample_variance',
            objective='analytic Gaussian CRPS',constraints='c,d >= 0; a,b unrestricted',
            calibration_coordinates=len(y),parameters=obj.parameters,reused_existing_fit=existing is not None)
        return obj
    def predict(self,rows):return gaussian_predict(self.parameters,rows)


def verify_checkpoint_reference(reference):
    from pathlib import Path
    path=Path(reference)
    if not path.exists():raise FileNotFoundError(path)
    if path.is_dir() and not any(p.is_file() for p in path.rglob('*')):
        raise FileNotFoundError('Empty native checkpoint directory: '+str(path))
    return True
