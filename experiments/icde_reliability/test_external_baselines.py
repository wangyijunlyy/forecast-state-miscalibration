"""Algorithm-level checks independent of any benchmark scores."""
import numpy as np
from qian_pit import SoftCRPSPIT

def brute(m,mu,sigma,z,k,levels):
    h=np.partition(np.abs(mu-m),max(1,len(mu)//k)-1)[max(1,len(mu)//k)-1]
    weights=sigma*(mu==m) if h==0 else sigma*np.maximum(0,1-((mu-m)/h)**2)
    order=np.argsort(z);c=np.cumsum(weights[order]);c/=c[-1]
    return z[order[np.searchsorted(c,levels,side='left')]]

def test_soft_crps_matches_direct_kernel_weighted_solution():
    rng=np.random.default_rng(12);n=157
    mu=rng.normal(size=n);sigma=rng.uniform(.1,2,n);z=rng.normal(size=n)
    model=SoftCRPSPIT.from_summaries(mu,sigma,mu+sigma*z)
    levels=np.array([.005,.1,.5,.9,.995]);query=np.array([-4.,-.15,.34,4.])
    for k in [1,2,7,20]:
        actual=model.standardized_quantiles(query,k,levels)
        expected=np.array([brute(m,mu,sigma,z,k,levels) for m in query])
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)

def test_soft_crps_cdf_and_zero_bandwidth_ties():
    mu=np.zeros(40);sigma=np.linspace(.2,2,40);z=np.linspace(-3,3,40)
    model=SoftCRPSPIT.from_summaries(mu,sigma,sigma*z)
    actual=model.cdf(np.zeros(40),z,20)
    expected=np.cumsum(sigma)/sigma.sum()
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    q=model.standardized_quantiles(np.array([0.]),20,np.array([.1,.5,.9]))[0]
    np.testing.assert_allclose(q,brute(0.,mu,sigma,z,20,np.array([.1,.5,.9])))

def test_soft_crps_retains_extreme_residuals_without_cdf_clipping():
    mu=np.linspace(-1,1,100);sigma=np.ones(100);z=np.linspace(-2,2,100);z[-1]=1e8
    model=SoftCRPSPIT.from_summaries(mu,sigma,mu+sigma*z)
    q=model.standardized_quantiles(np.array([1.]),20,np.array([.5,.999]))
    assert q[0,-1]==1e8

def test_quantile_features_and_pinball_definition():
    import torch
    from external_baselines import forecast_features,pinball,DenseQuantileNet
    rows=np.tile(np.arange(100,dtype=float),(3,1))
    q=forecast_features(rows)
    np.testing.assert_allclose(q[0],np.arange(1,10)*9.9)
    assert forecast_features(rows,True).shape==(3,11)
    assert float(pinball(torch.tensor([0.,2.]),torch.tensor([1.,1.]),torch.tensor([.25,.25])))==.5
    net=DenseQuantileNet();x=torch.ones(3,9);tau=torch.full((3,1),.5)
    assert net(x,tau).shape==(3,1)
    net(x,tau).sum().backward()
    assert all(p.grad is not None for p in net.parameters())

def test_qrf_models_leaf_distribution_instead_of_tree_means():
    import pytest
    pytest.importorskip('quantile_forest')
    from external_baselines import QRF
    rows=np.tile(np.linspace(-1,1,100),(80,1));y=np.r_[np.full(40,-10.),np.full(40,10.)]
    model=QRF.fit(rows,y,seed=27,trees=8)
    output=model.predict(rows[:2])
    assert output.shape==(2,100)
    assert np.all(output[:,:10]==-10.) and np.all(output[:,-10:]==10.)
    assert np.all(np.diff(output,axis=1)>=0)
    assert model.model.max_samples_leaf is None

def test_emos_uses_native_spread_and_affine_mean():
    from external_baselines import EMOS
    rows=np.array([np.linspace(0,2,100),np.linspace(1,4,100)])
    params=dict(kind='gaussian_affine',mean_intercept=.5,mean_multiplier=.8,
        variance_floor=.2,variance_multiplier=1.4,optimizer_success=True)
    model=EMOS.fit(rows,np.array([1.,2.]),existing=params);pred=model.predict(rows)
    np.testing.assert_allclose(pred.mean(1),.5+.8*rows.mean(1),atol=1e-12)
    from scipy.special import ndtri
    expected=np.sqrt(.2+1.4*rows.var(1,ddof=1))*(ndtri(.995)-ndtri(.005))
    np.testing.assert_allclose(pred[:,-1]-pred[:,0],expected)

def test_qian_selects_k_with_original_squared_pce():
    from external_baselines import SoftCRPSPIT
    rng=np.random.default_rng(31);mu=rng.normal(size=80);sigma=rng.uniform(.2,2,80)
    rows=mu[:,None]+sigma[:,None]*np.tile(np.linspace(-1,1,100),(80,1))
    y=mu+sigma*rng.normal(size=80);seed=773
    model=SoftCRPSPIT.fit(rows,y,seed)
    means=rows.mean(1);scales=rows.std(1,ddof=1);z=(y-means)/scales
    probabilities=np.sort(np.random.default_rng(seed).uniform(0,1,3000));expected=[]
    for k in range(1,21):
        pit=[]
        for m,target in zip(means,z):
            h=np.partition(abs(means-m),max(1,len(means)//k)-1)[max(1,len(means)//k)-1]
            w=scales*(means==m) if h==0 else scales*np.maximum(0,1-((means-m)/h)**2)
            pit.append(w[z<=target].sum()/w.sum())
        error=np.searchsorted(np.sort(pit),probabilities,side='right')/len(pit)-probabilities
        expected.append(float(np.mean(error**2)))
    np.testing.assert_allclose([r['calibration_PCE'] for r in model.history],expected,atol=1e-12)
    assert model.k==np.argmin(expected)+1

def test_ragged_qrf_matches_padded_library_with_bootstrap_leaf_weighting():
    import pytest
    pytest.importorskip('quantile_forest')
    from quantile_forest import RandomForestQuantileRegressor
    from ragged_qrf import RaggedQuantileForest
    rng=np.random.default_rng(17);x=rng.normal(size=(180,4));x[:90]=0
    y=rng.normal(size=180);q=(np.arange(100)+.5)/100;query=np.r_[x[:12],rng.normal(size=(20,4))]
    common=dict(n_estimators=13,min_samples_leaf=5,max_features=1,bootstrap=True,n_jobs=1,random_state=29)
    reference=RandomForestQuantileRegressor(max_samples_leaf=None,**common).fit(x,y)
    compact=RaggedQuantileForest(**common).fit(x,y)
    expected=reference.predict(query,quantiles=q.tolist(),weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True)
    actual=compact.predict(query,quantiles=q.tolist(),weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(compact.forest.apply(query),reference.apply(query))
    assert compact.members.size==len(x)*common['n_estimators']

def test_checkpoint_reference_can_be_native_directory(tmp_path):
    from external_baselines import verify_checkpoint_reference
    (tmp_path/'model.pt').write_bytes(b'checkpoint')
    assert verify_checkpoint_reference(str(tmp_path))
    assert verify_checkpoint_reference(str(tmp_path/'model.pt'))
    import pytest
    with pytest.raises(FileNotFoundError):verify_checkpoint_reference(str(tmp_path/'absent'))

def test_soft_crps_narrow_mean_cluster_matches_direct_kernel_solution():
    # A clipped diffusion forecast can put many means next to the boundary.
    # Prefix second moments then cancel after division by a tiny bandwidth.
    rng=np.random.default_rng(20261109)
    mu=np.r_[rng.uniform(-1,.8,600),1-rng.uniform(0,1e-6,400)]
    sigma=rng.uniform(.000001,.3,len(mu));z=rng.normal(size=len(mu))
    y=mu+sigma*z;z=(y-mu)/sigma
    model=SoftCRPSPIT.from_summaries(mu,sigma,y)
    query=mu[-12:];target=z[-12:];levels=np.array([.005,.1,.5,.9,.995])
    expected_q=np.array([brute(m,mu,sigma,z,20,levels) for m in query])
    expected_cdf=[]
    for m,v in zip(query,target):
        h=np.partition(abs(mu-m),len(mu)//20-1)[len(mu)//20-1]
        w=sigma*np.maximum(0,1-((mu-m)/h)**2)
        expected_cdf.append(w[z<=v].sum()/w.sum())
    np.testing.assert_allclose(model.cdf(query,target,20),expected_cdf,atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(model.standardized_quantiles(query,20,levels),expected_q,atol=1e-9,rtol=1e-9)

def test_soft_crps_coincident_neighbor_boundary_uses_continuous_kernel_limit():
    rng=np.random.default_rng(41)
    mu=np.r_[rng.uniform(-1,.8,600),np.ones(400)]
    sigma=rng.uniform(.1,1,len(mu));y=mu+sigma*rng.normal(size=len(mu))
    model=SoftCRPSPIT.from_summaries(mu,sigma,y);z=(y-mu)/sigma
    # All L nearest neighbors coincide at one: their Epanechnikov weights
    # reach zero together. The outward bandwidth limit retains sigma weights.
    query=np.array([1-1e-7,1-1e-6]);levels=np.array([.005,.1,.5,.9,.995])
    selected=mu==1;order=np.argsort(z[selected]);zz=z[selected][order]
    cumulative=np.cumsum(sigma[selected][order]);cumulative/=cumulative[-1]
    expected=zz[np.searchsorted(cumulative,levels)]
    np.testing.assert_allclose(model.standardized_quantiles(query,20,levels),np.tile(expected,(2,1)),atol=1e-12)
    values=np.array([-.25,.25]);expected_cdf=[sigma[selected & (z<=v)].sum()/sigma[selected].sum() for v in values]
    np.testing.assert_allclose(model.cdf(query,values,20),expected_cdf,atol=1e-12)
