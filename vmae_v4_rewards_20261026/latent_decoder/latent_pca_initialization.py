"""Training-only PCA mean compression and ridge fit of diagonal posterior marginals."""
import copy

import torch


@torch.no_grad()
def compress_pca(state, hidden, means, logvars, weights, dimension):
    source=state['config']['model']['latent_dim']
    assert 0<dimension<=source and hidden.ndim==means.ndim==logvars.ndim==2
    assert means.shape==logvars.shape and means.shape[1]==source and len(hidden)==len(means)==len(weights)
    assert torch.isfinite(hidden).all() and torch.isfinite(means).all() and torch.isfinite(logvars).all()
    assert (weights>0).all()
    weights=weights.double()/weights.sum()
    if dimension==source:
        return copy.deepcopy(state),dict(full_capacity_exact=True,retained_fraction=1.)
    hidden,means,logvars=hidden.double(),means.double(),logvars.double()
    center=(means*weights[:,None]).sum(0)
    centered=means-center
    covariance=centered.T@(weights[:,None]*centered)
    eigen,vectors=torch.linalg.eigh(covariance)
    order=eigen.argsort(descending=True)
    basis=vectors[:,order[:dimension]]
    pivots=basis.abs().argmax(0)
    basis=basis*basis[pivots,torch.arange(dimension)].sign()[None,:]
    torch.testing.assert_close(basis.T@basis,torch.eye(dimension,dtype=torch.float64),atol=1e-8,rtol=1e-8)
    # A rotated diagonal Gaussian is generally correlated; fit a NEW diagonal q, never call it exact.
    target=(logvars.exp()@basis.square()).clamp_min(1e-12).log()
    hmean=(hidden*weights[:,None]).sum(0);ymean=(target*weights[:,None]).sum(0)
    h=hidden-hmean;y=target-ymean
    gram=h.T@(weights[:,None]*h)
    ridge=(1e-4*gram.trace()/len(gram)).clamp_min(1e-8)
    coefficients=torch.linalg.solve(gram+ridge*torch.eye(len(gram),dtype=torch.float64),h.T@(weights[:,None]*y))
    bias=ymean-hmean@coefficients
    result=dict(config=copy.deepcopy(state['config']),model=dict(state['model']),recipe=state['recipe'])
    result['config']['model']['latent_dim']=dimension
    old=state['model'];dtype=old['mu.weight'].dtype
    result['model']['mu.weight']=(basis.T@old['mu.weight'].double()).to(dtype)
    result['model']['mu.bias']=(basis.T@(old['mu.bias'].double()-center)).to(dtype)
    result['model']['from_latent.weight']=(old['from_latent.weight'].double()@basis).to(dtype)
    result['model']['from_latent.bias']=(old['from_latent.bias'].double()+old['from_latent.weight'].double()@center).to(dtype)
    result['model']['logvar.weight']=coefficients.T.to(dtype)
    result['model']['logvar.bias']=bias.to(dtype)
    predicted=(hidden@coefficients+bias).clamp(-8,8)
    report=dict(full_capacity_exact=False,source_dimension=source,dimension=dimension,
        retained_fraction=float(eigen[order[:dimension]].clamp_min(0).sum()/eigen.clamp_min(0).sum()),
        training_logvar_rmse=float(((predicted-target).square()*weights[:,None]).sum().div(dimension).sqrt()),
        ridge=float(ridge),basis=basis.tolist(),mean=center.tolist(),
        posterior='New diagonal Gaussian fitted to projected marginal variances; correlations discarded; sampling requires separate verification')
    return result,report
