"""Batch independent TimeDiff trajectories through the unchanged native forward."""
import torch


def sample_timediff_batched(model,x,xm,y,ym,num_samples,chunk):
    pieces=[];remaining=num_samples
    while remaining:
        current=min(chunk,remaining)
        # Repeat each conditioning window; one independent original DPM
        # trajectory is sampled per expanded batch element.
        inputs=[v.repeat_interleave(current,dim=0) for v in [x,xm,y,ym]]
        draws=model(*inputs,None,None,None,1)[0]
        assert draws.shape[1]==1
        pieces.append(draws[:,0].reshape(len(x),current,*draws.shape[2:]).detach())
        remaining-=current
    return torch.cat(pieces,dim=1)
