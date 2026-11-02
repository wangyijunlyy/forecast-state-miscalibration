"""Numerically stable native de-stationary attention."""
import math
from types import MethodType
import torch

TMDM_NUMERICS_VERSION='native_fp32_attention_float64_overflow_fallback_v1'


def _guard_tau_exp(module,inputs,value):
    if torch.isfinite(value.detach().exp()).all():return value
    # Above double's exponent range, float32 dot-product differences already
    # yield the same saturated softmax; retain a finite representable factor.
    return value.double().clamp(max=math.log(torch.finfo(torch.float64).max)-2.)


def _attention_forward(mask_factory):
    def forward(self,queries,keys,values,attn_mask,tau=None,delta=None):
        batch,length,heads,width=queries.shape
        scale=self.scale or 1./math.sqrt(width)
        factor=1. if tau is None else tau.unsqueeze(1).unsqueeze(1)
        shift=0. if delta is None else delta.unsqueeze(1).unsqueeze(1)
        raw=torch.einsum('blhe,bshe->bhls',queries,keys)
        scores=raw*factor+shift
        promoted=torch.is_tensor(factor) and factor.dtype!=queries.dtype
        if promoted or not torch.isfinite(scores).all():
            wide=torch.einsum('blhe,bshe->bhls',queries.double(),keys.double())
            if self.mask_flag:
                if attn_mask is None:attn_mask=mask_factory(batch,length,device=queries.device)
                row_max=wide.masked_fill(attn_mask.mask,float('-inf')).amax(dim=-1,keepdim=True)
            else:row_max=wide.amax(dim=-1,keepdim=True)
            # Softmax is invariant to a row constant. Center before the large
            # positive temperature so overflow is confined to negative tails.
            wide=wide-row_max
            # Masked entries become -inf only after multiplication; otherwise
            # the temperature gradient includes the indeterminate0*(-inf).
            if self.mask_flag:wide=wide.masked_fill(attn_mask.mask,0.)
            scores=wide*(factor.double() if torch.is_tensor(factor) else factor)
            scores=scores+(shift.double() if torch.is_tensor(shift) else shift)
        if scores.dtype!=queries.dtype:self._overflow_fallbacks+=1
        if self.mask_flag:
            if attn_mask is None:attn_mask=mask_factory(batch,length,device=queries.device)
            scores.masked_fill_(attn_mask.mask,float('-inf'))
        attention=self.dropout(torch.softmax(scale*scores,dim=-1))
        if attention.dtype!=values.dtype:attention=attention.to(values.dtype)
        output=torch.einsum('bhls,bshd->blhd',attention,values).contiguous()
        return output,attention if self.output_attention else None
    return forward


def enable_stable_tmdm_attention(model):
    count=0
    for layer in model.modules():
        if type(layer).__name__!='DSAttention':continue
        if getattr(layer,'_native_attention_precision_guard',False):continue
        mask_factory=layer.forward.__func__.__globals__['TriangularCausalMask']
        layer.forward=MethodType(_attention_forward(mask_factory),layer)
        layer._native_attention_precision_guard=True;layer._overflow_fallbacks=0;count+=1
    tau_learner=getattr(model,'tau_learner',None)
    if tau_learner is not None and not getattr(tau_learner,'_native_tau_precision_guard',False):
        tau_learner.register_forward_hook(_guard_tau_exp)
        tau_learner._native_tau_precision_guard=True
    return count
