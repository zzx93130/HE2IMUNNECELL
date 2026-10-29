"""Optional inference-only attention kernel for Cellpose SAM; no weight edits."""
import types

import torch
from torch.nn import functional as F


def _sdpa_forward(self, x):
    from segment_anything.modeling.image_encoder import get_rel_pos
    batch, height, width, _ = x.shape
    length = height * width
    qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, -1).permute(2,0,3,1,4)
    q, k, v = qkv.unbind(0)
    bias = None
    if self.use_rel_pos:
        # Relative-position vectors depend only on the token grid and fixed
        # inference weights, so interpolation/indexing is cached per layer.
        key = (height, width, x.device, x.dtype)
        cache = getattr(self, "_inference_relative_position_cache", {})
        if key not in cache:
            cache[key] = (get_rel_pos(height, height, self.rel_pos_h),
                          get_rel_pos(width, width, self.rel_pos_w))
            self._inference_relative_position_cache = cache
        rh, rw = cache[key]
        grid = q.reshape(batch * self.num_heads, height, width, -1)
        rel_h = torch.einsum("bhwc,hkc->bhwk", grid, rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", grid, rw)
        bias = (rel_h[...,None] + rel_w[...,None,:]).reshape(batch,self.num_heads,length,length)
    output = F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.0,scale=self.scale)
    output = output.transpose(1,2).reshape(batch,height,width,-1)
    return self.proj(output)


def enable_sdpa_attention(model):
    """Patch only this loaded model; return the number of SAM attention layers."""
    from segment_anything.modeling.image_encoder import Attention
    if model.net.training:
        model.net.eval()
    count = 0
    for module in model.net.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(_sdpa_forward, module)
            count += 1
    if not count:
        raise ValueError("SDPA acceleration requires the Cellpose SAM backbone")
    return count
