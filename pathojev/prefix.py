"""Shared-prefix execution for the Qwen3.5 hybrid backbone.

The slide prefix is encoded once with an ordinary causal forward. Every question row then continues from the prefix's
per-layer state and never sees another row:

* softmax-attention layers: rows attend to [prefix K/V ; own causal past]; the prefix K/V are shared, not copied.
  Registered as the attention implementation "pathojev"; without `jev_prefix` it falls back to SDPA.
* Gated DeltaNet layers: each row starts its causal conv and recurrence from the prefix's final conv / recurrent state.
"""
import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

IMPL = "pathojev"


def shared_prefix_attention(q, k, v, pk, pv, scaling):
    """q: (B,Hq,L,D); k,v: (B,Hkv,L,D); pk,pv: (1,Hkv,P,D). Returns (B,L,Hq,D)."""
    B, Hq, L, D = q.shape
    Hkv, P = pk.shape[1], pk.shape[2]
    g = Hq // Hkv
    qh = q.view(B, Hkv, g, L, D).permute(1, 0, 2, 3, 4).reshape(Hkv, B * g * L, D)
    s_p = torch.bmm(qh, pk[0].transpose(1, 2)).view(Hkv, B, g, L, P).permute(1, 0, 2, 3, 4)
    s_s = torch.matmul(q.view(B, Hkv, g, L, D), k.unsqueeze(2).transpose(-1, -2))
    causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
    s_s = s_s.masked_fill(~causal, float("-inf"))
    w = torch.softmax(torch.cat([s_p, s_s], -1).float() * scaling, -1).to(q.dtype)
    w_p, w_s = w[..., :P], w[..., P:]
    o_p = torch.bmm(w_p.permute(1, 0, 2, 3, 4).reshape(Hkv, B * g * L, P), pv[0]).view(Hkv, B, g, L, D).permute(1, 0, 2, 3, 4)
    o_s = torch.matmul(w_s, v.unsqueeze(2))
    return (o_p + o_s).reshape(B, Hq, L, D).transpose(1, 2).contiguous()


def _attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    prefix = kwargs.pop("jev_prefix", None)
    if prefix is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    _, pk, pv = prefix[module.layer_idx]
    scaling = module.head_dim ** -0.5 if scaling is None else scaling
    return shared_prefix_attention(query, key, value, pk, pv, scaling), None


AttentionInterface.register(IMPL, _attention)
AttentionMaskInterface.register(IMPL, sdpa_mask)


class _RowStart:
    """Cache handed to one Gated DeltaNet layer: every row starts from the prefix state; updates are dropped."""

    class _Layer:
        def __init__(self, conv, rec):
            self.conv_states, self.recurrent_states = {0: conv}, {0: rec}

    def __init__(self, layer_idx, conv, rec, batch):
        self.layers = {layer_idx: self._Layer(conv.expand(batch, -1, -1), rec.expand(batch, -1, -1, -1))}

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return True

    def update_conv_state(self, conv_states, *a, **k):
        return conv_states

    def update_recurrent_state(self, recurrent_states, *a, **k):
        return recurrent_states


def _patch_gated_deltanet():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    cls = m.Qwen3_5GatedDeltaNet
    if getattr(cls, "_pathojev_patched", False):
        return
    orig = cls.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        prefix = kwargs.pop("jev_prefix", None)
        if prefix is not None:
            _, conv, rec = prefix[self.layer_idx]
            cache_params = _RowStart(self.layer_idx, conv, rec, hidden_states.shape[0])
        return orig(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

    cls.forward = forward
    cls._pathojev_patched = True


_patch_gated_deltanet()


def prefix_state(cache):
    """Per-layer state of the prefix pass: ("attn", K, V) or ("linear", conv_state, recurrent_state)."""
    out = []
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            out.append(("linear", layer.conv_states[0], layer.recurrent_states[0]))
        else:
            out.append(("attn", layer.keys, layer.values))
    return out
