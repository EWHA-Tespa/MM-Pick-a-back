from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
import models.layers as nl

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        # ✅ context 인자가 존재하는 경우에만 적용
        if exists(self.norm_context) and 'context' in kwargs:
            kwargs['context'] = self.norm_context(kwargs['context'])

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nl.SharableLinear(dim, dim * mult * 2, bias=True, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer'),
            GEGLU(),
            nl.SharableLinear(dim * mult, dim, bias=True, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer')
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nl.SharableLinear(query_dim, inner_dim, bias=False)
        self.to_kv = nl.SharableLinear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nl.SharableLinear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        seq_dropout_prob=0.,
        dataset_history=None,
        dataset2num_classes=None
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.input_proj = nl.SharableLinear(3, dim)

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(
            queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=latent_dim
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nl.SharableLinear(queries_dim, logits_dim, bias=False, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer')  

    def forward(self, data, mask=None, queries=None):
        b, c, h, w = data.shape  
        data = rearrange(data, 'b c h w -> b (h w) c') 
        data = self.input_proj(data)
        b, *_, device = *data.shape, data.device
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context=data, mask=mask) + x  # 🔹 수정된 `data` 사용
        x = cross_ff(x) + x

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if queries is None:
            return self.to_logits(x.mean(dim=1))  # ✅ Mean Pooling 후 Classification 수행

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)

        latents = self.decoder_cross_attn(queries, context=x)

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_logits(latents)

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if not hasattr(self, 'datasets'):
            self.datasets = []
            self.dataset2num_classes = {}

        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.to_logits = nl.SharableLinear(self.latents.shape[-1], num_classes)  # ✅ classifier 업데이트
            nn.init.normal_(self.to_logits.weight, 0, 0.01)
            nn.init.constant_(self.to_logits.bias, 0)

    def set_dataset(self, dataset):
        """Ensures the correct classification head is used"""
        assert dataset in self.datasets, f"Dataset {dataset} has not been added!"
        num_classes = self.dataset2num_classes[dataset]
        self.to_logits = nn.Linear(self.latents.shape[-1], num_classes)
