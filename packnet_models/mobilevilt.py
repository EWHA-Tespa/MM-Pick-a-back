import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, einsum
from typing import Callable, Tuple, Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

# --------------------------------------------------
# at.py
# --------------------------------------------------
def exists(val):
    return val is not None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9, **kwargs):
        b, t, device = *start_tokens.shape, start_tokens.device
        out = start_tokens
        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            if exists(eos_token):
                is_eos_token = out == eos_token
                if is_eos_token.any(dim=-1).all():
                    shifted = F.pad(is_eos_token, (1, -1))
                    mask = shifted.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break
        return out[:, t:]

    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        logits = rearrange(logits, "b c n -> b n c")
        return F.cross_entropy(logits, x_labels)

# --------------------------------------------------
# bit_attention.py
# --------------------------------------------------
def scaled_dot_product_gqa(query: Tensor, key: Tensor, value: Tensor, dropout: float = 0.0,
                           scale: Optional[float] = None, mask: Optional[Tensor] = None,
                           is_causal: Optional[bool] = None, need_weights: bool = False,
                           average_attn_weights: bool = False, force_grouped: bool = False):
    if (mask is not None) and (is_causal is not None):
        raise ValueError("Only one of 'mask' and 'is_causal' should be provided.")
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f"Expected 4D tensors, got {query.shape}, {key.shape}, {value.shape}.")

    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError("Mismatched dimensions between query, key and value.")
    if hq % hk != 0:
        raise ValueError("query heads must be a multiple of key/value heads.")

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    if is_causal:
        mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()

    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    out = rearrange(out, "b h n d -> b n h d")

    attn_weights = None
    if need_weights:
        attn_weights = rearrange(attention, "b h n s -> b n s h")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights

class BitMGQA(nn.Module):
    def __init__(self, embed_dim: int, query_heads: int = 8, kv_heads: int = 4, dropout: float = 0.1,
                 bias: bool = True, layer_norm: bool = True, layer_norm_eps: float = 1e-5,
                 gamma_init: float = 1.0, linear_groups: int = 1, *args, **kwargs):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError("embed_dim must be divisible by both query_heads and kv_heads")

        head_dim = embed_dim // query_heads
        if head_dim % 8 != 0 or head_dim > 128:
            raise ValueError("head_dim must be divisible by 8 and <= 128")

        # Query projection layer
        self.q_proj = BitLinear(embed_dim, embed_dim, bias=bias, *args, **kwargs)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = BitLinear(embed_dim, kv_embed_dim, bias=bias, *args, **kwargs)
        self.v_proj = BitLinear(embed_dim, kv_embed_dim, bias=bias, *args, **kwargs)
        self.norm = nn.LayerNorm(kv_embed_dim, eps=layer_norm_eps) if layer_norm else None
        self.out_proj = BitLinear(kv_embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                need_weights: bool = False,
                is_causal: bool = False,
                average_attn_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        x, attn_weights = scaled_dot_product_gqa(query=q, key=k, value=v,
                                                  is_causal=is_causal, need_weights=need_weights,
                                                  average_attn_weights=average_attn_weights)
        x = rearrange(x, "b n h d -> b n (h d)")
        if self.layer_norm and self.norm is not None:
            x = self.norm(x)
        x = self.out_proj(x)
        return x, attn_weights

# --------------------------------------------------
# bit_ffn.py
# --------------------------------------------------
def default(val, d):
    return val if val is not None else d

def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)

class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation, mult_bias: bool = False, linear: Optional[Callable] = None, *args, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias
        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear(dim_in, dim_out * 4, *args, **kwargs)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate) * self.mult_bias

class BitFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False,
                 glu_mult_bias: bool = False, swish: bool = False, post_act_ln: bool = False,
                 dropout: float = 0.0, no_bias: bool = False, zero_init_output: bool = False, *args, **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        activation = nn.SiLU() if swish else nn.GELU()
        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
        else:
            project_in = nn.Sequential(
                BitLinear(dim, inner_dim, bias=not no_bias, *args, **kwargs), activation
            )
        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            BitLinear(inner_dim, dim_out, bias=not no_bias, *args, **kwargs),
        )
        if zero_init_output:
            init_zero_(self.ff[-1].weight)

    def forward(self, x):
        return self.ff(x)

# --------------------------------------------------
# BitLinear 관련 코드 (bit_linear.py & bitlinear.py 통합)
# --------------------------------------------------
def activation_quant(x: Tensor):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u

class BitLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        w = self.weight
        x_norm = RMSNorm(d)(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        return F.linear(x_quant, w_quant)

# --------------------------------------------------
# bit_llama.py
# --------------------------------------------------
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

# RMSNorm 이미 정의됨

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# Attention, FeedForward, TransformerBlock, Transformer 구현 (bit_llama.py)
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size 관련 초기화는 생략 (단일 GPU 환경)
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # ColumnParallelLinear, RowParallelLinear, ParallelEmbedding 은 생략하거나 nn.Linear로 대체
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim,
                                        multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, params) for layer_id in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2)
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        return self.output(h).float()

# --------------------------------------------------
# bit_transformer.py
# --------------------------------------------------
def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)

class RMSNorm2(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0
    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

class TransformerAlt(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int, ff_mult: int = 2, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(BitMGQA(dim, heads, *args, **kwargs))
            self.ffn_layers.append(BitFeedForward(dim, dim, ff_mult, swish=True, post_act_ln=True, dropout=0.1))
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        skip = x
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x, _ = attn(x, x, x, is_causal=True, *args, **kwargs)
            x = x + skip
            x = ffn(x) + x
        return x

class BitNetTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, num_tokens: int, heads=8, ff_mult=4):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.transformer = TransformerAlt(dim=dim, depth=depth, heads=heads, ff_mult=ff_mult)
        self.to_logits = nn.Sequential(RMSNorm2(dim), nn.Linear(dim, num_tokens))
    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)

# --------------------------------------------------
# inference.py
# --------------------------------------------------
class BitNetInference:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = BitNetTransformer(num_tokens=256, dim=512, depth=8)
        self.model = AutoregressiveWrapper(self.model, max_seq_len=1024)
        self.model.to(self.device)
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    @staticmethod
    def decode_token(token):
        return str(chr(max(32, token)))
    @staticmethod
    def decode_tokens(tokens):
        return "".join(map(BitNetInference.decode_token, tokens))
    def generate(self, input_str, length):
        inp = torch.from_numpy(np.fromstring(input_str, dtype=np.uint8)).long().to(self.device)
        sample = self.model.generate(inp[None, ...], length)
        return self.decode_tokens(sample[0])

# --------------------------------------------------
# replace_hf.py
# --------------------------------------------------
def replace_linears_in_hf(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None))
        else:
            replace_linears_in_hf(module)

def replace_linears_in_pytorch_model(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None))
        else:
            replace_linears_in_hf(module)
