from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# --- Helper Functions and Classes (unchanged) ---

__all__ = [
    'perceiver',
    # 'combined_perceiver'
]

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class MultiModalAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        image_context_dim,
        text_context_dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        image_context_dim = default(image_context_dim, query_dim)
        text_context_dim = default(text_context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv_image = nn.Linear(image_context_dim, inner_dim * 2, bias=False)
        self.to_kv_text = nn.Linear(text_context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None, modality='image'):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        if modality == 'image':
            k, v = self.to_kv_image(context).chunk(2, dim = -1)
        else:
            k, v = self.to_kv_text(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


# main class

class perceiver(nn.Module):
    
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        dataset_history,
        dataset2num_classes,
        init_weights=True,
        image_input_channels = 3, # 이미지
        image_input_axis = 2,
        text_input_channels = 768, # 텍스트
        text_input_axis = 1,
        max_text_length=512,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.image_input_axis = image_input_axis
        self.text_input_axis = text_input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        self.current_modality = 'image'

        self.datasets = dataset_history
        self.dataset2num_classes = dataset2num_classes
        self.classifiers = []

        fourier_channels_image = (image_input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        image_input_dim = fourier_channels_image + image_input_channels
        
        # fourier_channels_text =  (text_input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        # text_input_dim = fourier_channels_text + text_input_channels
        text_input_dim = text_input_channels
        self.text_position_embeddings = nn.Embedding(max_text_length, text_input_channels)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, MultiModalAttention(query_dim=latent_dim, 
                                                                        image_context_dim=image_input_dim, 
                                                                        text_context_dim=text_input_dim,
                                                                        heads = cross_heads,
                                                                        dim_head = cross_dim_head,
                                                                        dropout = attn_dropout), context_dim = None) # 원래는 context_dim=input_dim해서, layernorm해야함. 하지만, 일단 생략함.
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

        self.proj = nn.Linear(latent_dim, 84)

        if self.datasets:
            self._reconstruct_classifiers()
        
        if init_weights:
            self._initialize_weights()
    
    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False,
        modality='image',
    ):
        if modality is not None:
            modality = self.current_modality

        if modality == 'image':
            self.input_axis = self.image_input_axis
            if data.ndim == 4:
                data = data.permute(0, 2, 3, 1)
        elif modality == 'text':
            self.input_axis = self.text_input_axis
        else:
            raise ValueError("Unknown modality: choose 'image' or 'text'.")

        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data and modality =='image':
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)
        else:
            if modality == 'text':
                seq_length = data.shape[1]
                position_ids = torch.arange(0, seq_length, device=device).unsqueeze(0)
                data = data + self.text_position_embeddings(position_ids)
        
        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask, modality = modality) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        if isinstance(self.to_logits, nn.Identity):
            features = x.mean(dim=1)  # [B, latent_dim]
            features = self.proj(features)  # [B, 84]
            return self.classifier(features)
        else:
            return self.to_logits(x)
        #return self.to_logits(x)
    
    def set_modality(self, modality='image'):
        self.current_modality = modality
        for layer in self.layers:
            cross_attn_block = layer[0].fn
            if isinstance(cross_attn_block, MultiModalAttention):
                if modality == 'image':
                    cross_attn_block.to_kv_image.requires_grad_(True)
                    cross_attn_block.to_kv_text.requires_grad_(False)
                else:
                    cross_attn_block.to_kv_image.requires_grad_(False)
                    cross_attn_block.to_kv_text.requires_grad_(True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            classifier = nn.Linear(84, num_classes)
            self.classifiers.append(classifier)
        # Optionally register the active classifier as a submodule:
        if self.classifiers:
            self.classifier = self.classifiers[0]
            

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(84, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

'''
class combined_perceiver(nn.Module):
    def __init__(self, input_dim, dataset_history, dataset2num_classes):
        super(combined_perceiver, self).__init__()
        self.input_projection = nn.Linear(input_dim, 3*32*32)
        self.perceiver = perceiver(input_channels=3,
                                    input_axis=2,
                                    num_freq_bands=6,
                                    depth=4,
                                    max_freq=10,
                                    num_latents=512,
                                    latent_dim=256,
                                    cross_heads=1,
                                    latent_heads=8,
                                    cross_dim_head=64,
                                    latent_dim_head=64,
                                    attn_dropout=0.,
                                    ff_dropout=0.,
                                    weight_tie_layers=False,
                                    fourier_encode_data=True,
                                    self_per_cross_attn=1,
                                    final_classifier_head=False,
                                    dataset_history=dataset_history,
                                    dataset2num_classes=dataset2num_classes)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x.view(-1, 3, 32, 32)
        x = self.perceiver(x)
        return x
    
    def add_dataset(self, dataset, num_classes):
        return self.perceiver.add_dataset(dataset, num_classes)
    
    def set_dataset(self, dataset):
        return self.perceiver.set_dataset(dataset)
    
    @property
    def datasets(self):
        return self.perceiver.datasets
    
    @property
    def dataset2num_classes(self):
        return self.perceiver.dataset2num_classes
    
    @property
    def classifiers(self):
        return self.perceiver.classifiers
    
    @property
    def classifier(self):
        return self.perceiver.classifier
    
    @classifier.setter
    def classifier(self, value):
        self.perceiver.classifier = value'''