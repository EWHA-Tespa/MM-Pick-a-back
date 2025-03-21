import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    return tensor

# Encoder 
class EncoderConfig:
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 vocab_size=64010,
                 multiway=True,
                 layernorm_embedding=False,
                 normalize_output=True,
                 no_output_layer=True,
                 drop_path_rate=0.0,
                 encoder_embed_dim=768,
                 encoder_attention_heads=12,
                 encoder_ffn_embed_dim=768*4,
                 encoder_layers=12,
                 checkpoint_activations=False):
        self.img_size = img_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.multiway = multiway
        self.layernorm_embedding = layernorm_embedding
        self.normalize_output = normalize_output
        self.no_output_layer = no_output_layer
        self.drop_path_rate = drop_path_rate
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_layers = encoder_layers
        self.checkpoint_activations = checkpoint_activations

def _get_base_config(**kwargs):
    return EncoderConfig(**kwargs)

# Transformer Block 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# BEiT3 백본 정의 
class BEiT3(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_dim = config.encoder_embed_dim
        
        self.patch_embed = nn.Conv2d(3, self.embed_dim,
                                     kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=self.embed_dim,
                             num_heads=config.encoder_attention_heads,
                             mlp_ratio=config.encoder_ffn_embed_dim / self.embed_dim,
                             dropout=0.0)
            for _ in range(config.encoder_layers)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        
        self.init_weights()
        
    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, textual_tokens, visual_tokens, **kwargs):
        B = visual_tokens.shape[0]
        x = self.patch_embed(visual_tokens)  # (B, embed_dim, H_patch, W_patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"encoder_out": x}

# BEiT3Wrapper
class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args  # args는 EncoderConfig 객체입니다.
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def get_num_layers(self):
        return self.args.encoder_layers
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(self, args, num_classes, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(args, **kwargs)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
        
        self.datasets = kwargs.get("dataset_history", [])
        self.dataset2num_classes = kwargs.get("dataset2num_classes", {})
        self.classifiers = []  
        if self.datasets:
            self._reconstruct_classifiers()
        else:
            self.classifier = self.head

    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        t = x[:, 1:, :]
        cls_x = self.fc_norm(t.mean(1))
        return self.classifier(cls_x)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            classifier = nn.Linear(self.fc_norm.normalized_shape[0], num_classes)
            nn.init.normal_(classifier.weight, 0, 0.01)
            nn.init.constant_(classifier.bias, 0)
            self.classifiers.append(classifier)
        if self.classifiers:
            self.classifier = self.classifiers[0]

    def add_dataset(self, dataset, num_classes):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            classifier = nn.Linear(self.fc_norm.normalized_shape[0], num_classes)
            nn.init.normal_(classifier.weight, 0, 0.01)
            nn.init.constant_(classifier.bias, 0)
            self.classifiers.append(classifier)
            self.add_module("classifier_{}".format(dataset), classifier)

    def set_dataset(self, dataset):
        assert dataset in self.datasets, f"Dataset {dataset} not found."
        self.classifier = self.classifiers[self.datasets.index(dataset)]
