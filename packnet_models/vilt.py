import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertPredictionHeadTransform

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super(ITMHead, self).__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc(x)

class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super(MLMHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class MPPHead(nn.Module):
    def __init__(self, config):
        super(MPPHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3) # 조정 

    def forward(self, x):
        x = self.transform(x)
        return self.decoder(x)


class JointTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_ratio, drop_rate):
        super(JointTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=int(hidden_size * mlp_ratio), 
            dropout=drop_rate, 
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, src_key_padding_mask=None):
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)
        return x


class ViLTModel(nn.Module):
    def __init__(self, config):
        super(ViLTModel, self).__init__()
        self.config = config

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=0,  
            num_attention_heads=0,
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        self.text_embeddings = BertEmbeddings(bert_config)
        self._init_weights(self.text_embeddings)

        self.image_encoder = timm.create_model(config["image_model"], pretrained=True)
        self.image_encoder.reset_classifier(0) 

        self.image_modality_embed = nn.Parameter(torch.zeros(1, 1, config["hidden_size"]))
        nn.init.normal_(self.image_modality_embed, mean=0.0, std=0.02)

        self.joint_transformer = JointTransformer(
            hidden_size=config["hidden_size"],
            num_layers=config["joint_num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["hidden_size"]))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        num_image_tokens = self.image_encoder.patch_embed.num_patches if hasattr(self.image_encoder, "patch_embed") else 196
        self.max_joint_length = 1 + config["max_text_len"] + num_image_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_joint_length, config["hidden_size"]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pooler = Pooler(config["hidden_size"])
        self._init_weights(self.pooler)

        if config["loss_names"].get("mlm", 0) > 0:
            self.mlm_head = MLMHead(bert_config)
            self._init_weights(self.mlm_head)
        if config["loss_names"].get("itm", 0) > 0:
            self.itm_head = ITMHead(config["hidden_size"])
            self._init_weights(self.itm_head)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, batch, mask_text=False):
        """
        batch: 딕셔너리로, 다음 키들을 포함해야 함.
            "text_ids": (B, L_text)
            "text_masks": (B, L_text)   [선택: 없으면 전부 1로 가정]
            "image": (B, 3, H, W)
            (MLM용) "text_labels": (B, L_text)
            (ITM용) "itm_labels": (B,)
        mask_text: 텍스트 마스킹 여부 (MLM loss 계산 시 True)
        """
        B = batch["text_ids"].size(0)

        # 텍스트 임베딩 (B, L_text, hidden_size)
        text_tokens = self.text_embeddings(batch["text_ids"])
        if "text_masks" in batch:
            text_mask = batch["text_masks"].bool()
        else:
            text_mask = torch.ones(batch["text_ids"].size(), dtype=torch.bool, device=batch["text_ids"].device)

        # 이미지 인코딩: timm ViT 모델은 forward_features()로 feature 추출
        image_features = self.image_encoder.forward_features(batch["image"])  # (B, num_tokens, hidden_size)
        # ViT의 첫 토큰은 [CLS]이므로 제거
        image_tokens = image_features[:, 1:, :]  # (B, num_image_tokens, hidden_size)
        image_tokens = image_tokens + self.image_modality_embed

        # joint sequence 구성: [CLS] 토큰, 텍스트 토큰, 이미지 토큰
        cls_tokens = self.cls_token.expand(B, -1, -1)
        joint_tokens = torch.cat([cls_tokens, text_tokens, image_tokens], dim=1)
        seq_len = joint_tokens.size(1)
        pos_embed = self.pos_embed[:, :seq_len, :]
        joint_tokens = joint_tokens + pos_embed

        # joint mask 구성: 텍스트는 batch["text_masks"], 이미지 토큰은 모두 valid, [CLS]도 valid
        num_text = text_tokens.size(1)
        num_image = image_tokens.size(1)
        joint_mask = torch.cat([
            torch.ones(B, 1, device=joint_tokens.device, dtype=torch.bool),
            text_mask,
            torch.ones(B, num_image, device=joint_tokens.device, dtype=torch.bool)
        ], dim=1)
        # nn.TransformerEncoder의 src_key_padding_mask는 False여야 하는 토큰이 attend됨
        src_key_padding_mask = ~joint_mask

        # 융합 transformer 통과
        joint_tokens = self.joint_transformer(joint_tokens, src_key_padding_mask=src_key_padding_mask)

        # pooled output: 첫 토큰([CLS]) 사용
        pooled_output = self.pooler(joint_tokens)

        output = {
            "joint_tokens": joint_tokens,
            "cls_feats": pooled_output,
            "text_tokens": text_tokens,
            "image_tokens": image_tokens
        }

        if mask_text and "text_labels" in batch:
            mlm_logits = self.mlm_head(text_tokens)
            output["mlm_logits"] = mlm_logits
            output["text_labels"] = batch["text_labels"]

        if self.config["loss_names"].get("itm", 0) > 0 and "itm_labels" in batch:
            itm_logits = self.itm_head(pooled_output)
            output["itm_logits"] = itm_logits
            output["itm_labels"] = batch["itm_labels"]

        return output

    def compute_loss(self, batch, output):
        loss = 0.0
        loss_dict = {}
        if self.config["loss_names"].get("mlm", 0) > 0 and "mlm_logits" in output:
            mlm_loss = F.cross_entropy(output["mlm_logits"].view(-1, self.config["vocab_size"]),
                                       output["text_labels"].view(-1),
                                       ignore_index=-100)
            loss += mlm_loss
            loss_dict["mlm_loss"] = mlm_loss.item()
        if self.config["loss_names"].get("itm", 0) > 0 and "itm_logits" in output:
            itm_loss = F.cross_entropy(output["itm_logits"], output["itm_labels"].long())
            loss += itm_loss
            loss_dict["itm_loss"] = itm_loss.item()
        return loss, loss_dict


class vilt(nn.Module):
    def __init__(self, pretrained=False, dataset_history=None, dataset2num_classes=None, **kwargs):
        super(vilt, self).__init__()
        self.dataset_history = dataset_history if dataset_history is not None else []
        self.dataset2num_classes = dataset2num_classes if dataset2num_classes is not None else {}
        config = {
            "vocab_size": 30522,
            "hidden_size": 768,
            "max_text_len": 40,
            "image_model": "vit_base_patch16_224",
            "joint_num_layers": 4,
            "num_heads": 12,
            "mlp_ratio": 4,
            "drop_rate": 0.1,
            "loss_names": {"mlm": 1, "itm": 1}
        }
        config.update(kwargs)
        self.model = ViLTModel(config)
        self.classifiers = nn.ModuleDict()
        self.current_dataset = None

    def add_dataset(self, dataset_name, num_classes):
        if dataset_name not in self.classifiers:
            hidden_size = self.model.config["hidden_size"]
            self.classifiers[dataset_name] = nn.Linear(hidden_size, num_classes)
            self.dataset2num_classes[dataset_name] = num_classes
            self.dataset_history.append(dataset_name)

    def set_dataset(self, dataset_name):
        self.current_dataset = dataset_name

    def forward(self, batch, mask_text=False):
        output = self.model(batch, mask_text=mask_text)
        if self.current_dataset and self.current_dataset in self.classifiers:
            logits = self.classifiers[self.current_dataset](output["cls_feats"])
            output["logits"] = logits
        if self.training:
            loss, loss_dict = self.model.compute_loss(batch, output)
            output["loss"] = loss
            output["loss_dict"] = loss_dict
        return output
    