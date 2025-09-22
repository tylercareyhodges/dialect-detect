from __future__ import annotations
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, freeze_backbone: bool = True, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        h = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # last_hidden_state: [B, T, H]
        last_hidden = self.backbone(input_values=input_values, attention_mask=attention_mask).last_hidden_state
        # mean pool over time with mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts
        else:
            pooled = last_hidden.mean(dim=1)
        logits = self.classifier(pooled)  # [B, C]
        return logits

