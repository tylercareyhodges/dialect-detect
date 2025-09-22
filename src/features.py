from __future__ import annotations
from typing import Callable, Dict, List
import torch
from transformers import Wav2Vec2FeatureExtractor


def make_feature_extractor(backbone_name: str, sample_rate: int) -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor.from_pretrained(backbone_name, sampling_rate=sample_rate)


def collate_fn_factory(feature_extractor: Wav2Vec2FeatureExtractor, label2id: Dict[str, int]) -> Callable:
    """
    Pads variable-length waveforms and creates attention masks using the HF feature extractor.
    """
    def collate(batch: List[Dict]):
        waves = [b["waveform"].numpy() for b in batch]  # list of [T]
        labels = torch.tensor([label2id[b["label"]] for b in batch], dtype=torch.long)
        inputs = feature_extractor(
            waves,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_values": inputs.input_values,      # [B, T]
            "attention_mask": inputs.attention_mask,  # [B, T]
            "labels": labels,
        }
    return collate

