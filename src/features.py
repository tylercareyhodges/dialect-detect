from __future__ import annotations
from typing import Callable, Dict, List
import torch
from transformers import Wav2Vec2FeatureExtractor


def make_feature_extractor(backbone_name: str, sample_rate: int) -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor.from_pretrained(backbone_name, sampling_rate=sample_rate)

from typing import Any
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: List[Dict[str, Any]], feat, label2id: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Top-level (picklable) collate function.
    - batch items look like: {"waveform": Tensor[T], "label": str, "speaker_id": str}
    - feat: your feature extractor (kept for compatibility; not required by Wav2Vec2)
    - label2id: mapping from label string -> class index
    Returns dict with keys matching your model.forward: input_values, attention_mask, labels
    """
    # Waveforms (already 16kHz, mono, fixed 5s by your dataset)
    waves = [b["waveform"] for b in batch]            # list of [T]
    # Make [B, T] (pad_sequence is harmless since most are equal length)
    inputs = pad_sequence([w for w in waves], batch_first=True)  # [B, T]

    # Attention mask (1 where real samples exist). With fixed-length, this is all ones.
    attn = torch.ones_like(inputs, dtype=torch.long)  # [B, T]

    labels = torch.tensor([label2id[b["label"]] for b in batch], dtype=torch.long)

    return {
        "input_values": inputs,      # [B, T]
        "attention_mask": attn,      # [B, T]
        "labels": labels             # [B]
    }

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

