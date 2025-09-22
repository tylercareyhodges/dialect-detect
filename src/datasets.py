from __future__ import annotations
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class AudioDataset(Dataset):
    """
    Loads waveforms from a metadata CSV with columns: path,label,speaker_id.
    Applies: resampling to target sample_rate, mono mixdown, simple crop/pad.
    Cropping is random during training; centre crop during evaluation.
    """

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        max_seconds: int = 5,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        required = {"path", "label", "speaker_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        self.sample_rate = sample_rate
        self.max_len = sample_rate * max_seconds
        self.train = train
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return wav
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
        return self._resamplers[sr](wav)

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n >= self.max_len:
            # random crop for train; centre for eval
            if self.train:
                start = torch.randint(0, n - self.max_len + 1, (1,)).item()
            else:
                start = max(0, (n - self.max_len) // 2)
            return wav[:, start : start + self.max_len]
        # pad to max_len
        pad = self.max_len - n
        return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.expanduser(str(row["path"]))
        label = str(row["label"])
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)  # mono
        wav = self._resample(wav, sr)
        wav = self._crop_or_pad(wav)  # [1, T]
        return {
            "waveform": wav.squeeze(0),  # [T]
            "label": label,
            "speaker_id": str(row["speaker_id"]),
        }


def build_label_mapping(labels_in_config: List[str], df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create a label ↔ id mapping; prefers config labels to keep order stable."""
    unique_in_data = [l for l in df["label"].astype(str).unique().tolist()]
    labels = labels_in_config if labels_in_config else sorted(unique_in_data)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def compute_class_weights(train_df: pd.DataFrame, label2id: Dict[str, int]) -> torch.Tensor:
    counts = train_df["label"].map(label2id).value_counts().sort_index()
    counts = counts.reindex(range(len(label2id)), fill_value=0).astype(float)
    total = counts.sum()
    # Inverse frequency normalised so average weight ≈ 1.0
    weights = total / (len(counts) * counts.clip(min=1.0))
    return torch.tensor(weights.values, dtype=torch.float32)

