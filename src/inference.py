from __future__ import annotations
import os, json, argparse
import torch
import torchaudio
import yaml
from transformers import Wav2Vec2FeatureExtractor
from .model import Wav2Vec2Classifier


@torch.no_grad()
def predict_one(wav_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    label2id = ckpt["label2id"]; id2label = {int(k): v for k, v in ckpt["id2label"].items()}
    sample_rate = ckpt["config"]["audio"]["sample_rate"]
    max_seconds = ckpt["config"]["audio"]["max_seconds"]
    max_len = sample_rate * max_seconds

    # Load and prepare audio
    wav, sr = torchaudio.load(os.path.expanduser(wav_path))  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    wav = wav.squeeze(0)  # [T]
    if wav.numel() >= max_len:
        start = (wav.numel() - max_len) // 2
        wav = wav[start : start + max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.numel()))

    # Features
    fe = Wav2Vec2FeatureExtractor.from_pretrained(ckpt["backbone_name"], sampling_rate=sample_rate)
    inputs = fe([wav.numpy()], sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Model
    model = Wav2Vec2Classifier(ckpt["backbone_name"], num_classes=len(label2id), freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    logits = model(input_values=inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device))
    probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    top = torch.topk(probs, k=min(3, probs.numel()))
    results = [(id2label[int(i)], float(probs[i])) for i in top.indices]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()
    preds = predict_one(args.wav, args.checkpoint)
    print("Top predictions:")
    for label, p in preds:
        print(f"{label:18s}  {p:.3f}")

