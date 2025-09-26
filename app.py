# app.py
from __future__ import annotations
import os
import torch
import torchaudio
import gradio as gr
import numpy as np

from src.model import Wav2Vec2Classifier  # uses your fixed mean-pooling forward
import yaml
import json

# ---- Config ----
CKPT_PATH = "checkpoints/best.pt"      # your trained checkpoint
CFG_PATH  = "configs/base.yaml"        # to read labels if needed
TARGET_SR = 16000
TARGET_SEC = 5                         # your training length
TOPK = 3

# ---- Load checkpoint + labels ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device)

# Prefer labels from checkpoint; otherwise fall back to YAML
id2label = {int(k): v for k, v in ckpt.get("id2label", {}).items()}
if not id2label:
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    labels = cfg["labels"]
    id2label = {i: l for i, l in enumerate(labels)}

label_list = [id2label[i] for i in sorted(id2label.keys())]
num_classes = len(label_list)

# Build model and load weights
backbone_name = ckpt.get("backbone_name", "facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Classifier(
    backbone_name=backbone_name,
    num_classes=num_classes,
    freeze_backbone=True,     # inference only; doesn’t matter
    hidden_dim=ckpt.get("config", {}).get("model", {}).get("hidden_dim", 256),
    dropout=ckpt.get("config", {}).get("model", {}).get("dropout", 0.4),
)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.to(device).eval()

# Preprocessors
resamplers = {}  # cache per-sr resamplers

def crop_or_pad(wav: torch.Tensor, seconds: int = TARGET_SEC, sr: int = TARGET_SR) -> torch.Tensor:
    """Center-crop or right-pad to fixed length [T]."""
    max_len = seconds * sr
    n = wav.shape[-1]
    if n >= max_len:
        start = max(0, (n - max_len) // 2)
        return wav[..., start:start+max_len]
    pad = max_len - n
    return torch.nn.functional.pad(wav, (0, pad))

def prepare_waveform(audio: tuple[int, np.ndarray] | None) -> torch.Tensor:
    """
    Gradio mic provides (sample_rate, np.ndarray [T] or [T, C]).
    Returns mono, 16 kHz, 5 s tensor [T].
    """
    if audio is None:
        raise gr.Error("No audio received. Please record again.")
    sr, data = audio
    # to torch [C, T]
    if data.ndim == 1:
        wav = torch.from_numpy(data).float().unsqueeze(0)
    else:
        wav = torch.from_numpy(data).float().T  # [T,C] -> [C,T]
        wav = wav.mean(0, keepdim=True)         # mono
    # resample if needed
    if sr != TARGET_SR:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
        wav = resamplers[sr](wav)
    # crop/pad and return [T]
    wav = crop_or_pad(wav.squeeze(0), TARGET_SEC, TARGET_SR)
    return wav

@torch.no_grad()
def predict(audio: tuple[int, np.ndarray] | None):
    wav = prepare_waveform(audio).unsqueeze(0).to(device)  # [1, T]
    logits = model(input_values=wav)                       # [1, C]
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Top-k
    idx = probs.argsort()[::-1][:TOPK]
    result = {label_list[i]: float(probs[i]) for i in idx}
    # Also return full distribution (optional)
    table = [[label_list[i], float(probs[i])] for i in np.argsort(-probs)]
    return result, table

# ---- UI ----
DESC = (
    "Say this sentence clearly for ~6 seconds:\n\n"
    "“I saw seven swans swimming by the silver shore.”\n\n"
    "Then release the record button — the model will guess your UK dialect."
)

with gr.Blocks() as demo:
    gr.Markdown("# UK Dialect Demo (Wav2Vec2)\n" + DESC)

    with gr.Row():
        audio = gr.Audio(sources=["microphone"], type="numpy", label="Record ~6s (mic)", waveform_options={"show_recording_waveform": True})
    btn = gr.Button("Analyze")
    topk_out = gr.JSON(label="Top predictions (probabilities)")
    table_out = gr.Dataframe(headers=["Dialect", "Probability"], label="All classes (sorted)", interactive=False)

    btn.click(fn=predict, inputs=audio, outputs=[topk_out, table_out])

if __name__ == "__main__":
    demo.launch(share=True)
