from __future__ import annotations
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from .datasets import AudioDataset
from .features import make_feature_extractor, collate_fn_factory
from .model import Wav2Vec2Classifier


@torch.no_grad()
def run_eval(cfg, checkpoint_path: str, split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load mapping from checkpoint (source of truth)
    ckpt = torch.load(checkpoint_path, map_location=device)
    label2id = ckpt["label2id"]; id2label = {int(k): v for k, v in ckpt["id2label"].items()}

    csv_path = cfg["paths"][f"{split}_csv"]
    ds = AudioDataset(csv_path, sample_rate=cfg["audio"]["sample_rate"], max_seconds=cfg["audio"]["max_seconds"], train=False)
    feat = make_feature_extractor(ckpt["backbone_name"], cfg["audio"]["sample_rate"])
    collate = collate_fn_factory(feat, label2id)
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                    num_workers=cfg["training"]["num_workers"], collate_fn=collate)

    model = Wav2Vec2Classifier(ckpt["backbone_name"], num_classes=len(label2id), freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    crit = nn.CrossEntropyLoss()
    losses, ys, ps = [], [], []

    for batch in tqdm(dl, desc=f"eval-{split}", leave=False):
        x = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        y = batch["labels"].to(device)
        logits = model(**x)
        loss = crit(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(preds.cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    print(f"{split}: loss {np.mean(losses):.4f} acc {acc:.4f} macroF1 {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(ys, ps, target_names=[id2label[i] for i in range(len(id2label))], digits=3))

    # Save report + confusion matrix
    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)
    report_txt = os.path.join(cfg["paths"]["reports_dir"], f"{split}_classification_report.txt")
    with open(report_txt, "w") as f:
        f.write(classification_report(ys, ps, target_names=[id2label[i] for i in range(len(id2label))], digits=3))
    print(f"Saved report → {report_txt}")

    cm = confusion_matrix(ys, ps, labels=list(range(len(id2label))))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix ({split})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(id2label))); ax.set_yticks(range(len(id2label)))
    ax.set_xticklabels([id2label[i] for i in range(len(id2label))], rotation=45, ha="right")
    ax.set_yticklabels([id2label[i] for i in range(len(id2label))])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    cm_path = os.path.join(cfg["paths"]["reports_dir"], f"{split}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix → {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val","test"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_eval(cfg, args.checkpoint, args.split)

