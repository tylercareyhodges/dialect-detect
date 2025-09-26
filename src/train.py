from __future__ import annotations
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from .datasets import AudioDataset, build_label_mapping, compute_class_weights
from .model import Wav2Vec2Classifier

from functools import partial
from .features import make_feature_extractor, collate_fn


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimiser, device, grad_accum_steps=1):
    model.train()
    losses = []
    all_y, all_p = [], []
    optimiser.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        x = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        y = batch["labels"].to(device)
        logits = model(**x)
        loss = criterion(logits, y) / grad_accum_steps
        loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        losses.append(loss.item() * grad_accum_steps)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(preds.cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p, average="macro")
    return float(np.mean(losses)), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_y, all_p = [], []
    for batch in tqdm(loader, desc="val", leave=False):
        x = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        y = batch["labels"].to(device)
        logits = model(**x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_y.extend(y.cpu().tolist())
        all_p.extend(preds.cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p, average="macro")
    return float(np.mean(losses)), acc, f1


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    # Load DataFrames
    train_df = pd.read_csv(cfg["paths"]["train_csv"])
    val_df = pd.read_csv(cfg["paths"]["val_csv"])

    # Label mapping
    label2id, id2label = build_label_mapping(cfg.get("labels", []), train_df)
    with open(os.path.join(cfg["paths"]["artifacts_dir"], "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # Datasets
    ds_train = AudioDataset(
        cfg["paths"]["train_csv"],
        sample_rate=cfg["audio"]["sample_rate"],
        max_seconds=cfg["audio"]["max_seconds"],
        train=True,
    )
    ds_val = AudioDataset(
        cfg["paths"]["val_csv"],
        sample_rate=cfg["audio"]["sample_rate"],
        max_seconds=cfg["audio"]["max_seconds"],
        train=False,
    )

    # Feature extractor + collate
    feat = make_feature_extractor(cfg["model"]["backbone_name"], cfg["audio"]["sample_rate"])
    collate = partial(collate_fn, feat=feat, label2id=label2id)

    # Loaders
    pin = (device.type == "cuda")
    dl_train = DataLoader(
    ds_train,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    num_workers=cfg["training"]["num_workers"],
    collate_fn=collate
    )
    dl_val = DataLoader(ds_val, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=cfg["training"]["num_workers"], collate_fn=collate, pin_memory=True)

    # Model
    model = Wav2Vec2Classifier(
        backbone_name=cfg["model"]["backbone_name"],
        num_classes=len(label2id),
        freeze_backbone=cfg["model"]["freeze_backbone"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Loss
    if cfg["training"]["class_balance"] == "weights":
        class_weights = compute_class_weights(train_df, label2id).to(device)
        print("Class weights:", class_weights.cpu().numpy().round(3).tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    best_f1 = -1.0
    patience = cfg["training"]["early_stop_patience"]
    wait = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, dl_train, criterion, optimiser, device, cfg["training"]["grad_accum_steps"])
        vl_loss, vl_acc, vl_f1 = evaluate(model, dl_val, criterion, device)
        print(f"train: loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f}")
        print(f"  val: loss {vl_loss:.4f} acc {vl_acc:.4f} f1 {vl_f1:.4f}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            wait = 0
            ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "best.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone_name": cfg["model"]["backbone_name"],
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint → {ckpt_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)

