"""Validation helper for trained TinyESPNet checkpoints."""

import argparse
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from config.config import TrainingConfig
from model.model import AudioDataset, TinyESPNet


def evaluate(weights_path: str, batch_size: Optional[int] = None, device_str: Optional[str] = None) -> None:
    cfg = TrainingConfig()
    if batch_size is not None:
        cfg.BATCH_SIZE = batch_size

    device = torch.device(device_str) if device_str else torch.device(cfg.DEVICE)

    df_valid = pd.read_csv(cfg.PROCESSED_VALID_METADATA)
    dataset = AudioDataset(df_valid, cfg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = TinyESPNet(
        num_classes=2,
        base_channels=cfg.base_channels,
        classifier_hidden=cfg.classifier_hidden,
        dropout=cfg.model_dropout,
    ).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = model(waveforms)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    if not all_labels:
        print("Validation loader is empty.")
        return

    labels_tensor = torch.cat(all_labels)
    preds_tensor = torch.cat(all_preds)

    avg_loss = total_loss / labels_tensor.size(0)
    acc = accuracy_score(labels_tensor, preds_tensor)
    precision = precision_score(labels_tensor, preds_tensor, zero_division=0)
    recall = recall_score(labels_tensor, preds_tensor, zero_division=0)
    f1 = f1_score(labels_tensor, preds_tensor, zero_division=0)

    print("--- TinyESPNet Validation Metrics ---")
    print(f"Weights: {weights_path}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TinyESPNet on validation set")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    cfg = TrainingConfig()
    weights_path = args.weights or str(cfg.MODEL_EXPORT)

    evaluate(weights_path, batch_size=args.batch_size, device_str=args.device)


if __name__ == "__main__":
    main()
