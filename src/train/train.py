"""End-to-end training entry point for the TinyESPNet classifier."""

import math
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import TrainingConfig
from model.model import AudioDataset, TinyESPNet
from utils.plotting import plot_metrics


class WarmupCosineScheduler:
    """Cosine decay learning-rate schedule with initial warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        if self.total_epochs <= 1:
            lr = self.base_lr
        elif epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", colour="green")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", colour="cyan"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "loss": avg_loss,
        "acc": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    print(
        f"Validation -> Loss:{metrics['loss']:.4f} | "
        f"Acc:{metrics['acc']:.4f} | "
        f"P:{metrics['precision']:.4f} | "
        f"R:{metrics['recall']:.4f} | "
        f"F1:{metrics['f1']:.4f}"
    )
    return metrics


def main() -> None:
    cfg = TrainingConfig()
    device = torch.device(cfg.DEVICE)
    print(f"Using device {device}")

    df_train = pd.read_csv(cfg.PROCESSED_TRAIN_METADATA)
    df_valid = pd.read_csv(cfg.PROCESSED_VALID_METADATA)

    train_loader = DataLoader(
        AudioDataset(df_train, cfg),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    valid_loader = DataLoader(
        AudioDataset(df_valid, cfg),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    model = TinyESPNet(
        num_classes=2,
        base_channels=cfg.base_channels,
        classifier_hidden=cfg.classifier_hidden,
        dropout=cfg.model_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1_000_000:.3f}M")

    label_counts = df_train["label"].value_counts()
    pos_weight = label_counts[0] / label_counts[1] if 1 in label_counts and label_counts[1] > 0 else 1.0
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    print(f"Using class weights: [1.0, {pos_weight:.2f}]")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-2)
    scheduler = WarmupCosineScheduler(optimizer, cfg.WARMUP_EPOCHS, cfg.EPOCHS, cfg.LEARNING_RATE, cfg.MIN_LR)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "lr": [],
    }
    best_f1 = 0.0

    for epoch in range(cfg.EPOCHS):
        current_lr = scheduler.step(epoch)
        print(f"\n===== Epoch {epoch + 1}/{cfg.EPOCHS} | LR: {current_lr:.6f} =====")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.GRAD_CLIP)
        val_metrics = validate_one_epoch(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["lr"].append(current_lr)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), cfg.MODEL_EXPORT)
            print(f"New best model saved with F1 {best_f1:.4f}")

    print(f"\nTraining complete. Best model saved to {cfg.MODEL_EXPORT}")
    plot_metrics(history)


if __name__ == "__main__":
    main()
