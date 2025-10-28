"""Simple training metric plotting helper."""

import os
from typing import Dict, Iterable

import matplotlib.pyplot as plt


def plot_metrics(history: Dict[str, Iterable[float]], save_path: str = ".") -> None:
    """Plot loss and validation metrics over training."""
    os.makedirs(save_path, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Training and Validation Metrics", fontsize=16)

    ax1.plot(epochs, history["train_loss"], "g", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "b", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_acc"], "r", label="Validation Accuracy")
    ax2.plot(epochs, history["val_precision"], "c", label="Validation Precision")
    ax2.plot(epochs, history["val_recall"], "m", label="Validation Recall")
    ax2.plot(epochs, history["val_f1"], "y", label="Validation F1")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(save_path, "training_metrics.png")
    plt.savefig(plot_filename)
    print(f"Metric plot saved to {plot_filename}")
    plt.close()
