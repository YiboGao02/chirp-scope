"""Quantise a trained TinyESPNet model and export an ESP-DL artefact."""

import argparse
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor

from config.config import TrainingConfig
from model.model import TinyESPNet


class WaveformDataset(Dataset):
    """Dataset feeding cached waveform tensors for calibration and evaluation."""

    def __init__(self, cfg: TrainingConfig, csv_path: str, return_label: bool, limit: Optional[int] = None):
        self.cfg = cfg
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit)
        self.records = df.reset_index(drop=True)
        self.return_label = return_label
        self.root = os.getcwd()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        rel_path = os.path.normpath(row["path"])
        sample_path = os.path.join(self.root, rel_path)
        data = torch.load(sample_path, map_location="cpu")
        waveform = data["waveform"].float()
        target_len = int(self.cfg.SEGMENT_SAMPLES)
        if waveform.numel() < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.numel()))
        elif waveform.numel() > target_len:
            waveform = waveform[:target_len]
        if self.return_label:
            return waveform, int(row["label"])
        return waveform


def build_model(cfg: TrainingConfig, device: torch.device, weights_path: str) -> TinyESPNet:
    model = TinyESPNet(
        num_classes=2,
        base_channels=cfg.base_channels,
        classifier_hidden=cfg.classifier_hidden,
        dropout=cfg.model_dropout,
    ).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def build_collate(device: torch.device) -> Callable[[List[torch.Tensor]], torch.Tensor]:
    def _collate(batch: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch).to(device)

    return _collate


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = model(waveforms)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def evaluate_executor(executor: TorchExecutor, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = executor(waveforms)[0]
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantise TinyESPNet and export ESP-DL model")
    parser.add_argument("--output", type=str, default="model.espdl", help="Output ESP-DL filename")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained TinyESPNet weights")
    parser.add_argument("--calib-split", choices=["train", "valid"], default="train", help="Metadata split for calibration")
    parser.add_argument("--calib-count", type=int, default=256, help="Number of samples for calibration")
    parser.add_argument("--eval-split", choices=["train", "valid"], default="valid", help="Metadata split for evaluation")
    parser.add_argument("--eval-count", type=int, default=512, help="Number of samples for evaluation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for calibration/evaluation")
    parser.add_argument("--calib-steps", type=int, default=64, help="Calibration steps for espdl_quantize_torch")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bit width")
    parser.add_argument("--target", type=str, default="esp32s3", choices=["c", "esp32s3", "esp32p4"], help="Target ESP-DL platform")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--skip-export", action="store_true", help="Skip ESP-DL export (debug only)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity for espdl_quantize_torch")
    parser.add_argument("--test-index", type=int, default=None, help="Embed one calibration sample as test input/output")
    args = parser.parse_args()

    cfg = TrainingConfig()
    device = torch.device(args.device)

    weights_path = args.weights or str(cfg.MODEL_EXPORT)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    calib_csv = cfg.PROCESSED_TRAIN_METADATA if args.calib_split == "train" else cfg.PROCESSED_VALID_METADATA
    eval_csv = cfg.PROCESSED_TRAIN_METADATA if args.eval_split == "train" else cfg.PROCESSED_VALID_METADATA

    calib_dataset = WaveformDataset(cfg, calib_csv, return_label=False, limit=args.calib_count)
    eval_dataset = WaveformDataset(cfg, eval_csv, return_label=True, limit=args.eval_count)

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate(device),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(cfg, device, weights_path)

    base_loss, base_acc = evaluate_model(model, eval_loader, device)
    print(f"Original model -> loss: {base_loss:.4f}, acc: {base_acc:.4f}")

    test_inputs = None
    test_outputs = None
    if args.test_index is not None and len(calib_dataset) > 0:
        idx = max(0, min(args.test_index, len(calib_dataset) - 1))
        sample = calib_dataset[idx]
        waveform = sample[0] if isinstance(sample, tuple) else sample
        waveform = waveform.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(waveform)
        test_inputs = [waveform.cpu()]
        test_outputs = [logits.cpu()]

    calib_steps = min(args.calib_steps, max(1, len(calib_loader)))
    quant_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=args.output,
        calib_dataloader=calib_loader,
        calib_steps=calib_steps,
        input_shape=[1, int(cfg.SEGMENT_SAMPLES)],
        inputs=test_inputs,
        outputs=test_outputs,
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=None,
        dispatching_override=None,
        device=device.type,
        error_report=True,
        skip_export=args.skip_export,
        export_test_values=True,
        verbose=args.verbose,
    )

    executor = TorchExecutor(graph=quant_graph, device=device.type)
    quant_loss, quant_acc = evaluate_executor(executor, eval_loader, device)
    print(f"Quantized model -> loss: {quant_loss:.4f}, acc: {quant_acc:.4f}")


if __name__ == "__main__":
    main()
