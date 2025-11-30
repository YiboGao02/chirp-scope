"""Convert raw audio metadata into cached waveform tensors."""

import os
from typing import Tuple, Optional
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pyloudnorm as pyln 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrainingConfig



def _normalize_peak(
    waveform: torch.Tensor,
    target_peak: float = 0.891,
    only_attenuate: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    if waveform.dim() != 2 or waveform.size(0) != 1:
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.dim() > 1 else waveform.unsqueeze(0)
    x = waveform.squeeze(0)
    peak = float(x.abs().max().item())
    if peak < eps:
        return waveform
    scale = target_peak / peak
    if only_attenuate:
        scale = min(scale, 1.0)
    y = x * scale
    return y.unsqueeze(0).to(waveform.dtype)


def _normalize_loudness(
    waveform: torch.Tensor,
    sr: int,
    target_lufs: float,
    fallback_rms: float,
    peak_ceiling_dbfs: float = -1.0,
    true_peak_oversample: Optional[int] = None,
) -> torch.Tensor:
    """Apply loudness normalization with peak ceiling to mono waveform [1, T].
    """
    if waveform.dim() != 2 or waveform.size(0) != 1:
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.dim() > 1 else waveform.unsqueeze(0)

    x = waveform.squeeze(0)
    x_np = x.detach().cpu().float().numpy()

    if np.max(np.abs(x_np)) < 1e-6:
        return waveform

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x_np)

    if not np.isfinite(loudness):
        rms = float(np.sqrt(np.mean(x_np**2)))
        if rms < 1e-9:
            return waveform
        gain_linear = float(fallback_rms) / rms
        y = x_np * gain_linear
        ceiling_linear = 10 ** (peak_ceiling_dbfs / 20.0)
        peak_after = float(np.max(np.abs(y)))
        if peak_after > ceiling_linear:
            y *= (ceiling_linear / peak_after) * 0.999
        y_t = torch.from_numpy(y).to(waveform.dtype)
        return y_t.unsqueeze(0)

    gain_db_lufs = float(target_lufs - loudness)

    if true_peak_oversample and true_peak_oversample > 1:
        with torch.no_grad():
            x_t = torch.from_numpy(x_np).unsqueeze(0)
            try:
                x_os = torchaudio.functional.resample(x_t, sr, sr * int(true_peak_oversample))
                current_peak = float(x_os.abs().max().item())
            except Exception:
                current_peak = float(np.max(np.abs(x_np)))
    else:
        current_peak = float(np.max(np.abs(x_np)))

    current_peak_dbfs = 20.0 * float(np.log10(max(current_peak, 1e-12)))
    allowed_gain_db_peak = float(peak_ceiling_dbfs - current_peak_dbfs)

    final_gain_db = min(gain_db_lufs, allowed_gain_db_peak)
    gain_linear = 10 ** (final_gain_db / 20.0)
    y = x_np * gain_linear

    ceiling_linear = 10 ** (peak_ceiling_dbfs / 20.0)
    peak_after = float(np.max(np.abs(y)))
    if peak_after > ceiling_linear:
        y *= (ceiling_linear / peak_after) * 0.999

    y_t = torch.from_numpy(y).to(waveform.dtype)
    return y_t.unsqueeze(0)


class WaveformDataset(Dataset):
    """Dataset that loads audio files and extracts training-ready segments."""

    def __init__(self, dataframe: pd.DataFrame, cfg: TrainingConfig):
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        path = row["path"]
        label = int(row["label"])

        waveform, sr = torchaudio.load(path)

        if sr != self.cfg.TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.TARGET_SR)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if getattr(self.cfg, "ENABLE_PEAK_NORM", False):
            waveform = _normalize_peak(
                waveform,
                getattr(self.cfg, "TARGET_PEAK", 0.891),
                getattr(self.cfg, "PEAK_ONLY_ATTENUATE", True),
            )
        elif getattr(self.cfg, "ENABLE_LOUDNESS_NORM", False):
            waveform = _normalize_loudness(
                waveform, self.cfg.TARGET_SR, getattr(self.cfg, "TARGET_LUFS", -23.0), getattr(self.cfg, "TARGET_RMS", 0.1)
            )

        segment = self._select_segment(waveform, label)
        return segment.squeeze(0), torch.tensor(label, dtype=torch.long)

    def _select_segment(self, waveform: torch.Tensor, label: int) -> torch.Tensor:
        segment_samples = self.cfg.SEGMENT_SAMPLES
        num_samples = waveform.shape[1]

        if num_samples <= segment_samples:
            return F.pad(waveform, (0, segment_samples - num_samples))

        max_start = num_samples - segment_samples
        if label == 1 and max_start > 0:
            energy_window = min(self.cfg.ENERGY_WINDOW, num_samples)
            pooled = F.avg_pool1d(
                waveform.unsqueeze(0).abs().pow(2), kernel_size=energy_window, stride=self.cfg.ENERGY_STRIDE, ceil_mode=True
            )

            if pooled.numel() > 0:
                peak_index = int(pooled.squeeze().argmax().item())
                peak_start = min(peak_index * self.cfg.ENERGY_STRIDE, max_start)
                center_offset = segment_samples // 2
                start_idx = max(0, min(peak_start - center_offset, max_start))

                jitter_limit = min(int(self.cfg.POS_JITTER_SECONDS * self.cfg.TARGET_SR), max_start)
                if jitter_limit > 0:
                    jitter = torch.randint(-jitter_limit, jitter_limit + 1, (1,)).item()
                    start_idx = max(0, min(start_idx + jitter, max_start))
            else:
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start_idx = torch.randint(0, max_start + 1, (1,)).item()

        return waveform[:, start_idx : start_idx + segment_samples]


def run_preprocessing(df: pd.DataFrame, output_dir: os.PathLike[str], cfg: TrainingConfig) -> pd.DataFrame:
    """Process audio rows into cached tensors and return new metadata."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = WaveformDataset(df, cfg)

    new_metadata = []
    for idx in tqdm(range(len(dataset)), desc=f"Processing {os.path.basename(str(output_dir))}"):
        segment, label = dataset[idx]
        original_path = df.iloc[idx]["path"]
        base_filename = os.path.splitext(os.path.basename(original_path))[0]
        new_filepath = os.path.join(output_dir, f"{base_filename}.pt")

        torch.save({"waveform": segment.clone(), "label": label.clone()}, new_filepath)

        record = {"path": new_filepath, "label": label.item()}
        if "sample_weight" in df.columns:
            record["sample_weight"] = float(df.iloc[idx].get("sample_weight", 1.0))
        new_metadata.append(record)

    return pd.DataFrame(new_metadata)


def main() -> None:
    cfg = TrainingConfig()

    print("Running waveform preprocessing...")
    train_df = pd.read_csv(cfg.TRAIN_METADATA)
    valid_df = pd.read_csv(cfg.VALID_METADATA)

    processed_root = os.path.join(os.path.dirname(__file__), "processed_data")
    train_output = os.path.join(processed_root, "train")
    valid_output = os.path.join(processed_root, "valid")

    new_train = run_preprocessing(train_df, train_output, cfg)
    new_valid = run_preprocessing(valid_df, valid_output, cfg)

    os.makedirs(Path(cfg.PROCESSED_TRAIN_METADATA).parent, exist_ok=True)
    os.makedirs(Path(cfg.PROCESSED_VALID_METADATA).parent, exist_ok=True)
    new_train.to_csv(cfg.PROCESSED_TRAIN_METADATA, index=False)
    new_valid.to_csv(cfg.PROCESSED_VALID_METADATA, index=False)
    print("Waveform preprocessing complete.")


if __name__ == "__main__":
    main()
