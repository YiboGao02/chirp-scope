"""Convert raw audio metadata into cached waveform tensors."""

import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from config.config import TrainingConfig


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

        try:
            waveform, sr = torchaudio.load(path)
            if sr != self.cfg.TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.TARGET_SR)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            segment = self._select_segment(waveform, label)
            return segment.squeeze(0), torch.tensor(label, dtype=torch.long)
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")
            return torch.zeros(self.cfg.SEGMENT_SAMPLES, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

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
        new_metadata.append({"path": new_filepath, "label": label.item()})

    return pd.DataFrame(new_metadata)


def main() -> None:
    cfg = TrainingConfig()

    print("Running waveform preprocessing...")
    train_df = pd.read_csv(cfg.TRAIN_METADATA)
    valid_df = pd.read_csv(cfg.VALID_METADATA)

    processed_root = os.path.join("processed_data")
    train_output = os.path.join(processed_root, "train")
    valid_output = os.path.join(processed_root, "valid")

    new_train = run_preprocessing(train_df, train_output, cfg)
    new_valid = run_preprocessing(valid_df, valid_output, cfg)

    new_train.to_csv(cfg.PROCESSED_TRAIN_METADATA, index=False)
    new_valid.to_csv(cfg.PROCESSED_VALID_METADATA, index=False)
    print("Waveform preprocessing complete.")


if __name__ == "__main__":
    main()
