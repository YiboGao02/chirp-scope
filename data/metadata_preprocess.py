"""Utilities to build and split metadata for bird-song training."""

from __future__ import annotations

from pathlib import Path
import sys

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Make project root importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DataConfig


def build_metadata(cfg: DataConfig) -> pd.DataFrame:
    """Create a shuffled metadata frame covering positive bird calls and noise."""
    records = []

    if not cfg.TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training metadata not found: {cfg.TRAIN_CSV}")
    df_meta = pd.read_csv(cfg.TRAIN_CSV)
    for filename in tqdm(df_meta["filename"], desc="Collecting bird audio"):
        audio_path = cfg.TRAIN_AUDIO_DIR / filename
        if audio_path.exists():
            records.append({"path": str(audio_path), "label": cfg.POSITIVE_LABEL})

    noise_root = cfg.NOISE_ROOT_DIR
    if noise_root.exists():
        for subdir in sorted(p for p in noise_root.iterdir() if p.is_dir()):
            for audio_file in tqdm(
                subdir.iterdir(),
                desc=f"Collecting noise from {subdir.name}",
                leave=False,
            ):
                if audio_file.suffix.lower() in cfg.VALID_AUDIO_EXTENSIONS:
                    records.append({"path": str(audio_file), "label": cfg.NEGATIVE_LABEL})

    if not records:
        raise RuntimeError("No audio files found while building metadata.")

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=cfg.RANDOM_STATE).reset_index(drop=True)

    if df["label"].nunique() > 1:
        counts = df["label"].value_counts()
        weights = 1.0 / counts
        df["sample_weight"] = df["label"].map(weights)
        df["sample_weight"] = df["sample_weight"] / df["sample_weight"].sum() * len(df)
    else:
        df["sample_weight"] = 1.0

    return df


def split_metadata(
    df: pd.DataFrame, cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata into train/valid/test according to config proportions."""
    stratify = df["label"] if df["label"].nunique() > 1 else None
    df_rest, df_test = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=stratify,
    )

    remaining = 1 - cfg.TEST_SIZE
    if remaining <= 0:
        raise ValueError("TEST_SIZE leaves no samples for training/validation.")
    valid_ratio = cfg.VALID_SIZE / remaining

    stratify_rest = df_rest["label"] if df_rest["label"].nunique() > 1 else None
    df_train, df_valid = train_test_split(
        df_rest,
        test_size=valid_ratio,
        random_state=cfg.RANDOM_STATE,
        stratify=stratify_rest,
    )
    return df_train, df_valid, df_test


def export_splits(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, cfg: DataConfig) -> None:
    """Write split metadata frames to disk."""
    train.to_csv(cfg.TRAIN_METADATA, index=False)
    valid.to_csv(cfg.VALID_METADATA, index=False)
    test.to_csv(cfg.TEST_METADATA, index=False)


def main() -> None:
    cfg = DataConfig()
    df_full = build_metadata(cfg)
    train_df, valid_df, test_df = split_metadata(df_full, cfg)
    export_splits(train_df, valid_df, test_df, cfg)

    print(
        f"Saved metadata splits -> "
        f"train:{len(train_df)} valid:{len(valid_df)} test:{len(test_df)}"
    )


if __name__ == "__main__":
    main()
