"""Reusable dataset and model definitions for ESP-friendly bird song training."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset wrapper for cached waveform tensors generated during preprocessing."""

    def __init__(self, dataframe, cfg):
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pt_path = row["path"]
        label = int(row["label"])
        try:
            data = torch.load(pt_path, map_location="cpu")
            waveform = data["waveform"]
            return waveform, torch.tensor(label, dtype=torch.long)
        except Exception as exc:
            print(f"Warning: failed to load cached tensor {pt_path}: {exc}")
            return torch.zeros(self.cfg.SEGMENT_SAMPLES, dtype=torch.float32), torch.tensor(0, dtype=torch.long)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable conv block used in the TinyESPNet backbone."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ResidualTemporalBlock(nn.Module):
    """Residual temporal block with depthwise + pointwise convolutions."""

    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.bn(y)
        y = self.dropout(y)
        return self.act(y + x)


class TinyESPNet(nn.Module):
    """Compact 1D conv network tailored for ESP-DL deployments."""

    def __init__(self, num_classes, base_channels=12, classifier_hidden=24, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=15, stride=6, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=4, padding=2),
        )
        mid_channels = base_channels * 2
        high_channels = base_channels * 4
        self.block1 = DepthwiseSeparableConv1d(base_channels, mid_channels, kernel_size=5, stride=2)
        self.block2 = DepthwiseSeparableConv1d(mid_channels, high_channels, kernel_size=5, stride=2)
        self.block3 = DepthwiseSeparableConv1d(high_channels, high_channels, kernel_size=3, stride=2)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(high_channels, kernel_size=5, dilation=1, dropout=dropout),
            ResidualTemporalBlock(high_channels, kernel_size=5, dilation=2, dropout=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(high_channels, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.temporal(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
