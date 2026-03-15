"""
1D Convolutional Neural Network for EMG gesture classification.
Treats each EMG window as a multi-channel 1-D time series.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable conv to reduce parameters."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class CNN1D(nn.Module):
    """
    1-D CNN for temporal EMG feature extraction.

    Input  : (B, W, C)  — window_size × n_channels
    Output : (B, n_classes)
    """

    def __init__(
        self,
        n_channels:  int = 10,
        window_size: int = 200,
        n_classes:   int = 52,
        base_filters: int = 64,
        dropout:     float = 0.5,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                             # W → W/2
        )

        self.block1 = nn.Sequential(
            DepthwiseSeparableConv1d(base_filters, base_filters * 2, 5, padding=2),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                             # W/2 → W/4
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv1d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                             # W/4 → W/8
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv1d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.BatchNorm1d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),                     # → 4 time steps
        )

        flat_dim = base_filters * 8 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, W, C) → (B, C, W) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
