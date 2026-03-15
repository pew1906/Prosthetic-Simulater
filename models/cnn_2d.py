"""
2D CNN for EMG gesture classification.
Each window is converted to a Short-Time Fourier Transform (STFT) spectrogram,
giving a (freq_bins, time_frames, n_channels) image processed by 2-D convolutions.
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# STFT / spectrogram conversion (runs on CPU with numpy)
# ---------------------------------------------------------------------------

def emg_to_spectrogram(
    windows: np.ndarray,
    n_fft:   int = 32,
    hop:     int = 8,
) -> np.ndarray:
    """
    Convert EMG windows to magnitude spectrograms.

    Parameters
    ----------
    windows : (N, W, C)  float32
    n_fft   : FFT size
    hop     : hop length between frames

    Returns
    -------
    specs   : (N, freq_bins, time_frames, C)  float32
    """
    from scipy.signal import stft as scipy_stft

    N, W, C    = windows.shape
    freq_bins  = n_fft // 2 + 1                   # 17 for n_fft=32
    time_frames = (W - n_fft) // hop + 1          # approx

    results = []
    for i in range(N):
        ch_specs = []
        for ch in range(C):
            sig = windows[i, :, ch]
            _, _, Zxx = scipy_stft(sig, nperseg=n_fft, noverlap=n_fft - hop)
            mag = np.abs(Zxx).astype(np.float32)   # (freq_bins, time_frames)
            ch_specs.append(mag)
        # Stack: (C, freq_bins, time_frames) → (freq_bins, time_frames, C)
        spec = np.stack(ch_specs, axis=0)          # (C, F, T)
        results.append(spec)

    specs = np.array(results, dtype=np.float32)    # (N, C, F, T)
    # Normalize per sample
    mins  = specs.min(axis=(2, 3), keepdims=True)
    maxs  = specs.max(axis=(2, 3), keepdims=True)
    specs = (specs - mins) / (maxs - mins + 1e-8)
    return specs                                   # (N, C, F, T)


# ---------------------------------------------------------------------------
# 2-D CNN model
# ---------------------------------------------------------------------------

class CNN2D(nn.Module):
    """
    2-D CNN that operates on STFT spectrograms of EMG signals.

    Input  : (B, C, F, T)  — channels × freq_bins × time_frames
    Output : (B, n_classes)
    """

    def __init__(
        self,
        n_channels:  int = 10,
        n_classes:   int = 52,
        base_filters: int = 32,
        dropout:     float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(n_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        flat_dim = base_filters * 8 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, F, T)
        x = self.features(x)
        return self.classifier(x)
