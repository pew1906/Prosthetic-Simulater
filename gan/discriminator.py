"""
Conditional GAN – Discriminator
Evaluates whether an EMG window is real or generator-produced,
conditioned on the gesture label.
"""

import torch
import torch.nn as nn


class SpectralNormConv1d(nn.Module):
    """Conv1d with spectral normalization for training stability."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=True)
        )

    def forward(self, x):
        return self.conv(x)


class EMGDiscriminator(nn.Module):
    """
    Conditional PatchGAN-style discriminator for 1-D EMG windows.

    Architecture
    ------------
    1. Embed gesture label → tile across time → concatenate with EMG
    2. Series of strided 1-D convolutions (downsampling)
    3. Global average pooling
    4. Dense → sigmoid
    """

    def __init__(
        self,
        n_classes: int    = 52,
        window_size: int  = 200,
        n_channels: int   = 10,
        base_filters: int = 64,
    ):
        super().__init__()
        self.n_classes   = n_classes
        self.window_size = window_size
        self.n_channels  = n_channels

        # Label embedding (projected to n_channels, then tiled along time)
        self.label_emb   = nn.Embedding(n_classes, n_channels)

        in_ch = n_channels * 2   # EMG channels + label channels

        self.conv_blocks = nn.Sequential(
            # Block 1
            SpectralNormConv1d(in_ch,           base_filters,     4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2
            SpectralNormConv1d(base_filters,    base_filters * 2, 4, stride=2, padding=1),
            nn.InstanceNorm1d(base_filters * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3
            SpectralNormConv1d(base_filters*2,  base_filters * 4, 4, stride=2, padding=1),
            nn.InstanceNorm1d(base_filters * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4
            SpectralNormConv1d(base_filters*4,  base_filters * 8, 4, stride=2, padding=1),
            nn.InstanceNorm1d(base_filters * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, emg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        emg    : (B, window_size, n_channels)
        labels : (B,) long

        Returns
        -------
        validity : (B, 1)  — probability of being real
        """
        # Build label map: (B, n_channels, window_size)
        lbl = self.label_emb(labels)              # (B, n_channels)
        lbl = lbl.unsqueeze(2).expand(            # (B, n_channels, W)
            -1, -1, self.window_size)

        # Transpose EMG to (B, n_channels, W)
        x   = emg.permute(0, 2, 1)               # (B, n_ch, W)

        # Concatenate along channel dimension
        x   = torch.cat([x, lbl], dim=1)         # (B, n_ch*2, W)

        x   = self.conv_blocks(x)
        x   = self.pool(x)
        return self.head(x)
