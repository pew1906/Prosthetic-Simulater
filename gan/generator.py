"""
Conditional GAN – Generator
Transforms a latent noise vector (conditioned on gesture label) into a
realistic EMG window of shape (window_size, n_channels).
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1-D residual block with two convolutional layers and skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class EMGGenerator(nn.Module):
    """
    Conditional generator for EMG windows.

    Architecture
    ------------
    1. Embed gesture label → concatenate with noise vector
    2. Dense projection → reshape to (channels, time/8)
    3. Three 1-D transposed-conv upsampling stages
    4. Residual refinement blocks
    5. Final conv → tanh → (window_size, n_channels)
    """

    def __init__(
        self,
        latent_dim: int   = 100,
        n_classes: int    = 52,
        window_size: int  = 200,
        n_channels: int   = 10,
        base_filters: int = 128,
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.n_classes   = n_classes
        self.window_size = window_size
        self.n_channels  = n_channels

        # Label embedding
        self.label_emb = nn.Embedding(n_classes, n_classes)

        # Initial dense layer: (latent_dim + n_classes) → (base_filters*4 × init_len)
        self.init_len  = window_size // 8          # 200//8 = 25
        self.fc        = nn.Sequential(
            nn.Linear(latent_dim + n_classes, base_filters * 4 * self.init_len),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsampling backbone
        self.upsample = nn.Sequential(
            # (base_filters*4, 25) → (base_filters*2, 50)
            nn.ConvTranspose1d(base_filters * 4, base_filters * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (base_filters*2, 50) → (base_filters, 100)
            nn.ConvTranspose1d(base_filters * 2, base_filters,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # (base_filters, 100) → (base_filters//2, 200)
            nn.ConvTranspose1d(base_filters, base_filters // 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_filters // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Residual refinement
        self.res_blocks = nn.Sequential(
            ResBlock1D(base_filters // 2),
            ResBlock1D(base_filters // 2),
        )

        # Output projection: → n_channels, tanh
        self.out_conv = nn.Sequential(
            nn.Conv1d(base_filters // 2, n_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noise  : (B, latent_dim)
        labels : (B,) long

        Returns
        -------
        emg    : (B, window_size, n_channels)
        """
        lbl_emb = self.label_emb(labels)                        # (B, n_classes)
        x       = torch.cat([noise, lbl_emb], dim=1)            # (B, latent+n_cls)
        x       = self.fc(x)                                     # (B, filters*4*len)
        x       = x.view(x.size(0), -1, self.init_len)          # (B, C, init_len)
        x       = self.upsample(x)                               # (B, C/2, W)
        # Trim/pad to exact window_size
        if x.size(2) != self.window_size:
            x   = x[:, :, :self.window_size]
        x       = self.res_blocks(x)                             # (B, C/2, W)
        x       = self.out_conv(x)                               # (B, n_ch, W)
        return x.permute(0, 2, 1)                                # (B, W, n_ch)
