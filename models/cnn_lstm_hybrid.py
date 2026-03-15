"""
CNN-LSTM Hybrid model for EMG gesture classification.
CNN extracts local spatial/temporal features; LSTM captures long-range dependencies.
"""

import torch
import torch.nn as nn


class CNNLSTMHybrid(nn.Module):
    """
    Hybrid architecture:
      EMG (B,W,C) → 1D-CNN feature extractor → LSTM sequence modeller → classifier

    Input  : (B, W, C)  — window_size × n_channels
    Output : (B, n_classes)
    """

    def __init__(
        self,
        n_channels:  int   = 10,
        window_size: int   = 200,
        n_classes:   int   = 52,
        cnn_filters: int   = 128,
        lstm_hidden: int   = 256,
        lstm_layers: int   = 2,
        dropout:     float = 0.4,
    ):
        super().__init__()

        # ── 1-D CNN front-end ──────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Stage 1: local muscle burst detection
            nn.Conv1d(n_channels, cnn_filters // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cnn_filters // 2),
            nn.ReLU(inplace=True),

            # Stage 2: inter-channel interaction
            nn.Conv1d(cnn_filters // 2, cnn_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(inplace=True),

            # Stage 3: higher-level patterns
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(inplace=True),
        )

        cnn_out_channels = cnn_filters

        # ── Bidirectional LSTM back-end ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = cnn_out_channels,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
            bidirectional = True,
        )

        lstm_out_dim = lstm_hidden * 2   # bidirectional

        # ── Multi-scale pooling ────────────────────────────────────────
        # Concatenate: last hidden + mean-pooled + max-pooled
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, W, C)
        """
        # CNN: (B, W, C) → (B, C, W) → CNN → (B, cnn_filters, W)
        x_cnn = self.cnn(x.permute(0, 2, 1))           # (B, F, W)

        # Prepare for LSTM: (B, W, F)
        x_lstm_in = x_cnn.permute(0, 2, 1)              # (B, W, F)
        lstm_out, (hn, _) = self.lstm(x_lstm_in)        # (B, W, H*2)

        # Multi-scale aggregation
        last     = lstm_out[:, -1, :]                    # (B, H*2)
        mean_p   = lstm_out.mean(dim=1)                  # (B, H*2)
        max_p    = lstm_out.max(dim=1).values            # (B, H*2)

        combined = torch.cat([last, mean_p, max_p], dim=1)  # (B, H*6)
        return self.classifier(combined)
