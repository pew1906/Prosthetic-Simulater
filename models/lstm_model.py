"""
Bidirectional LSTM for EMG gesture classification.
Models the temporal dependencies in multi-channel EMG signals.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Stacked bidirectional LSTM.

    Input  : (B, W, C)  — window_size × n_channels
    Output : (B, n_classes)
    """

    def __init__(
        self,
        n_channels:  int   = 10,
        hidden_size: int   = 256,
        num_layers:  int   = 3,
        n_classes:   int   = 52,
        dropout:     float = 0.4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        directions         = 2 if bidirectional else 1

        # Input projection to larger space
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_size // 2),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size   = hidden_size // 2,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        lstm_out_dim = hidden_size * directions

        # Attention over time steps
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, W, C)
        """
        x    = self.input_proj(x)                    # (B, W, H/2)
        out, _ = self.lstm(x)                        # (B, W, H*dirs)

        # Soft attention pooling over time
        scores = self.attn(out)                      # (B, W, 1)
        weights = torch.softmax(scores, dim=1)       # (B, W, 1)
        ctx    = (out * weights).sum(dim=1)          # (B, H*dirs)

        return self.classifier(ctx)
