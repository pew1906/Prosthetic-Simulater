"""
Utility helpers for the prosthetic simulator project.
Covers: feature extraction, dataset statistics, reproducibility, logging.
"""

import os
import random
import time
import numpy as np
import torch
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA: {name}")
        return "cuda"
    print("[Device] Using CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def dataset_summary(X: np.ndarray, y: np.ndarray, name: str = "Dataset"):
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Samples    : {len(X)}")
    print(f"  Shape      : {X.shape}")
    print(f"  Classes    : {len(classes)}  (labels {classes[0]}–{classes[-1]})")
    print(f"  Per class  : min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")
    print(f"  Signal     : mean={X.mean():.4f}  std={X.std():.4f}  "
          f"min={X.min():.3f}  max={X.max():.3f}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Time-domain feature extraction (for classic ML baselines)
# ---------------------------------------------------------------------------

def extract_td_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract 5 classical time-domain features per channel:
    MAV, ZC, SSC, WL, RMS

    Parameters
    ----------
    windows : (N, W, C)

    Returns
    -------
    features : (N, 5 * C)
    """
    N, W, C = windows.shape
    eps     = 1e-8
    feats   = []

    for i in range(N):
        w = windows[i]                              # (W, C)
        mav = np.mean(np.abs(w), axis=0)            # mean absolute value
        rms = np.sqrt(np.mean(w ** 2, axis=0))      # root mean square

        # Zero crossing count (with dead band threshold)
        th  = 0.01
        zc  = np.sum(
            ((w[:-1] * w[1:] < 0) &
             (np.abs(w[:-1] - w[1:]) >= th)), axis=0).astype(float)

        # Slope sign changes
        diff = np.diff(w, axis=0)
        ssc  = np.sum(
            ((diff[:-1] * diff[1:] < 0) &
             (np.abs(diff[:-1] - diff[1:]) >= th)), axis=0).astype(float)

        # Waveform length
        wl   = np.sum(np.abs(diff), axis=0)

        feats.append(np.concatenate([mav, rms, zc, ssc, wl]))

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Simple timer context manager
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self.t0
        print(f"[Timer] {self.label}: {elapsed:.2f}s")
        self.elapsed = elapsed


# ---------------------------------------------------------------------------
# Model parameter count
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters — total={total:,}  trainable={trainable:,}")
    return trainable


# ---------------------------------------------------------------------------
# Quick sanity check: forward pass
# ---------------------------------------------------------------------------

def sanity_check(model: torch.nn.Module, window_size: int = 200,
                 n_channels: int = 10, batch_size: int = 4,
                 is_2d: bool = False) -> bool:
    """Run a dummy forward pass to verify model shapes."""
    model.eval()
    try:
        if is_2d:
            x = torch.randn(batch_size, n_channels, 17, 24)
        else:
            x = torch.randn(batch_size, window_size, n_channels)
        with torch.no_grad():
            out = model(x)
        print(f"  Sanity check OK — output shape: {tuple(out.shape)}")
        return True
    except Exception as e:
        print(f"  Sanity check FAILED: {e}")
        return False
