"""
Synthetic EMG Data Generator
Produces physiologically plausible EMG windows without requiring real NinaPro data.
Used to bootstrap training when the .mat files are not yet available.

Signal model
------------
Each channel is a burst of motor-unit action potential (MUAP) trains modelled as
amplitude-modulated bandpass noise, with gesture-specific spatial activation
patterns and frequency profiles derived from the EMG literature.
"""

import numpy as np
import os
import pickle
from typing import Tuple

# ---------------------------------------------------------------------------
# Per-gesture activation templates  (10 channels, 52 gestures)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

def _make_activation_templates(n_gestures: int = 52, n_channels: int = 10) -> np.ndarray:
    """
    Create a (n_gestures, n_channels) template matrix.
    Each row encodes which muscles are primary (high activation) or synergistic
    (mid activation) for that gesture.
    Values in [0, 1]: 0 = inactive, 1 = maximal activation.
    """
    templates = np.zeros((n_gestures, n_channels), dtype=np.float32)
    rng = np.random.default_rng(7)
    for g in range(n_gestures):
        # 2-4 primary channels
        n_primary = rng.integers(2, 5)
        primary   = rng.choice(n_channels, size=n_primary, replace=False)
        templates[g, primary] = rng.uniform(0.7, 1.0, size=n_primary)
        # 1-3 synergistic channels
        n_syn  = rng.integers(1, 4)
        others = [c for c in range(n_channels) if c not in primary]
        if others:
            syn = rng.choice(others, size=min(n_syn, len(others)), replace=False)
            templates[g, syn] = rng.uniform(0.2, 0.5, size=len(syn))
    return templates


_ACTIVATION_TEMPLATES = _make_activation_templates()


# ---------------------------------------------------------------------------
# Core signal synthesis
# ---------------------------------------------------------------------------

def _bandpass_noise(size: int, fs: float = 100.0,
                    low: float = 20.0, high: float = 45.0) -> np.ndarray:
    """White noise shaped by a Butterworth bandpass filter."""
    from scipy.signal import butter, filtfilt
    nyq  = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    noise = np.random.randn(size)
    return filtfilt(b, a, noise)


def _amplitude_envelope(size: int, onset_frac: float = 0.1,
                         offset_frac: float = 0.9,
                         ramp_frac: float = 0.05) -> np.ndarray:
    """Trapezoid envelope: ramp up → plateau → ramp down."""
    env    = np.zeros(size)
    onset  = int(onset_frac  * size)
    offset = int(offset_frac * size)
    ramp   = max(1, int(ramp_frac * size))
    env[onset:onset + ramp]  = np.linspace(0, 1, ramp)
    env[onset + ramp:offset] = 1.0
    if offset + ramp <= size:
        env[offset:offset + ramp] = np.linspace(1, 0, ramp)
    return env


def generate_emg_window(
    gesture_id: int,
    window_size: int = 200,
    n_channels: int  = 10,
    fs: float        = 100.0,
    snr_db: float    = 20.0,
    add_artifacts: bool = True,
) -> np.ndarray:
    """
    Generate one synthetic EMG window for a given gesture.

    Parameters
    ----------
    gesture_id   : int  — zero-indexed gesture label (0-51)
    window_size  : int  — number of time samples
    n_channels   : int  — number of EMG electrodes
    fs           : float — sampling rate in Hz
    snr_db       : float — signal-to-noise ratio in dB
    add_artifacts: bool  — add random motion artefacts

    Returns
    -------
    window : np.ndarray shape (window_size, n_channels) — normalised to [-1, 1]
    """
    activation = _ACTIVATION_TEMPLATES[gesture_id % 52]      # (n_channels,)
    signal_noise_ratio = 10 ** (snr_db / 20.0)

    window = np.zeros((window_size, n_channels), dtype=np.float32)
    env    = _amplitude_envelope(window_size,
                                  onset_frac=np.random.uniform(0.05, 0.15),
                                  offset_frac=np.random.uniform(0.80, 0.95))

    for ch in range(n_channels):
        amp = activation[ch]
        if amp < 0.05:
            # Mostly silent channel – just noise
            raw = 0.02 * np.random.randn(window_size)
        else:
            # Frequency band shifts slightly with activation level
            low  = max(10.0, 20.0 - amp * 5)
            high = min(49.0, 30.0 + amp * 15)
            raw  = amp * _bandpass_noise(window_size, fs, low, high) * env

        # Add thermal noise
        noise_power  = np.var(raw) / signal_noise_ratio if np.var(raw) > 1e-8 else 1e-6
        raw          = raw + np.sqrt(noise_power) * np.random.randn(window_size)

        # Optional: low-frequency motion artefact (<5 Hz)
        if add_artifacts and np.random.rand() < 0.15:
            from scipy.signal import butter, filtfilt
            nyq      = fs / 2.0
            ba       = butter(2, 4.0 / nyq, btype="low")
            artifact = 0.05 * filtfilt(*ba, np.random.randn(window_size))
            raw      = raw + artifact

        window[:, ch] = raw.astype(np.float32)

    # Per-window normalisation to [-1, 1]
    max_val = np.max(np.abs(window))
    if max_val > 1e-8:
        window /= max_val

    return window


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples_per_class: int = 200,
    n_classes: int            = 52,
    window_size: int          = 200,
    n_channels: int           = 10,
    fs: float                 = 100.0,
    snr_db: float             = 20.0,
    seed: int                 = 0,
    verbose: bool             = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a fully synthetic, class-balanced EMG dataset.

    Returns
    -------
    X : np.ndarray  shape (N, window_size, n_channels)  float32
    y : np.ndarray  shape (N,)                          int64   (0-indexed)
    """
    np.random.seed(seed)
    total = n_samples_per_class * n_classes
    X     = np.zeros((total, window_size, n_channels), dtype=np.float32)
    y     = np.zeros(total, dtype=np.int64)

    idx = 0
    for cls in range(n_classes):
        for _ in range(n_samples_per_class):
            # Vary SNR per sample to improve diversity
            snr = snr_db + np.random.uniform(-5.0, 5.0)
            X[idx] = generate_emg_window(cls, window_size, n_channels, fs,
                                          snr_db=snr)
            y[idx] = cls
            idx   += 1

    # Shuffle
    perm = np.random.permutation(total)
    X, y = X[perm], y[perm]

    if verbose:
        print(f"[SyntheticEMG] Generated {total} windows "
              f"({n_classes} classes × {n_samples_per_class} samples), "
              f"shape={X.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Augmentation helpers (used after GAN training to further expand data)
# ---------------------------------------------------------------------------

def augment_window(window: np.ndarray,
                   scale_range: Tuple[float, float] = (0.85, 1.15),
                   noise_std: float = 0.02,
                   time_shift_max: int = 10) -> np.ndarray:
    """Apply random scale, noise, and time-shift augmentation."""
    w = window.copy()
    # Amplitude scaling
    scale = np.random.uniform(*scale_range)
    w    *= scale
    # Additive noise
    w    += noise_std * np.random.randn(*w.shape).astype(np.float32)
    # Time shift (roll with zero-padding)
    shift = np.random.randint(-time_shift_max, time_shift_max + 1)
    if shift != 0:
        w = np.roll(w, shift, axis=0)
        if shift > 0:
            w[:shift, :] = 0
        else:
            w[shift:, :]  = 0
    return np.clip(w, -1.0, 1.0)


def augment_dataset(X: np.ndarray, y: np.ndarray,
                    factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiply dataset size by `factor` using online augmentation.
    """
    aug_X = [X]
    aug_y = [y]
    for _ in range(factor - 1):
        aug_X.append(np.array([augment_window(w) for w in X], dtype=np.float32))
        aug_y.append(y.copy())
    X_out = np.vstack(aug_X)
    y_out = np.concatenate(aug_y)
    perm  = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_synthetic(X: np.ndarray, y: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    print(f"[SyntheticEMG] Saved to {path}")


def load_synthetic(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"[SyntheticEMG] Loaded {len(d['X'])} windows from {path}")
    return d["X"], d["y"]
