"""
EMG Preprocessor for NinaPro DB1 Dataset
Handles loading, filtering, normalization, and windowed segmentation.
"""

import numpy as np
import os
import pickle
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler

try:
    import scipy.io as sio
    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False


class EMGPreprocessor:
    """
    Full preprocessing pipeline for NinaPro DB1 EMG signals.

    NinaPro DB1 specs:
      - 10 EMG channels (Otto Bock electrode)
      - 100 Hz sampling rate
      - 52 hand/wrist gestures across 3 exercise sets
      - 27 subjects
    """

    def __init__(self, fs: int = 100, window_size: int = 200, overlap: float = 0.5):
        self.fs          = fs
        self.window_size = window_size
        self.overlap     = overlap
        self.scaler      = StandardScaler()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_ninapro(self, filepath: str):
        """Load a NinaPro DB1 .mat file and return (emg, labels)."""
        if not HAS_SCIPY_IO:
            raise ImportError("scipy is required to load .mat files.")
        data   = sio.loadmat(filepath)
        emg    = data["emg"].astype(np.float32)       # (T, 10)
        labels = data["restimulus"].flatten().astype(int)
        return emg, labels

    # ------------------------------------------------------------------
    # Signal filtering
    # ------------------------------------------------------------------

    def bandpass_filter(self, signal: np.ndarray,
                        low: float = 20.0, high: float = 450.0) -> np.ndarray:
        """4th-order Butterworth bandpass 20-450 Hz."""
        nyq  = self.fs / 2.0
        b, a = butter(4, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, signal, axis=0)

    def notch_filter(self, signal: np.ndarray, freq: float = 50.0) -> np.ndarray:
        """IIR notch filter for powerline interference (50/60 Hz)."""
        nyq  = self.fs / 2.0
        b, a = iirnotch(freq / nyq, Q=30)
        return filtfilt(b, a, signal, axis=0)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def rectify_smooth(self, signal: np.ndarray, cutoff: float = 5.0) -> np.ndarray:
        """Full-wave rectification followed by low-pass envelope smoothing."""
        rect = np.abs(signal)
        nyq  = self.fs / 2.0
        b, a = butter(2, cutoff / nyq, btype="low")
        return filtfilt(b, a, rect, axis=0)

    # ------------------------------------------------------------------
    # Windowed segmentation
    # ------------------------------------------------------------------

    def segment_signal(self, emg: np.ndarray, labels: np.ndarray):
        """
        Sliding-window segmentation.
        Returns X (N, window_size, n_channels) and y (N,) zero-indexed.
        Rest periods (label == 0) are excluded.
        """
        step = max(1, int(self.window_size * (1.0 - self.overlap)))
        X, y = [], []
        n    = len(emg)
        for start in range(0, n - self.window_size, step):
            end   = start + self.window_size
            label = labels[end - 1]
            if label > 0:                         # skip rest
                X.append(emg[start:end])
                y.append(label - 1)               # 0-indexed
        if len(X) == 0:
            raise ValueError("No valid windows found. Check your label file.")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(self, filepath: str):
        """
        Load → bandpass → notch → normalize → segment.
        Returns X (N, W, C) and y (N,).
        """
        emg, labels = self.load_ninapro(filepath)
        emg         = self.bandpass_filter(emg)
        emg         = self.notch_filter(emg)
        emg         = self.scaler.fit_transform(emg)
        X, y        = self.segment_signal(emg, labels)
        return X, y

    def preprocess_multiple(self, filepaths: list):
        """Process and concatenate several subject/exercise .mat files."""
        all_X, all_y = [], []
        for fp in filepaths:
            X, y = self.preprocess(fp)
            all_X.append(X)
            all_y.append(y)
        return np.vstack(all_X), np.concatenate(all_y)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_processed(self, X: np.ndarray, y: np.ndarray, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"X": X, "y": y}, f)
        print(f"Saved {len(X)} windows to {path}")

    @staticmethod
    def load_processed(path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["X"], d["y"]
