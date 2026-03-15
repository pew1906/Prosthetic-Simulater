"""
test_pipeline.py
Smoke-tests every module that does NOT require PyTorch (pure numpy/scipy).
Run before the full main.py to verify the data and preprocessing layers.

Usage:
    python test_pipeline.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

PASS = "PASS"
FAIL = "FAIL"

results = []


def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [{PASS}] {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  [{FAIL}] {name}: {e}")


# ── 1. Synthetic EMG window ────────────────────────────────────────────────
def t_single_window():
    from data.synthetic_emg import generate_emg_window
    win = generate_emg_window(gesture_id=0, window_size=200, n_channels=10)
    assert win.shape == (200, 10), f"Expected (200,10) got {win.shape}"
    assert np.abs(win).max() <= 1.0 + 1e-6, "Values outside [-1,1]"

test("Single EMG window generation", t_single_window)


# ── 2. Full synthetic dataset ─────────────────────────────────────────────
def t_dataset():
    from data.synthetic_emg import generate_synthetic_dataset
    X, y = generate_synthetic_dataset(
        n_samples_per_class=5, n_classes=10, window_size=200, n_channels=10,
        verbose=False)
    assert X.shape == (50, 200, 10), f"Bad shape: {X.shape}"
    assert y.shape == (50,)
    assert set(np.unique(y)) == set(range(10))

test("Synthetic dataset generation (10 classes × 5 samples)", t_dataset)


# ── 3. Dataset augmentation ───────────────────────────────────────────────
def t_augment():
    from data.synthetic_emg import generate_synthetic_dataset, augment_dataset
    X, y = generate_synthetic_dataset(5, 4, 200, 10, verbose=False)
    X2, y2 = augment_dataset(X, y, factor=3)
    assert len(X2) == len(X) * 3
    assert X2.shape[1:] == X.shape[1:]

test("Data augmentation (factor=3)", t_augment)


# ── 4. Save / load synthetic data ─────────────────────────────────────────
def t_save_load():
    import tempfile
    from data.synthetic_emg import (generate_synthetic_dataset,
                                     save_synthetic, load_synthetic)
    X, y = generate_synthetic_dataset(3, 4, 200, 10, verbose=False)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    save_synthetic(X, y, path)
    X2, y2 = load_synthetic(path)
    assert np.allclose(X, X2) and np.array_equal(y, y2)
    os.unlink(path)

test("Save / load synthetic dataset", t_save_load)


# ── 5. Time-domain feature extraction ────────────────────────────────────
def t_td_features():
    from utils.helpers import extract_td_features
    X = np.random.randn(20, 200, 10).astype(np.float32)
    feats = extract_td_features(X)
    assert feats.shape == (20, 50), f"Expected (20,50) got {feats.shape}"

test("Time-domain feature extraction (5 feats × 10 ch)", t_td_features)


# ── 6. STFT spectrogram ───────────────────────────────────────────────────
def t_spectrogram():
    from models.cnn_2d import emg_to_spectrogram
    X = np.random.randn(8, 200, 10).astype(np.float32)
    specs = emg_to_spectrogram(X, n_fft=32, hop=8)
    N, C, F, T = specs.shape
    assert N == 8
    assert C == 10
    assert F == 17           # n_fft//2 + 1
    assert specs.min() >= 0 and specs.max() <= 1 + 1e-5

test("STFT spectrogram conversion", t_spectrogram)


# ── 7. Gesture name map ───────────────────────────────────────────────────
def t_gesture_map():
    from simulator.prosthetic_interface import GESTURE_NAMES
    assert len(GESTURE_NAMES) == 52
    assert GESTURE_NAMES[0]  == "Index finger flexion"
    assert GESTURE_NAMES[12] == "Wrist flexion"

test("Gesture name map (52 entries)", t_gesture_map)


# ── 8. EMGStreamSimulator (no torch) ─────────────────────────────────────
def t_stream():
    from simulator.prosthetic_interface import EMGStreamSimulator
    stream = EMGStreamSimulator(n_channels=10, window_size=200, update_hz=50)
    stream.set_gesture(5)
    stream.start()
    import time; time.sleep(0.15)
    win = stream.get_window(timeout=0.5)
    stream.stop()
    assert win is not None
    assert win.shape == (200, 10)

test("EMG stream simulator (3 windows @ 50 Hz)", t_stream)


# ── 9. Activation templates shape ─────────────────────────────────────────
def t_templates():
    from data.synthetic_emg import _ACTIVATION_TEMPLATES
    assert _ACTIVATION_TEMPLATES.shape == (52, 10)
    assert _ACTIVATION_TEMPLATES.min() >= 0
    assert _ACTIVATION_TEMPLATES.max() <= 1

test("Activation templates (52 × 10)", t_templates)


# ── 10. helpers.py utils ──────────────────────────────────────────────────
def t_helpers():
    from utils.helpers import set_seed, Timer
    set_seed(0)
    with Timer("noop"):
        pass

test("set_seed + Timer context manager", t_helpers)


# ── Summary ───────────────────────────────────────────────────────────────
print()
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
print(f"  {'─'*40}")
print(f"  Results: {passed}/{len(results)} passed", end="")
if failed:
    print(f"  ({failed} failed)")
    sys.exit(1)
else:
    print("  — all good!")
