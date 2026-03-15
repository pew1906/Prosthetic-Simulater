# Adaptive AI-Powered Smart Prosthetic Simulator

End-to-end pipeline for EMG-based gesture classification using NinaPro DB1.
Includes synthetic data generation, Conditional GAN training, and four deep
learning classifiers (1D CNN, 2D CNN, LSTM, CNN-LSTM Hybrid).

---

## Project Structure

```
prosthetic_simulator/
├── main.py                          ← Full pipeline entry point
├── requirements.txt
│
├── data/
│   ├── synthetic_emg.py             ← Physics-based EMG signal synthesiser
│   ├── raw/                         ← Place NinaPro DB1 .mat files here
│   ├── processed/                   ← Cached numpy arrays (auto-generated)
│   └── synthetic/                   ← GAN output (auto-generated)
│
├── preprocessing/
│   └── emg_preprocessor.py          ← Bandpass/notch filter, windowing
│
├── gan/
│   ├── generator.py                 ← Conditional 1D conv generator
│   ├── discriminator.py             ← Spectral-norm PatchGAN discriminator
│   └── trainer.py                   ← Adversarial training loop
│
├── models/
│   ├── cnn_1d.py                    ← Depthwise-separable 1D CNN
│   ├── cnn_2d.py                    ← STFT spectrogram + 2D CNN
│   ├── lstm_model.py                ← Bidirectional LSTM + attention
│   └── cnn_lstm_hybrid.py           ← CNN front-end + BiLSTM back-end
│
├── training/
│   └── train_classifier.py          ← Training loop, early stopping, AMP
│
├── evaluation/
│   ├── evaluator.py                 ← Plots: CM, curves, comparison
│   └── output/                      ← Saved figures (auto-generated)
│
├── simulator/
│   └── prosthetic_interface.py      ← Real-time EMG stream + prediction
│
└── utils/
    └── helpers.py                   ← Seeds, device, feature extraction
```

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.8.0
pandas>=2.0.0
tqdm>=4.65.0
```

---

## Quick Start

```bash
# Full pipeline (synthetic data + GAN + all 4 models)
python main.py

# Fast test (10 classes, 30 epochs, skip GAN)
python main.py --n-classes 10 --clf-epochs 30 --skip-gan

# Skip slow 2D CNN
python main.py --skip-2dcnn

# All options
python main.py --help
```

---

## Using Real NinaPro DB1 Data

1. Download from https://ninapro.unige.ch (free registration required)
2. Place `.mat` files in `data/raw/`
3. Use the preprocessor:

```python
from preprocessing.emg_preprocessor import EMGPreprocessor

pre = EMGPreprocessor(fs=100, window_size=200, overlap=0.5)

# Single file
X, y = pre.preprocess("data/raw/S1_A1_E1.mat")

# Multiple subjects / exercises
X, y = pre.preprocess_multiple([
    "data/raw/S1_A1_E1.mat",
    "data/raw/S1_A1_E2.mat",
    "data/raw/S1_A1_E3.mat",
])
pre.save_processed(X, y, "data/processed/subject1.pkl")
```

Then replace `X_synth, y_synth` in `main.py` Stage 1 with your real data.

---

## Module Reference

### `data/synthetic_emg.py`

```python
from data.synthetic_emg import generate_synthetic_dataset, augment_dataset

# Generate class-balanced dataset
X, y = generate_synthetic_dataset(
    n_samples_per_class=200,   # samples per gesture
    n_classes=52,
    window_size=200,           # time steps (200 @ 100Hz = 2s window)
    n_channels=10,             # EMG electrode channels
    snr_db=20,                 # signal-to-noise ratio
)
# X: (10400, 200, 10)   y: (10400,)

# Augment existing data (scale, noise, time shift)
X_aug, y_aug = augment_dataset(X, y, factor=3)
```

**Signal model:** Each channel is amplitude-modulated bandpass noise with
gesture-specific muscle activation patterns derived from the EMG literature.
Motor-unit firing, inter-channel crosstalk, and motion artefacts are simulated.

---

### `gan/`

```python
from gan.generator     import EMGGenerator
from gan.discriminator import EMGDiscriminator
from gan.trainer       import GANTrainer

G = EMGGenerator(latent_dim=100, n_classes=52)
D = EMGDiscriminator(n_classes=52)

trainer = GANTrainer(G, D, latent_dim=100, device="cuda")
trainer.train(X, y, epochs=200, batch_size=64)

# Generate 10,400 new windows (200 per class)
X_gan, y_gan = trainer.generate_synthetic(n_samples=10400, n_classes=52)

# Save / resume training
trainer.save_checkpoint("checkpoints", epoch=200)
trainer.load_checkpoint("checkpoints/gan_epoch_0200.pt")
```

**GAN architecture:**
- Generator: dense projection → 3× transposed-conv upsampling → residual blocks → tanh
- Discriminator: label-tiled input → spectral-norm strided convs → sigmoid
- Training: label smoothing (real=0.85–1.0), n_critic=2, cosine LR decay

---

### `models/`

All models accept `(B, W, C)` input (batch × window × channels) and return `(B, n_classes)`.

```python
from models.cnn_1d          import CNN1D
from models.cnn_2d          import CNN2D, emg_to_spectrogram
from models.lstm_model      import LSTMModel
from models.cnn_lstm_hybrid import CNNLSTMHybrid

# 1D CNN — depthwise-separable, ~480K params
model = CNN1D(n_channels=10, window_size=200, n_classes=52)

# 2D CNN — requires STFT preprocessing
X_spec = emg_to_spectrogram(X)   # (N, C, freq_bins, time_frames)
model  = CNN2D(n_channels=10, n_classes=52)

# Bidirectional LSTM with attention pooling
model = LSTMModel(n_channels=10, hidden_size=256, num_layers=3, n_classes=52)

# CNN-LSTM hybrid — CNN feature extraction + BiLSTM sequence modelling
model = CNNLSTMHybrid(n_channels=10, window_size=200, n_classes=52)
```

---

### `training/train_classifier.py`

```python
from training.train_classifier import ClassifierTrainer

trainer = ClassifierTrainer(model, device="cuda", lr=1e-3)
history = trainer.train(
    X_train, y_train, X_val, y_val,
    epochs=80, batch_size=64, patience=15,
    save_path="checkpoints/models/cnn1d.pt",
)

# Full test-set evaluation
results = trainer.evaluate_full(X_test, y_test)
print(results["accuracy"], results["macro_f1"])
print(results["report"])
```

Features: class-weighted cross-entropy, gradient clipping, cosine annealing,
early stopping, mixed-precision (CUDA only).

---

### `simulator/prosthetic_interface.py`

```python
from simulator.prosthetic_interface import ProstheticSimulator

sim = ProstheticSimulator(model, device="cpu", n_classes=52, update_hz=10.0)
sim.start()
sim.set_target_gesture(12)       # wrist flexion

result = sim.step()
print(result["gesture_name"])    # "Wrist flexion"
print(result["confidence"])      # 0.94

sim.stop()

# CLI demo
sim.run_demo(n_steps=20, target_gesture=12)
```

---

## Evaluation Outputs

All plots are saved to `evaluation/output/`:

| File                         | Contents                                |
|------------------------------|-----------------------------------------|
| `synthetic_emg_samples.png`  | Synthetic EMG windows (4 gestures)      |
| `gan_emg_samples.png`        | GAN-generated EMG windows               |
| `gan_training.png`           | G/D loss curves + discriminator acc     |
| `curves_1D_CNN.png`          | Train/val loss, accuracy, F1            |
| `curves_LSTM.png`            | (same)                                  |
| `curves_CNN-LSTM.png`        | (same)                                  |
| `curves_2D_CNN.png`          | (same)                                  |
| `cm_1D_CNN.png`              | Normalised confusion matrix             |
| `model_comparison.png`       | Accuracy + Macro F1 bar chart           |

---

## NinaPro DB1 — Dataset Details

| Property         | Value                                    |
|------------------|------------------------------------------|
| Subjects         | 27 able-bodied                           |
| EMG channels     | 10 (Otto Bock electrode)                 |
| Sampling rate    | 100 Hz                                   |
| Gestures         | 52 (17 finger, 17 wrist/hand, 18 grasp) |
| Exercise sets    | A (12), B (17), C (23)                   |
| File format      | MATLAB .mat                              |

---

## Pipeline Diagram

```
NinaPro DB1 (.mat)
       │
       ▼
EMGPreprocessor ──── bandpass (20–450 Hz)
       │         └── notch (50 Hz)
       │         └── StandardScaler
       │         └── sliding window (200 samples, 50% overlap)
       │
       ▼
SyntheticEMGGenerator  ←─ (if no .mat files: physics-based bootstrap)
       │
       ▼
  GAN Training
  ┌─────────────────────────┐
  │  Generator              │
  │  noise+label → EMG win  │
  │         ↕ adversarial   │
  │  Discriminator          │
  │  real/fake classifier   │
  └─────────────────────────┘
       │
       ▼
Combined Dataset (real + GAN-augmented + classical augmentation)
       │
       ├──→  1D CNN  ──┐
       ├──→  2D CNN  ──┤
       ├──→  LSTM    ──┼──→ Model Comparison → Best Model
       └──→ CNN-LSTM ──┘
                              │
                              ▼
                    ProstheticSimulator
                    real-time EMG stream
                    + smoothed prediction
                    + gesture name + confidence
```

---

## Extending the Project

**Add a new model:**
1. Create `models/my_model.py` with input `(B, W, C)` → output `(B, n_classes)`
2. Add it to `model_configs` dict in `main.py`

**Use real electrodes (e.g. Myo Armband):**
Replace `EMGStreamSimulator` in `prosthetic_interface.py` with a driver that
reads from your device and calls `controller.predict(window)`.

**Add WGAN-GP:**
In `gan/trainer.py`, replace BCE loss with Wasserstein loss and add a gradient
penalty term to `loss_D`.

**Export to ONNX:**
```python
import torch
dummy = torch.randn(1, 200, 10)
torch.onnx.export(model, dummy, "model.onnx", input_names=["emg"],
                  output_names=["logits"])
```
