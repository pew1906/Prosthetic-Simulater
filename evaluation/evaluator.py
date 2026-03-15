"""
Evaluation & Visualisation
Generates confusion matrices, training curves, and model comparison charts.
Saves all figures to the evaluation/output/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix",
                           filename: str = "confusion_matrix.png",
                           max_classes: int = 30):
    _ensure_output()
    # Truncate for readability if > max_classes
    if cm.shape[0] > max_classes:
        cm = cm[:max_classes, :max_classes]

    fig, ax = plt.subplots(figsize=(14, 11))
    # Normalise row-wise for better readability
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    # seaborn >=0.13 requires fmt to match the actual annotated array dtype.
    # Pass integer cm with fmt="d"; for large matrices skip annotation.
    if cm.shape[0] <= 15:
        sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    annot=cm.astype(int), fmt="d", linewidths=0.2,
                    cbar_kws={"label": "Normalised count"})
    else:
        sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    annot=False, linewidths=0.2,
                    cbar_kws={"label": "Normalised count"})
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title(title,              fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, model_name: str = "Model",
                          filename: str = None):
    _ensure_output()
    filename = filename or f"training_curves_{model_name.replace(' ', '_')}.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#2196F3")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#F44336")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["train_f1"], label="Train", color="#2196F3")
    axes[2].plot(epochs, history["val_f1"],   label="Val",   color="#F44336")
    axes[2].set_title("Macro F1"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, filename: str = "model_comparison.png"):
    """
    Parameters
    ----------
    results : {model_name: {"accuracy": float, "macro_f1": float}}
    """
    _ensure_output()
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] * 100 for n in names]
    f1s    = [results[n]["macro_f1"]  * 100 for n in names]

    x      = np.arange(len(names))
    width  = 0.35
    colors_acc = ["#1565C0", "#0D47A1", "#1976D2", "#42A5F5"]
    colors_f1  = ["#B71C1C", "#C62828", "#D32F2F", "#EF5350"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1   = ax.bar(x - width/2, accs, width, label="Accuracy (%)",
                     color=colors_acc[:len(names)], zorder=3)
    bars2   = ax.bar(x + width/2, f1s,  width, label="Macro F1 (%)",
                     color=colors_f1[:len(names)],  zorder=3)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Test Set", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# GAN training curves
# ---------------------------------------------------------------------------

def plot_gan_history(history: dict, filename: str = "gan_training.png"):
    _ensure_output()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GAN Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["d_loss"]) + 1)
    axes[0].plot(epochs, history["d_loss"], label="D loss", color="#9C27B0")
    axes[0].plot(epochs, history["g_loss"], label="G loss", color="#FF9800")
    axes[0].set_title("Generator vs Discriminator Loss")
    axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["d_acc"], color="#4CAF50")
    axes[1].set_title("Discriminator Accuracy")
    axes[1].set_ylim(0, 1); axes[1].set_xlabel("Epoch")
    axes[1].axhline(0.5, color="red", linestyle="--", label="Random baseline")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# EMG signal visualisation
# ---------------------------------------------------------------------------

def plot_emg_windows(X: np.ndarray, y: np.ndarray,
                     n_gestures: int = 4, title_prefix: str = "EMG",
                     filename: str = "emg_samples.png"):
    """
    Plot one EMG window per gesture (first n_gestures classes).
    """
    _ensure_output()
    unique = np.unique(y)[:n_gestures]
    fig, axes = plt.subplots(n_gestures, 1, figsize=(12, 3 * n_gestures), sharex=True)
    if n_gestures == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    for i, cls in enumerate(unique):
        idx  = np.where(y == cls)[0][0]
        win  = X[idx]                              # (W, C)
        t    = np.arange(win.shape[0])
        for ch in range(win.shape[1]):
            axes[i].plot(t, win[:, ch] + ch * 2.5,
                         color=cmap(ch / win.shape[1]), linewidth=0.8)
        axes[i].set_ylabel(f"Gesture {cls+1}", fontsize=9)
        axes[i].set_yticks([])

    axes[-1].set_xlabel("Sample")
    fig.suptitle(f"{title_prefix} — 10 channels stacked", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved → {path}")
    return path
