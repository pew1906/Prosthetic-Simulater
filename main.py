"""
main.py — Full end-to-end pipeline for the Adaptive AI-Powered Smart Prosthetic Simulator

Stages
------
1.  Generate (or load) synthetic EMG dataset  [NinaPro DB1 style]
2.  Train GAN on synthetic data to learn realistic signal distribution
3.  Generate GAN-augmented data and merge with original synthetic data
4.  Train four classifiers: 1D CNN, 2D CNN, LSTM, CNN-LSTM Hybrid
5.  Evaluate and compare all models
6.  Run prosthetic simulator demo

Usage
-----
python main.py                     # full run
python main.py --skip-gan          # skip GAN training (faster)
python main.py --epochs 30         # fewer epochs (quick test)
python main.py --n-classes 10      # fewer gesture classes
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers        import set_seed, get_device, dataset_summary, Timer, count_parameters, sanity_check
from data.synthetic_emg   import generate_synthetic_dataset, augment_dataset, save_synthetic, load_synthetic
from gan.generator        import EMGGenerator
from gan.discriminator    import EMGDiscriminator
from gan.trainer          import GANTrainer
from models.cnn_1d        import CNN1D
from models.cnn_2d        import CNN2D, emg_to_spectrogram
from models.lstm_model    import LSTMModel
from models.cnn_lstm_hybrid import CNNLSTMHybrid
from training.train_classifier import ClassifierTrainer
from evaluation.evaluator import (plot_confusion_matrix, plot_training_curves,
                                   plot_model_comparison, plot_gan_history,
                                   plot_emg_windows)
from simulator.prosthetic_interface import ProstheticSimulator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prosthetic EMG Simulator")
    p.add_argument("--n-classes",        type=int,   default=52)
    p.add_argument("--n-samples-class",  type=int,   default=150,
                   help="Synthetic samples per class before GAN augmentation")
    p.add_argument("--window-size",      type=int,   default=200)
    p.add_argument("--n-channels",       type=int,   default=10)
    p.add_argument("--gan-epochs",       type=int,   default=100)
    p.add_argument("--clf-epochs",       type=int,   default=50)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--latent-dim",       type=int,   default=100)
    p.add_argument("--skip-gan",         action="store_true")
    p.add_argument("--skip-2dcnn",       action="store_true",
                   help="Skip 2D CNN (slower due to STFT conversion)")
    p.add_argument("--demo-gesture",     type=int,   default=12,
                   help="Gesture ID to demonstrate in simulator")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = get_device()

    os.makedirs("checkpoints",         exist_ok=True)
    os.makedirs("checkpoints/models",  exist_ok=True)
    os.makedirs("data/processed",      exist_ok=True)
    os.makedirs("evaluation/output",   exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1 — Synthetic EMG Dataset (NinaPro DB1 style)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  STAGE 1 — Synthetic EMG Data Generation")
    print("═"*60)

    synth_path = "data/processed/synthetic_base.pkl"
    if os.path.exists(synth_path):
        print("[Stage 1] Loading cached synthetic data …")
        X_synth, y_synth = load_synthetic(synth_path)
    else:
        with Timer("Synthetic data generation"):
            X_synth, y_synth = generate_synthetic_dataset(
                n_samples_per_class = args.n_samples_class,
                n_classes           = args.n_classes,
                window_size         = args.window_size,
                n_channels          = args.n_channels,
                seed                = args.seed,
            )
        save_synthetic(X_synth, y_synth, synth_path)

    dataset_summary(X_synth, y_synth, "Synthetic Base Dataset")
    plot_emg_windows(X_synth, y_synth, n_gestures=4,
                     title_prefix="Synthetic EMG", filename="synthetic_emg_samples.png")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2 — GAN Training
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  STAGE 2 — GAN Training")
    print("═"*60)

    X_gan_aug, y_gan_aug = None, None

    if not args.skip_gan:
        generator     = EMGGenerator(latent_dim=args.latent_dim,
                                     n_classes=args.n_classes,
                                     window_size=args.window_size,
                                     n_channels=args.n_channels)
        discriminator = EMGDiscriminator(n_classes=args.n_classes,
                                         window_size=args.window_size,
                                         n_channels=args.n_channels)

        print(f"\n  Generator:")
        count_parameters(generator)
        print(f"  Discriminator:")
        count_parameters(discriminator)

        trainer = GANTrainer(generator, discriminator,
                             latent_dim=args.latent_dim, device=device)

        with Timer("GAN training"):
            trainer.train(
                X_synth, y_synth,
                epochs=args.gan_epochs,
                batch_size=args.batch_size,
                save_dir="checkpoints",
                save_every=max(10, args.gan_epochs // 5),
            )

        plot_gan_history(trainer.history, "gan_training.png")

        print("\n[Stage 2] Generating GAN-augmented samples …")
        n_gan = len(X_synth)               # match base dataset size
        X_gan_aug, y_gan_aug = trainer.generate_synthetic(
            n_samples=n_gan, n_classes=args.n_classes)
        save_synthetic(X_gan_aug, y_gan_aug, "data/processed/gan_augmented.pkl")
        dataset_summary(X_gan_aug, y_gan_aug, "GAN-Augmented Dataset")
        plot_emg_windows(X_gan_aug, y_gan_aug, n_gestures=4,
                         title_prefix="GAN-Generated EMG",
                         filename="gan_emg_samples.png")
    else:
        print("[Stage 2] Skipped (--skip-gan flag)")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3 — Combine & Split
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  STAGE 3 — Data Combination & Splitting")
    print("═"*60)

    # Augment base with classical augmentation (×2)
    X_aug, y_aug = augment_dataset(X_synth, y_synth, factor=2)

    if X_gan_aug is not None:
        X_all = np.vstack([X_aug, X_gan_aug])
        y_all = np.concatenate([y_aug, y_gan_aug])
        print(f"[Stage 3] Combined: synthetic aug + GAN aug = {len(X_all)} windows")
    else:
        X_all, y_all = X_aug, y_aug
        print(f"[Stage 3] Using augmented synthetic data only: {len(X_all)} windows")

    dataset_summary(X_all, y_all, "Final Training Dataset")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval,
        random_state=args.seed)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 4 — Train Classifiers
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  STAGE 4 — Classifier Training")
    print("═"*60)

    model_configs = {
        "1D CNN": {
            "model": CNN1D(n_channels=args.n_channels,
                           window_size=args.window_size,
                           n_classes=args.n_classes),
            "is_2d": False,
        },
        "LSTM": {
            "model": LSTMModel(n_channels=args.n_channels,
                               n_classes=args.n_classes),
            "is_2d": False,
        },
        "CNN-LSTM": {
            "model": CNNLSTMHybrid(n_channels=args.n_channels,
                                   window_size=args.window_size,
                                   n_classes=args.n_classes),
            "is_2d": False,
        },
    }

    if not args.skip_2dcnn:
        print("\n[Stage 4] Computing spectrograms for 2D CNN …")
        with Timer("STFT conversion"):
            X_spec_train = emg_to_spectrogram(X_train)
            X_spec_val   = emg_to_spectrogram(X_val)
            X_spec_test  = emg_to_spectrogram(X_test)
        print(f"  Spectrogram shape: {X_spec_train.shape}")
        n_ch_spec = X_spec_train.shape[1]
        model_configs["2D CNN"] = {
            "model":    CNN2D(n_channels=n_ch_spec, n_classes=args.n_classes),
            "is_2d":    True,
            "X_train":  X_spec_train,
            "X_val":    X_spec_val,
            "X_test":   X_spec_test,
        }

    trained_models = {}
    all_results    = {}
    all_histories  = {}

    for name, cfg in model_configs.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")

        model = cfg["model"]
        is_2d = cfg["is_2d"]
        X_tr  = cfg.get("X_train", X_train)
        X_vl  = cfg.get("X_val",   X_val)
        X_te  = cfg.get("X_test",  X_test)

        count_parameters(model)
        sanity_check(model, args.window_size, args.n_channels, is_2d=is_2d)

        clf_trainer = ClassifierTrainer(model, device=device)
        save_path   = f"checkpoints/models/{name.replace(' ', '_')}.pt"

        with Timer(f"{name} training"):
            history = clf_trainer.train(
                X_tr, y_train, X_vl, y_val,
                epochs     = args.clf_epochs,
                batch_size = args.batch_size,
                patience   = 12,
                save_path  = save_path,
                is_2d      = is_2d,
            )

        # Reload best checkpoint
        model.load_state_dict(
            __import__("torch").load(save_path, map_location=device))
        clf_trainer.model = model

        results = clf_trainer.evaluate_full(X_te, y_test)
        print(f"\n  ── {name} Test Results ──")
        print(f"     Accuracy : {results['accuracy']*100:.2f}%")
        print(f"     Macro F1 : {results['macro_f1']*100:.2f}%")
        print(results["report"][:600])          # first 600 chars

        # Plots
        plot_confusion_matrix(
            results["confusion_matrix"],
            title    = f"{name} — Confusion Matrix",
            filename = f"cm_{name.replace(' ', '_')}.png",
        )
        plot_training_curves(
            history,
            model_name = name,
            filename   = f"curves_{name.replace(' ', '_')}.png",
        )

        trained_models[name]  = model
        all_results[name]     = results
        all_histories[name]   = history

    # ══════════════════════════════════════════════════════════════════
    # STAGE 5 — Model Comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  STAGE 5 — Model Comparison")
    print("═"*60)

    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'─'*42}")
    for name, res in all_results.items():
        print(f"  {name:<20} {res['accuracy']*100:>9.2f}%  {res['macro_f1']*100:>9.2f}%")

    plot_model_comparison(
        {n: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
         for n, r in all_results.items()}
    )

    best_name  = max(all_results, key=lambda n: all_results[n]["accuracy"])
    best_model = trained_models[best_name]
    print(f"\n  Best model: {best_name} "
          f"({all_results[best_name]['accuracy']*100:.2f}%)")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 6 — Prosthetic Simulator Demo
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  STAGE 6 — Prosthetic Simulator Demo")
    print("═"*60)

    sim = ProstheticSimulator(best_model, device=device,
                              n_classes=args.n_classes)
    sim.run_demo(n_steps=20, target_gesture=args.demo_gesture)

    print("\n[Pipeline] All stages complete.")
    print(f"[Pipeline] Evaluation plots saved to: evaluation/output/")
    print(f"[Pipeline] Model checkpoints saved to: checkpoints/models/")


if __name__ == "__main__":
    main()
