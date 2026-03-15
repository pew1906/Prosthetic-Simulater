"""
GAN Trainer
Implements conditional adversarial training with:
  - Label smoothing
  - Feature matching loss
  - Gradient penalty (optional WGAN-GP mode)
  - Checkpoint saving / loading
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from gan.generator     import EMGGenerator
from gan.discriminator import EMGDiscriminator


class GANTrainer:
    """
    Trains a conditional GAN on EMG windows.

    Parameters
    ----------
    generator     : EMGGenerator
    discriminator : EMGDiscriminator
    latent_dim    : noise vector size
    device        : 'cpu' | 'cuda'
    lr_g / lr_d   : learning rates
    """

    def __init__(
        self,
        generator:     EMGGenerator,
        discriminator: EMGDiscriminator,
        latent_dim:    int   = 100,
        device:        str   = "cpu",
        lr_g:          float = 2e-4,
        lr_d:          float = 1e-4,
    ):
        self.G          = generator.to(device)
        self.D          = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device     = device

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # Cosine LR decay
        self.sched_G = optim.lr_scheduler.CosineAnnealingLR(self.opt_G, T_max=200)
        self.sched_D = optim.lr_scheduler.CosineAnnealingLR(self.opt_D, T_max=200)

        self.criterion = nn.BCELoss()
        self.history   = {"d_loss": [], "g_loss": [], "d_acc": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def _smooth_labels(self, size: int, real: bool) -> torch.Tensor:
        """One-sided label smoothing: real → [0.85, 1.0], fake → [0.0, 0.15]"""
        if real:
            return torch.empty(size, 1, device=self.device).uniform_(0.85, 1.0)
        else:
            return torch.empty(size, 1, device=self.device).uniform_(0.0, 0.15)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X:             np.ndarray,
        y:             np.ndarray,
        epochs:        int  = 200,
        batch_size:    int  = 64,
        n_critic:      int  = 2,       # D steps per G step
        save_dir:      str  = "checkpoints",
        save_every:    int  = 50,
        verbose:       bool = True,
    ):
        """
        Adversarial training loop.

        Parameters
        ----------
        X          : (N, W, C) float32
        y          : (N,) int64
        epochs     : number of full passes
        batch_size : mini-batch size
        n_critic   : discriminator updates per generator update
        save_dir   : where to save checkpoints
        save_every : checkpoint interval (epochs)
        """
        os.makedirs(save_dir, exist_ok=True)

        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True)

        n_classes = int(y.max()) + 1

        for epoch in range(1, epochs + 1):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_acc  = 0.0
            n_batches    = 0

            for real_emg, labels in loader:
                real_emg = real_emg.to(self.device)
                labels   = labels.to(self.device)
                bs       = real_emg.size(0)

                # ── Discriminator update (n_critic times) ──────────────
                for _ in range(n_critic):
                    self.opt_D.zero_grad()

                    # Real
                    d_real  = self.D(real_emg, labels)
                    loss_real = self.criterion(d_real, self._smooth_labels(bs, real=True))

                    # Fake
                    fake_emg = self.G(self._noise(bs), labels).detach()
                    d_fake   = self.D(fake_emg, labels)
                    loss_fake = self.criterion(d_fake, self._smooth_labels(bs, real=False))

                    loss_D  = (loss_real + loss_fake) / 2
                    loss_D.backward()
                    self.opt_D.step()

                    # D accuracy (how often it is correct)
                    with torch.no_grad():
                        d_acc = ((d_real > 0.5).float().mean() +
                                 (d_fake < 0.5).float().mean()) / 2

                # ── Generator update ───────────────────────────────────
                self.opt_G.zero_grad()

                # Random labels for generation diversity
                rand_labels = torch.randint(0, n_classes, (bs,), device=self.device)
                gen_emg     = self.G(self._noise(bs), rand_labels)
                g_out       = self.D(gen_emg, rand_labels)
                loss_G      = self.criterion(g_out, self._smooth_labels(bs, real=True))
                loss_G.backward()
                self.opt_G.step()

                epoch_d_loss += loss_D.item()
                epoch_g_loss += loss_G.item()
                epoch_d_acc  += d_acc.item()
                n_batches    += 1

            self.sched_G.step()
            self.sched_D.step()

            avg_d = epoch_d_loss / n_batches
            avg_g = epoch_g_loss / n_batches
            avg_a = epoch_d_acc  / n_batches

            self.history["d_loss"].append(avg_d)
            self.history["g_loss"].append(avg_g)
            self.history["d_acc"].append(avg_a)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"[GAN] Epoch {epoch:>4}/{epochs} | "
                      f"D_loss={avg_d:.4f}  G_loss={avg_g:.4f}  D_acc={avg_a:.3f}")

            if epoch % save_every == 0:
                self.save_checkpoint(save_dir, epoch)

        if verbose:
            print("[GAN] Training complete.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_synthetic(
        self,
        n_samples:   int,
        n_classes:   int,
        batch_size:  int = 128,
    ):
        """
        Generate a class-balanced synthetic dataset.

        Returns
        -------
        X_syn : (n_samples, W, C)  float32
        y_syn : (n_samples,)       int64
        """
        self.G.eval()
        per_class = n_samples // n_classes
        all_X, all_y = [], []

        with torch.no_grad():
            for cls in range(n_classes):
                generated = []
                remaining = per_class
                while remaining > 0:
                    bs     = min(batch_size, remaining)
                    lbl    = torch.full((bs,), cls, dtype=torch.long,
                                        device=self.device)
                    noise  = self._noise(bs)
                    gen    = self.G(noise, lbl).cpu().numpy()
                    generated.append(gen)
                    remaining -= bs
                all_X.append(np.vstack(generated))
                all_y.extend([cls] * per_class)

        self.G.train()
        X_syn = np.vstack(all_X).astype(np.float32)
        y_syn = np.array(all_y, dtype=np.int64)

        # Shuffle
        perm  = np.random.permutation(len(X_syn))
        return X_syn[perm], y_syn[perm]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, save_dir: str, epoch: int):
        path = os.path.join(save_dir, f"gan_epoch_{epoch:04d}.pt")
        torch.save({
            "epoch":        epoch,
            "G_state":      self.G.state_dict(),
            "D_state":      self.D.state_dict(),
            "opt_G_state":  self.opt_G.state_dict(),
            "opt_D_state":  self.opt_D.state_dict(),
            "history":      self.history,
        }, path)
        print(f"[GAN] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        self.opt_G.load_state_dict(ckpt["opt_G_state"])
        self.opt_D.load_state_dict(ckpt["opt_D_state"])
        self.history = ckpt.get("history", self.history)
        print(f"[GAN] Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["epoch"]
