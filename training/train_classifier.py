"""
Classifier Trainer
Trains any of the four EMG models with:
  - Class-weighted cross-entropy (handles imbalance)
  - Cosine annealing LR scheduler
  - Early stopping
  - Mixed-precision training (AMP) when CUDA is available
  - Per-epoch metrics logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.triggered = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter  += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class ClassifierTrainer:
    """
    Generic trainer for all EMG classification models.

    Parameters
    ----------
    model      : nn.Module
    device     : 'cpu' | 'cuda'
    lr         : initial learning rate
    weight_decay: L2 regularisation
    """

    def __init__(
        self,
        model:        nn.Module,
        device:       str   = "cpu",
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model        = model.to(device)
        self.device       = device
        self.lr           = lr
        self.weight_decay = weight_decay
        self.history      = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "train_f1":   [], "val_f1":   [],
        }

    def _make_loaders(self, X_train, y_train, X_val, y_val,
                      batch_size: int, is_2d: bool = False):
        def to_tensor(X, y):
            return TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).long(),
            )
        train_ds = to_tensor(X_train, y_train)
        val_ds   = to_tensor(X_val,   y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                                   shuffle=False, num_workers=0, pin_memory=False)
        return train_loader, val_loader

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        classes, counts = np.unique(y, return_counts=True)
        weights = 1.0 / counts.astype(np.float32)
        weights /= weights.sum()
        # Map back to full class range
        n_cls = int(y.max()) + 1
        w = np.ones(n_cls, dtype=np.float32)
        for cls, wt in zip(classes, weights):
            w[cls] = wt
        return torch.from_numpy(w).to(self.device)

    @staticmethod
    def _evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b  = X_b.to(device), y_b.to(device)
                logits    = model(X_b)
                loss      = criterion(logits, y_b)
                total_loss += loss.item() * len(y_b)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_b.cpu().numpy())

        avg_loss = total_loss / len(all_true)
        acc      = accuracy_score(all_true, all_preds)
        f1       = f1_score(all_true, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, f1

    def train(
        self,
        X_train:      np.ndarray,
        y_train:      np.ndarray,
        X_val:        np.ndarray,
        y_val:        np.ndarray,
        epochs:       int   = 80,
        batch_size:   int   = 64,
        patience:     int   = 15,
        save_path:    str   = None,
        is_2d:        bool  = False,
        verbose:      bool  = True,
    ) -> dict:
        """
        Train the model.

        Parameters
        ----------
        is_2d : set True for CNN2D (input already in (N, C, F, T) format)
        """
        train_loader, val_loader = self._make_loaders(
            X_train, y_train, X_val, y_val, batch_size, is_2d)

        class_weights = self._compute_class_weights(y_train)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        optimizer     = optim.AdamW(self.model.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stop    = EarlyStopping(patience=patience)

        use_amp = self.device == "cuda" and torch.cuda.is_available()
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            t0 = time.time()

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(X_b)
                        loss   = criterion(logits, y_b)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(X_b)
                    loss   = criterion(logits, y_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

            scheduler.step()

            tr_loss, tr_acc, tr_f1 = self._evaluate(
                self.model, train_loader, criterion, self.device)
            vl_loss, vl_acc, vl_f1 = self._evaluate(
                self.model, val_loader,   criterion, self.device)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(vl_acc)
            self.history["train_f1"].append(tr_f1)
            self.history["val_f1"].append(vl_f1)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)

            if verbose and (epoch % 5 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:>3}/{epochs} | "
                      f"train acc={tr_acc:.3f} f1={tr_f1:.3f} | "
                      f"val acc={vl_acc:.3f} f1={vl_f1:.3f} | "
                      f"{elapsed:.1f}s")

            if early_stop(vl_loss):
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        return self.history

    def evaluate_full(self, X_test: np.ndarray, y_test: np.ndarray,
                      batch_size: int = 128) -> dict:
        """Full evaluation: accuracy, macro-F1, per-class report, confusion matrix."""
        self.model.eval()
        ds     = TensorDataset(torch.from_numpy(X_test).float(),
                               torch.from_numpy(y_test).long())
        loader = DataLoader(ds, batch_size=batch_size)

        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                logits = self.model(X_b.to(self.device))
                preds  = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_b.numpy())

        acc  = accuracy_score(all_true, all_preds)
        f1   = f1_score(all_true, all_preds, average="macro", zero_division=0)
        cm   = confusion_matrix(all_true, all_preds)
        rep  = classification_report(all_true, all_preds, zero_division=0)
        return {"accuracy": acc, "macro_f1": f1,
                "confusion_matrix": cm, "report": rep}
