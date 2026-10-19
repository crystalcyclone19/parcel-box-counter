# model/training/trainer.py
"""
Stage 3 (Part A): Training Loop with Early Stopping & Checkpointing
=====================================================================
ดำเนินการเทรน MLP Classifier สำหรับ 1 Configuration (Trial) โดย:
  - รับ Feature Matrix ที่ผ่าน SVD Projection + Statistical Features แล้ว (Z_train, Z_val)
  - รัน Mini-batch SGD Loop สูงสุด `max_epochs` รอบ
  - ติดตาม Validation Cross-Entropy ทุก Epoch → Early Stopping เมื่อไม่ดีขึ้น
  - บันทึก Best Weights ณ Epoch ที่ Val Loss ต่ำสุด (Checkpoint)

ใช้โมดูลจาก mlp/mlp_classifier.py:
  - init_weights_xavier()  : Initialize W1, b1, W2, b2
  - train_step()           : Forward + Backward + SGD Update
  - forward()              : Inference (no dropout)
  - cross_entropy_loss()   : Loss computation
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── เพิ่ม mlp/ เข้า sys.path ─────────────────────────────────────────────────
_MLP_DIR = Path(__file__).resolve().parent.parent / "mlp"
if str(_MLP_DIR) not in sys.path:
    sys.path.insert(0, str(_MLP_DIR))

from mlp_classifier import init_weights_xavier, train_step, forward, cross_entropy_loss


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """ผลลัพธ์จากการเทรน 1 Trial (Multi-class Classifier)"""
    W1: np.ndarray                  # Best Layer-1 weights (input_dim, hidden_dim)
    b1: np.ndarray                  # Best Layer-1 bias   (hidden_dim,)
    W2: np.ndarray                  # Best Layer-2 weights (hidden_dim, n_classes)
    b2: np.ndarray                  # Best Layer-2 bias   (n_classes,)
    best_val_ce: float              # Validation Cross-Entropy ณ Best Epoch
    best_epoch: int                 # Epoch ที่ได้ Best Val CE
    train_loss_history: List[float] # Train CE ทุก Epoch
    val_loss_history: List[float]   # Val CE ทุก Epoch
    stopped_early: bool             # True = หยุดก่อนครบ max_epochs
    total_epochs_run: int           # จำนวน Epoch ที่รันจริง
    n_classes: int = 9              # จำนวนคลาส
    hidden_dim: int = 64            # Hidden layer size


# ─────────────────────────────────────────────────────────────────────────────
# Batch Generator
# ─────────────────────────────────────────────────────────────────────────────

def _iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """สุ่ม Shuffle แล้วแบ่ง Mini-batches"""
    n = X.shape[0]
    if batch_size >= n:
        return [(X, y)]
    perm = rng.permutation(n)
    X_s, y_s = X[perm], y[perm]
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((X_s[start:end], y_s[start:end]))
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Validation Loss (Cross-Entropy)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_val_ce(
    X_val: np.ndarray,
    y_idx_val: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> float:
    """คำนวณ Cross-Entropy บน Validation Set (ไม่มี Dropout)"""
    probs, _, _, _ = forward(X_val, W1, b1, W2, b2, dropout_rate=0.0, training=False)
    return cross_entropy_loss(probs, y_idx_val)


# ─────────────────────────────────────────────────────────────────────────────
# Label Conversion Helper
# ─────────────────────────────────────────────────────────────────────────────

def labels_to_class_idx(y: np.ndarray, n_classes: int = 9) -> np.ndarray:
    """แปลง label (จำนวนกล่อง 1-9) → class index 0-indexed

    Parameters
    ----------
    y        : float32 (N, 1) หรือ (N,) — จำนวนกล่อง (1-9)
    n_classes: จำนวนคลาส (default 9)

    Returns
    -------
    y_idx : int32 (N,) — class index 0 to n_classes-1
    """
    y_flat = y.reshape(-1).astype(int)
    y_idx = np.clip(y_flat - 1, 0, n_classes - 1).astype(np.int32)
    return y_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer Class
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Training Loop Handler สำหรับ 1 Trial (Multi-class Classifier)

    Parameters
    ----------
    n_classes   : จำนวนคลาส (default: 9 สำหรับ 1-9 กล่อง)
    hidden_dim  : ขนาด Hidden Layer (default: 64)
    dropout_rate: Dropout rate (default: 0.3)
    max_epochs  : จำนวน Epoch สูงสุด (default: 500)
    patience    : Early Stopping patience (default: 30)
    random_seed : seed สำหรับ reproducibility
    verbose     : แสดง Progress ทุกกี่ Epoch (0 = ปิด)
    """

    def __init__(
        self,
        n_classes: int = 9,
        hidden_dim: int = 64,
        dropout_rate: float = 0.3,
        max_epochs: int = 500,
        patience: int = 30,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        self.n_classes    = n_classes
        self.hidden_dim   = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.rng          = np.random.default_rng(random_seed)
        self.verbose      = verbose

    def fit(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate: float = 0.01,
        batch_size: int = 16,
    ) -> TrainResult:
        """เทรน MLP Classifier สำหรับ 1 Configuration พร้อม Early Stopping

        Parameters
        ----------
        Z_train       : float32 (N_train, n_features) — SVD + Stat features
        y_train       : float32 (N_train, 1)          — จำนวนกล่อง (1-9)
        Z_val         : float32 (N_val, n_features)
        y_val         : float32 (N_val, 1)
        learning_rate : Learning Rate
        batch_size    : Mini-batch size

        Returns
        -------
        TrainResult : W1, b1, W2, b2 ที่ดีที่สุด และ Loss History
        """
        input_dim = Z_train.shape[1]

        # แปลง label → class index
        y_idx_train = labels_to_class_idx(y_train, self.n_classes)
        y_idx_val   = labels_to_class_idx(y_val,   self.n_classes)

        # Xavier Initialization
        W1, b1, W2, b2 = init_weights_xavier(
            input_dim, self.hidden_dim, self.n_classes, self.rng
        )

        best_W1 = W1.copy(); best_b1 = b1.copy()
        best_W2 = W2.copy(); best_b2 = b2.copy()
        best_val  = float("inf")
        best_ep   = 0
        no_improve = 0

        train_history: List[float] = []
        val_history:   List[float] = []

        for epoch in range(1, self.max_epochs + 1):
            batches = _iter_batches(Z_train, y_idx_train, batch_size, self.rng)
            epoch_losses: List[float] = []

            for X_b, y_b in batches:
                W1, b1, W2, b2, batch_loss = train_step(
                    X_b, y_b,
                    W1, b1, W2, b2,
                    learning_rate=learning_rate,
                    dropout_rate=self.dropout_rate,
                    rng=self.rng,
                )
                epoch_losses.append(batch_loss)

            train_ce = float(np.mean(epoch_losses))
            val_ce   = _compute_val_ce(Z_val, y_idx_val, W1, b1, W2, b2)

            train_history.append(train_ce)
            val_history.append(val_ce)

            # Checkpoint
            if val_ce < best_val - 1e-8:
                best_val  = val_ce
                best_ep   = epoch
                best_W1 = W1.copy(); best_b1 = b1.copy()
                best_W2 = W2.copy(); best_b2 = b2.copy()
                no_improve = 0
            else:
                no_improve += 1

            # Early Stopping
            if no_improve >= self.patience:
                if self.verbose > 0:
                    print(
                        f"    Early Stop @ epoch {epoch:4d} | "
                        f"Best Val CE: {best_val:.6f} @ epoch {best_ep}"
                    )
                return TrainResult(
                    W1=best_W1, b1=best_b1, W2=best_W2, b2=best_b2,
                    best_val_ce=best_val, best_epoch=best_ep,
                    train_loss_history=train_history,
                    val_loss_history=val_history,
                    stopped_early=True,
                    total_epochs_run=epoch,
                    n_classes=self.n_classes,
                    hidden_dim=self.hidden_dim,
                )

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(
                    f"    Epoch {epoch:4d}/{self.max_epochs} | "
                    f"Train CE: {train_ce:.6f} | Val CE: {val_ce:.6f}"
                )

        return TrainResult(
            W1=best_W1, b1=best_b1, W2=best_W2, b2=best_b2,
            best_val_ce=best_val, best_epoch=best_ep,
            train_loss_history=train_history,
            val_loss_history=val_history,
            stopped_early=False,
            total_epochs_run=self.max_epochs,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
        )
