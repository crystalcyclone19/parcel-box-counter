# model/training/trainer.py
"""
Stage 3 (Part A): Training Loop with Early Stopping & Checkpointing
=====================================================================
ดำเนินการเทรน MLP สำหรับ 1 Configuration (Trial) โดย:
  - รับ Feature Matrix ที่ผ่าน SVD Projection แล้ว (Z_train, Z_val)
  - รัน Gradient Descent Loop สูงสุด `max_epochs` รอบ
  - ติดตาม Validation MSE ทุก Epoch → Early Stopping เมื่อไม่ดีขึ้น
  - บันทึก Best Weights ณ Epoch ที่ Val Loss ต่ำสุด (Checkpoint)

ใช้โมดูลจาก mlp/mlp_regression.py:
  - sigmoid() สำหรับ Scaled Sigmoid Activation
  - train_step() สำหรับ Forward + Backward Pass
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

from mlp_regression import sigmoid, train_step


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """ผลลัพธ์จากการเทรน 1 Trial"""
    W: np.ndarray                   # Best weights  shape (input_dim, 1)
    b: np.ndarray                   # Best bias      shape (1,)
    best_val_mse: float             # Validation MSE ณ Best Epoch
    best_epoch: int                 # Epoch ที่ได้ Best Val MSE
    train_loss_history: List[float] # Train MSE ทุก Epoch (ก่อน Early Stop)
    val_loss_history: List[float]   # Val MSE ทุก Epoch
    stopped_early: bool             # True = หยุดก่อนครบ max_epochs
    total_epochs_run: int           # จำนวน Epoch ที่รันจริง


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialization
# ─────────────────────────────────────────────────────────────────────────────

def _init_weights_xavier(input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Xavier/Glorot Normal Initialization

    W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    สำหรับ Linear Layer → Scaled Sigmoid ซึ่งมี fan_out = 1
    """
    std = np.sqrt(2.0 / (input_dim + 1))
    W = rng.standard_normal((input_dim, 1)).astype(np.float32) * std
    b = np.zeros((1,), dtype=np.float32)
    return W, b


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
        # Full Batch
        return [(X, y)]
    perm = rng.permutation(n)
    X_s, y_s = X[perm], y[perm]
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((X_s[start:end], y_s[start:end]))
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# MSE Validation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_val_mse(
    X_val: np.ndarray,
    y_val: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    k: float,
) -> float:
    """คำนวณ MSE บน Validation Set (ไม่มี Gradient)"""
    y_hat = k * sigmoid(np.dot(X_val, W) + b)
    return float(0.5 * np.mean((y_hat - y_val) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer Class
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Training Loop Handler สำหรับ 1 Trial

    Parameters
    ----------
    k           : Scaled Sigmoid Max (จำนวนกล่องสูงสุดที่คาดหวัง)
    max_epochs  : จำนวน Epoch สูงสุด (default: 500)
    patience    : Early Stopping patience — หยุดหาก Val MSE ไม่ลดลง N Epochs (default: 25)
    random_seed : seed สำหรับ reproducibility
    verbose     : แสดง Progress ทุกกี่ Epoch (0 = ปิด)
    """

    def __init__(
        self,
        k: float = 10.0,
        max_epochs: int = 500,
        patience: int = 25,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        self.k = k
        self.max_epochs = max_epochs
        self.patience = patience
        self.rng = np.random.default_rng(random_seed)
        self.verbose = verbose

    def fit(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate: float = 0.01,
        batch_size: int = 16,
    ) -> TrainResult:
        """เทรน MLP สำหรับ 1 Configuration พร้อม Early Stopping

        Parameters
        ----------
        Z_train       : float32 (N_train, n_features) — SVD-projected features
        y_train       : float32 (N_train, 1)          — labels
        Z_val         : float32 (N_val, n_features)
        y_val         : float32 (N_val, 1)
        learning_rate : Learning Rate สำหรับ Gradient Descent
        batch_size    : ขนาด Mini-batch (ถ้า >= N จะเป็น Full Batch)

        Returns
        -------
        TrainResult : ผลลัพธ์รวม W, b ที่ดีที่สุด และ Loss History
        """
        input_dim = Z_train.shape[1]

        # Xavier Initialization (กำหนดเป็นค่าคงที่ตามเอกสาร)
        W, b = _init_weights_xavier(input_dim, self.rng)

        best_W    = W.copy()
        best_b    = b.copy()
        best_val  = float("inf")
        best_ep   = 0
        no_improve = 0

        train_history: List[float] = []
        val_history:   List[float] = []

        for epoch in range(1, self.max_epochs + 1):
            # ── Mini-batch Training ─────────────────────────────────────────
            batches = _iter_batches(Z_train, y_train, batch_size, self.rng)
            epoch_losses: List[float] = []

            for X_b, y_b in batches:
                _, W, b, batch_loss = train_step(X_b, y_b, W, b, k=self.k, learning_rate=learning_rate)
                epoch_losses.append(batch_loss)

            train_mse = float(np.mean(epoch_losses))
            val_mse   = _compute_val_mse(Z_val, y_val, W, b, self.k)

            train_history.append(train_mse)
            val_history.append(val_mse)

            # ── Checkpoint: บันทึก Best Weights ─────────────────────────────
            if val_mse < best_val - 1e-8:
                best_val  = val_mse
                best_ep   = epoch
                best_W    = W.copy()
                best_b    = b.copy()
                no_improve = 0
            else:
                no_improve += 1

            # ── Early Stopping ───────────────────────────────────────────────
            if no_improve >= self.patience:
                if self.verbose > 0:
                    print(
                        f"    Early Stop @ epoch {epoch:4d} | "
                        f"Best Val MSE: {best_val:.6f} @ epoch {best_ep}"
                    )
                return TrainResult(
                    W=best_W, b=best_b,
                    best_val_mse=best_val, best_epoch=best_ep,
                    train_loss_history=train_history,
                    val_loss_history=val_history,
                    stopped_early=True,
                    total_epochs_run=epoch,
                )

            # ── Verbose Logging ──────────────────────────────────────────────
            if self.verbose > 0 and epoch % self.verbose == 0:
                print(
                    f"    Epoch {epoch:4d}/{self.max_epochs} | "
                    f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}"
                )

        return TrainResult(
            W=best_W, b=best_b,
            best_val_mse=best_val, best_epoch=best_ep,
            train_loss_history=train_history,
            val_loss_history=val_history,
            stopped_early=False,
            total_epochs_run=self.max_epochs,
        )
