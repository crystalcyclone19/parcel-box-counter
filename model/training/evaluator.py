# model/training/evaluator.py
"""
Stage 4 (Part A): Metrics Computation & Model Ranking
======================================================
คำนวณ Metrics หลายตัวพร้อมกันสำหรับการจัดอันดับโมเดล:
  - MSE (Mean Squared Error)     — Primary Metric (ต่ำ = ดี)
  - MAE (Mean Absolute Error)    — ความคลาดเคลื่อนเฉลี่ยในหน่วย "จำนวนกล่อง"
  - R² Score                     — Coefficient of Determination
  - Accuracy Exact               — % ที่ปัดเศษแล้วตรงพอดี
  - Accuracy ±1                  — % ที่คลาดเคลื่อนไม่เกิน 1 กล่อง
  - Generalization Gap           — |Train MSE - Val MSE|
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_MLP_DIR = Path(__file__).resolve().parent.parent / "mlp"
if str(_MLP_DIR) not in sys.path:
    sys.path.insert(0, str(_MLP_DIR))

from mlp_regression import sigmoid


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    """ผลลัพธ์ Metrics ครบชุดสำหรับ 1 Trial"""
    mse:            float
    mae:            float
    r2:             float
    acc_exact:      float   # % ปัดเศษแล้วตรงพอดี (|round(y_hat) - y| == 0)
    acc_within_1:   float   # % คลาดเคลื่อน ≤ 1 กล่อง
    gen_gap:        Optional[float] = None   # |train_mse - val_mse| (ถ้าระบุ)

    def summary(self) -> str:
        gap_str = f"{self.gen_gap:.6f}" if self.gen_gap is not None else "N/A"
        return (
            f"  MSE             : {self.mse:.6f}\n"
            f"  MAE             : {self.mae:.6f} boxes\n"
            f"  R2              : {self.r2:.4f}\n"
            f"  Acc (Exact)     : {self.acc_exact*100:.2f}%\n"
            f"  Acc (+-1 box)   : {self.acc_within_1*100:.2f}%\n"
            f"  Gen. Gap        : {gap_str}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def _predict(Z: np.ndarray, W: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """คำนวณ y_hat = k * sigmoid(Z @ W + b)"""
    return k * sigmoid(np.dot(Z, W) + b)


def compute_metrics(
    Z: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    k: float,
    train_mse: Optional[float] = None,
) -> MetricsResult:
    """คำนวณ Metrics ครบชุดสำหรับ Dataset ที่กำหนด

    Parameters
    ----------
    Z          : float32 (N, n_features) — SVD-projected feature matrix
    y          : float32 (N, 1)          — Ground Truth labels
    W, b       : Best Model weights & bias
    k          : Scaled Sigmoid Max
    train_mse  : ถ้าระบุ จะคำนวณ Generalization Gap ด้วย

    Returns
    -------
    MetricsResult : Metrics ครบชุด
    """
    y_hat = _predict(Z, W, b, k)   # (N, 1)
    y_true = y.reshape(-1)
    y_pred = y_hat.reshape(-1)

    n = len(y_true)

    # ── Primary Metrics ──────────────────────────────────────────────────────
    residuals = y_pred - y_true
    mse = float(0.5 * np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))

    # ── R² Score ─────────────────────────────────────────────────────────────
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    # ── Accuracy Metrics ─────────────────────────────────────────────────────
    rounded = np.round(y_pred)
    acc_exact    = float(np.mean(np.abs(rounded - y_true) == 0))
    acc_within_1 = float(np.mean(np.abs(y_pred - y_true) <= 1.0))

    # ── Generalization Gap ───────────────────────────────────────────────────
    gen_gap = abs(train_mse - mse) if train_mse is not None else None

    return MetricsResult(
        mse=mse,
        mae=mae,
        r2=r2,
        acc_exact=acc_exact,
        acc_within_1=acc_within_1,
        gen_gap=gen_gap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator Class
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """จัดการการประเมินและจัดอันดับโมเดลทั้งหมด

    Parameters
    ----------
    primary_metric : metric หลักสำหรับจัดอันดับ ('mse', 'mae', 'r2', 'acc_within_1')
                     'mse' และ 'mae' → ค่าน้อย = ดี
                     'r2' และ 'acc_within_1' → ค่ามาก = ดี
    """

    def __init__(self, primary_metric: str = "mse"):
        self.primary_metric = primary_metric
        self._higher_is_better = primary_metric in ("r2", "acc_within_1", "acc_exact")

    def is_better(self, new_metric: float, current_best: float) -> bool:
        """เปรียบเทียบว่า new_metric ดีกว่า current_best หรือไม่"""
        if self._higher_is_better:
            return new_metric > current_best + 1e-8
        else:
            return new_metric < current_best - 1e-8

    def get_primary_value(self, metrics: MetricsResult) -> float:
        """ดึงค่า Primary Metric จาก MetricsResult"""
        return getattr(metrics, self.primary_metric)
