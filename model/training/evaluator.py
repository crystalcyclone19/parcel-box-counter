# model/training/evaluator.py
"""
Stage 4 (Part A): Classification Metrics Computation & Model Ranking
======================================================================
คำนวณ Metrics สำหรับ Multi-class Classification ครบชุด:
  - Cross-Entropy Loss      — Primary Metric สำหรับ Early Stopping (ต่ำ = ดี)
  - Accuracy (Exact)        — % ที่ทำนายถูกต้องพอดี
  - Accuracy ±1             — % ที่ทำนายคลาดเคลื่อนไม่เกิน 1 กล่อง
  - Macro F1-Score          — F1 เฉลี่ยทุกคลาส (สำคัญเมื่อ dataset imbalanced)
  - Confusion Matrix        — สำหรับ Debug Pattern
  - Generalization Gap      — |Train CE - Val CE|
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

_MLP_DIR = Path(__file__).resolve().parent.parent / "mlp"
if str(_MLP_DIR) not in sys.path:
    sys.path.insert(0, str(_MLP_DIR))

from mlp_classifier import predict as mlp_predict, cross_entropy_loss


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    """ผลลัพธ์ Metrics ครบชุดสำหรับ 1 Trial (Classification)"""
    cross_entropy:  float           # Cross-Entropy Loss (ต่ำ = ดี)
    acc_exact:      float           # % ทำนายคลาสถูกต้องพอดี
    acc_within_1:   float           # % คลาดเคลื่อน ≤ 1 กล่อง (Ordinal Tolerance)
    macro_f1:       float           # Macro-averaged F1 Score
    confusion_matrix: np.ndarray    # (n_classes, n_classes)
    gen_gap:        Optional[float] = None   # |train_ce - val_ce|

    # Convenience: สร้างทางลัดชื่อ mse/mae เพื่อ Backward Compatibility
    @property
    def mse(self) -> float:
        """Alias → cross_entropy (ใช้เพื่อ Compatibility กับ ModelRegistry)"""
        return self.cross_entropy

    @property
    def mae(self) -> float:
        """Placeholder — ไม่มีความหมายสำหรับ Classification, คืนค่า 0"""
        return 0.0

    @property
    def r2(self) -> float:
        """Placeholder — ไม่มีความหมายสำหรับ Classification, คืนค่า 0"""
        return 0.0

    def summary(self) -> str:
        gap_str = f"{self.gen_gap:.6f}" if self.gen_gap is not None else "N/A"
        return (
            f"  Cross-Entropy   : {self.cross_entropy:.6f}\n"
            f"  Acc (Exact)     : {self.acc_exact*100:.2f}%\n"
            f"  Acc (±1 box)    : {self.acc_within_1*100:.2f}%\n"
            f"  Macro F1        : {self.macro_f1:.4f}\n"
            f"  Gen. Gap (CE)   : {gap_str}\n"
        )

    def confusion_str(self, n_classes: int = 9) -> str:
        """แสดง Confusion Matrix แบบข้อความ"""
        header = "Pred→ " + "  ".join(f"{i+1:2d}" for i in range(n_classes))
        rows = [header, "-" * len(header)]
        for i, row in enumerate(self.confusion_matrix):
            rows.append(f"True {i+1}: " + "  ".join(f"{v:2d}" for v in row))
        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_macro_f1(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 Score (ไม่ใช้ sklearn เพื่อลด Dependencies)"""
    f1_scores: List[float] = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred_idx == c) & (y_true_idx == c)))
        fp = int(np.sum((y_pred_idx == c) & (y_true_idx != c)))
        fn = int(np.sum((y_pred_idx != c) & (y_true_idx == c)))

        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _compute_confusion_matrix(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """สร้าง Confusion Matrix (n_classes × n_classes)"""
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true_idx, y_pred_idx):
        cm[t, p] += 1
    return cm


def compute_metrics(
    Z: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    n_classes: int = 9,
    train_ce: Optional[float] = None,
) -> MetricsResult:
    """คำนวณ Metrics ครบชุดสำหรับ Dataset ที่กำหนด

    Parameters
    ----------
    Z          : float32 (N, n_features) — SVD+Stat feature matrix
    y          : float32 (N, 1)          — Ground Truth (จำนวนกล่อง 1-9)
    W1, b1     : Layer 1 weights & bias
    W2, b2     : Layer 2 weights & bias
    n_classes  : จำนวนคลาส (default: 9)
    train_ce   : ถ้าระบุ จะคำนวณ Generalization Gap

    Returns
    -------
    MetricsResult : Metrics ครบชุด
    """
    from trainer import labels_to_class_idx
    y_true_idx = labels_to_class_idx(y, n_classes)

    # Predict
    probs, y_pred_idx = mlp_predict(Z, W1, b1, W2, b2)

    # Cross-Entropy
    ce = cross_entropy_loss(probs, y_true_idx)

    # Accuracy (Exact)
    acc_exact = float(np.mean(y_pred_idx == y_true_idx))

    # Accuracy ±1 (Ordinal Tolerance — คลาดเคลื่อนไม่เกิน 1 คลาส = 1 กล่อง)
    diff = np.abs(y_pred_idx.astype(int) - y_true_idx.astype(int))
    acc_within_1 = float(np.mean(diff <= 1))

    # Macro F1
    macro_f1 = _compute_macro_f1(y_true_idx, y_pred_idx, n_classes)

    # Confusion Matrix
    cm = _compute_confusion_matrix(y_true_idx, y_pred_idx, n_classes)

    # Generalization Gap
    gen_gap = abs(train_ce - ce) if train_ce is not None else None

    return MetricsResult(
        cross_entropy=ce,
        acc_exact=acc_exact,
        acc_within_1=acc_within_1,
        macro_f1=macro_f1,
        confusion_matrix=cm,
        gen_gap=gen_gap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator Class
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """จัดการการประเมินและจัดอันดับโมเดลทั้งหมด

    Parameters
    ----------
    primary_metric : metric หลักสำหรับจัดอันดับ
        'cross_entropy' → ค่าน้อย = ดี  (default)
        'acc_exact'     → ค่ามาก = ดี
        'acc_within_1'  → ค่ามาก = ดี
        'macro_f1'      → ค่ามาก = ดี
    """

    HIGHER_IS_BETTER = {"acc_exact", "acc_within_1", "macro_f1", "r2"}

    def __init__(self, primary_metric: str = "cross_entropy"):
        self.primary_metric = primary_metric
        self._higher_is_better = primary_metric in self.HIGHER_IS_BETTER

    def is_better(self, new_metric: float, current_best: float) -> bool:
        """เปรียบเทียบว่า new_metric ดีกว่า current_best หรือไม่"""
        if self._higher_is_better:
            return new_metric > current_best + 1e-8
        else:
            return new_metric < current_best - 1e-8

    def get_primary_value(self, metrics: MetricsResult) -> float:
        """ดึงค่า Primary Metric จาก MetricsResult"""
        return getattr(metrics, self.primary_metric)
