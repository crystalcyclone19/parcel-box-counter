# model/training/tuner.py
"""
Stage 3 (Part B): Grid Search Hyperparameter Tuner (Multi-class Classification)
================================================================================
วนลูปทดสอบทุก Configuration ใน Search Space อัตโนมัติ (Full Grid Search)

Search Space:
  clip_percentile : [95.0, 98.0, 99.0]           — 3 ค่า
  learning_rate   : [0.001, 0.005, 0.01, 0.05]   — 4 ค่า
  batch_size      : [8, 16, 32]                   — 3 ค่า
  hidden_dim      : [32, 64, 128]                 — 3 ค่า
  ───────────────────────────────────────────────────────
  รวม: 3 × 4 × 3 × 3 = 108 Trials

Fixed Parameters:
  n_svd_features  : 64            (Top-64 SVD components)
  stat_features   : 3             (total_edge_area, log_contour_count, aspect_ratio)
  input_dim       : 64 + 3 = 67
  n_classes       : 8             (กล่อง 1-8)
  blur_ksize      : 5
  epochs          : 500 (max)
  patience        : 30
  dropout_rate    : 0.3
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))
_PREPROCESSING_DIR = _TRAINING_DIR.parent / "preprocessing"
if str(_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_DIR))

try:
    from .dataset import extract_edge_features, DatasetSplit
except ImportError:
    from dataset import extract_edge_features, DatasetSplit

from trainer import Trainer, TrainResult
from evaluator import Evaluator, compute_metrics, MetricsResult


# ─────────────────────────────────────────────────────────────────────────────
# Default Search Space
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "clip_percentile": [95.0, 98.0, 99.0],
    "learning_rate":   [0.001, 0.005, 0.01, 0.05],
    "batch_size":      [8, 16, 32],
    "hidden_dim":      [32, 64, 128],
}

# Fixed Constants
FIXED_N_SVD_FEATURES:  int   = 64
FIXED_N_STAT_FEATURES: int   = 3       # total_edge_area, log_contour_count, aspect_ratio
FIXED_N_CLASSES:       int   = 8       # กล่อง 1-8
FIXED_BLUR_KSIZE:      int   = 5
FIXED_MAX_EPOCHS:      int   = 500
FIXED_PATIENCE:        int   = 30
FIXED_DROPOUT_RATE:    float = 0.3
FIXED_WEIGHT_INIT:     str   = "xavier"
FIXED_IMAGE_SIZE:      int   = 128
FIXED_RESIZE_TARGET:   Tuple = (512, 512)


def build_search_space(
    clip_percentile: Optional[List[float]] = None,
    learning_rate:   Optional[List[float]] = None,
    batch_size:      Optional[List[int]]   = None,
    hidden_dim:      Optional[List[int]]   = None,
) -> Dict[str, List[Any]]:
    """สร้าง Search Space พร้อม override ค่าเฉพาะได้"""
    return {
        "clip_percentile": clip_percentile or DEFAULT_SEARCH_SPACE["clip_percentile"],
        "learning_rate":   learning_rate   or DEFAULT_SEARCH_SPACE["learning_rate"],
        "batch_size":      batch_size      or DEFAULT_SEARCH_SPACE["batch_size"],
        "hidden_dim":      hidden_dim      or DEFAULT_SEARCH_SPACE["hidden_dim"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trial Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """ผลลัพธ์ของ 1 Trial"""
    trial_id:        int
    clip_percentile: float
    learning_rate:   float
    batch_size:      int
    hidden_dim:      int
    # Fixed params
    n_classes:       int   = FIXED_N_CLASSES
    n_svd_features:  int   = FIXED_N_SVD_FEATURES
    n_stat_features: int   = FIXED_N_STAT_FEATURES
    blur_ksize:      int   = FIXED_BLUR_KSIZE
    max_epochs:      int   = FIXED_MAX_EPOCHS
    weight_init:     str   = FIXED_WEIGHT_INIT

    # Results
    train_result:    Optional[TrainResult]   = field(default=None, repr=False)
    val_metrics:     Optional[MetricsResult] = None
    train_ce_final:  float = float("inf")
    elapsed_sec:     float = 0.0

    @property
    def val_ce(self) -> float:
        return self.val_metrics.cross_entropy if self.val_metrics else float("inf")

    @property
    def val_acc(self) -> float:
        return self.val_metrics.acc_exact if self.val_metrics else 0.0

    @property
    def val_f1(self) -> float:
        return self.val_metrics.macro_f1 if self.val_metrics else 0.0

    # Backward-compat alias
    @property
    def val_mse(self) -> float:
        return self.val_ce

    @property
    def val_mae(self) -> float:
        return 0.0

    def summary_line(self, total: int) -> str:
        es = "✓ES" if (self.train_result and self.train_result.stopped_early) else "   "
        return (
            f"Trial {self.trial_id:03d}/{total:03d} {es} | "
            f"Clip={self.clip_percentile:.0f}, LR={self.learning_rate:.4f}, "
            f"Batch={self.batch_size:2d}, Hidden={self.hidden_dim:3d} | "
            f"Val CE: {self.val_ce:.4f} | "
            f"Acc: {self.val_acc*100:.1f}% | "
            f"F1: {self.val_f1:.3f} | "
            f"{self.elapsed_sec:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SVD Fitting & Projection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVDModel:
    """SVD Projection Matrix + Stat Feature Normalizer สำหรับ 1 ค่า clip_percentile"""
    clip_percentile: float
    mean: np.ndarray       # Train pixel mean   (D_pixel,)
    Vt:   np.ndarray       # Top-k right vectors (n_svd, D_pixel)
    S:    np.ndarray       # Singular values     (n_svd,)
    stat_mean: np.ndarray  # Train stat mean    (3,)
    stat_std:  np.ndarray  # Train stat std     (3,)

    def project(self, X_pixel: np.ndarray, X_stat: np.ndarray) -> np.ndarray:
        """Project raw features → combined feature vector Z

        Parameters
        ----------
        X_pixel : (N, D_pixel) — Normalized pixel features
        X_stat  : (N, 3)       — Statistical edge features

        Returns
        -------
        Z : (N, n_svd + 3) — Combined feature vector
        """
        # SVD Projection
        X_centered = X_pixel.astype(np.float32) - self.mean[None, :]
        Z_svd = (X_centered @ self.Vt.T).astype(np.float32)

        # Normalize Statistical Features (z-score ด้วย train stats)
        Z_stat = ((X_stat - self.stat_mean[None, :]) /
                  (self.stat_std[None, :] + 1e-8)).astype(np.float32)

        return np.concatenate([Z_svd, Z_stat], axis=1)


def _fit_svd(
    X_train_pixel: np.ndarray,
    X_train_stat:  np.ndarray,
    n_features: int = FIXED_N_SVD_FEATURES,
    random_state: int = 42,
) -> SVDModel:
    """Fit SVD + Stat Normalizer บน Train Set เท่านั้น (ป้องกัน Data Leakage)"""
    from sklearn.utils.extmath import randomized_svd as _rsvd

    # SVD on pixel features
    mean = X_train_pixel.mean(axis=0, dtype=np.float32)
    X_c = (X_train_pixel - mean[None, :]).astype(np.float32)
    _, S, Vt = _rsvd(X_c, n_components=n_features, n_iter=5, random_state=random_state)

    # Stat feature normalization
    stat_mean = X_train_stat.mean(axis=0, dtype=np.float32)
    stat_std  = X_train_stat.std(axis=0,  dtype=np.float32)

    return SVDModel(
        clip_percentile=0.0,
        mean=mean,
        Vt=Vt.astype(np.float32),
        S=S.astype(np.float32),
        stat_mean=stat_mean,
        stat_std=stat_std,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grid Search Tuner
# ─────────────────────────────────────────────────────────────────────────────

class GridSearchTuner:
    """Full Grid Search ผ่านทุก Configuration ใน Search Space

    หลักการ:
      1. สำหรับแต่ละค่า clip_percentile → Preprocess + Cache Feature Matrix
      2. Fit SVD + Stat Normalizer บน X_train
      3. Project (X_train, X_val, X_test) → Z_train, Z_val, Z_test
      4. วนลูปทุกคู่ (lr × batch_size × hidden_dim) → เทรน + ประเมิน

    Parameters
    ----------
    split             : DatasetSplit
    search_space      : dict ของ Tunable Parameters
    evaluator         : Evaluator instance
    n_svd_features    : จำนวน SVD components (default: 64)
    n_classes         : จำนวนคลาส (default: 8)
    random_seed       : seed สำหรับ reproducibility
    augment_train     : True = ทำ Augmentation บน Train Set (default: True)
    """

    def __init__(
        self,
        split: DatasetSplit,
        k: float = 8.0,            # ไม่ใช้แล้วใน Classification แต่เก็บไว้เพื่อ Compat
        search_space: Optional[Dict[str, List[Any]]] = None,
        evaluator: Optional[Evaluator] = None,
        n_svd_features: int = FIXED_N_SVD_FEATURES,
        n_classes: int = FIXED_N_CLASSES,
        random_seed: int = 42,
        augment_train: bool = True,
    ):
        self.split = split
        self.k = k
        self.search_space = search_space or build_search_space()
        self.evaluator = evaluator or Evaluator(primary_metric="cross_entropy")
        self.n_svd_features = n_svd_features
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.augment_train = augment_train
        self.n_train_effective = split.n_train

        self.all_trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        self.best_svd: Optional[SVDModel] = None

    # ── Build Configurations ──────────────────────────────────────────────────
    def _build_configs(self) -> List[Dict[str, Any]]:
        keys = ["clip_percentile", "learning_rate", "batch_size", "hidden_dim"]
        vals = [self.search_space.get(k, DEFAULT_SEARCH_SPACE[k]) for k in keys]
        return [dict(zip(keys, combo)) for combo in product(*vals)]

    # ── Preprocess + Cache ────────────────────────────────────────────────────
    def _preprocess_and_cache(
        self, clip_percentile: float
    ) -> Tuple[SVDModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Feature Extraction + SVD สำหรับ 1 ค่า clip_percentile

        Returns (svd_model, Z_train, y_train, Z_val, y_val, Z_test, y_test)
        """
        print(f"\n  [Cache] Preprocessing clip_percentile={clip_percentile:.0f}% ...")
        t0 = time.time()

        # 1. Train Set (with optional Augmentation)
        X_train_pixel, X_train_stat, y_train = extract_edge_features(
            self.split.train,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=clip_percentile,
            image_size=FIXED_IMAGE_SIZE,
            augment=self.augment_train,
        )
        self.n_train_effective = len(X_train_pixel)

        # 2. Validation Set (Clean — No Augmentation)
        X_val_pixel, X_val_stat, y_val = extract_edge_features(
            self.split.val,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=clip_percentile,
            image_size=FIXED_IMAGE_SIZE,
            augment=False,
        )

        # 3. Test Set (Clean — No Augmentation)
        X_test_pixel, X_test_stat, y_test = extract_edge_features(
            self.split.test,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=clip_percentile,
            image_size=FIXED_IMAGE_SIZE,
            augment=False,
        )

        # Fit SVD + Stat Normalizer บน Train Set เท่านั้น
        svd = _fit_svd(X_train_pixel, X_train_stat, n_features=self.n_svd_features, random_state=self.random_seed)
        svd.clip_percentile = clip_percentile

        Z_train = svd.project(X_train_pixel, X_train_stat)
        Z_val   = svd.project(X_val_pixel,   X_val_stat)
        Z_test  = svd.project(X_test_pixel,  X_test_stat)

        elapsed = time.time() - t0
        print(
            f"  [Cache] สำเร็จ | Z_train={Z_train.shape}, Z_val={Z_val.shape}, "
            f"Z_test={Z_test.shape} | {elapsed:.1f}s"
        )
        return svd, Z_train, y_train, Z_val, y_val, Z_test, y_test

    # ── Main Search Loop ──────────────────────────────────────────────────────
    def run(self) -> "GridSearchTuner":
        """รัน Full Grid Search"""
        configs = self._build_configs()
        total = len(configs)

        print(f"\n{'='*75}")
        print(f"  PARCEL BOX COUNTER — MULTI-CLASS CLASSIFICATION TRAINING")
        print(f"{'='*75}")
        print(f"  Total Trials : {total}")
        print(f"  Classes      : {self.n_classes} (กล่อง 1-{self.n_classes})")
        print(f"  n_svd        : {self.n_svd_features} | n_stat: {FIXED_N_STAT_FEATURES} | input_dim: {self.n_svd_features + FIXED_N_STAT_FEATURES}")
        print(f"  Max Epochs   : {FIXED_MAX_EPOCHS} | Patience: {FIXED_PATIENCE} | Dropout: {FIXED_DROPOUT_RATE}")
        print(f"{'='*75}\n")

        # Cache Feature Matrices per clip_percentile
        clip_values = self.search_space.get("clip_percentile", DEFAULT_SEARCH_SPACE["clip_percentile"])
        clip_cache: Dict[float, Tuple] = {}
        print(f"[Stage 1] Feature Extraction & SVD Caching ({len(clip_values)} percentile values)...")
        for cp in clip_values:
            clip_cache[cp] = self._preprocess_and_cache(cp)

        # Trial Loop
        print(f"\n[Stage 2] Grid Search ({total} Trials)...")
        trial_id = 0
        best_primary = float("inf") if not self.evaluator._higher_is_better else float("-inf")

        for config in configs:
            trial_id += 1
            cp  = config["clip_percentile"]
            lr  = config["learning_rate"]
            bs  = config["batch_size"]
            hd  = config["hidden_dim"]

            svd, Z_train, y_train, Z_val, y_val, Z_test, y_test = clip_cache[cp]

            t_start = time.time()

            # Train
            trainer = Trainer(
                n_classes=self.n_classes,
                hidden_dim=hd,
                dropout_rate=FIXED_DROPOUT_RATE,
                max_epochs=FIXED_MAX_EPOCHS,
                patience=FIXED_PATIENCE,
                random_seed=self.random_seed + trial_id,
                verbose=0,
            )
            train_result = trainer.fit(
                Z_train, y_train,
                Z_val,   y_val,
                learning_rate=lr,
                batch_size=bs,
            )

            # Evaluate on Validation Set
            train_ce_final = train_result.train_loss_history[-1] if train_result.train_loss_history else float("inf")
            val_metrics = compute_metrics(
                Z_val, y_val,
                train_result.W1, train_result.b1,
                train_result.W2, train_result.b2,
                n_classes=self.n_classes,
                train_ce=train_ce_final,
            )

            elapsed = time.time() - t_start

            trial = TrialResult(
                trial_id=trial_id,
                clip_percentile=cp,
                learning_rate=lr,
                batch_size=bs,
                hidden_dim=hd,
                n_classes=self.n_classes,
                train_result=train_result,
                val_metrics=val_metrics,
                train_ce_final=train_ce_final,
                elapsed_sec=elapsed,
            )
            self.all_trials.append(trial)

            # Check Best
            primary_val = self.evaluator.get_primary_value(val_metrics)
            if self.evaluator.is_better(primary_val, best_primary):
                best_primary = primary_val
                self.best_trial = trial
                self.best_svd   = svd
                marker = " ★ New Best!"
            else:
                marker = ""

            print(f"  {trial.summary_line(total)}{marker}")

        metric_label = self.evaluator.primary_metric.replace("_", " ").title()
        print(f"\n[Stage 2] Grid Search เสร็จสิ้น! Best Val {metric_label}: {best_primary:.6f}")
        return self

    # ── Leaderboard ───────────────────────────────────────────────────────────
    def get_leaderboard(self) -> List[TrialResult]:
        """คืน Trial ทั้งหมดเรียงจากดีสุดไปแย่สุดตาม Primary Metric"""
        reverse = self.evaluator._higher_is_better
        return sorted(
            self.all_trials,
            key=lambda t: self.evaluator.get_primary_value(t.val_metrics) if t.val_metrics else float("inf"),
            reverse=reverse,
        )
