# model/training/tuner.py
"""
Stage 3 (Part B): Grid Search Hyperparameter Tuner
====================================================
วนลูปทดสอบทุก Configuration ใน Search Space อัตโนมัติ (Full Grid Search)

Search Space (ตามเอกสาร auto_training_pipeline_design.md):
  clip_percentile : [95.0, 98.0, 99.0]      — 3 ค่า (Tunable)
  learning_rate   : [0.001, 0.005, 0.01, 0.05, 0.1]  — 5 ค่า (Tunable)
  batch_size      : [4, 8, 16, 32]           — 4 ค่า (Tunable)
  ───────────────────────────────────────────
  รวม: 3 × 5 × 4 = 60 Trials

Fixed Parameters:
  n_svd_features  : 64            (8×8 Matrix)
  k               : กำหนดโดยเจ้าของ (default: 10.0)
  weight_init     : 'xavier'      (คงที่ตลอด)
  blur_ksize      : 5             (kernel 5×5)
  epochs          : 500           (max)
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
_PREPROCESSING_DIR = _TRAINING_DIR.parent / "preprocessing"
if str(_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_DIR))

from dataset import extract_edge_features, DatasetSplit
from trainer import Trainer, TrainResult
from evaluator import Evaluator, compute_metrics, MetricsResult


# ─────────────────────────────────────────────────────────────────────────────
# Default Search Space
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "clip_percentile": [95.0, 98.0, 99.0],
    "learning_rate":   [0.001, 0.005, 0.01, 0.05, 0.1],
    "batch_size":      [4, 8, 16, 32],
}

# Fixed Constants (กำหนดตายตัวตามเอกสาร)
FIXED_N_SVD_FEATURES: int   = 64
FIXED_BLUR_KSIZE:     int   = 5
FIXED_MAX_EPOCHS:     int   = 500
FIXED_PATIENCE:       int   = 25
FIXED_WEIGHT_INIT:    str   = "xavier"
FIXED_IMAGE_SIZE:     int   = 128   # ขนาดก่อน flatten เข้า SVD
FIXED_RESIZE_TARGET:  Tuple = (512, 512)


def build_search_space(
    clip_percentile: Optional[List[float]] = None,
    learning_rate:   Optional[List[float]] = None,
    batch_size:      Optional[List[int]]   = None,
) -> Dict[str, List[Any]]:
    """สร้าง Search Space พร้อม override ค่าเฉพาะได้

    ถ้าไม่ระบุ ใช้ค่า Default ตามเอกสาร
    """
    return {
        "clip_percentile": clip_percentile or DEFAULT_SEARCH_SPACE["clip_percentile"],
        "learning_rate":   learning_rate   or DEFAULT_SEARCH_SPACE["learning_rate"],
        "batch_size":      batch_size      or DEFAULT_SEARCH_SPACE["batch_size"],
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
    # Fixed params (บันทึกไว้เพื่อ reproducibility)
    k:               float
    n_svd_features:  int   = FIXED_N_SVD_FEATURES
    blur_ksize:      int   = FIXED_BLUR_KSIZE
    max_epochs:      int   = FIXED_MAX_EPOCHS
    weight_init:     str   = FIXED_WEIGHT_INIT

    # Training results
    train_result:    Optional[TrainResult]  = field(default=None, repr=False)
    val_metrics:     Optional[MetricsResult] = None
    train_mse_final: float = float("inf")

    # Timing
    elapsed_sec:     float = 0.0

    @property
    def val_mse(self) -> float:
        return self.val_metrics.mse if self.val_metrics else float("inf")

    @property
    def val_mae(self) -> float:
        return self.val_metrics.mae if self.val_metrics else float("inf")

    def summary_line(self, total: int) -> str:
        es = "✓ES" if (self.train_result and self.train_result.stopped_early) else "   "
        return (
            f"Trial {self.trial_id:02d}/{total:02d} {es} | "
            f"Clip={self.clip_percentile:.0f}, "
            f"LR={self.learning_rate:.4f}, "
            f"Batch={self.batch_size:2d} | "
            f"Val MSE: {self.val_mse:.4f} | "
            f"Val MAE: {self.val_mae:.4f} | "
            f"{self.elapsed_sec:.2f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SVD Fitting & Projection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVDModel:
    """SVD Projection Matrix สำหรับ 1 ค่า clip_percentile"""
    clip_percentile: float
    mean: np.ndarray
    Vt:   np.ndarray
    S:    np.ndarray

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project X → Z โดยใช้ Mean_train และ Vt_train

        Parameters
        ----------
        X : float32 (N, D) — raw pixel features (Normalized 0-1)

        Returns
        -------
        Z : float32 (N, n_features)
        """
        X_centered = X.astype(np.float32) - self.mean[None, :]
        return (X_centered @ self.Vt.T).astype(np.float32)


def _fit_svd(
    X_train: np.ndarray,
    n_features: int = FIXED_N_SVD_FEATURES,
    random_state: int = 42,
) -> SVDModel:
    """Fit SVD บน Train Set เท่านั้น (ป้องกัน Data Leakage)"""
    from sklearn.utils.extmath import randomized_svd as _rsvd
    mean = X_train.mean(axis=0, dtype=np.float32)
    X_c = (X_train - mean[None, :]).astype(np.float32)
    U, S, Vt = _rsvd(X_c, n_components=n_features, n_iter=5, random_state=random_state)
    return SVDModel(clip_percentile=0.0, mean=mean, Vt=Vt.astype(np.float32), S=S.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Grid Search Tuner
# ─────────────────────────────────────────────────────────────────────────────

class GridSearchTuner:
    """Full Grid Search ผ่านทุก Configuration ใน Search Space

    หลักการ:
      1. สำหรับแต่ละค่า clip_percentile ที่ต่างกัน → Preprocess + Cache Feature Matrix
         (ทำเพียง 3 ครั้ง ไม่ต้อง Preprocess ซ้ำทุก Trial)
      2. Fit SVD บน X_train → ได้ (mean_train, Vt_train)
      3. Project X_train, X_val, X_test → Z_train, Z_val, Z_test
      4. วนลูปทุกคู่ (learning_rate × batch_size) → เทรน + ประเมิน

    Parameters
    ----------
    split             : DatasetSplit (train / val / test lists)
    k                 : Scaled Sigmoid Max (กำหนดโดยเจ้าของ)
    search_space      : dict ของ Tunable Parameters
    evaluator         : Evaluator instance (default: ใช้ MSE เป็น Primary)
    n_svd_features    : จำนวน SVD components (default: 64)
    random_seed       : seed สำหรับ reproducibility
    """

    def __init__(
        self,
        split: DatasetSplit,
        k: float = 10.0,
        search_space: Optional[Dict[str, List[Any]]] = None,
        evaluator: Optional[Evaluator] = None,
        n_svd_features: int = FIXED_N_SVD_FEATURES,
        random_seed: int = 42,
    ):
        self.split = split
        self.k = k
        self.search_space = search_space or build_search_space()
        self.evaluator = evaluator or Evaluator(primary_metric="mse")
        self.n_svd_features = n_svd_features
        self.random_seed = random_seed

        # ผลลัพธ์ทั้งหมด
        self.all_trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        self.best_svd: Optional[SVDModel] = None

    # ── Build Configurations ──────────────────────────────────────────────────
    def _build_configs(self) -> List[Dict[str, Any]]:
        keys = ["clip_percentile", "learning_rate", "batch_size"]
        vals = [self.search_space[k] for k in keys]
        configs = []
        for combo in product(*vals):
            configs.append(dict(zip(keys, combo)))
        return configs

    # ── Preprocess + Cache ────────────────────────────────────────────────────
    def _preprocess_and_cache(self, clip_percentile: float) -> Tuple[SVDModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ทำ Feature Extraction + SVD สำหรับ 1 ค่า clip_percentile

        Returns (svd_model, Z_train, y_train, Z_val, y_val, Z_test, y_test)
        """
        print(f"\n  [Cache] Preprocessing clip_percentile={clip_percentile:.0f}% ...")
        t0 = time.time()

        X_train, y_train = extract_edge_features(
            self.split.train,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=clip_percentile,
            image_size=FIXED_IMAGE_SIZE,
        )
        X_val, y_val = extract_edge_features(
            self.split.val,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=clip_percentile,
            image_size=FIXED_IMAGE_SIZE,
        )
        X_test, y_test = extract_edge_features(
            self.split.test,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=clip_percentile,
            image_size=FIXED_IMAGE_SIZE,
        )

        # Fit SVD บน Train Set เท่านั้น (Data Leakage Prevention)
        svd = _fit_svd(X_train, n_features=self.n_svd_features, random_state=self.random_seed)
        svd.clip_percentile = clip_percentile

        Z_train = svd.project(X_train)
        Z_val   = svd.project(X_val)
        Z_test  = svd.project(X_test)

        elapsed = time.time() - t0
        print(
            f"  [Cache] สำเร็จ | Z_train={Z_train.shape}, Z_val={Z_val.shape}, Z_test={Z_test.shape} | {elapsed:.1f}s"
        )
        return svd, Z_train, y_train, Z_val, y_val, Z_test, y_test

    # ── Main Search Loop ──────────────────────────────────────────────────────
    def run(self) -> "GridSearchTuner":
        """รัน Full Grid Search

        Returns
        -------
        self (สำหรับ method chaining)
        """
        configs = self._build_configs()
        total = len(configs)

        print(f"\n{'='*70}")
        print(f"  PARCEL BOX COUNTER — AUTOMATED TRAINING & MODEL SELECTION")
        print(f"{'='*70}")
        print(f"  Total Trials: {total} | Max Epochs: {FIXED_MAX_EPOCHS} | Patience: {FIXED_PATIENCE}")
        print(f"  k={self.k} | n_svd={self.n_svd_features} | blur={FIXED_BLUR_KSIZE} | init={FIXED_WEIGHT_INIT}")
        print(f"{'='*70}\n")

        # ── Cache Feature Matrices per clip_percentile ─────────────────────
        clip_values = self.search_space["clip_percentile"]
        clip_cache: Dict[float, Tuple] = {}
        print(f"[Stage 1] Feature Extraction & SVD Caching ({len(clip_values)} percentile values)...")
        for cp in clip_values:
            clip_cache[cp] = self._preprocess_and_cache(cp)

        # ── Trial Loop ─────────────────────────────────────────────────────
        print(f"\n[Stage 2] Grid Search ({total} Trials)...")
        trial_id = 0
        best_primary = float("inf") if not self.evaluator._higher_is_better else float("-inf")

        for config in configs:
            trial_id += 1
            cp   = config["clip_percentile"]
            lr   = config["learning_rate"]
            bs   = config["batch_size"]

            svd, Z_train, y_train, Z_val, y_val, Z_test, y_test = clip_cache[cp]

            t_start = time.time()

            # ── Train ──────────────────────────────────────────────────────
            trainer = Trainer(
                k=self.k,
                max_epochs=FIXED_MAX_EPOCHS,
                patience=FIXED_PATIENCE,
                random_seed=self.random_seed + trial_id,
                verbose=0,
            )
            train_result = trainer.fit(
                Z_train, y_train,
                Z_val, y_val,
                learning_rate=lr,
                batch_size=bs,
            )

            # ── Evaluate on Validation Set ─────────────────────────────────
            train_mse_final = train_result.train_loss_history[-1] if train_result.train_loss_history else float("inf")
            val_metrics = compute_metrics(
                Z_val, y_val,
                train_result.W, train_result.b, self.k,
                train_mse=train_mse_final,
            )

            elapsed = time.time() - t_start

            # ── Record Trial ───────────────────────────────────────────────
            trial = TrialResult(
                trial_id=trial_id,
                clip_percentile=cp,
                learning_rate=lr,
                batch_size=bs,
                k=self.k,
                train_result=train_result,
                val_metrics=val_metrics,
                train_mse_final=train_mse_final,
                elapsed_sec=elapsed,
            )
            self.all_trials.append(trial)

            # ── Check Best ─────────────────────────────────────────────────
            primary_val = self.evaluator.get_primary_value(val_metrics)
            is_new_best = self.evaluator.is_better(primary_val, best_primary)
            if is_new_best:
                best_primary = primary_val
                self.best_trial = trial
                self.best_svd   = svd
                marker = " ★ New Best!"
            else:
                marker = ""

            print(f"  {trial.summary_line(total)}{marker}")

        print(f"\n[Stage 2] Grid Search เสร็จสิ้น! Best Val MSE: {best_primary:.6f}")
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
