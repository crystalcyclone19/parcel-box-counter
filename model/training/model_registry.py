# model/training/model_registry.py
"""
Stage 4 (Part B): Model Registry & Deployment Packaging
=========================================================
หลังจากได้ Champion Model จาก Grid Search:
  1. ประเมินบน Test Set (Unseen Data) ครั้งสุดท้าย
  2. บันทึก Model Bundle แบบ Self-Contained พร้อมใช้งานทันที
  3. สร้าง Leaderboard CSV สรุปผลทุก Trial
  4. บันทึก Training Report (.md)

โครงสร้างไฟล์ที่บันทึก (ใน trained_models/session_<timestamp>/):
  best_weights.json      — W, b, k, architecture
  svd_model.npz          — mean, Vt, S, preprocessing config
  pipeline_config.json   — ค่า config ที่ใช้ใน Pipeline ของ Champion
  manifest.json          — Metadata ครบชุด (วันที่, metrics, hyperparams)
  leaderboard.csv        — ตารางผลทุก Trial เรียงลำดับ
  training_report.md     — รายงาน Markdown อ่านง่าย
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from tuner import GridSearchTuner, TrialResult, FIXED_N_SVD_FEATURES, FIXED_BLUR_KSIZE, FIXED_RESIZE_TARGET
from evaluator import compute_metrics, MetricsResult


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """จัดการบันทึก Champion Model และ Artifacts ทั้งหมด

    Parameters
    ----------
    output_dir      : โฟลเดอร์ trained_models/ (บันทึก session แต่ละครั้งแยกโฟลเดอร์)
    session_name    : ชื่อ session (default: auto timestamp)
    """

    def __init__(
        self,
        output_dir: Path,
        session_name: Optional[str] = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"
        self.session_dir = output_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Registry] Session directory: {self.session_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # Test Set Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_champion_on_test(
        self,
        tuner: GridSearchTuner,
    ) -> MetricsResult:
        """ประเมิน Champion Model บน Test Set (Unseen)

        ดึง Z_test ที่ Cache ไว้ใน tuner (ด้วย clip_percentile ของ Champion)
        """
        if tuner.best_trial is None or tuner.best_svd is None:
            raise RuntimeError("[Registry] ไม่มี Champion Model — รัน tuner.run() ก่อน")

        best = tuner.best_trial

        # ดึง X_test จาก split ด้วย clip_percentile ของ Champion
        from dataset import extract_edge_features
        X_test, y_test = extract_edge_features(
            tuner.split.test,
            resize_target=FIXED_RESIZE_TARGET,
            blur_ksize=FIXED_BLUR_KSIZE,
            clip_percentile=best.clip_percentile,
            image_size=128,
        )
        Z_test = tuner.best_svd.project(X_test)

        train_mse_final = best.train_mse_final
        test_metrics = compute_metrics(
            Z_test, y_test,
            best.train_result.W, best.train_result.b, best.k,
            train_mse=train_mse_final,
        )

        print(f"\n{'='*70}")
        print(f"  CHAMPION MODEL — TEST SET EVALUATION (UNSEEN DATA)")
        print(f"{'='*70}")
        print(test_metrics.summary())
        return test_metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Save Bundle
    # ─────────────────────────────────────────────────────────────────────────

    def save_champion_bundle(
        self,
        tuner: GridSearchTuner,
        test_metrics: MetricsResult,
    ) -> Path:
        """บันทึก Champion Model Bundle ครบชุดลงใน session_dir

        Returns
        -------
        session_dir : Path ของโฟลเดอร์ session ที่บันทึกไว้
        """
        best = tuner.best_trial

        # ── 1. best_weights.json ─────────────────────────────────────────────
        weights_path = self.session_dir / "best_weights.json"
        weights_data = {
            "W": best.train_result.W.tolist(),
            "b": best.train_result.b.tolist(),
            "k": best.k,
            "architecture": {
                "input_size":    int(best.n_svd_features),
                "output_size":   1,
                "activation":    "scaled_sigmoid",
                "weight_init":   best.weight_init,
            },
        }
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=2)
        print(f"  ✓ บันทึก best_weights.json")

        # ── 2. svd_model.npz ─────────────────────────────────────────────────
        svd_path = self.session_dir / "svd_model.npz"
        np.savez_compressed(
            svd_path,
            mean=tuner.best_svd.mean,
            Vt=tuner.best_svd.Vt,
            S=tuner.best_svd.S,
            image_size=np.int32(128),
            n_features=np.int32(best.n_svd_features),
        )
        print(f"  ✓ บันทึก svd_model.npz")

        # ── 3. pipeline_config.json ──────────────────────────────────────────
        config_path = self.session_dir / "pipeline_config.json"
        pipeline_cfg = {
            "preprocessing": {
                "resize_target":    list(FIXED_RESIZE_TARGET),
                "blur_ksize":       best.blur_ksize,
                "clip_percentile":  best.clip_percentile,
                "image_size_svd":   128,
            },
            "training": {
                "learning_rate":    best.learning_rate,
                "batch_size":       best.batch_size,
                "max_epochs":       best.max_epochs,
                "weight_init":      best.weight_init,
                "patience":         25,
            },
            "model": {
                "k":                best.k,
                "n_svd_features":   best.n_svd_features,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_cfg, f, indent=2)
        print(f"  ✓ บันทึก pipeline_config.json")

        # ── 4. manifest.json ─────────────────────────────────────────────────
        manifest_path = self.session_dir / "manifest.json"
        manifest = {
            "session":          self.session_name,
            "trained_at":       datetime.now().isoformat(),
            "dataset": {
                "n_train_raw":       tuner.split.n_train,
                "n_train_effective": getattr(tuner, "n_train_effective", tuner.split.n_train),
                "n_val":             tuner.split.n_val,
                "n_test":            tuner.split.n_test,
                "train_augmented":   getattr(tuner, "augment_train", True),
                "val_augmented":     False,
                "test_augmented":    False,
            },
            "champion": {
                "trial_id":       best.trial_id,
                "clip_percentile": best.clip_percentile,
                "learning_rate":  best.learning_rate,
                "batch_size":     best.batch_size,
                "k":              best.k,
                "best_epoch":     best.train_result.best_epoch,
                "stopped_early":  best.train_result.stopped_early,
                "total_epochs":   best.train_result.total_epochs_run,
            },
            "val_metrics": {
                "mse":          best.val_mse,
                "mae":          best.val_mae,
                "r2":           best.val_metrics.r2,
                "acc_exact":    best.val_metrics.acc_exact,
                "acc_within_1": best.val_metrics.acc_within_1,
            },
            "test_metrics": {
                "mse":          test_metrics.mse,
                "mae":          test_metrics.mae,
                "r2":           test_metrics.r2,
                "acc_exact":    test_metrics.acc_exact,
                "acc_within_1": test_metrics.acc_within_1,
            },
            "search_summary": {
                "total_trials": len(tuner.all_trials),
                "search_space": {k: v for k, v in tuner.search_space.items()},
            },
            "artifacts": [
                "best_weights.json",
                "svd_model.npz",
                "pipeline_config.json",
                "manifest.json",
                "leaderboard.csv",
                "training_report.md",
            ],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✓ บันทึก manifest.json")

        return self.session_dir

    # ─────────────────────────────────────────────────────────────────────────
    # Leaderboard CSV
    # ─────────────────────────────────────────────────────────────────────────

    def save_leaderboard(self, tuner: GridSearchTuner) -> Path:
        """บันทึก leaderboard.csv เรียงจาก Val MSE ต่ำสุด"""
        lb_path = self.session_dir / "leaderboard.csv"
        ranked = tuner.get_leaderboard()

        fieldnames = [
            "rank", "trial_id",
            "clip_percentile", "learning_rate", "batch_size",
            "val_mse", "val_mae", "val_r2", "val_acc_exact", "val_acc_within_1",
            "gen_gap", "train_mse_final",
            "best_epoch", "total_epochs", "stopped_early",
            "elapsed_sec",
        ]

        with open(lb_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, trial in enumerate(ranked, 1):
                vm = trial.val_metrics
                writer.writerow({
                    "rank":             rank,
                    "trial_id":         trial.trial_id,
                    "clip_percentile":  trial.clip_percentile,
                    "learning_rate":    trial.learning_rate,
                    "batch_size":       trial.batch_size,
                    "val_mse":          f"{vm.mse:.6f}" if vm else "",
                    "val_mae":          f"{vm.mae:.6f}" if vm else "",
                    "val_r2":           f"{vm.r2:.4f}"  if vm else "",
                    "val_acc_exact":    f"{vm.acc_exact:.4f}"    if vm else "",
                    "val_acc_within_1": f"{vm.acc_within_1:.4f}" if vm else "",
                    "gen_gap":          f"{vm.gen_gap:.6f}" if vm and vm.gen_gap else "",
                    "train_mse_final":  f"{trial.train_mse_final:.6f}",
                    "best_epoch":       trial.train_result.best_epoch if trial.train_result else "",
                    "total_epochs":     trial.train_result.total_epochs_run if trial.train_result else "",
                    "stopped_early":    trial.train_result.stopped_early if trial.train_result else "",
                    "elapsed_sec":      f"{trial.elapsed_sec:.2f}",
                })
        print(f"  ✓ บันทึก leaderboard.csv ({len(ranked)} trials)")
        return lb_path

    # ─────────────────────────────────────────────────────────────────────────
    # Training Report (.md)
    # ─────────────────────────────────────────────────────────────────────────

    def save_training_report(
        self,
        tuner: GridSearchTuner,
        test_metrics: MetricsResult,
    ) -> Path:
        """บันทึก training_report.md ที่อ่านง่าย"""
        report_path = self.session_dir / "training_report.md"
        best = tuner.best_trial
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Training Report — {self.session_name}",
            f"",
            f"**Generated at**: {now}  ",
            f"",
            f"---",
            f"",
            f"## Dataset",
            f"| Split | Samples |",
            f"|---|:---:|",
            f"| Train | {tuner.split.n_train} |",
            f"| Validation | {tuner.split.n_val} |",
            f"| Test (Unseen) | {tuner.split.n_test} |",
            f"| **Total** | **{tuner.split.total}** |",
            f"",
            f"---",
            f"",
            f"## Search Configuration",
            f"| Category | Parameter | Value |",
            f"|---|---|---|",
            f"| **Tunable** | `clip_percentile` | `{tuner.search_space['clip_percentile']}` |",
            f"| | `learning_rate` | `{tuner.search_space['learning_rate']}` |",
            f"| | `batch_size` | `{tuner.search_space['batch_size']}` |",
            f"| **Fixed** | `k` | `{best.k}` |",
            f"| | `n_svd_features` | `{best.n_svd_features}` |",
            f"| | `blur_ksize` | `{best.blur_ksize}` |",
            f"| | `max_epochs` | `{best.max_epochs}` |",
            f"| | `weight_init` | `{best.weight_init}` |",
            f"| **Total Trials** | — | **{len(tuner.all_trials)}** |",
            f"",
            f"---",
            f"",
            f"## 🏆 Champion Model — Trial {best.trial_id}",
            f"",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| `clip_percentile` | `{best.clip_percentile}` |",
            f"| `learning_rate` | `{best.learning_rate}` |",
            f"| `batch_size` | `{best.batch_size}` |",
            f"| Best Epoch | {best.train_result.best_epoch} / {best.train_result.total_epochs_run} |",
            f"| Early Stopped | {'Yes' if best.train_result.stopped_early else 'No'} |",
            f"",
            f"### Validation Metrics",
            f"| Metric | Value |",
            f"|---|---|",
            f"| MSE | `{best.val_mse:.6f}` |",
            f"| MAE | `{best.val_mae:.4f}` กล่อง |",
            f"| R² | `{best.val_metrics.r2:.4f}` |",
            f"| Accuracy (Exact) | `{best.val_metrics.acc_exact*100:.2f}%` |",
            f"| Accuracy (±1 box) | `{best.val_metrics.acc_within_1*100:.2f}%` |",
            f"",
            f"### Test Set Metrics (Unseen Data)",
            f"| Metric | Value |",
            f"|---|---|",
            f"| MSE | `{test_metrics.mse:.6f}` |",
            f"| MAE | `{test_metrics.mae:.4f}` กล่อง |",
            f"| R² | `{test_metrics.r2:.4f}` |",
            f"| Accuracy (Exact) | `{test_metrics.acc_exact*100:.2f}%` |",
            f"| Accuracy (±1 box) | `{test_metrics.acc_within_1*100:.2f}%` |",
            f"",
            f"---",
            f"",
            f"## Artifacts",
            f"```",
            f"{self.session_dir}/",
            f"├── best_weights.json",
            f"├── svd_model.npz",
            f"├── pipeline_config.json",
            f"├── manifest.json",
            f"├── leaderboard.csv",
            f"└── training_report.md",
            f"```",
        ]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  ✓ บันทึก training_report.md")
        return report_path

    # ─────────────────────────────────────────────────────────────────────────
    # All-in-One: Save All Artifacts
    # ─────────────────────────────────────────────────────────────────────────

    def save_all(self, tuner: GridSearchTuner) -> Path:
        """บันทึก Artifacts ครบชุดสำหรับ Session นี้

        Returns
        -------
        session_dir : Path ของโฟลเดอร์ที่บันทึกทั้งหมด
        """
        print(f"\n[Registry] กำลังบันทึก Model Bundle ...")
        test_metrics = self.evaluate_champion_on_test(tuner)
        self.save_champion_bundle(tuner, test_metrics)
        self.save_leaderboard(tuner)
        self.save_training_report(tuner, test_metrics)

        print(f"\n{'='*70}")
        print(f"  ✓ Exported Champion Model to: {self.session_dir}")
        print(f"{'='*70}")
        return self.session_dir
