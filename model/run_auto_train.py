# model/run_auto_train.py
"""
Main Entry Point: Multi-class Classification — Automated Training & Model Selection
=====================================================================================
รันกระบวนการทั้งหมดตั้งแต่โหลดข้อมูลจนบันทึก Champion Model ด้วยคำสั่งเดียว

การใช้งาน:
  python model/run_auto_train.py \\
      --images-dir dataset/boxes \\
      --labels dataset/labels.csv \\
      --output-dir model/trained_models

  # หรือใช้ Quick Mode (Search Space เล็กกว่า, เทรนเร็ว):
  python model/run_auto_train.py \\
      --images-dir dataset/boxes \\
      --labels dataset/labels.csv \\
      --clip-percentiles 98.0 \\
      --learning-rates 0.005 0.01 \\
      --batch-sizes 16 \\
      --hidden-dims 64

สภาพแวดล้อมที่รองรับ:
  - Windows (Local)
  - Google Colab (Free Tier)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── เพิ่ม path ที่จำเป็นทั้งหมด ────────────────────────────────────────────────
_MODEL_DIR        = Path(__file__).resolve().parent
_TRAINING_DIR     = _MODEL_DIR / "training"
_PREPROCESSING_DIR = _MODEL_DIR / "preprocessing"

for _p in [str(_TRAINING_DIR), str(_PREPROCESSING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_auto_train",
        description="Parcel Box Counter — Multi-class Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--images-dir", type=Path, required=True,
        help="Directory of pre-aligned images",
    )
    parser.add_argument(
        "--labels", type=Path, required=True,
        help="Path to labels.csv with columns: filename, count",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=Path,
        default=_MODEL_DIR / "trained_models",
        help="Output directory for trained model sessions",
    )
    parser.add_argument(
        "--session-name", type=str, default=None,
        help="Session name (default: session_<timestamp>)",
    )

    # ── Dataset Split ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # ── Model Parameters ─────────────────────────────────────────────────────
    parser.add_argument(
        "--n-classes", type=int, default=9,
        help="จำนวนคลาส (จำนวนกล่องสูงสุด, default: 9)",
    )

    # ── Search Space Override (Optional) ─────────────────────────────────────
    parser.add_argument(
        "--clip-percentiles", type=float, nargs="+",
        default=[95.0, 98.0, 99.0],
        help="clip_percentile values to search",
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+",
        default=[0.001, 0.005, 0.01, 0.05],
        help="learning_rate values to search",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[8, 16, 32],
        help="batch_size values to search",
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+",
        default=[32, 64, 128],
        help="hidden_dim values for MLP hidden layer",
    )

    # ── Primary Metric ────────────────────────────────────────────────────────
    parser.add_argument(
        "--metric", type=str, default="cross_entropy",
        choices=["cross_entropy", "acc_exact", "acc_within_1", "macro_f1"],
        help="Primary metric for champion model selection",
    )

    # ── Augmentation ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-augment", action="store_true",
        help="ปิดการทำ Data Augmentation บน Train Set",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_auto_train(args: argparse.Namespace) -> None:
    """รันกระบวนการ End-to-End ทั้งหมด"""
    from dataset import load_dataset, split_dataset
    from tuner import GridSearchTuner, build_search_space
    from evaluator import Evaluator
    from model_registry import ModelRegistry

    total_start = time.time()
    augment_train = not args.no_augment

    # ── Stage 1: Load & Split Dataset ─────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  PARCEL BOX COUNTER — MULTI-CLASS CLASSIFICATION TRAINING")
    print(f"{'='*75}")
    print(f"\n[Stage 1] Loading data & splitting...")
    print(f"  • Model Type             : Multi-class Classification (กล่อง 1–{args.n_classes})")
    print(f"  • Feature Vector         : SVD(64) + StatFeatures(3) = 67 dims")
    print(f"  • Data Augmentation (Train) : {'ENABLED (9x independent variants)' if augment_train else 'DISABLED'}")
    print(f"  • Data Augmentation (Val)   : DISABLED")
    print(f"  • Data Augmentation (Test)  : DISABLED")

    dataset = load_dataset(
        images_dir=args.images_dir,
        labels_csv=args.labels,
    )
    split = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )

    # ── Stage 2 & 3: Grid Search ──────────────────────────────────────────────
    search_space = build_search_space(
        clip_percentile=args.clip_percentiles,
        learning_rate=args.learning_rates,
        batch_size=args.batch_sizes,
        hidden_dim=args.hidden_dims,
    )
    evaluator = Evaluator(primary_metric=args.metric)

    tuner = GridSearchTuner(
        split=split,
        search_space=search_space,
        evaluator=evaluator,
        n_classes=args.n_classes,
        random_seed=args.seed,
        augment_train=augment_train,
    )
    tuner.run()

    if tuner.best_trial is None:
        print("[ERROR] ไม่สามารถค้นหา Best Model ได้ — ตรวจสอบข้อมูล")
        sys.exit(1)

    # ── Stage 4: Save Champion Model Bundle ───────────────────────────────────
    print(f"\n[Stage 3] Saving Champion Model Bundle...")
    registry = ModelRegistry(
        output_dir=args.output_dir,
        session_name=args.session_name,
    )
    session_dir = registry.save_all(tuner)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    best = tuner.best_trial
    vm = best.val_metrics

    print(f"\n{'='*75}")
    print(f"  SELECTION SUMMARY & TEST SET VERIFICATION")
    print(f"{'='*75}")
    print(f"  🏆 Champion Model: Trial {best.trial_id}")
    print(f"     Clip={best.clip_percentile:.0f}%, LR={best.learning_rate}, "
          f"Batch={best.batch_size}, Hidden={best.hidden_dim}")
    print(f"     Val CE: {best.val_ce:.4f} | "
          f"Val Acc: {vm.acc_exact*100:.1f}% | "
          f"Val F1: {vm.macro_f1:.3f}" if vm else "")
    print(f"     Best Epoch: {best.train_result.best_epoch} / {best.train_result.total_epochs_run}")
    print(f"\n  ✓ Exported to: {session_dir}")
    print(f"  ⏱ Total time: {total_elapsed:.1f}s")
    print(f"{'='*75}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์ภาพ: {args.images_dir}")
        sys.exit(1)
    if not args.labels.exists():
        print(f"[ERROR] ไม่พบไฟล์ labels: {args.labels}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_auto_train(args)
