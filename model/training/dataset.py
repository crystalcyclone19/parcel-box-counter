# model/training/dataset.py
"""
Stage 1: Dataset Preparation & Splitting
=========================================
โหลดและจับคู่ภาพกับ Label พร้อมแบ่งข้อมูล Train / Validation / Test

รูปแบบ Label ที่รองรับ:
  - labels.csv ที่มีคอลัมน์ 'filename' (ชื่อไฟล์ภาพ) และ 'count' (จำนวนกล่อง)

หมายเหตุ Data Leakage Prevention:
  SVD Fit ต้องทำบน Train Set เท่านั้น แล้วนำ Mean_train, Vt_train
  ไปใช้ Project ข้อมูล Val/Test — ห้าม Fit บน Full Dataset

Feature Vector (สำหรับ Multi-class Classification):
  [SVD_features(64) | total_edge_area | contour_count | aspect_ratio]
  → input_dim = 64 + 3 = 67
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

# ── เพิ่ม preprocessing/, augmentation/, tools/ เข้า sys.path ─────────────────
_TRAINING_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TRAINING_DIR.parent
_PREPROCESSING_DIR = _MODEL_DIR / "preprocessing"
_AUGMENTATION_DIR = _MODEL_DIR / "augmentation"
_TOOLS_DIR = _MODEL_DIR.parent / "tools"

for p in (_PREPROCESSING_DIR, _AUGMENTATION_DIR, _TOOLS_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from augment_image import run_all_augmentations
except ImportError:
    try:
        from model.augmentation.augment_image import run_all_augmentations
    except ImportError:
        from tools.augment_image import run_all_augmentations


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageLabel:
    """จับคู่ภาพหนึ่งใบกับ Label (จำนวนกล่องพัสดุ)"""
    image_path: Path
    label: float       # จำนวนกล่องพัสดุ (Ground Truth) — ใช้ int(label)-1 เป็น class index
    filename: str = field(init=False)

    def __post_init__(self):
        self.filename = self.image_path.name


@dataclass
class DatasetSplit:
    """ผลลัพธ์จากการแบ่ง Dataset เป็น 3 ชุด"""
    train: List[ImageLabel]
    val:   List[ImageLabel]
    test:  List[ImageLabel]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_val(self) -> int:
        return len(self.val)

    @property
    def n_test(self) -> int:
        return len(self.test)

    @property
    def total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def summary(self) -> str:
        pct = lambda n: f"{n} ({100*n/self.total:.1f}%)" if self.total > 0 else str(n)
        return (
            f"Dataset Split Summary\n"
            f"  Total  : {self.total} images\n"
            f"  Train  : {pct(self.n_train)}\n"
            f"  Val    : {pct(self.n_val)}\n"
            f"  Test   : {pct(self.n_test)}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loading & Pairing
# ─────────────────────────────────────────────────────────────────────────────

def load_labels_csv(csv_path: Path) -> dict[str, float]:
    """อ่านไฟล์ labels.csv → dict {filename: count}"""
    if not csv_path.exists():
        raise FileNotFoundError(f"[Dataset] ไม่พบไฟล์ label: {csv_path}")

    labels: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"filename", "count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"[Dataset] labels.csv ต้องมีคอลัมน์: {required}\n"
                f"  พบคอลัมน์: {reader.fieldnames}"
            )
        for row in reader:
            raw_fname = row["filename"].strip().replace("\\", "/")
            val = float(row["count"])
            labels[raw_fname] = val
            labels[Path(raw_fname).name] = val

    print(f"[Dataset] Loaded labels: {len(labels)} entries from {csv_path.name}")
    return labels


def load_dataset(
    images_dir: Path,
    labels_csv: Path,
    supported_extensions: Optional[set] = None,
) -> List[ImageLabel]:
    """จับคู่รูปภาพทุกไฟล์ใน images_dir กับ Label ใน labels_csv"""
    if supported_extensions is None:
        supported_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

    label_map = load_labels_csv(labels_csv)

    dataset: List[ImageLabel] = []
    unmatched: List[str] = []

    img_candidates: List[Path] = []
    direct_files = [
        f for f in sorted(images_dir.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if direct_files:
        img_candidates = direct_files
    else:
        img_candidates = [
            f for f in sorted(images_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

    for img_path in img_candidates:
        fname = img_path.name
        rel_str = str(img_path.relative_to(images_dir)).replace("\\", "/")

        matched_label = None
        if rel_str in label_map:
            matched_label = label_map[rel_str]
        elif fname in label_map:
            matched_label = label_map[fname]

        if matched_label is not None:
            dataset.append(ImageLabel(image_path=img_path, label=matched_label))
        else:
            unmatched.append(rel_str)

    if unmatched:
        print(f"[Dataset] Warning: No labels found for {len(unmatched)} files (skipped):")
        for f in unmatched[:5]:
            print(f"    - {f}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched)-5} more files")

    if not dataset:
        raise ValueError(
            "[Dataset] No images matched with labels! Check images_dir and labels_csv"
        )

    print(f"[Dataset] Matched successfully: {len(dataset)} images")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    dataset: List[ImageLabel],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplit:
    """แบ่งข้อมูลเป็น Train / Validation / Test ด้วย Stratified Random Split"""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("[Dataset] train_ratio และ val_ratio ต้องอยู่ระหว่าง 0 ถึง 1")
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("[Dataset] train_ratio + val_ratio > 1.0")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(dataset)).tolist()

    n = len(dataset)
    n_train = max(1, int(round(n * train_ratio)))
    n_val   = max(1, int(round(n * val_ratio)))
    n_test  = max(1, n - n_train - n_val)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    split = DatasetSplit(
        train=[dataset[i] for i in train_idx],
        val  =[dataset[i] for i in val_idx],
        test =[dataset[i] for i in test_idx],
    )

    print(f"[Dataset] แบ่งชุดข้อมูลเสร็จสิ้น (seed={random_seed})")
    print(split.summary())
    return split


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Edge Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_stat_features(edge_map: "np.ndarray") -> np.ndarray:
    """สกัด 3 สถิติ Geometric/Edge จาก Edge Map

    Features:
      1. total_edge_area   — สัดส่วน pixel ที่เป็น edge (normalized 0-1)
      2. contour_count     — จำนวน contour จาก cv2.findContours (log-scaled)
      3. aspect_ratio      — อัตราส่วน bounding-box ของ contour ทั้งหมด

    Parameters
    ----------
    edge_map : uint8 ndarray (H, W)  — Edge Map ก่อน normalize

    Returns
    -------
    stat_vec : float32 ndarray (3,)
    """
    import cv2

    h, w = edge_map.shape

    # 1. Total Edge Area
    binary = (edge_map > 0).astype(np.uint8) * 255
    total_edge_area = float(np.count_nonzero(binary)) / float(h * w)

    # 2. Contour Count + Aspect Ratio จาก bounding box ของ contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    if contour_count > 0:
        # รวม bounding box ของทุก contour → compute overall aspect ratio
        all_x = np.concatenate([c[:, 0, 0] for c in contours])
        all_y = np.concatenate([c[:, 0, 1] for c in contours])
        bw = float(all_x.max() - all_x.min() + 1)
        bh = float(all_y.max() - all_y.min() + 1)
        aspect_ratio = bw / (bh + 1e-6)
    else:
        aspect_ratio = 1.0

    # Log-scale contour count (ทำให้ scale ใกล้เคียงกัน)
    contour_count_log = float(np.log1p(contour_count))

    return np.array([total_edge_area, contour_count_log, aspect_ratio], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction (Preprocessing Pipeline Integration)
# ─────────────────────────────────────────────────────────────────────────────

def extract_edge_features(
    items: List[ImageLabel],
    resize_target: Tuple[int, int] = (512, 512),
    blur_ksize: int = 5,
    clip_percentile: float = 98.0,
    image_size: int = 128,
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """รันกระบวนการ Preprocessing และสกัด Feature Vector ครบชุดจากรูปภาพ

    Feature Vector ต่อ 1 รูป:
      [pixel_flat(image_size²)]   → ใช้เป็น Raw Input สำหรับ SVD ในภายหลัง
      + stat_features(3)          → [total_edge_area, log_contour_count, aspect_ratio]

    Pipeline: BGR → Grayscale → Resize(512) → GaussianBlur → Laplacian Edge →
              Resize(image_size) → Normalize → Flatten
              + extract stat features จาก edge map ขนาด 512×512

    Parameters
    ----------
    items           : รายการ ImageLabel
    resize_target   : ขนาดลดความละเอียดสำหรับ Edge Detection (512×512)
    blur_ksize      : kernel Gaussian Blur (default: 5)
    clip_percentile : เปอร์เซ็นไทล์ตัด Outlier ใน Edge Map
    image_size      : ขนาดภาพที่จะนำเข้า SVD (128×128 → flatten 16384)
    augment         : True สำหรับ Train Set เท่านั้น (ห้ามใช้กับ Val หรือ Test)

    Returns
    -------
    X_pixel  : float32 ndarray shape (N, image_size²)  — pixel matrix สำหรับ SVD
    X_stat   : float32 ndarray shape (N, 3)             — stat features
    y        : float32 ndarray shape (N, 1)             — labels (จำนวนกล่อง)
    """
    import cv2
    from edge_processing import process_image_edges
    from data_ingestion import read_image

    X_pixel_list: List[np.ndarray] = []
    X_stat_list:  List[np.ndarray] = []
    y_list: List[float] = []
    skipped = 0

    mode_tag = "TRAIN (with 11x Augmentation)" if augment else "EVALUATION (Clean / No Augmentation)"
    print(f"[Dataset] กำลังสกัด Feature: {len(items)} ภาพต้นฉบับ [{mode_tag}] ...")

    for item in items:
        bgr = read_image(item.image_path)
        if bgr is None:
            print(f"[Dataset] ⚠ อ่านรูปไม่ได้ ข้ามไป: {item.filename}")
            skipped += 1
            continue

        images_to_process: List[np.ndarray] = [bgr]

        if augment:
            bgr_base = cv2.resize(bgr, resize_target, interpolation=cv2.INTER_AREA)
            try:
                _, _, aug_results = run_all_augmentations(bgr_base, resize_crops_to_original=True)
                for aug_item in aug_results:
                    images_to_process.append(aug_item.image)
            except Exception as e:
                print(f"[Dataset] ⚠ เกิดข้อผิดพลาดขณะทำ Augmentation {item.filename}: {e}")

        for img in images_to_process:
            # Preprocessing: Gray → Resize → Blur → Edge Map (512×512 for stat)
            _, _, _, edge_map = process_image_edges(
                img,
                resize_target=resize_target,
                blur_ksize=blur_ksize,
                clip_percentile=clip_percentile,
            )

            # Statistical features จาก edge_map ขนาดเต็ม (512×512)
            stat_vec = _extract_stat_features(edge_map)

            # Resize to image_size for SVD input
            small = cv2.resize(edge_map, (image_size, image_size), interpolation=cv2.INTER_AREA)
            normalized = small.astype(np.float32) / 255.0

            X_pixel_list.append(normalized.flatten())
            X_stat_list.append(stat_vec)
            y_list.append(item.label)

    if not X_pixel_list:
        raise RuntimeError("[Dataset] ไม่สามารถประมวลผลได้เลยแม้แต่รูปเดียว")

    if skipped > 0:
        print(f"[Dataset] ข้ามรูปที่อ่านไม่ได้: {skipped} รูป")

    X_pixel = np.stack(X_pixel_list, axis=0).astype(np.float32)
    X_stat  = np.stack(X_stat_list,  axis=0).astype(np.float32)
    y       = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    aug_mult = f" ({len(items)} × 12 = {len(X_pixel)} ตัวอย่าง)" if augment else ""
    print(f"[Dataset] สกัด Feature สำเร็จ: {len(X_pixel)} ตัวอย่าง{aug_mult}")
    print(f"[Dataset] X_pixel: {X_pixel.shape}, X_stat: {X_stat.shape}")
    return X_pixel, X_stat, y
