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
    label: float       # จำนวนกล่องพัสดุ (Ground Truth)
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
    """อ่านไฟล์ labels.csv → dict {filename: count}

    Parameters
    ----------
    csv_path : Path
        พาธของ labels.csv ที่มีคอลัมน์ 'filename' และ 'count'

    Returns
    -------
    labels : dict[str, float]
        ชื่อไฟล์ → จำนวนกล่อง
    """
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
            # บันทึกด้วยชื่อไฟล์เดี่ยวด้วย เพื่อรองรับทั้ง relative path และ flat name
            labels[Path(raw_fname).name] = val

    print(f"[Dataset] Loaded labels: {len(labels)} entries from {csv_path.name}")
    return labels


def load_dataset(
    images_dir: Path,
    labels_csv: Path,
    supported_extensions: Optional[set] = None,
) -> List[ImageLabel]:
    """จับคู่รูปภาพทุกไฟล์ใน images_dir กับ Label ใน labels_csv
    (รองรับทั้ง flat directory และ recursive subdirectories เช่น 1_box/, 2_box/)

    Parameters
    ----------
    images_dir          : ไดเรกทอรีของรูปภาพที่ผ่านการเตรียมข้อมูลแล้ว
    labels_csv          : ไฟล์ labels.csv (filename, count)
    supported_extensions: นามสกุลที่ยอมรับ (default: .jpg, .jpeg, .png, .heic, .heif)

    Returns
    -------
    dataset : รายการ ImageLabel ที่จับคู่ได้ครบถ้วน
    """
    if supported_extensions is None:
        supported_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

    label_map = load_labels_csv(labels_csv)

    dataset: List[ImageLabel] = []
    unmatched: List[str] = []

    # 1. ค้นหาไฟล์ภาพ (ตรวจทั้งระดับ root และ subdirectories)
    img_candidates: List[Path] = []
    direct_files = [
        f for f in sorted(images_dir.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if direct_files:
        img_candidates = direct_files
    else:
        # หากไม่มีไฟล์ใน root ให้สแกนลึกลงไปในโฟลเดอร์ย่อย (เช่น 1_box/, 2_box/)
        img_candidates = [
            f for f in sorted(images_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

    # 2. จับคู่กับ Label
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
    """แบ่งข้อมูลเป็น Train / Validation / Test ด้วย Stratified Random Split

    หลักการ:
      test_ratio = 1 - train_ratio - val_ratio
      ลำดับการแบ่ง: สุ่ม Shuffle → แบ่ง Train | Val | Test

    Parameters
    ----------
    dataset      : รายการ ImageLabel ที่จับคู่ได้แล้ว
    train_ratio  : สัดส่วน Train (default: 0.70 = 70%)
    val_ratio    : สัดส่วน Validation (default: 0.15 = 15%)
    random_seed  : seed สำหรับ reproducibility

    Returns
    -------
    DatasetSplit : ประกอบด้วย train, val, test lists
    """
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
    """รันกระบวนการ Preprocessing และสกัด Raw Pixel Feature จากรูปภาพ

    Pipeline: BGR → Grayscale → Resize(512) → GaussianBlur → Laplacian Edge →
              Resize(image_size) → Normalize → Flatten

    Parameters
    ----------
    items           : รายการ ImageLabel
    resize_target   : ขนาดลดความละเอียดสำหรับ Edge Detection (512×512)
    blur_ksize      : kernel Gaussian Blur (default: 5)
    clip_percentile : เปอร์เซ็นไทล์ตัด Outlier ใน Edge Map
    image_size      : ขนาดภาพที่จะนำเข้า SVD (128×128 → flatten 16384)
    augment         : True สำหรับ Train Set เท่านั้น (ห้ามใช้กับ Val หรือ Test)
                      สร้าง 11 ภาพเพิ่ม: หมุน ±15° (crop), Flip แนวนอน,
                      สว่าง ±25%, Contrast ±25%, Shear X ±10% (crop), Shear Y ±10% (crop)

    Returns
    -------
    X        : float32 ndarray shape (N, image_size²)   — raw feature matrix
    y        : float32 ndarray shape (N, 1)              — labels
    """
    import cv2
    from edge_processing import process_image_edges
    from data_ingestion import read_image

    X_list: List[np.ndarray] = []
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
            # ย่อขนาดเป็น resize_target (512, 512) เพื่อความรวดเร็วและคงรูปทรง
            bgr_base = cv2.resize(bgr, resize_target, interpolation=cv2.INTER_AREA)
            try:
                _, _, aug_results = run_all_augmentations(bgr_base, resize_crops_to_original=True)
                for aug_item in aug_results:
                    images_to_process.append(aug_item.image)
            except Exception as e:
                print(f"[Dataset] ⚠ เกิดข้อผิดพลาดขณะทำ Augmentation {item.filename}: {e}")

        for img in images_to_process:
            # Preprocessing: Gray → Resize → Blur → Edge Map
            _, _, _, edge_map = process_image_edges(
                img,
                resize_target=resize_target,
                blur_ksize=blur_ksize,
                clip_percentile=clip_percentile,
            )

            # Resize to image_size for SVD input
            small = cv2.resize(edge_map, (image_size, image_size), interpolation=cv2.INTER_AREA)
            normalized = small.astype(np.float32) / 255.0
            X_list.append(normalized.flatten())
            y_list.append(item.label)

    if not X_list:
        raise RuntimeError("[Dataset] ไม่สามารถประมวลผลได้เลยแม้แต่รูปเดียว")

    if skipped > 0:
        print(f"[Dataset] ข้ามรูปที่อ่านไม่ได้: {skipped} รูป")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    
    aug_mult = f" ({len(items)} × 12 = {len(X)} ตัวอย่าง)" if augment else ""
    print(f"[Dataset] สกัด Feature สำเร็จ: {len(X)} ตัวอย่าง{aug_mult}")
    return X, y
