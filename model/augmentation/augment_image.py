# tools/augment_image.py
"""
Independent Image Augmentation Suite
====================================
เครื่องมือสำหรับทำ Data Augmentation แยกแต่ละวิธีโดยอิสระจากภาพต้นฉบับ (ห้ามทำรวมหรือต่อๆ กัน):
  2.1 หมุน +-15 องศา (Crop เอาเฉพาะพื้นที่จริงที่เหลือ ไม่ติดขอบดำ)
  2.2 Horizontal Flip (พลิกภาพแนวนอน ซ้าย-ขวา)
  2.3 ปรับความสว่าง +-25% (Brightness adjustment)
  2.4 ปรับ Contrast +-25% (Contrast adjustment)
  2.5 Shear X +-10% (Crop เอาเฉพาะพื้นที่จริงที่เหลือ ไม่ติดขอบดำ)
  2.6 Shear Y +-10% (Crop เอาเฉพาะพื้นที่จริงที่เหลือ ไม่ติดขอบดำ)

ผลลัพธ์:
  - บันทึกภาพผลลัพธ์แยกแต่ละวิธี (รวม 11 ภาพ Augmentation + 1 ภาพ Original)
  - บันทึกภาพรวมสรุป (Summary Grid Comparison) สวยงามความละเอียดสูง
  - บันทึกรายงาน metadata ในไฟล์ augmentation_manifest.json

Usage:
------
# 1. รันกับไฟล์ภาพเดี่ยว:
python tools/augment_image.py --image dataset/dataset/1_box/IMG_6510.jpg

# 2. กำหนดโฟลเดอร์ผลลัพธ์เอง:
python tools/augment_image.py --image test.jpg --output-dir my_augmented_results

# 3. นำเข้าเป็น Python Module:
from tools.augment_image import ImageAugmenter
augmenter = ImageAugmenter()
results = augmenter.run("test.jpg", output_dir="output/")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Enable UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Robust File I/O Helpers (Windows UTF-8 Safe)
# ─────────────────────────────────────────────────────────────────────────────

def read_image(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """อ่านรูปภาพรองรับ JPG, PNG, HEIC, WEBP, BMP และ Path ภาษาไทย/UTF-8 บน Windows"""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {p}")

    # Read binary buffer to support Unicode path on Windows
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)

    # Fallback to PIL for HEIC or other special formats
    if bgr is None:
        try:
            from PIL import Image
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            pil_img = Image.open(p).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"ไม่สามารถอ่านไฟล์ภาพ '{p.name}': {e}")

    if bgr is None or bgr.size == 0:
        raise ValueError(f"ไฟล์ภาพว่างเปล่าหรือเสียหาย: {p}")

    h, w, c = bgr.shape
    meta = {
        "filename": p.name,
        "path": str(p),
        "height": h,
        "width": w,
        "channels": c,
        "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
    }
    return bgr, meta


def save_image(image: np.ndarray, path: Union[str, Path], quality: int = 95) -> Path:
    """บันทึกภาพรองรับ Path ภาษาไทย/UTF-8 บน Windows"""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = []

    success, enc = cv2.imencode(suffix, image, params)
    if not success:
        raise RuntimeError(f"ไม่สามารถ Encode รูปภาพ: {p.name}")
    enc.tofile(str(p))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 2. Individual Independent Augmentation Functions
# ─────────────────────────────────────────────────────────────────────────────

def augment_rotate_and_crop(
    image: np.ndarray,
    angle_deg: float,
    resize_to_original: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2.1 หมุนภาพตามมุม angle_deg (เช่น +15 หรือ -15 องศา)

    และทำการ Crop เฉพาะพื้นที่ภาพจริงด้านใน (Largest Inscribed Rectangle)
    โดยรักษา Aspect Ratio เดิม ไม่ติดขอบสีดำรอบข้างอย่างแม่นยำ
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    alpha = abs(angle_deg) * np.pi / 180.0

    # 1. Rotate around center
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 2. Analytical formula for largest inscribed rectangle with same aspect ratio
    # s = scale factor of inscribed rectangle
    denom1 = w * np.cos(alpha) + h * np.sin(alpha)
    denom2 = w * np.sin(alpha) + h * np.cos(alpha)
    scale = min(w / denom1, h / denom2)

    crop_w = max(1, int(scale * w))
    crop_h = max(1, int(scale * h))

    x1 = max(0, int(center[0] - crop_w / 2.0))
    y1 = max(0, int(center[1] - crop_h / 2.0))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    cropped = rotated[y1:y2, x1:x2]

    # 3. Resize back to original dimensions for consistent resolution
    if resize_to_original and (crop_w != w or crop_h != h):
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = cropped

    meta = {
        "operation": f"rotate_{angle_deg:+0.0f}deg",
        "angle_deg": angle_deg,
        "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": crop_w, "h": crop_h},
        "crop_scale_ratio": round(float(scale), 4),
        "resized_to_original": resize_to_original,
    }
    return result, meta


def augment_horizontal_flip(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2.2 พลิกภาพแนวนอน (Horizontal Flip: ซ้าย-ขวา)"""
    flipped = cv2.flip(image, 1)
    meta = {
        "operation": "horizontal_flip",
        "flip_code": 1,
    }
    return flipped, meta


def augment_brightness(
    image: np.ndarray,
    factor: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2.3 ปรับความสว่าง +-25%

    Parameters
    ----------
    factor : float, e.g. 1.25 (+25%) หรือ 0.75 (-25%)
    """
    adjusted = np.clip(image.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
    pct_change = int(round((factor - 1.0) * 100))
    meta = {
        "operation": f"brightness_{pct_change:+d}%",
        "factor": factor,
        "percent_change": pct_change,
    }
    return adjusted, meta


def augment_contrast(
    image: np.ndarray,
    factor: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2.4 ปรับ Contrast +-25%

    ปรับความเปรียบต่างโดยอิงจากค่าเฉลี่ยความสว่างของภาพ (Per-channel mean)
    เพื่อคงระดับแสงเฉลี่ยของภาพไว้ และขยาย/หด Dynamic Range อย่างเป็นธรรมชาติ

    Parameters
    ----------
    factor : float, e.g. 1.25 (+25%) หรือ 0.75 (-25%)
    """
    mean_val = np.mean(image.astype(np.float32), axis=(0, 1), keepdims=True)
    adjusted = np.clip(mean_val + factor * (image.astype(np.float32) - mean_val), 0.0, 255.0).astype(np.uint8)
    pct_change = int(round((factor - 1.0) * 100))
    meta = {
        "operation": f"contrast_{pct_change:+d}%",
        "factor": factor,
        "percent_change": pct_change,
    }
    return adjusted, meta


def augment_shear_x_and_crop(
    image: np.ndarray,
    shear_factor: float,
    resize_to_original: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2.5 Shear X +-10% พร้อม Crop พื้นที่จริงที่เหลือ (ไม่ติดขอบดำ)

    Parameters
    ----------
    shear_factor : float, เช่น +0.10 (+10%) หรือ -0.10 (-10%)
    """
    h, w = image.shape[:2]
    # Shear centered at center (w/2, h/2)
    # x' = x + s_x * (y - h/2)
    shift_offset = -shear_factor * (h / 2.0)
    mat_x = np.float32([[1.0, shear_factor, shift_offset], [0.0, 1.0, 0.0]])

    sheared = cv2.warpAffine(
        image,
        mat_x,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Inscribed crop maintaining aspect ratio:
    # Common overlap across all y in [0, h] has width = w - |s_x| * h
    scale = max(0.1, 1.0 - abs(shear_factor) * (h / w))
    crop_w = max(1, int(scale * w))
    crop_h = max(1, int(scale * h))

    x1 = max(0, int(w / 2.0 - crop_w / 2.0))
    y1 = max(0, int(h / 2.0 - crop_h / 2.0))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    cropped = sheared[y1:y2, x1:x2]

    if resize_to_original and (crop_w != w or crop_h != h):
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = cropped

    pct_change = int(round(shear_factor * 100))
    meta = {
        "operation": f"shear_x_{pct_change:+d}%",
        "shear_factor": shear_factor,
        "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": crop_w, "h": crop_h},
        "crop_scale_ratio": round(float(scale), 4),
        "resized_to_original": resize_to_original,
    }
    return result, meta


def augment_shear_y_and_crop(
    image: np.ndarray,
    shear_factor: float,
    resize_to_original: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2.6 Shear Y +-10% พร้อม Crop พื้นที่จริงที่เหลือ (ไม่ติดขอบดำ)

    Parameters
    ----------
    shear_factor : float, เช่น +0.10 (+10%) หรือ -0.10 (-10%)
    """
    h, w = image.shape[:2]
    # Shear centered at center (w/2, h/2)
    # y' = y + s_y * (x - w/2)
    shift_offset = -shear_factor * (w / 2.0)
    mat_y = np.float32([[1.0, 0.0, 0.0], [shear_factor, 1.0, shift_offset]])

    sheared = cv2.warpAffine(
        image,
        mat_y,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Inscribed crop maintaining aspect ratio:
    # Common overlap across all x in [0, w] has height = h - |s_y| * w
    scale = max(0.1, 1.0 - abs(shear_factor) * (w / h))
    crop_w = max(1, int(scale * w))
    crop_h = max(1, int(scale * h))

    x1 = max(0, int(w / 2.0 - crop_w / 2.0))
    y1 = max(0, int(h / 2.0 - crop_h / 2.0))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    cropped = sheared[y1:y2, x1:x2]

    if resize_to_original and (crop_w != w or crop_h != h):
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = cropped

    pct_change = int(round(shear_factor * 100))
    meta = {
        "operation": f"shear_y_{pct_change:+d}%",
        "shear_factor": shear_factor,
        "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": crop_w, "h": crop_h},
        "crop_scale_ratio": round(float(scale), 4),
        "resized_to_original": resize_to_original,
    }
    return result, meta


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Structures & Master Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugmentationResult:
    """ผลลัพธ์ของ 1 การทำ Augmentation"""
    key: str
    title: str
    tag: str
    image: np.ndarray = field(repr=False)
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_all_augmentations(
    image_input: Union[str, Path, np.ndarray],
    resize_crops_to_original: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], List[AugmentationResult]]:
    """รันกระบวนการ Augmentation ทุกรูปแบบโดยแยกกันเป็นอิสระ (ห้ามทำรวมหรือต่อๆ กัน)

    Returns
    -------
    original_image : np.ndarray
    original_meta  : dict
    results        : List[AugmentationResult] รวม 11 วิธี
    """
    if isinstance(image_input, (str, Path)):
        orig_img, orig_meta = read_image(image_input)
    else:
        orig_img = image_input.copy()
        orig_meta = {
            "filename": "raw_input.jpg",
            "height": orig_img.shape[0],
            "width": orig_img.shape[1],
            "channels": orig_img.shape[2] if orig_img.ndim > 2 else 1,
        }

    results: List[AugmentationResult] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1 หมุน +-15 องศา (crop เอาพื้นที่จริงที่เหลือ)
    # ─────────────────────────────────────────────────────────────────────────
    img_rot_p15, meta_rot_p15 = augment_rotate_and_crop(
        orig_img, angle_deg=+15.0, resize_to_original=resize_crops_to_original
    )
    results.append(AugmentationResult(
        key="01_rotate_plus_15",
        title="2.1 Rotate +15 deg (Cropped)",
        tag="2.1 Rotate +15°",
        image=img_rot_p15,
        metadata=meta_rot_p15,
    ))

    img_rot_m15, meta_rot_m15 = augment_rotate_and_crop(
        orig_img, angle_deg=-15.0, resize_to_original=resize_crops_to_original
    )
    results.append(AugmentationResult(
        key="02_rotate_minus_15",
        title="2.1 Rotate -15 deg (Cropped)",
        tag="2.1 Rotate -15°",
        image=img_rot_m15,
        metadata=meta_rot_m15,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2 Horizontal Flip
    # ─────────────────────────────────────────────────────────────────────────
    img_hflip, meta_hflip = augment_horizontal_flip(orig_img)
    results.append(AugmentationResult(
        key="03_horizontal_flip",
        title="2.2 Horizontal Flip",
        tag="2.2 Horizontal Flip",
        image=img_hflip,
        metadata=meta_hflip,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3 ปรับความสว่าง +-25%
    # ─────────────────────────────────────────────────────────────────────────
    img_bright_p25, meta_bright_p25 = augment_brightness(orig_img, factor=1.25)
    results.append(AugmentationResult(
        key="04_brightness_plus_25",
        title="2.3 Brightness +25%",
        tag="2.3 Brightness +25%",
        image=img_bright_p25,
        metadata=meta_bright_p25,
    ))

    img_bright_m25, meta_bright_m25 = augment_brightness(orig_img, factor=0.75)
    results.append(AugmentationResult(
        key="05_brightness_minus_25",
        title="2.3 Brightness -25%",
        tag="2.3 Brightness -25%",
        image=img_bright_m25,
        metadata=meta_bright_m25,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4 ปรับ Contrast +-25%
    # ─────────────────────────────────────────────────────────────────────────
    img_contrast_p25, meta_contrast_p25 = augment_contrast(orig_img, factor=1.25)
    results.append(AugmentationResult(
        key="06_contrast_plus_25",
        title="2.4 Contrast +25%",
        tag="2.4 Contrast +25%",
        image=img_contrast_p25,
        metadata=meta_contrast_p25,
    ))

    img_contrast_m25, meta_contrast_m25 = augment_contrast(orig_img, factor=0.75)
    results.append(AugmentationResult(
        key="07_contrast_minus_25",
        title="2.4 Contrast -25%",
        tag="2.4 Contrast -25%",
        image=img_contrast_m25,
        metadata=meta_contrast_m25,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5 Shear X +-10% (crop เอาพื้นที่จริงที่เหลือ)
    # ─────────────────────────────────────────────────────────────────────────
    img_shear_x_p10, meta_shear_x_p10 = augment_shear_x_and_crop(
        orig_img, shear_factor=+0.10, resize_to_original=resize_crops_to_original
    )
    results.append(AugmentationResult(
        key="08_shear_x_plus_10",
        title="2.5 Shear X +10% (Cropped)",
        tag="2.5 Shear X +10%",
        image=img_shear_x_p10,
        metadata=meta_shear_x_p10,
    ))

    img_shear_x_m10, meta_shear_x_m10 = augment_shear_x_and_crop(
        orig_img, shear_factor=-0.10, resize_to_original=resize_crops_to_original
    )
    results.append(AugmentationResult(
        key="09_shear_x_minus_10",
        title="2.5 Shear X -10% (Cropped)",
        tag="2.5 Shear X -10%",
        image=img_shear_x_m10,
        metadata=meta_shear_x_m10,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6 Shear Y +-10% (crop เอาพื้นที่จริงที่เหลือ)
    # ─────────────────────────────────────────────────────────────────────────
    img_shear_y_p10, meta_shear_y_p10 = augment_shear_y_and_crop(
        orig_img, shear_factor=+0.10, resize_to_original=resize_crops_to_original
    )
    results.append(AugmentationResult(
        key="10_shear_y_plus_10",
        title="2.6 Shear Y +10% (Cropped)",
        tag="2.6 Shear Y +10%",
        image=img_shear_y_p10,
        metadata=meta_shear_y_p10,
    ))

    img_shear_y_m10, meta_shear_y_m10 = augment_shear_y_and_crop(
        orig_img, shear_factor=-0.10, resize_to_original=resize_crops_to_original
    )
    results.append(AugmentationResult(
        key="11_shear_y_minus_10",
        title="2.6 Shear Y -10% (Cropped)",
        tag="2.6 Shear Y -10%",
        image=img_shear_y_m10,
        metadata=meta_shear_y_m10,
    ))

    return orig_img, orig_meta, results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary Grid Visualizer
# ─────────────────────────────────────────────────────────────────────────────

def create_summary_grid(
    original_image: np.ndarray,
    results: List[AugmentationResult],
    cell_width: int = 400,
    cell_height: int = 500,
    cols: int = 4,
    main_title: str = "INDEPENDENT IMAGE AUGMENTATION SUITE (ALL SEPARATE)",
) -> np.ndarray:
    """สร้างภาพรวมสรุป (Summary Grid Comparison) รวม 12 ภาพ (Original + 11 Augmentations)

    Grid 3 แถว x 4 คอลัมน์ พร้อมแถบชื่อกำกับแต่ละภาพและขอบตกแต่งอย่างสวยงาม
    """
    # Item 0 is Original
    all_items = [
        {"title": "0. Original (Reference)", "image": original_image, "tag": "Original"}
    ] + [
        {"title": res.title, "image": res.image, "tag": res.tag}
        for res in results
    ]

    total_items = len(all_items)
    rows = int(np.ceil(total_items / cols))

    banner_height = 42
    padding = 12
    header_height = 80
    border_thick = 2

    grid_w = cols * cell_width + (cols + 1) * padding
    grid_h = header_height + rows * (cell_height + banner_height) + (rows + 1) * padding

    # Create dark slate canvas
    canvas = np.full((grid_h, grid_w, 3), 28, dtype=np.uint8)

    # ── Master Header Banner ──
    cv2.rectangle(canvas, (0, 0), (grid_w, header_height - 10), (18, 20, 24), -1)
    cv2.line(canvas, (0, header_height - 10), (grid_w, header_height - 10), (70, 75, 85), 2)

    cv2.putText(
        canvas,
        main_title,
        (padding + 10, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Total 12 Variants | Each operation applied independently to original image (no chaining)",
        (padding + 10, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 180, 195),
        1,
        cv2.LINE_AA,
    )

    # ── Draw Grid Cells ──
    for idx, item in enumerate(all_items):
        r = idx // cols
        c = idx % cols

        x_cell = padding + c * (cell_width + padding)
        y_cell = header_height + padding + r * (cell_height + banner_height + padding)

        # 1. Title Banner for Cell
        is_original = (idx == 0)
        banner_color = (40, 90, 160) if is_original else (45, 50, 60)
        cv2.rectangle(
            canvas,
            (x_cell, y_cell),
            (x_cell + cell_width, y_cell + banner_height),
            banner_color,
            -1,
        )

        title_text = item["title"]
        cv2.putText(
            canvas,
            title_text,
            (x_cell + 8, y_cell + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 2. Resize Thumbnail maintaining aspect ratio within cell
        img = item["image"]
        ih, iw = img.shape[:2]
        scale = min(cell_width / iw, cell_height / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Centering thumbnail inside cell
        tx = x_cell + (cell_width - nw) // 2
        ty = y_cell + banner_height + (cell_height - nh) // 2

        # Background for thumbnail area
        cv2.rectangle(
            canvas,
            (x_cell, y_cell + banner_height),
            (x_cell + cell_width, y_cell + banner_height + cell_height),
            (15, 16, 18),
            -1,
        )

        # Paste thumbnail
        canvas[ty : ty + nh, tx : tx + nw] = thumb

        # Cell border
        border_color = (65, 140, 240) if is_original else (60, 65, 75)
        cv2.rectangle(
            canvas,
            (x_cell, y_cell),
            (x_cell + cell_width, y_cell + banner_height + cell_height),
            border_color,
            border_thick,
        )

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 5. ImageAugmenter Class
# ─────────────────────────────────────────────────────────────────────────────

class ImageAugmenter:
    """Class จัดการ Augmentation ภาพพัสดุครบวงจร"""

    def __init__(self, resize_crops_to_original: bool = True):
        self.resize_crops_to_original = resize_crops_to_original

    def run(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_dir: Optional[Union[str, Path]] = None,
        save_summary_grid: bool = True,
        grid_filename: str = "summary_augmentation_grid.jpg",
    ) -> Dict[str, Any]:
        """รันกระบวนการทำ Augmentation ทั้งหมดและบันทึกภาพผลลัพธ์

        Parameters
        ----------
        image_input : Path, str ของไฟล์ภาพ หรือ np.ndarray BGR
        output_dir  : โฟลเดอร์ปลายทางสำหรับบันทึกภาพผลลัพธ์
        save_summary_grid : บันทึกภาพตารางสรุปรวมหรือไม่
        grid_filename : ชื่อไฟล์ภาพตารางสรุป
        """
        orig_img, orig_meta, results = run_all_augmentations(
            image_input, resize_crops_to_original=self.resize_crops_to_original
        )

        saved_files: Dict[str, str] = {}
        out_path: Optional[Path] = None

        if output_dir is not None:
            out_path = Path(output_dir).resolve()
            out_path.mkdir(parents=True, exist_ok=True)

            # 1. Save original reference
            orig_save_path = out_path / "00_original.jpg"
            save_image(orig_img, orig_save_path)
            saved_files["00_original"] = str(orig_save_path)

            # 2. Save each individual augmented image
            for res in results:
                file_path = out_path / f"{res.key}.jpg"
                save_image(res.image, file_path)
                res.output_path = file_path
                saved_files[res.key] = str(file_path)

            # 3. Save Summary Grid
            if save_summary_grid:
                grid_img = create_summary_grid(
                    original_image=orig_img,
                    results=results,
                    main_title=f"AUGMENTATION SUITE: {orig_meta['filename']}",
                )
                grid_save_path = out_path / grid_filename
                save_image(grid_img, grid_save_path, quality=92)
                saved_files["summary_grid"] = str(grid_save_path)

            # 4. Save Manifest JSON
            manifest = {
                "source_image": orig_meta,
                "timestamp": datetime.now().isoformat(),
                "resize_crops_to_original": self.resize_crops_to_original,
                "total_variants": len(results) + 1,
                "augmentations": [
                    {
                        "key": r.key,
                        "title": r.title,
                        "output_file": str(r.output_path) if r.output_path else None,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "saved_files": saved_files,
            }
            manifest_path = out_path / "augmentation_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            saved_files["manifest"] = str(manifest_path)

        return {
            "source_meta": orig_meta,
            "original_image": orig_img,
            "results": results,
            "saved_files": saved_files,
            "output_dir": str(out_path) if out_path else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Command Line Interface (CLI)
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Image Augmentation Suite (Separate operations, no chaining)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Path ไปยังไฟล์ภาพต้นฉบับที่ต้องการทำ Augmentation",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="โฟลเดอร์สำหรับบันทึกผลลัพธ์ (Default: tools/output_augmented/<image_stem>)",
    )
    parser.add_argument(
        "--grid-name", type=str, default="summary_augmentation_grid.jpg",
        help="ชื่อไฟล์ภาพตารางสรุปผลลัพธ์",
    )
    parser.add_argument(
        "--no-resize-crops", action="store_true",
        help="ไม่ Resize ภาพที่ Crop กลับเป็นขนาดเดิม (เก็บเฉพาะ Resolution ตามจริงที่ Crop ได้)",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.image.exists():
        print(f"[ERROR] ไม่พบไฟล์ภาพ: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Determine default output directory
    if args.output_dir is None:
        out_dir = Path(__file__).resolve().parent / "output_augmented" / args.image.stem
    else:
        out_dir = args.output_dir

    print("\n" + "=" * 74)
    print("      🎨 PARCEL BOX COUNTER — INDEPENDENT IMAGE AUGMENTATION SUITE")
    print("=" * 74)
    print(f"  • Source Image : {args.image}")
    print(f"  • Output Dir   : {out_dir}")
    print(f"  • Constraint   : แต่ละวิธีทำแยกจากภาพต้นฉบับอย่างอิสระ (ห้ามทำรวมกัน)")
    print("-" * 74)

    augmenter = ImageAugmenter(resize_crops_to_original=not args.no_resize_crops)
    res_dict = augmenter.run(
        image_input=args.image,
        output_dir=out_dir,
        save_summary_grid=True,
        grid_filename=args.grid_name,
    )

    results: List[AugmentationResult] = res_dict["results"]
    print(f"  ✓ [00] Original Reference Image")
    for idx, r in enumerate(results, 1):
        print(f"  ✓ [{idx:02d}] {r.title:<34} → {r.key}.jpg")

    summary_file = res_dict["saved_files"].get("summary_grid")
    if summary_file:
        print(f"\n  🏆 บันทึกภาพรวมสรุป (Summary Grid) สำเร็จที่:")
        print(f"     >> {summary_file}")

    manifest_file = res_dict["saved_files"].get("manifest")
    if manifest_file:
        print(f"  📋 บันทึกรายงานพารามิเตอร์ที่:")
        print(f"     >> {manifest_file}")

    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
