# geometric_transform.py — Geometric Transformation
"""
ส่วนที่ 2: การแปลงทางเรขาคณิต (Geometric Transformation)
ตามข้อกำหนดใน documents/model_training_pipeline.md (ขั้นตอนที่ 3)

หมายเหตุการใช้งาน:
  - ในกระบวนการเทรนโมเดลจริง (Training Pipeline):
    จะข้าม (Skip) การทำ Transformation อัตโนมัติด้วยค่าคงที่ เนื่องจากภาพแต่ละภาพ
    ต้องการการปรับแต่งที่แตกต่างกัน และการจัดตำแหน่ง/Crop ภาพเป็นหน้าที่ของ "ผู้นำเข้าข้อมูล"
    (Data Ingestor / Ingestion Specialist) ก่อนนำเข้าสู่ระบบเทรน
  - ในระบบ Web Application (Production / Inference):
    ฟังก์ชันในไฟล์นี้จะถูกนำไปใช้ใน Web Application ให้ผู้ใช้งาน (User)
    ปรับแต่งและจัดเฟรมภาพบนหน้าเว็บ (Interactive Slider/Canvas) ก่อนส่งรูปภาพและ
    พารามิเตอร์มายัง Backend เพื่อรันการทำนายผล

ประกอบด้วย 3 สเต็ปย่อย:
  1. Translation  (เลื่อนพิกเซลแกน X และแกน Y)
  2. Scaling      (ขยาย/ย่อขนาดภาพจากจุดกึ่งกลาง)
  3. Rotation     (หมุนภาพทวนเข็มนาฬิกาจากจุดกึ่งกลาง)
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

# เพิ่ม preprocessing/ เข้า sys.path อัตโนมัติ
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (
    TRANSLATE_X,
    TRANSLATE_Y,
    SCALE_FACTOR,
    ROTATION_DEG,
)


def translate_image(
    image: np.ndarray,
    shift_x: float = TRANSLATE_X,
    shift_y: float = TRANSLATE_Y,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, ...] = (0, 0, 0),
) -> np.ndarray:
    """เลื่อนตำแหน่งภาพตามแกน X และแกน Y"""
    height, width = image.shape[:2]
    move_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(
        image,
        move_matrix,
        (width, height),
        borderMode=border_mode,
        borderValue=border_value,
    )


def scale_image(
    image: np.ndarray,
    scale_factor: float = SCALE_FACTOR,
    center: tuple[float, float] | None = None,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, ...] = (0, 0, 0),
) -> np.ndarray:
    """ขยาย/ย่อขนาดภาพจากจุดศูนย์กลาง (หรือจุดที่ระบุ)"""
    height, width = image.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    scale_matrix = cv2.getRotationMatrix2D(center, 0.0, scale_factor)
    return cv2.warpAffine(
        image,
        scale_matrix,
        (width, height),
        borderMode=border_mode,
        borderValue=border_value,
    )


def rotate_image(
    image: np.ndarray,
    angle_deg: float = ROTATION_DEG,
    center: tuple[float, float] | None = None,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, ...] = (0, 0, 0),
) -> np.ndarray:
    """หมุนภาพทวนเข็มนาฬิกาตามองศาที่ระบุจากจุดศูนย์กลาง"""
    height, width = image.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        rotate_matrix,
        (width, height),
        borderMode=border_mode,
        borderValue=border_value,
    )


def transform_image(
    bgr: np.ndarray,
    translate_x: float = TRANSLATE_X,
    translate_y: float = TRANSLATE_Y,
    scale_factor: float = SCALE_FACTOR,
    rotation_deg: float = ROTATION_DEG,
) -> np.ndarray:
    """เลื่อน → ขยาย → หมุน; ขนาดภาพคงเดิมตามขนาดภาพต้นฉบับ

    ลำดับการแปลง:
      1. Translation  (+translate_x, +translate_y) pixels
      2. Scaling       scale_factor เท่า (จากจุดกึ่งกลาง)
      3. Rotation      rotation_deg องศา (ทวนเข็มนาฬิกา)

    Parameters
    ----------
    bgr          : np.ndarray
        ภาพ BGR ต้นฉบับ
    translate_x  : float, default=TRANSLATE_X
        ระยะเลื่อนแกน X (พิกเซล)
    translate_y  : float, default=TRANSLATE_Y
        ระยะเลื่อนแกน Y (พิกเซล)
    scale_factor : float, default=SCALE_FACTOR
        อัตราส่วนการขยายภาพ
    rotation_deg : float, default=ROTATION_DEG
        องศาการหมุนภาพ (ทวนเข็มนาฬิกา)

    Returns
    -------
    transformed : np.ndarray
        ภาพที่ผ่านการแปลงทางเรขาคณิตแล้ว (มิติเดียวกับภาพต้นฉบับ)
    """
    height, width = bgr.shape[:2]
    size = (width, height)
    center = (width / 2.0, height / 2.0)

    # 1. Translation
    moved = translate_image(bgr, shift_x=translate_x, shift_y=translate_y)

    # 2. Scaling
    scaled = scale_image(moved, scale_factor=scale_factor, center=center)

    # 3. Rotation
    rotated = rotate_image(scaled, angle_deg=rotation_deg, center=center)

    return rotated
