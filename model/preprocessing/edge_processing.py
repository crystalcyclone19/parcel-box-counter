# edge_processing.py — Grayscale, Resizing, Gaussian Blur & Edge Detection
"""
ส่วนที่ 3: การแปลงภาพขาวดำ ลดสัญญาณรบกวน และตรวจจับเส้นขอบ (Edge Processing)
ตามข้อกำหนดใน documents/model_training_pipeline.md (ขั้นตอนที่ 4-5)

ประกอบด้วย:
  1. Grayscale Conversion   (แปลง BGR เป็น Grayscale)
  2. Downscaling (Resize)   (ย่อขนาดลด Micro-noise ให้เข้ากับ Receptive field 3×3)
  3. Gaussian Blur          (กรอง Noise ความถี่สูงบนผิวกล่องพัสดุ)
  4. Edge Convolution       (Laplacian 3×3 พร้อม Percentile Clipping ป้องกันภาพมืด)
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

from config import RESIZE_TARGET, BLUR_KSIZE, CLIP_PERCENTILE

# ─── Laplacian-style Edge Detection Kernel (8-neighbor) ───────────────────────
EDGE_KERNEL = np.array(
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]],
    dtype=np.float32,
)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """แปลงภาพสี (BGR) เป็นภาพขาวดำ (Grayscale) 2D uint8

    Parameters
    ----------
    image : np.ndarray
        ภาพ BGR หรือภาพ Grayscale อยู่แล้ว

    Returns
    -------
    gray : np.ndarray (H, W) uint8
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] | int = RESIZE_TARGET,
    keep_aspect_ratio: bool = False,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """ลดขนาดความละเอียดของภาพ (Downscale) เพื่อลด Micro-noise และปรับสเกลขอบให้เหมาะกับ Kernel 3x3

    Parameters
    ----------
    image             : np.ndarray
        ภาพนำเข้า (Grayscale หรือ BGR)
    target_size       : tuple[int, int] หรือ int, default=(512, 512)
        ขนาดปลายทาง (width, height) ในหน่วยพิกเซล
        หากส่งเป็นจำนวนเต็มเดี่ยว เช่น 512 จะหมายถึง (512, 512)
    keep_aspect_ratio : bool, default=False
        True  = รักษาสัดส่วนภาพเดิม (คำนวณความกว้าง/ยาวใหม่โดยยึดด้านยาวสุดไม่เกิน target_size)
        False = ย่อ/ขยายตาม target_size โดยตรง
    interpolation     : int, default=cv2.INTER_AREA
        วิธี Interpolation สำหรับการย่อภาพ แนะนำ cv2.INTER_AREA สำหรับการ Downsampling เพื่อลด Moire effect และ noise

    Returns
    -------
    resized : np.ndarray
        ภาพที่ถูกลดขนาดความละเอียดแล้ว
    """
    h, w = image.shape[:2]

    if isinstance(target_size, int):
        target_w, target_h = target_size, target_size
    else:
        target_w, target_h = target_size

    if keep_aspect_ratio:
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        dsize = (new_w, new_h)
    else:
        dsize = (target_w, target_h)

    return cv2.resize(image, dsize, interpolation=interpolation)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int] | int = BLUR_KSIZE,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
) -> np.ndarray:
    """ลด Noise และสัญญาณรบกวนความถี่สูงด้วย Gaussian Blur ก่อนส่งเข้า Edge Detector

    Parameters
    ----------
    image       : np.ndarray
        ภาพนำเข้า (รองรับทั้ง Grayscale 2D หรือ BGR 3D uint8)
    kernel_size : tuple[int, int] หรือ int, default=5
        ขนาดของ Gaussian kernel (ต้องเป็นเลขคี่บวก เช่น 3, 5, 7, 9)
        หากส่งเป็นจำนวนเต็มเดี่ยว เช่น 5 จะแปลงเป็น (5, 5) ให้อัตโนมัติ
    sigma_x     : float, default=0.0
        Gaussian standard deviation ในแนวแกน X (0 = คำนวณอัตโนมัติตาม kernel size)
    sigma_y     : float, default=0.0
        Gaussian standard deviation ในแนวแกน Y (0 = เท่ากับ sigma_x)

    Returns
    -------
    blurred : np.ndarray
        ภาพที่ผ่านการกรองลดสัญญาณรบกวนแล้ว
    """
    if isinstance(kernel_size, int):
        ksize = (kernel_size, kernel_size)
    else:
        ksize = kernel_size

    # บังคับให้เป็นเลขคี่บวก
    kx = ksize[0] if ksize[0] % 2 != 0 else ksize[0] + 1
    ky = ksize[1] if ksize[1] % 2 != 0 else ksize[1] + 1
    ksize = (max(1, kx), max(1, ky))

    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x, sigmaY=sigma_y)


def create_edge_map(
    gray: np.ndarray,
    clip_percentile: float = CLIP_PERCENTILE,
    blur_ksize: int = 0,
) -> np.ndarray:
    """สกัดเส้นขอบด้วย Laplacian-style 3×3 kernel พร้อม Percentile Clipping

    Parameters
    ----------
    gray            : np.ndarray
        ภาพ Grayscale (2D uint8)
    clip_percentile : float, default=98.0
        เปอร์เซ็นไทล์สำหรับตัด Outlier ความสว่าง เพื่อป้องกันไม่ให้แสงสะท้อนดึงภาพให้มืด
    blur_ksize      : int, default=0
        ขนาด Gaussian blur kernel เสริม (0 = ไม่เบลอซ้ำ, หรือระบุเลขคี่ เช่น 3, 5)

    Returns
    -------
    edge_map : np.ndarray (uint8, ค่า 0-255)
    """
    src = gray
    if blur_ksize > 1:
        src = apply_gaussian_blur(src, kernel_size=blur_ksize)

    response = cv2.filter2D(src.astype(np.float32), cv2.CV_32F, EDGE_KERNEL)
    absolute = np.abs(response)

    # ตัด Outlier ที่เปอร์เซ็นไทล์ที่กำหนด เพื่อไม่ให้จุดสะท้อนแสงดึงสเกลภาพให้มืด
    p_high = np.percentile(absolute, clip_percentile)
    if p_high > 1e-5:
        scaled = (absolute / p_high) * 255.0
    else:
        scaled = absolute

    return np.clip(scaled, 0, 255).astype(np.uint8)


def process_image_edges(
    image: np.ndarray,
    resize_target: tuple[int, int] | int = RESIZE_TARGET,
    blur_ksize: int = BLUR_KSIZE,
    clip_percentile: float = CLIP_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper ฟังก์ชันแปลงรวดเดียว: Grayscale → Resize → Gaussian Blur → Edge Map

    Returns
    -------
    (gray, resized, blurred, edge_map)
    """
    gray = to_grayscale(image)
    resized = resize_image(gray, target_size=resize_target)
    blurred = apply_gaussian_blur(resized, kernel_size=blur_ksize)
    edge_map = create_edge_map(blurred, clip_percentile=clip_percentile)
    return gray, resized, blurred, edge_map
