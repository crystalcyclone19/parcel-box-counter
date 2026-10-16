# image_pipeline.py — Upload, Image Transform, Grayscale, Convolution, Visualize
import shutil
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (
    INPUT_DIR, GRAY_DIR, CONV_DIR,
    SUPPORTED_EXTENSIONS,
    TRANSLATE_X, TRANSLATE_Y, SCALE_FACTOR, ROTATION_DEG,
    MAX_PLOTS,
)

# ─── Edge detection kernel ────────────────────────────────────────────────────
EDGE_KERNEL = np.array(
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]],
    dtype=np.float32
)


# ─────────────────────────────────────────────────────────────────────────────
# Upload helpers (Colab-specific)
# ─────────────────────────────────────────────────────────────────────────────

def upload_zip() -> list[Path]:
    """อัปโหลด .zip → แตกไฟล์ → คัดลอกรูปภาพลง INPUT_DIR
    คืนรายการ Path ของรูปภาพทั้งหมดที่นำเข้าสำเร็จ
    """
    from google.colab import files  # type: ignore[import]

    print('กรุณาอัปโหลดไฟล์ .zip ที่รวบรวมไฟล์ภาพของคุณ')
    uploaded_zip = files.upload()

    for zip_name, zip_bytes in uploaded_zip.items():
        if Path(zip_name).suffix.lower() != '.zip':
            print(f'ข้ามไฟล์เนื่องจากไม่ใช่ .zip: {zip_name}')
            continue

        print(f'กำลังดำเนินการกับไฟล์: {zip_name}')
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = Path(temp_dir) / zip_name
            temp_zip_path.write_bytes(zip_bytes)

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

                count = 0
                for file_path in Path(temp_dir).rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        destination = INPUT_DIR / file_path.name
                        shutil.copy2(str(file_path), str(destination))
                        count += 1

            print(f'แตกไฟล์สำเร็จ: นำเข้า {count} รูปภาพไปยัง {INPUT_DIR.name}')

    return _collect_image_paths()


def upload_images() -> list[Path]:
    """อัปโหลดรูป JPG/PNG หลายไฟล์โดยตรง → บันทึกลง INPUT_DIR
    คืนรายการ Path ของรูปภาพทั้งหมดใน INPUT_DIR
    """
    from google.colab import files  # type: ignore[import]

    print('อัปโหลดไฟล์ JPG/JPEG/PNG ได้หลายไฟล์พร้อมกัน')
    uploaded = files.upload()

    for uploaded_name, file_bytes in uploaded.items():
        safe_name = Path(uploaded_name).name
        if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f'ข้ามไฟล์ที่ไม่รองรับ: {safe_name}')
            continue

        destination = INPUT_DIR / safe_name
        source = Path.cwd() / safe_name
        if source.is_file() and source.resolve() != destination.resolve():
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
        else:
            destination.write_bytes(file_bytes)
        print('บันทึก:', destination.name)

    image_paths = _collect_image_paths()
    if not image_paths:
        raise ValueError('ไม่พบรูปภาพใน input_images กรุณาอัปโหลดไฟล์ JPG/PNG')
    return image_paths


def _collect_image_paths() -> list[Path]:
    """คืนรายการ Path รูปภาพที่รองรับทั้งหมดใน INPUT_DIR (เรียงชื่อ)"""
    paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f'พบภาพทั้งหมดใน input_images: {len(paths)} รูป')
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Image transformation functions
# ─────────────────────────────────────────────────────────────────────────────

def transform_image(bgr: np.ndarray) -> np.ndarray:
    """เลื่อน → ขยาย → หมุน; ขนาดภาพคงเดิม

    ลำดับการแปลง:
      1. Translation  (+TRANSLATE_X, +TRANSLATE_Y) pixels
      2. Scaling       SCALE_FACTOR เท่า (จากจุดกึ่งกลาง)
      3. Rotation      ROTATION_DEG องศา (ทวนเข็มนาฬิกา)
    """
    height, width = bgr.shape[:2]
    size   = (width, height)
    center = (width / 2.0, height / 2.0)

    # 1. Translation
    move_matrix = np.float32([[1, 0, TRANSLATE_X], [0, 1, TRANSLATE_Y]])
    moved = cv2.warpAffine(bgr, move_matrix, size,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 2. Scaling
    scale_matrix = cv2.getRotationMatrix2D(center, 0, SCALE_FACTOR)
    scaled = cv2.warpAffine(moved, scale_matrix, size,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 3. Rotation
    rotate_matrix = cv2.getRotationMatrix2D(center, ROTATION_DEG, 1.0)
    rotated = cv2.warpAffine(scaled, rotate_matrix, size,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


# def create_edge_map(gray: np.ndarray) -> np.ndarray:
#     """ใช้ Laplacian-style edge kernel → คืน edge map (uint8, 0-255)"""
#     response   = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, EDGE_KERNEL)
#     absolute   = np.abs(response)
#     normalized = cv2.normalize(absolute, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized.astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Noise Reduction and Preprocessing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int] | int = (5, 5),
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
) -> np.ndarray:
    """ลด Noise และสัญญาณรบกวนความถี่สูงด้วย Gaussian Blur ก่อนส่งเข้า Edge Detector

    Parameters
    ----------
    image       : np.ndarray
        ภาพนำเข้า (รองรับทั้ง Grayscale 2D หรือ BGR 3D uint8)
    kernel_size : tuple[int, int] หรือ int, default=(5, 5)
        ขนาดของ Gaussian kernel (ต้องเป็นเลขคี่บวก เช่น 3, 5, 7, 9)
        หากส่งเป็นจำนวนเต็มเดี่ยว เช่น 5 จะแปลงเป็น (5, 5) ให้อัตโนมัติ
    sigma_x     : float, default=0.0
        Gaussian standard deviation ในแนวแกน X
        หากตั้งเป็น 0.0 OpenCV จะคำนวณจากขนาด kernel ให้อัตโนมัติ
    sigma_y     : float, default=0.0
        Gaussian standard deviation ในแนวแกน Y
        หากตั้งเป็น 0.0 จะใช้ค่าเดียวกับ sigma_x

    Returns
    -------
    blurred : np.ndarray
        ภาพที่ผ่านการกรองลดสัญญาณรบกวนแล้ว
    """
    if isinstance(kernel_size, int):
        ksize = (kernel_size, kernel_size)
    else:
        ksize = kernel_size

    # ปรับให้เป็นเลขคี่และอย่างน้อย 1 พิกเซล
    kx = ksize[0] if ksize[0] % 2 != 0 else ksize[0] + 1
    ky = ksize[1] if ksize[1] % 2 != 0 else ksize[1] + 1
    ksize = (max(1, kx), max(1, ky))

    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x, sigmaY=sigma_y)


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] | int = (512, 512),
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


def create_edge_map(
    gray: np.ndarray,
    clip_percentile: float = 98.0,
    blur_ksize: int = 0,
) -> np.ndarray:
    """ใช้ Laplacian-style edge kernel พร้อม Percentile Clipping และ Gaussian Blur เพื่อลด Noise

    Parameters
    ----------
    gray            : np.ndarray
        ภาพ Grayscale (2D uint8)
    clip_percentile : float, default=98.0
        เปอร์เซ็นไทล์สำหรับตัด Outlier ความสว่าง
    blur_ksize      : int, default=0
        ขนาด Gaussian blur kernel (0 = ไม่เบลอ, เช่น 3 หรือ 5 สำหรับตัด noise ก่อนทำ edge)

    Returns
    -------
    edge_map : np.ndarray (uint8, 0-255)
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



# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline: process all images
# ─────────────────────────────────────────────────────────────────────────────

def process_images(image_paths: list[Path]) -> list[tuple[str, Path]]:
    """Transform → Grayscale → Edge Convolution → บันทึก → (optionally) Visualize

    Parameters
    ----------
    image_paths : รายการ Path ของรูปภาพต้นฉบับ

    Returns
    -------
    processed : รายการ (original_filename, gray_path) ที่สำเร็จ
    """
    processed: list[tuple[str, Path]] = []

    for path in image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            print(f'[SKIP] อ่านรูปไม่ได้/รูปว่าง: {path.name}')
            continue

        rotated  = transform_image(bgr)
        gray     = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        edge_map = create_edge_map(gray)

        gray_path = GRAY_DIR / f'gray_{path.name}'
        conv_path = CONV_DIR / f'conv_{path.name}'
        gray_ok   = cv2.imwrite(str(gray_path), gray)
        conv_ok   = cv2.imwrite(str(conv_path), edge_map)
        if not (gray_ok and conv_ok):
            print(f'[SKIP] บันทึกผลลัพธ์ไม่สำเร็จ: {path.name}')
            continue

        processed.append((path.name, gray_path))
        _maybe_visualize(bgr, gray, edge_map, path.name, len(processed))

    print(f'สร้างภาพ Grayscale และ Convolution สำเร็จ: {len(processed)} รูป')
    return processed


def _maybe_visualize(bgr: np.ndarray, gray: np.ndarray,
                     edge_map: np.ndarray, title: str, count: int) -> None:
    """แสดงกราฟ Original / Grayscale / Edge Map (ถ้าไม่เกิน MAX_PLOTS)"""
    if MAX_PLOTS is not None and count > MAX_PLOTS:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original RGB')
    axes[1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Grayscale Matrix (after transform)')
    axes[2].imshow(edge_map, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Convolution Feature Map')
    for axis in axes:
        axis.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
