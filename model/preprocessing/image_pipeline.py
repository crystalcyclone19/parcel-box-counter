# image_pipeline.py — Upload, Image Transform, Grayscale, Convolution, Visualize
import shutil
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (
    INPUT_DIR, GRAY_DIR, RESIZED_DIR, BLUR_DIR, CONV_DIR,
    BASE_DIR,
    SUPPORTED_EXTENSIONS,
    TRANSLATE_X, TRANSLATE_Y, SCALE_FACTOR, ROTATION_DEG,
    MAX_PLOTS,
    RESIZE_TARGET, BLUR_KSIZE, CLIP_PERCENTILE,
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

def process_images(
    image_paths: list[Path],
    resize_target: tuple[int, int] | int = RESIZE_TARGET,
    blur_ksize: int = BLUR_KSIZE,
    clip_percentile: float = CLIP_PERCENTILE,
    save_intermediate: bool = True,
    save_viz: bool = True,
) -> list[tuple[str, Path]]:
    """Transform → Grayscale → Resize → Gaussian Blur → Edge Convolution → บันทึก → Visualize

    Parameters
    ----------
    image_paths       : รายการ Path ของรูปภาพต้นฉบับ
    resize_target     : ขนาดที่ต้องการย่อภาพ (width, height) เพื่อความเหมาะสมกับ Kernel 3x3
    blur_ksize        : ขนาด Gaussian blur kernel เพื่อลด Noise (เช่น 5)
    clip_percentile   : เปอร์เซ็นไทล์สำหรับตัด Outlier ใน Edge Map (เช่น 98.0)
    save_intermediate : True = บันทึกไฟล์ภาพของแต่ละขั้นตอน (Resized, Blurred)
    save_viz          : True = บันทึกภาพ Visualization รวมทุกขั้นตอนลง pipeline_visualizations/

    Returns
    -------
    processed : รายการ (original_filename, gray_path) ที่สำเร็จ
    """
    # ตรวจสอบและสร้างโฟลเดอร์สำหรับผลลัพธ์ทุกขั้นตอน
    GRAY_DIR.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    if save_intermediate:
        RESIZED_DIR.mkdir(parents=True, exist_ok=True)
        BLUR_DIR.mkdir(parents=True, exist_ok=True)

    viz_dir = (BASE_DIR / "pipeline_visualizations") if save_viz else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    processed: list[tuple[str, Path]] = []

    for path in image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            print(f'[SKIP] อ่านรูปไม่ได้/รูปว่าง: {path.name}')
            continue

        # ── Step 1: Geometric Transform ──────────────────────────────
        rotated = transform_image(bgr)

        # ── Step 2: Grayscale Conversion ─────────────────────────────
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        # ── Step 3: Resize (Downscaling to fit kernel) ────────────────
        resized = resize_image(gray, target_size=resize_target)

        # ── Step 4: Gaussian Blur (Noise Filtering) ──────────────────
        blurred = apply_gaussian_blur(resized, kernel_size=blur_ksize)

        # ── Step 5: Edge Convolution (Laplacian + Percentile) ────────
        edge_map = create_edge_map(blurred, clip_percentile=clip_percentile)

        # ── บันทึกผลลัพธ์แต่ละขั้นตอน ──────────────────────────────────
        gray_path = GRAY_DIR / f'gray_{path.name}'
        conv_path = CONV_DIR / f'conv_{path.name}'

        gray_ok = cv2.imwrite(str(gray_path), gray)
        conv_ok = cv2.imwrite(str(conv_path), edge_map)

        if save_intermediate:
            resized_path = RESIZED_DIR / f'resized_{path.name}'
            blur_path = BLUR_DIR / f'blur_{path.name}'
            cv2.imwrite(str(resized_path), resized)
            cv2.imwrite(str(blur_path), blurred)

        if not (gray_ok and conv_ok):
            print(f'[SKIP] บันทึกผลลัพธ์ไม่สำเร็จ: {path.name}')
            continue

        processed.append((path.name, gray_path))
        _maybe_visualize(
            original_bgr=bgr,
            transformed_bgr=rotated,
            gray=gray,
            resized=resized,
            blurred=blurred,
            edge_map=edge_map,
            title=path.name,
            count=len(processed),
            save_dir=viz_dir,
        )

    print(f'ประมวลผลสำเร็จ: {len(processed)} รูป (บันทึก Grayscale, Resized, Blurred, Convolution)')
    return processed


def _maybe_visualize(
    original_bgr: np.ndarray,
    transformed_bgr: np.ndarray,
    gray: np.ndarray,
    resized: np.ndarray,
    blurred: np.ndarray,
    edge_map: np.ndarray,
    title: str,
    count: int,
    save_dir: Path | None = None,
) -> None:
    """แสดงกราฟและบันทึกภาพผลลัพธ์ทุกขั้นตอนใน pipeline แบบ side-by-side (6 คอลัมน์)"""
    if MAX_PLOTS is not None and count > MAX_PLOTS:
        return

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.patch.set_facecolor("#0b0f19")

    steps = [
        (cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), "① Original RGB", None, "#4fc3f7"),
        (cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB), "② Transform", None, "#81c995"),
        (gray, "③ Grayscale", "gray", "#ffb74d"),
        (resized, f"④ Resized\n({resized.shape[1]}×{resized.shape[0]})", "gray", "#a78bfa"),
        (blurred, f"⑤ Gaussian Blur\n(K={BLUR_KSIZE})", "gray", "#38bdf8"),
        (edge_map, "⑥ Edge Map\n(Laplacian)", "inferno", "#f48fb1"),
    ]

    for ax, (img, step_title, cmap, border_color) in zip(axes, steps):
        kwargs = {"vmin": 0, "vmax": 255} if img.ndim == 2 else {}
        ax.imshow(img, cmap=cmap, **kwargs)
        ax.set_title(step_title, color="white", fontsize=9.5, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0)

    fig.suptitle(f"Preprocessing Pipeline Steps | {title}", color="white", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"pipeline_{Path(title).stem}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        print(f"  💾 บันทึกภาพขั้นตอน Pipeline → {out_path}")

    plt.show()
    plt.close(fig)
