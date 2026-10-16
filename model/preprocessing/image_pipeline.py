# image_pipeline.py — Image Processing Pipeline Orchestrator
"""
โมดูลรวมกระบวนการประมวลผลรูปภาพ (Pipeline Orchestrator)

ไฟล์นี้ทำหน้าที่เป็นตัวประสาน (Orchestrator) และรวบรวมองค์ประกอบของ Pipeline 
ที่ถูกแยกออกเป็นโมดูลย่อยตาม documents/model_training_pipeline.md:
  1. Data Ingestion & Image Reading                  → data_ingestion.py
  2. Geometric Transformation                        → geometric_transform.py
  3. Grayscale, Resizing, Gaussian Blur & Edge Map   → edge_processing.py
  4. SVD Feature Extraction                          → svd_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# เพิ่ม preprocessing/ เข้า sys.path อัตโนมัติ
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (
    INPUT_DIR,
    GRAY_DIR,
    RESIZED_DIR,
    BLUR_DIR,
    CONV_DIR,
    BASE_DIR,
    SUPPORTED_EXTENSIONS,
    TRANSLATE_X,
    TRANSLATE_Y,
    SCALE_FACTOR,
    ROTATION_DEG,
    MAX_PLOTS,
    RESIZE_TARGET,
    BLUR_KSIZE,
    CLIP_PERCENTILE,
)

# ── Re-export จากโมดูลย่อย (เพื่อความเข้ากันได้และการเรียกใช้ที่สะดวก) ───────────
from data_ingestion import (
    upload_zip,
    upload_images,
    collect_image_paths,
    _collect_image_paths,
    read_image,
    load_from_local_paths,
    load_from_zip_path,
)

from geometric_transform import (
    transform_image,
    translate_image,
    scale_image,
    rotate_image,
)

from edge_processing import (
    EDGE_KERNEL,
    to_grayscale,
    resize_image,
    apply_gaussian_blur,
    create_edge_map,
    process_image_edges,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline: Process all images
# ─────────────────────────────────────────────────────────────────────────────

def process_images(
    image_paths: List[Path],
    resize_target: Tuple[int, int] | int = RESIZE_TARGET,
    blur_ksize: int = BLUR_KSIZE,
    clip_percentile: float = CLIP_PERCENTILE,
    apply_transform: bool = False,
    save_intermediate: bool = True,
    save_viz: bool = True,
) -> List[Tuple[str, Path]]:
    """รันกระบวนการประมวลผลรูปภาพแบบ End-to-End
    (Optional Transform) → Grayscale → Resize → Gaussian Blur → Edge Convolution → บันทึก → Visualize

    หมายเหตุ:
      ในกระบวนการเทรนโมเดลจริง ค่าเริ่มต้น apply_transform=False เพราะภาพแต่ละภาพได้รับการ
      Crop / จัดตำแหน่งล่วงหน้าโดยผู้นำเข้าข้อมูลเรียบร้อยแล้ว ส่วนการทำ Transform จะนำไปใช้
      ในระบบ Web Application ให้ผู้ใช้งานปรับแต่งบนหน้าเว็บก่อนส่งมาทำนาย

    Parameters
    ----------
    image_paths       : รายการ Path ของรูปภาพต้นฉบับ
    resize_target     : ขนาดที่ต้องการย่อภาพ (width, height) เพื่อความเหมาะสมกับ Kernel 3x3
    blur_ksize        : ขนาด Gaussian blur kernel เพื่อลด Noise (เช่น 5)
    clip_percentile   : เปอร์เซ็นไทล์สำหรับตัด Outlier ใน Edge Map (เช่น 98.0)
    apply_transform   : False = ข้าม Transform (ภาพถูกจัดตำแหน่งมาแล้วโดยผู้นำเข้าข้อมูล),
                        True = รัน transform_image (เลื่อน/ขยาย/หมุน)
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

    processed: List[Tuple[str, Path]] = []

    for path in image_paths:
        bgr = read_image(path)
        if bgr is None:
            print(f"[SKIP] อ่านรูปไม่ได้/รูปว่าง: {path.name}")
            continue

        # ── Step 1: Geometric Transform (ข้ามในระบบเทรนจริง โดยทำโดยผู้นำเข้าข้อมูล) ──
        if apply_transform:
            transformed = transform_image(bgr)
        else:
            transformed = bgr

        # ── Step 2-5: Grayscale, Resize, Blur, Edge Convolution ──────
        gray, resized, blurred, edge_map = process_image_edges(
            transformed,
            resize_target=resize_target,
            blur_ksize=blur_ksize,
            clip_percentile=clip_percentile,
        )

        # ── บันทึกผลลัพธ์แต่ละขั้นตอน ──────────────────────────────────
        gray_path = GRAY_DIR / f"gray_{path.name}"
        conv_path = CONV_DIR / f"conv_{path.name}"

        gray_ok = cv2.imwrite(str(gray_path), gray)
        conv_ok = cv2.imwrite(str(conv_path), edge_map)

        if save_intermediate:
            resized_path = RESIZED_DIR / f"resized_{path.name}"
            blur_path = BLUR_DIR / f"blur_{path.name}"
            cv2.imwrite(str(resized_path), resized)
            cv2.imwrite(str(blur_path), blurred)

        if not (gray_ok and conv_ok):
            print(f"[SKIP] บันทึกผลลัพธ์ไม่สำเร็จ: {path.name}")
            continue

        processed.append((path.name, gray_path))
        _maybe_visualize(
            original_bgr=bgr,
            transformed_bgr=transformed,
            gray=gray,
            resized=resized,
            blurred=blurred,
            edge_map=edge_map,
            title=path.name,
            count=len(processed),
            apply_transform=apply_transform,
            save_dir=viz_dir,
        )

    print(f"ประมวลผลสำเร็จ: {len(processed)} รูป (บันทึก Grayscale, Resized, Blurred, Convolution)")
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
    apply_transform: bool = False,
    save_dir: Optional[Path] = None,
) -> None:
    """แสดงกราฟและบันทึกภาพผลลัพธ์ทุกขั้นตอนใน pipeline แบบ side-by-side (6 คอลัมน์)"""
    if MAX_PLOTS is not None and count > MAX_PLOTS:
        return

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.patch.set_facecolor("#0b0f19")

    transform_title = "② Transformed\n(Opt-in)" if apply_transform else "② Pre-aligned\n(Ingestor / As-is)"

    steps = [
        (cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), "① Original RGB", None, "#4fc3f7"),
        (cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB), transform_title, None, "#81c995"),
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
