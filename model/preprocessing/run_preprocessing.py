"""
run_preprocessing.py — รัน preprocessing pipeline แบบ end-to-end (Local / Colab)
================================================================================

Pipeline:
    1. รับ input (Data Ingestion — ภาพผ่านการทำ Geometric Transformation โดยผู้นำเข้าข้อมูลมาแล้ว)
    2. Grayscale conversion
    3. Resizing (Downscaling to 512x512 px)
    4. Gaussian Blur (ลด Noise)
    5. Edge Convolution (Laplacian-style kernel + Percentile Clipping)
    6. บันทึกและแสดงผล output ในทุกขั้นตอน

วิธีรัน (Local):
    python run_preprocessing.py --input path/to/images
    python run_preprocessing.py --input path/to/archive.zip
    python run_preprocessing.py --input img1.jpg img2.png

วิธีรัน (ไฟล์อยู่บน google drive)
    python run_preprocessing.py --zip-path /content/drive/MyDrive/data.zip

วิธีรัน (Colab — ระบุ path ล่วงหน้า):
    python run_preprocessing.py --zip-path /content/my_images.zip
    python run_preprocessing.py --zip-path /content/drive/MyDrive/data.zip

วิธีรัน (Colab — ใช้ interactive uploader):
    python run_preprocessing.py --colab-zip     # popup อัปโหลด .zip
    python run_preprocessing.py --colab-images  # popup อัปโหลดรูปภาพโดยตรง

Options เพิ่มเติม
    python run_preprocessing.py --zip-path /content/data.zip --max-display 3  # plot แค่ 3 รูปแรก
    python run_preprocessing.py --zip-path /content/data.zip --clear           # ล้างไฟล์เดิมก่อนรัน
    python run_preprocessing.py --zip-path /content/data.zip --no-save-viz     # ไม่บันทึก PNG
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
import tempfile
import time
from pathlib import Path

# ป้องกัน UnicodeEncodeError บน Windows console
if sys.platform == "win32":
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── import โมดูลของโปรเจกต์ ──────────────────────────────────────────────────
# เพิ่ม preprocessing/ เข้า sys.path อัตโนมัติ
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg
from config import (
    SUPPORTED_EXTENSIONS,
    RESIZE_TARGET, BLUR_KSIZE, CLIP_PERCENTILE,
)
from data_ingestion import (
    load_from_local_paths,
    load_from_zip_path,
    upload_zip,
    upload_images,
    read_image,
    collect_image_paths,
)
from edge_processing import (
    to_grayscale,
    resize_image,
    apply_gaussian_blur,
    create_edge_map,
    process_image_edges,
)


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 1: ตั้งค่า output directories
# ══════════════════════════════════════════════════════════════════════════════

def setup_directories() -> None:
    """สร้างโฟลเดอร์ output ทั้งหมด (ถ้ายังไม่มี)"""
    cfg.ensure_dirs()
    print("✓ ตรวจสอบและสร้างโฟลเดอร์ output เรียบร้อย")
    for d in cfg.ALL_DIRS:
        print(f"  └─ {d}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 2: Wrapper สำหรับ Colab Input
# ══════════════════════════════════════════════════════════════════════════════

def load_from_colab_zip() -> list[Path]:
    """Colab: อัปโหลด .zip แล้วแตกไฟล์"""
    return upload_zip()


def load_from_colab_images() -> list[Path]:
    """Colab: อัปโหลดรูปภาพโดยตรง"""
    return upload_images()


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 3: แสดงผลทุกขั้นตอน
# ══════════════════════════════════════════════════════════════════════════════

def visualize_all_steps(
    original_bgr: np.ndarray,
    gray: np.ndarray,
    resized: np.ndarray,
    blurred: np.ndarray,
    edge_map: np.ndarray,
    image_name: str,
    save_dir: Path | None = None,
) -> None:
    """
    แสดงและบันทึกภาพทุกขั้นตอนใน pipeline แบบ side-by-side (5 คอลัมน์)

    Columns:
        ① Input RGB (Pre-transformed by Data Ingestion)
        ② Grayscale
        ③ Resized (Downscaled to fit 3×3 kernel)
        ④ Gaussian Blur (Noise reduction)
        ⑤ Edge Convolution (Laplacian kernel + Percentile clip)
    """
    fig = plt.figure(figsize=(19, 4))
    fig.patch.set_facecolor("#0f0f1a")

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    steps = [
        (cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
         "① Input RGB\n(Pre-transformed)", None, "#4fc3f7"),
        (gray,
         "② Grayscale", "gray", "#ffb74d"),
        (resized,
         f"③ Resized\n({resized.shape[1]}×{resized.shape[0]} px)", "gray", "#a78bfa"),
        (blurred,
         f"④ Gaussian Blur\n(K={BLUR_KSIZE}×{BLUR_KSIZE})", "gray", "#38bdf8"),
        (edge_map,
         "⑤ Edge Convolution\n(Cleaned Laplacian)", "inferno", "#f48fb1"),
    ]

    for i, (img, title, cmap, border_color) in enumerate(steps):
        ax = fig.add_subplot(gs[i])
        kwargs = {"vmin": 0, "vmax": 255} if img.ndim == 2 else {}
        ax.imshow(img, cmap=cmap, **kwargs)
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0)

    fig.suptitle(
        f"Preprocessing Pipeline Steps  |  {image_name}",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"pipeline_{Path(image_name).stem}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=120,
                    facecolor=fig.get_facecolor())
        print(f"  💾 บันทึก visualization → {out_path}")

    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 4: pipeline หลัก
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_paths: list[Path],
    save_viz: bool = True,
    max_display: int | None = 5,
    **kwargs,
) -> list[tuple[str, Path]]:
    """
    รัน preprocessing pipeline ครบทุกขั้นตอนและแสดง output ทุกรูป
    (Input Pre-transformed → Grayscale → Resize → Gaussian Blur → Edge Convolution)

    Parameters
    ----------
    image_paths : รายการ Path รูปภาพ input (ที่ผ่านการทำ Geometric Transform โดยผู้เตรียมข้อมูลมาแล้ว)
    save_viz    : True = บันทึกภาพ visualization ลงโฟลเดอร์
    max_display : จำนวนสูงสุดของรูปที่จะ plot (None = ทั้งหมด)

    Returns
    -------
    processed : รายการ (original_filename, gray_path)
    """
    viz_dir = cfg.BASE_DIR / "pipeline_visualizations" if save_viz else None
    processed: list[tuple[str, Path]] = []

    print("=" * 60)
    print(f"ขั้นตอนที่ 2–5 : ประมวลผลภาพ ({len(image_paths)} รูป)")
    print("=" * 60)
    print(f"  หมายเหตุ     : ภาพผ่านการทำ Geometric Transform มาแล้วตั้งแต่ขั้นตอนนำเข้าข้อมูล")
    print(f"  Resize target: {cfg.RESIZE_TARGET}")
    print(f"  Gaussian Blur: Kernel {cfg.BLUR_KSIZE}×{cfg.BLUR_KSIZE}")
    print(f"  Edge kernel  : Laplacian-style 3×3 (Clip percentile: {cfg.CLIP_PERCENTILE}%)")
    print(f"  GRAY_DIR     : {cfg.GRAY_DIR}")
    print(f"  RESIZED_DIR  : {cfg.RESIZED_DIR}")
    print(f"  BLUR_DIR     : {cfg.BLUR_DIR}")
    print(f"  CONV_DIR     : {cfg.CONV_DIR}")
    print()

    t_start = time.perf_counter()

    for idx, path in enumerate(image_paths, start=1):
        header = f"[{idx:03d}/{len(image_paths):03d}] {path.name}"
        print(f"  {header}")
        print(f"  {'─' * len(header)}")

        # ── อ่านภาพ ──────────────────────────────────────────────────
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            print(f"    ✗ อ่านรูปไม่ได้ → ข้าม\n")
            continue

        h, w = bgr.shape[:2]
        print(f"    ขนาดภาพนำเข้า : {w}×{h} px")

        # ── Step 2: Grayscale ────────────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        print(f"    ② Grayscale : เสร็จสิ้น  (min={gray.min()}, max={gray.max()})")

        # ── Step 3: Resize (Downscaling to fit 3×3 kernel) ───────────
        resized = resize_image(gray, target_size=cfg.RESIZE_TARGET)
        print(f"    ③ Resized   : เสร็จสิ้น  ({resized.shape[1]}×{resized.shape[0]} px)")

        # ── Step 4: Gaussian Blur (Noise Reduction) ──────────────────
        blurred = apply_gaussian_blur(resized, kernel_size=cfg.BLUR_KSIZE)
        print(f"    ④ Blur      : เสร็จสิ้น  (K={cfg.BLUR_KSIZE}×{cfg.BLUR_KSIZE})")

        # ── Step 5: Edge Convolution ─────────────────────────────────
        edge_map = create_edge_map(blurred, clip_percentile=cfg.CLIP_PERCENTILE)
        print(f"    ⑤ Edge map  : เสร็จสิ้น  (min={edge_map.min()}, max={edge_map.max()})")

        # ── บันทึก output แต่ละขั้นตอน ───────────────────────────────
        gray_path = cfg.GRAY_DIR / f"gray_{path.name}"
        resized_path = cfg.RESIZED_DIR / f"resized_{path.name}"
        blur_path = cfg.BLUR_DIR / f"blur_{path.name}"
        conv_path = cfg.CONV_DIR / f"conv_{path.name}"

        gray_ok = cv2.imwrite(str(gray_path), gray)
        resized_ok = cv2.imwrite(str(resized_path), resized)
        blur_ok = cv2.imwrite(str(blur_path), blurred)
        conv_ok = cv2.imwrite(str(conv_path), edge_map)

        if not (gray_ok and resized_ok and blur_ok and conv_ok):
            print(f"    ✗ บันทึกไฟล์ไม่สำเร็จ → ข้าม\n")
            continue

        processed.append((path.name, gray_path))
        print(f"    ✓ Saved: {gray_path.name}")
        print(f"    ✓ Saved: {resized_path.name}")
        print(f"    ✓ Saved: {blur_path.name}")
        print(f"    ✓ Saved: {conv_path.name}")

        # ── Visualization ────────────────────────────────────────────
        should_display = (max_display is None) or (idx <= max_display)
        if should_display:
            visualize_all_steps(
                original_bgr=bgr,
                gray=gray,
                resized=resized,
                blurred=blurred,
                edge_map=edge_map,
                image_name=path.name,
                save_dir=viz_dir,
            )
        print()

    elapsed = time.perf_counter() - t_start

    print("=" * 60)
    print("สรุปผลการประมวลผล Pipeline")
    print("=" * 60)
    print(f"  รูปทั้งหมด    : {len(image_paths)}")
    print(f"  สำเร็จ         : {len(processed)}")
    print(f"  ล้มเหลว        : {len(image_paths) - len(processed)}")
    print(f"  เวลาที่ใช้      : {elapsed:.2f} วินาที")
    print()
    print("Output directories:")
    print(f"  📂 Grayscale      → {cfg.GRAY_DIR}")
    print(f"  📂 Resized        → {cfg.RESIZED_DIR}")
    print(f"  📂 Blurred        → {cfg.BLUR_DIR}")
    print(f"  📂 Convolution    → {cfg.CONV_DIR}")
    if viz_dir:
        print(f"  📂 Visualization  → {viz_dir}")
    print()

    return processed


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 5: สรุปผลและแสดง output summary grid
# ══════════════════════════════════════════════════════════════════════════════

def print_output_table(processed: list[tuple[str, Path]]) -> None:
    """แสดงตารางสรุปไฟล์ output ทั้งหมดของทุกขั้นตอน"""
    if not processed:
        print("[WARN] ไม่มีไฟล์ที่ประมวลผลสำเร็จ")
        return

    print("=" * 95)
    print("รายการ Output Files (ทุกขั้นตอนใน Pipeline)")
    print("=" * 95)
    fmt = "  {:<23} {:<23} {:<23} {:<23}"
    print(fmt.format("Grayscale", "Resized", "Blurred", "Convolution (Edge)"))
    print("  " + "─" * 90)
    for orig_name, gray_path in processed:
        resized_path = cfg.RESIZED_DIR / f"resized_{orig_name}"
        blur_path = cfg.BLUR_DIR / f"blur_{orig_name}"
        conv_path = cfg.CONV_DIR / f"conv_{orig_name}"

        gray_sz = f"{gray_path.stat().st_size / 1024:.1f}KB" if gray_path.exists() else "N/A"
        resized_sz = f"{resized_path.stat().st_size / 1024:.1f}KB" if resized_path.exists() else "N/A"
        blur_sz = f"{blur_path.stat().st_size / 1024:.1f}KB" if blur_path.exists() else "N/A"
        conv_sz = f"{conv_path.stat().st_size / 1024:.1f}KB" if conv_path.exists() else "N/A"

        print(fmt.format(
            f"{gray_path.name} ({gray_sz})",
            f"{resized_path.name} ({resized_sz})",
            f"{blur_path.name} ({blur_sz})",
            f"{conv_path.name} ({conv_sz})",
        ))
    print()


def show_output_grid(processed: list[tuple[str, Path]], max_rows: int = 5) -> None:
    """
    แสดง output grid (Resized + Blurred + Edge Convolution) ทั้งหมดในหน้าเดียว

    Parameters
    ----------
    processed : รายการ (original_filename, gray_path)
    max_rows  : จำนวนแถวสูงสุด (แต่ละแถว = 1 รูป × 3 คอลัมน์)
    """
    n = min(len(processed), max_rows)
    if n == 0:
        return

    print(f"แสดง Output Grid ({n}/{len(processed)} รูป) ...")

    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    fig.patch.set_facecolor("#0d1117")

    # กรณีมีแถวเดียว axes จะเป็น 1D
    if n == 1:
        axes = [axes]

    # หัวคอลัมน์
    col_labels = ["Resized (512×512)", f"Blurred (K={cfg.BLUR_KSIZE})", "Edge Convolution"]
    colors = ["#a78bfa", "#38bdf8", "#f48fb1"]
    for col_idx, (label, color) in enumerate(zip(col_labels, colors)):
        axes[0][col_idx].set_title(label, color=color, fontsize=13,
                                   fontweight="bold", pad=12)

    for row_idx, (orig_name, _) in enumerate(processed[:n]):
        resized_path = cfg.RESIZED_DIR / f"resized_{orig_name}"
        blur_path = cfg.BLUR_DIR / f"blur_{orig_name}"
        conv_path = cfg.CONV_DIR / f"conv_{orig_name}"

        resized_img = cv2.imread(str(resized_path), cv2.IMREAD_GRAYSCALE)
        blur_img = cv2.imread(str(blur_path), cv2.IMREAD_GRAYSCALE)
        conv_img = cv2.imread(str(conv_path), cv2.IMREAD_GRAYSCALE)

        items = [("gray", resized_img), ("gray", blur_img), ("inferno", conv_img)]
        for col_idx, (cmap, img) in enumerate(items):
            ax = axes[row_idx][col_idx]
            if img is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="red", fontsize=14, transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle("Output Summary — Resized, Blurred & Edge Convolution",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ── Entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_preprocessing",
        description="Preprocessing pipeline: รูปภาพ/zip → transform → grayscale → edge convolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่างการใช้งาน (Local):
  python run_preprocessing.py --input ./images
  python run_preprocessing.py --input archive.zip
  python run_preprocessing.py --input img1.jpg img2.png --max-display 3

ตัวอย่างการใช้งาน (Google Colab — ระบุ path ล่วงหน้า):
  python run_preprocessing.py --zip-path /content/data.zip
  python run_preprocessing.py --zip-path /content/drive/MyDrive/Box_Project/images.zip

ตัวอย่างการใช้งาน (Google Colab — interactive uploader):
  python run_preprocessing.py --colab-zip
  python run_preprocessing.py --colab-images
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i", nargs="+", metavar="PATH",
        help="(Local) โฟลเดอร์, .zip, หรือไฟล์รูปภาพ",
    )
    input_group.add_argument(
        "--zip-path", metavar="ZIP",
        help="(Colab/Local) ระบุ path ของไฟล์ .zip โดยตรง "
             "เช่น /content/data.zip หรือ /content/drive/MyDrive/data.zip",
    )
    input_group.add_argument(
        "--colab-zip", action="store_true",
        help="(Colab) อัปโหลด .zip ผ่าน Colab interactive uploader",
    )
    input_group.add_argument(
        "--colab-images", action="store_true",
        help="(Colab) อัปโหลดรูปภาพโดยตรงผ่าน Colab interactive uploader",
    )
    parser.add_argument(
        "--max-display", type=int, default=5, metavar="N",
        help="จำนวนสูงสุดของรูปที่จะ plot (default: 5, 0 = ไม่แสดง)",
    )
    parser.add_argument(
        "--no-save-viz", action="store_true",
        help="ไม่บันทึกภาพ visualization ลงไฟล์",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="ล้างไฟล์เดิมใน pipeline folders ก่อนรัน",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    max_display: int | None = args.max_display if args.max_display >= 0 else None

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Preprocessing Pipeline — Parcel Box Counter     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 0. ตั้งค่า directories
    setup_directories()

    # ล้างไฟล์เดิม (ถ้าขอ)
    if args.clear:
        from svd_features import clear_pipeline_folders  # type: ignore
        clear_pipeline_folders()
        print()

    # 1. รับ input
    if args.input:
        image_paths = load_from_local_paths(args.input)
    elif args.zip_path:
        image_paths = load_from_zip_path(args.zip_path)
    elif args.colab_zip:
        image_paths = load_from_colab_zip()
    else:
        image_paths = load_from_colab_images()

    # 2-5. รัน pipeline
    processed = run_pipeline(
        image_paths=image_paths,
        save_viz=not args.no_save_viz,
        max_display=max_display,
    )

    # 7. แสดงสรุปผล
    print_output_table(processed)
    if max_display != 0:
        show_output_grid(processed, max_rows=max_display or len(processed))

    print("✅ Pipeline เสร็จสมบูรณ์!")
    print()


if __name__ == "__main__":
    main()
