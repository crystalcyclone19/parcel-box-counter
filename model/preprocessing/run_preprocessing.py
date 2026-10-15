"""
run_preprocessing.py — รัน preprocessing pipeline แบบ end-to-end (Local / Colab)
================================================================================

Pipeline:
    1. รับ input (โฟลเดอร์รูปภาพ หรือ .zip หรือ รูปภาพแต่ละไฟล์)
    2. Geometric transform  (Translation → Scaling → Rotation)
    3. Grayscale conversion
    4. Edge Convolution      (Laplacian-style kernel)
    5. บันทึกและแสดงผล output ในทุกขั้นตอน

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
    TRANSLATE_X, TRANSLATE_Y, SCALE_FACTOR, ROTATION_DEG,
)
from image_pipeline import (
    transform_image,
    create_edge_map,
    _collect_image_paths,
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
# ── ส่วนที่ 2: รับ input รูปภาพ
# ══════════════════════════════════════════════════════════════════════════════

def load_from_local_paths(paths: list[str]) -> list[Path]:
    """
    รับ path แบบ local (โฟลเดอร์, .zip, หรือไฟล์รูปภาพ) → คัดลอกลง INPUT_DIR

    Parameters
    ----------
    paths : รายการ path ที่ผู้ใช้ระบุ

    Returns
    -------
    รายการ Path รูปภาพทั้งหมดใน INPUT_DIR
    """
    print("=" * 60)
    print("ขั้นตอนที่ 1 : รับ Input รูปภาพ")
    print("=" * 60)

    copied = 0
    for raw in paths:
        p = Path(raw).resolve()

        if not p.exists():
            print(f"  [WARN] ไม่พบ path: {p}")
            continue

        # กรณี: โฟลเดอร์
        if p.is_dir():
            print(f"  📁 สแกนโฟลเดอร์: {p}")
            for img_path in p.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    dest = cfg.INPUT_DIR / img_path.name
                    shutil.copy2(img_path, dest)
                    copied += 1
                    print(f"     คัดลอก: {img_path.name}")

        # กรณี: .zip
        elif p.suffix.lower() == ".zip":
            print(f"  📦 แตกไฟล์ zip: {p.name}")
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(tmp)
                for img_path in Path(tmp).rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        dest = cfg.INPUT_DIR / img_path.name
                        shutil.copy2(img_path, dest)
                        copied += 1
                        print(f"     คัดลอก: {img_path.name}")

        # กรณี: ไฟล์รูปภาพเดี่ยว
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            dest = cfg.INPUT_DIR / p.name
            shutil.copy2(p, dest)
            copied += 1
            print(f"  🖼  คัดลอก: {p.name}")

        else:
            print(f"  [SKIP] ไม่รองรับประเภทไฟล์: {p.name}")

    print(f"\n  → นำเข้ารูปภาพสำเร็จ {copied} ไฟล์ → {cfg.INPUT_DIR}\n")

    image_paths = _collect_image_paths()
    if not image_paths:
        print("[ERROR] ไม่พบรูปภาพใน INPUT_DIR กรุณาตรวจสอบ path ที่ระบุ")
        sys.exit(1)
    return image_paths


def load_from_colab_zip() -> list[Path]:
    """Colab: อัปโหลด .zip แล้วแตกไฟล์"""
    from image_pipeline import upload_zip  # type: ignore
    return upload_zip()


def load_from_colab_images() -> list[Path]:
    """Colab: อัปโหลดรูปภาพโดยตรง"""
    from image_pipeline import upload_images  # type: ignore
    return upload_images()


def load_from_zip_path(zip_path: str) -> list[Path]:
    """
    แตกไฟล์ .zip จาก path ที่ระบุล่วงหน้า → คัดลอกรูปภาพลง INPUT_DIR

    ใช้งานบน Colab เมื่อ zip ถูกอัปโหลดไว้แล้ว (เช่น /content/data.zip
    หรือ /content/drive/MyDrive/data.zip) โดยไม่ต้องใช้ interactive uploader

    Parameters
    ----------
    zip_path : path ของไฟล์ .zip (local หรือ Colab path)

    Returns
    -------
    รายการ Path รูปภาพทั้งหมดใน INPUT_DIR
    """
    print("=" * 60)
    print("ขั้นตอนที่ 1 : รับ Input จาก ZIP (path ที่ระบุล่วงหน้า)")
    print("=" * 60)

    p = Path(zip_path).resolve()
    if not p.exists():
        print(f"[ERROR] ไม่พบไฟล์ zip: {p}")
        sys.exit(1)
    if p.suffix.lower() != ".zip":
        print(f"[ERROR] ไฟล์ต้องเป็น .zip เท่านั้น ได้รับ: {p.name}")
        sys.exit(1)

    print(f"  📦 แตกไฟล์ zip: {p}")
    copied = 0
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp)
        for img_path in Path(tmp).rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                dest = cfg.INPUT_DIR / img_path.name
                shutil.copy2(img_path, dest)
                copied += 1
                print(f"     คัดลอก: {img_path.name}")

    print(f"\n  → นำเข้ารูปภาพสำเร็จ {copied} ไฟล์ → {cfg.INPUT_DIR}\n")

    image_paths = _collect_image_paths()
    if not image_paths:
        print("[ERROR] ไม่พบรูปภาพใน zip กรุณาตรวจสอบไฟล์")
        sys.exit(1)
    return image_paths


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 3: แสดงผลทุกขั้นตอน
# ══════════════════════════════════════════════════════════════════════════════

def visualize_all_steps(
    original_bgr: np.ndarray,
    transformed_bgr: np.ndarray,
    gray: np.ndarray,
    edge_map: np.ndarray,
    image_name: str,
    save_dir: Path | None = None,
) -> None:
    """
    แสดงและบันทึกภาพทุกขั้นตอนใน pipeline แบบ side-by-side (4 คอลัมน์)

    Columns:
        ① Original RGB
        ② After Transform (Translate → Scale → Rotate)
        ③ Grayscale
        ④ Edge Convolution (Laplacian kernel)
    """
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor("#0f0f1a")

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    steps = [
        (cv2.cvtColor(original_bgr,    cv2.COLOR_BGR2RGB),
         "① Original RGB", None, "#4fc3f7"),
        (cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB),
         f"② After Transform\n(T{int(TRANSLATE_X)},{int(TRANSLATE_Y)} | S×{SCALE_FACTOR} | R{int(ROTATION_DEG)}°)",
         None, "#81c995"),
        (gray,
         "③ Grayscale", "gray", "#ffb74d"),
        (edge_map,
         "④ Edge Convolution\n(Laplacian 3×3)", "inferno", "#f48fb1"),
    ]

    for i, (img, title, cmap, border_color) in enumerate(steps):
        ax = fig.add_subplot(gs[i])
        kwargs = {"vmin": 0, "vmax": 255} if img.ndim == 2 else {}
        ax.imshow(img, cmap=cmap, **kwargs)
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

    fig.suptitle(
        f"Preprocessing Pipeline  |  {image_name}",
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
) -> list[tuple[str, Path]]:
    """
    รัน preprocessing pipeline ครบทุกขั้นตอนและแสดง output ทุกรูป

    Parameters
    ----------
    image_paths : รายการ Path รูปภาพ input
    save_viz    : True = บันทึกภาพ visualization ลงโฟลเดอร์
    max_display : จำนวนสูงสุดของรูปที่จะ plot (None = ทั้งหมด)

    Returns
    -------
    processed : รายการ (original_filename, gray_path)
    """
    viz_dir = cfg.BASE_DIR / "pipeline_visualizations" if save_viz else None
    processed: list[tuple[str, Path]] = []

    print("=" * 60)
    print(f"ขั้นตอนที่ 2–4 : ประมวลผลภาพ ({len(image_paths)} รูป)")
    print("=" * 60)
    print(f"  Transform params:")
    print(f"    Translation : ({TRANSLATE_X}px, {TRANSLATE_Y}px)")
    print(f"    Scaling     : ×{SCALE_FACTOR}")
    print(f"    Rotation    : {ROTATION_DEG}°")
    print(f"  Edge kernel  : Laplacian-style 3×3")
    print(f"  GRAY_DIR     : {cfg.GRAY_DIR}")
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
        print(f"    ขนาดภาพ    : {w}×{h} px")

        # ── Step 2: Geometric Transform ──────────────────────────────
        transformed = transform_image(bgr)
        print(f"    ② Transform : เสร็จสิ้น")

        # ── Step 3: Grayscale ────────────────────────────────────────
        gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        print(f"    ③ Grayscale : เสร็จสิ้น  (min={gray.min()}, max={gray.max()})")

        # ── Step 4: Edge Convolution ─────────────────────────────────
        edge_map = create_edge_map(gray)
        print(f"    ④ Edge map  : เสร็จสิ้น  (min={edge_map.min()}, max={edge_map.max()})")

        # ── บันทึก output ────────────────────────────────────────────
        gray_path = cfg.GRAY_DIR / f"gray_{path.name}"
        conv_path = cfg.CONV_DIR / f"conv_{path.name}"

        gray_ok = cv2.imwrite(str(gray_path), gray)
        conv_ok = cv2.imwrite(str(conv_path), edge_map)

        if not (gray_ok and conv_ok):
            print(f"    ✗ บันทึกไฟล์ไม่สำเร็จ → ข้าม\n")
            continue

        processed.append((path.name, gray_path))
        print(f"    ✓ Saved: {gray_path.name}")
        print(f"    ✓ Saved: {conv_path.name}")

        # ── Visualization ────────────────────────────────────────────
        should_display = (max_display is None) or (idx <= max_display)
        if should_display:
            visualize_all_steps(
                original_bgr=bgr,
                transformed_bgr=transformed,
                gray=gray,
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
    print(f"  📂 Convolution    → {cfg.CONV_DIR}")
    if viz_dir:
        print(f"  📂 Visualization  → {viz_dir}")
    print()

    return processed


# ══════════════════════════════════════════════════════════════════════════════
# ── ส่วนที่ 5: สรุปผลและแสดง output summary grid
# ══════════════════════════════════════════════════════════════════════════════

def print_output_table(processed: list[tuple[str, Path]]) -> None:
    """แสดงตารางสรุปไฟล์ output ทั้งหมด"""
    if not processed:
        print("[WARN] ไม่มีไฟล์ที่ประมวลผลสำเร็จ")
        return

    print("=" * 60)
    print("รายการ Output Files")
    print("=" * 60)
    fmt = "  {:<40} {:<40}"
    print(fmt.format("Grayscale", "Convolution (Edge)"))
    print("  " + "─" * 78)
    for orig_name, gray_path in processed:
        conv_path = cfg.CONV_DIR / f"conv_{orig_name}"
        gray_sz = f"{gray_path.stat().st_size / 1024:.1f}KB" if gray_path.exists() else "N/A"
        conv_sz = f"{conv_path.stat().st_size / 1024:.1f}KB" if conv_path.exists() else "N/A"
        print(fmt.format(
            f"{gray_path.name} ({gray_sz})",
            f"{conv_path.name} ({conv_sz})",
        ))
    print()


def show_output_grid(processed: list[tuple[str, Path]], max_rows: int = 5) -> None:
    """
    แสดง output grid (Grayscale + Edge Convolution) ทั้งหมดในหน้าเดียว

    Parameters
    ----------
    processed : รายการ (original_filename, gray_path)
    max_rows  : จำนวนแถวสูงสุด (แต่ละแถว = 1 รูป × 2 คอลัมน์)
    """
    n = min(len(processed), max_rows)
    if n == 0:
        return

    print(f"แสดง Output Grid ({n}/{len(processed)} รูป) ...")

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    fig.patch.set_facecolor("#0d1117")

    # กรณีมีแถวเดียว axes จะเป็น 1D
    if n == 1:
        axes = [axes]

    # หัวคอลัมน์
    col_labels = ["Grayscale", "Edge Convolution"]
    colors = ["#ffb74d", "#f48fb1"]
    for col_idx, (label, color) in enumerate(zip(col_labels, colors)):
        axes[0][col_idx].set_title(label, color=color, fontsize=13,
                                   fontweight="bold", pad=12)

    for row_idx, (orig_name, gray_path) in enumerate(processed[:n]):
        conv_path = cfg.CONV_DIR / f"conv_{orig_name}"

        gray_img = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        conv_img = cv2.imread(str(conv_path), cv2.IMREAD_GRAYSCALE)

        for col_idx, (cmap, img) in enumerate([("gray", gray_img), ("inferno", conv_img)]):
            ax = axes[row_idx][col_idx]
            if img is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="red", fontsize=14, transform=ax.transAxes)
            ax.set_ylabel(orig_name, color="#cccccc", fontsize=8,
                          rotation=0, labelpad=120, va="center")
            ax.axis("off")

    fig.suptitle("Output Summary — Grayscale & Edge Convolution",
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

    max_display: int | None = args.max_display if args.max_display > 0 else None

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

    # 2-4. รัน pipeline
    processed = run_pipeline(
        image_paths=image_paths,
        save_viz=not args.no_save_viz,
        max_display=max_display,
    )

    # 5. แสดงสรุปผล
    print_output_table(processed)
    show_output_grid(processed, max_rows=max_display or len(processed))

    print("✅ Pipeline เสร็จสมบูรณ์!")
    print()


if __name__ == "__main__":
    main()
