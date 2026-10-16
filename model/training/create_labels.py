"""
Dataset Label Generator
=======================
สคริปต์สำหรับอ่านไฟล์รูปภาพเฉพาะในโฟลเดอร์ที่เป็น *_box 
(เช่น 1_box -> 1 กล่อง, 2_box -> 2 กล่อง) ในโฟลเดอร์ dataset
และสกัดจำนวนกล่องเพื่อสร้างไฟล์ labels.csv อัตโนมัติ

Usage:
------
# รันแบบ default (อ่านจาก ./dataset และบันทึกไปที่ ./dataset/labels.csv)
python model/training/create_labels.py

# กำหนดโฟลเดอร์ dataset และตำแหน่งไฟล์ output เอง
python model/training/create_labels.py --dataset-dir dataset --output dataset/labels.csv

# ตัวเลือกใช้เฉพาะชื่อไฟล์ (ไม่รวมชื่อโฟลเดอร์ย่อยใน filename)
python model/training/create_labels.py --flat-filename
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Enable UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Supported image extensions (case-insensitive)
DEFAULT_IMAGE_EXTENSIONS = {
    ".heic", ".heif",
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff", ".tif",
}


def parse_box_count_from_dir(folder_name: str) -> Optional[int]:
    """สกัดจำนวนกล่องจากชื่อโฟลเดอร์ที่ลงท้ายด้วย _box
    
    Examples:
    ---------
    >>> parse_box_count_from_dir("1_box")
    1
    >>> parse_box_count_from_dir("2_box")
    2
    >>> parse_box_count_from_dir("10_box")
    10
    >>> parse_box_count_from_dir("test_system")
    None
    """
    match = re.search(r"(\d+)_box$", folder_name.strip(), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def scan_box_folders(
    dataset_dir: Path,
    image_extensions: Optional[set] = None,
    relative_path: bool = True,
) -> Tuple[List[Dict[str, any]], Dict[str, int], List[str]]:
    """สแกนโฟลเดอร์ dataset เพื่อหาโฟลเดอร์ *_box และรวบรวมรายการไฟล์ภาพพร้อม label
    
    Parameters
    ----------
    dataset_dir : Path
        โฟลเดอร์หลักของ dataset
    image_extensions : Optional[set]
        เซ็ตของนามสกุลภาพที่รองรับ (default: DEFAULT_IMAGE_EXTENSIONS)
    relative_path : bool
        ถ้า True จะบันทึก filename เป็น '1_box/IMG_xxxx.HEIC'
        ถ้า False จะบันทึกเฉพาะชื่อไฟล์ 'IMG_xxxx.HEIC'
        
    Returns
    -------
    records : List[Dict[str, any]]
        รายการข้อมูล [{'filename': ..., 'count': ...}, ...]
    stats_per_folder : Dict[str, int]
        สถิติจำนวนภาพแยกตามชื่อโฟลเดอร์
    skipped_folders : List[str]
        รายชื่อโฟลเดอร์ที่ไม่ตรงกับ *_box และถูกข้าม
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"[Error] ไม่พบโฟลเดอร์ dataset: {dataset_dir}")

    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS

    records: List[Dict[str, any]] = []
    stats_per_folder: Dict[str, int] = {}
    skipped_folders: List[str] = []

    # ค้นหาทุก subdirectory ใน dataset_dir
    subdirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    for subdir in subdirs:
        box_count = parse_box_count_from_dir(subdir.name)
        if box_count is None:
            skipped_folders.append(subdir.name)
            continue

        # ค้นหาไฟล์ภาพในโฟลเดอร์นี้
        img_files = sorted([
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        stats_per_folder[subdir.name] = len(img_files)

        for img_file in img_files:
            if relative_path:
                fname = f"{subdir.name}/{img_file.name}"
            else:
                fname = img_file.name

            records.append({
                "filename": fname,
                "count": box_count,
            })

    return records, stats_per_folder, skipped_folders


def generate_labels_csv(
    dataset_dir: Path,
    output_csv: Path,
    image_extensions: Optional[set] = None,
    relative_path: bool = True,
) -> List[Dict[str, any]]:
    """สร้างไฟล์ labels.csv จากโฟลเดอร์ *_box
    
    Parameters
    ----------
    dataset_dir : Path
        ไดเรกทอรีของ dataset
    output_csv : Path
        ตำแหน่งที่จะบันทึกไฟล์ labels.csv
    image_extensions : Optional[set]
        นามสกุลไฟล์ภาพที่ต้องการ
    relative_path : bool
        ใช้ relative path เช่น '1_box/img.heic' หรือไม่
        
    Returns
    -------
    records : List[Dict[str, any]]
        รายการข้อมูลที่บันทึกลง CSV
    """
    records, stats, skipped = scan_box_folders(
        dataset_dir=dataset_dir,
        image_extensions=image_extensions,
        relative_path=relative_path,
    )

    if not records:
        print(f"[Warning] ไม่พบไฟล์ภาพในโฟลเดอร์ *_box ภายใต้ {dataset_dir}")
        return []

    # ตรวจสอบ parent directory ของ output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # บันทึกเป็น CSV (utf-8-sig เพื่อรองรับโปรแกรมทั่วไปรวมถึง Excel)
    fieldnames = ["filename", "count"]
    with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return records, stats, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate labels.csv automatically by scanning *_box folders in dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to the dataset directory (default: 'dataset')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output labels.csv (default: '<dataset-dir>/labels.csv')",
    )
    parser.add_argument(
        "--flat-filename",
        action="store_true",
        help="Write only filename without folder prefix in CSV (e.g. 'IMG_001.jpg')",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if args.output is None:
        output_csv = dataset_dir / "labels.csv"
    else:
        output_csv = args.output.resolve()

    print("=" * 60)
    print("Parcel Box Dataset - Auto Label Generator")
    print("=" * 60)
    print(f"Dataset Directory : {dataset_dir}")
    print(f"Output CSV File   : {output_csv}")
    print(f"Filename Format   : {'Flat (name only)' if args.flat_filename else 'Relative (folder/name)'}")
    print("-" * 60)

    try:
        records, stats, skipped = generate_labels_csv(
            dataset_dir=dataset_dir,
            output_csv=output_csv,
            relative_path=not args.flat_filename,
        )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[Folders] Scanned *_box directories:")
    total_images = 0
    for folder_name, count in stats.items():
        box_num = parse_box_count_from_dir(folder_name)
        print(f"  + {folder_name:<15} : {count:>4} images  ({box_num} box/image)")
        total_images += count

    if skipped:
        print("\n[Skipped] Non *_box directories ignored:")
        for folder_name in skipped:
            print(f"  - {folder_name}")

    print("\n" + "=" * 60)
    print(f"[Done] Successfully saved: {output_csv}")
    print(f"[Stats] Total: {total_images} images across {len(stats)} folders")
    print("=" * 60)

    if records:
        print("\nPreview first 5 rows in labels.csv:")
        print(f"  {'filename':<30} | {'count'}")
        print("  " + "-" * 38)
        for r in records[:5]:
            print(f"  {r['filename']:<30} | {r['count']}")
        if len(records) > 5:
            print(f"  ... and {len(records) - 5} more rows")


if __name__ == "__main__":
    main()
