"""
Dataset ZIP Extractor
=====================
สคริปต์สำหรับแตกไฟล์ .zip ไปยังโฟลเดอร์ที่กำหนด
เหมาะสำหรับแตกไฟล์ Dataset (เช่น zip ที่มีโฟลเดอร์ 1_box, 2_box หรือไฟล์ภาพ)

Usage:
------
# 1. แตกไฟล์ระบุต้นทางและปลายทางผ่าน CLI:
python extract_dataset.py --zip path/to/dataset.zip --dest dataset

# 2. รันแบบ Interactive (ระบบจะให้กรอกพาธหากไม่ได้ระบุ):
python extract_dataset.py

# 3. ล้างโฟลเดอร์ปลายทางก่อนแตกไฟล์ใหม่:
python extract_dataset.py --zip dataset.zip --dest dataset --clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# Enable UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ไฟล์ขยะระบบที่ไม่จำเป็นต้องแตกออกมา
IGNORED_PATTERNS = {
    "__macosx",
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
}


def is_ignored(name: str) -> bool:
    """ตรวจสอบว่าเป็นไฟล์ขยะของ OS หรือไม่"""
    lower = name.lower()
    return any(p in lower for p in IGNORED_PATTERNS)


def safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, target_dir: Path) -> Optional[Path]:
    """แตกไฟล์ทีละไฟล์อย่างปลอดภัย ป้องกัน Zip Slip (Path Traversal Vulnerability)"""
    # แก้ไข encoding ชื่อไฟล์ภาษาไทยหรือ unicode หากจำเป็น
    filename = member.filename
    try:
        # หาก flag bit 11 ไม่ได้ตั้ง (ไม่ใช่ utf-8 แท้) อาจเป็น cp437 หรือ tis-620
        if not (member.flag_bits & 0x800):
            filename = member.filename.encode("cp437").decode("cp874")
    except Exception:
        filename = member.filename

    if is_ignored(filename):
        return None

    # ป้องกัน Directory Traversal
    resolved_target = (target_dir / filename).resolve()
    if not str(resolved_target).startswith(str(target_dir.resolve())):
        print(f"  [Security Skip] Unsafe path detected: {member.filename}")
        return None

    if member.is_dir():
        resolved_target.mkdir(parents=True, exist_ok=True)
        return None

    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, open(resolved_target, "wb") as target:
        shutil.copyfileobj(source, target)

    return resolved_target


def extract_zip(
    zip_path: Path | str,
    dest_dir: Path | str,
    clean_dest: bool = False,
) -> Tuple[int, List[Path]]:
    """แตกไฟล์ .zip ไปยัง dest_dir โดยคงโครงสร้างโฟลเดอร์เดิมไว้

    Parameters
    ----------
    zip_path : Path | str
        ตำแหน่งไฟล์ .zip ต้นทาง
    dest_dir : Path | str
        โฟลเดอร์ปลายทาง
    clean_dest : bool
        หาก True จะล้างไฟล์เดิมใน dest_dir ก่อนแตกไฟล์ใหม่

    Returns
    -------
    total_extracted : int
        จำนวนไฟล์ทั้งหมดที่แตกออกมา
    extracted_files : List[Path]
        รายชื่อ Path ของไฟล์ทั้งหมดที่แตกออกมา
    """
    zip_path = Path(zip_path).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ zip: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"ไฟล์ที่ระบุไม่ใช่ .zip หรือไฟล์เสียหาย: {zip_path}")

    if clean_dest and dest_dir.exists():
        print(f"[Clean] ล้างโฟลเดอร์ปลายทางเดิม: {dest_dir}")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dataset ZIP Extractor")
    print("=" * 60)
    print(f"Source ZIP  : {zip_path}")
    print(f"Destination : {dest_dir}")
    print("-" * 60)

    extracted_files: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_members = len(members)
        print(f"📦 เริ่มแตกไฟล์ (ทั้งหมด {total_members} รายการใน zip)...")

        for member in members:
            out_file = safe_extract_member(zf, member, dest_dir)
            if out_file is not None and out_file.is_file():
                extracted_files.append(out_file)

    # สรุปโครงสร้างโฟลเดอร์ที่ได้
    subdirs = [d for d in dest_dir.iterdir() if d.is_dir()]
    print("\n[Summary] สรุปโฟลเดอร์ที่แตกออกมา:")
    for d in sorted(subdirs):
        file_count = len([f for f in d.rglob("*") if f.is_file()])
        print(f"  + {d.name:<20} : {file_count:>4} ไฟล์")

    root_files = len([f for f in dest_dir.iterdir() if f.is_file()])
    if root_files > 0:
        print(f"  + (root files)         : {root_files:>4} ไฟล์")

    print("=" * 60)
    print(f"[Done] แตกไฟล์สำเร็จทั้งหมด {len(extracted_files)} ไฟล์ ไปยัง {dest_dir}")
    print("=" * 60)

    return len(extracted_files), extracted_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract a dataset .zip file to a specified directory."
    )
    parser.add_argument(
        "-z", "--zip",
        dest="zip_path",
        type=Path,
        default=None,
        help="Path to the source .zip file",
    )
    parser.add_argument(
        "-d", "--dest",
        dest="dest_dir",
        type=Path,
        default=None,
        help="Destination directory to extract files into (default: 'dataset')",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean/delete destination directory before extracting",
    )

    args = parser.parse_args()

    # หากไม่ได้ใส่ผ่าน CLI ให้ถามแบบ Interactive
    zip_path = args.zip_path
    if zip_path is None:
        try:
            user_input = input("ระบุ Path ของไฟล์ .zip ต้นทาง: ").strip().strip('"').strip("'")
            if not user_input:
                print("❌ ไม่ได้ระบุไฟล์ .zip ยกเลิกการทำงาน")
                sys.exit(1)
            zip_path = Path(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nยกเลิกการทำงาน")
            sys.exit(0)

    dest_dir = args.dest_dir
    if dest_dir is None:
        try:
            default_dest = "dataset"
            prompt = f"ระบุโฟลเดอร์ปลายทางที่ต้องการนำไฟล์ไปใส่ [default: {default_dest}]: "
            user_dest = input(prompt).strip().strip('"').strip("'")
            dest_dir = Path(user_dest) if user_dest else Path(default_dest)
        except (KeyboardInterrupt, EOFError):
            print("\nยกเลิกการทำงาน")
            sys.exit(0)

    try:
        extract_zip(
            zip_path=zip_path,
            dest_dir=dest_dir,
            clean_dest=args.clean,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
