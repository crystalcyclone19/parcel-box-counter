# data_ingestion.py — Data Ingestion & Image Reading
"""
ส่วนที่ 1: การนำเข้าข้อมูลและการอ่านรูปภาพ (Data Ingestion & Geometric Pre-transformation)
ตามข้อกำหนดใน documents/model_training_pipeline.md (ขั้นตอนที่ 2)

หมายเหตุ:
  ขั้นตอน Geometric Transformation (Translation, Scaling, Rotation, Cropping) 
  ถูกย้ายมาดำเนินการตั้งแต่ขั้นตอนการเตรียมและนำเข้าข้อมูล โดยผู้นำเข้าข้อมูล
  (Data Ingestor / Ingestion Specialist) จะทำการจัดตำแหน่งและ crop ภาพมาให้เรียบร้อย
  ก่อนนำเข้าสู่ระบบ เพื่อให้แต่ละภาพมีความเหมาะสมกับสภาพแวดล้อมเฉพาะตัวของภาพนั้นๆ
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# เพิ่ม preprocessing/ เข้า sys.path อัตโนมัติ
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import INPUT_DIR, SUPPORTED_EXTENSIONS


def read_image(path: Path | str) -> Optional[np.ndarray]:
    """อ่านไฟล์รูปภาพด้วย OpenCV ในระบบสี BGR พร้อมตรวจสอบความถูกต้อง

    Parameters
    ----------
    path : Path หรือ str ของไฟล์ภาพ

    Returns
    -------
    bgr : np.ndarray (H, W, 3) หรือ None หากอ่านไม่ได้/ภาพว่าง
    """
    p = Path(path).resolve()
    if not p.is_file():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return None
    return bgr


def collect_image_paths(input_dir: Path = INPUT_DIR) -> List[Path]:
    """คืนรายการ Path รูปภาพทั้งหมดที่รองรับใน input_dir (เรียงตามชื่อไฟล์)"""
    if not input_dir.exists():
        return []
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"พบภาพทั้งหมดใน {input_dir.name}: {len(paths)} รูป")
    return paths


# Alias สำหรับความเข้ากันได้กับโค้ดเดิม
_collect_image_paths = collect_image_paths


def load_from_local_paths(paths: List[str], dest_dir: Path = INPUT_DIR) -> List[Path]:
    """รับ path แบบ local (โฟลเดอร์, .zip, หรือไฟล์รูปภาพ) → คัดลอกลง dest_dir

    Parameters
    ----------
    paths    : รายการ path ที่ผู้ใช้ระบุ
    dest_dir : โฟลเดอร์ปลายทางสำหรับจัดเก็บภาพนำเข้า

    Returns
    -------
    รายการ Path รูปภาพทั้งหมดใน dest_dir
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ขั้นตอนที่ 1 : รับ Input รูปภาพ (Data Ingestion)")
    print("=" * 60)

    copied = 0
    for raw in paths:
        p = Path(raw).resolve()
        if not p.exists():
            print(f"  [WARN] ไม่พบ path: {p}")
            continue

        # กรณี 1: เป็นโฟลเดอร์
        if p.is_dir():
            print(f"  📁 สแกนโฟลเดอร์: {p}")
            for img_path in p.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    dest = dest_dir / img_path.name
                    shutil.copy2(img_path, dest)
                    copied += 1
                    print(f"     คัดลอก: {img_path.name}")

        # กรณี 2: เป็นไฟล์ .zip
        elif p.suffix.lower() == ".zip":
            print(f"  📦 แตกไฟล์ zip: {p.name}")
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(tmp)
                for img_path in Path(tmp).rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        dest = dest_dir / img_path.name
                        shutil.copy2(img_path, dest)
                        copied += 1
                        print(f"     คัดลอก: {img_path.name}")

        # กรณี 3: เป็นไฟล์รูปภาพเดี่ยว
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            dest = dest_dir / p.name
            shutil.copy2(p, dest)
            copied += 1
            print(f"  🖼  คัดลอก: {p.name}")

        else:
            print(f"  [SKIP] ไม่รองรับประเภทไฟล์: {p.name}")

    print(f"\n  → นำเข้ารูปภาพสำเร็จ {copied} ไฟล์ → {dest_dir}\n")

    image_paths = collect_image_paths(dest_dir)
    if not image_paths:
        print(f"[ERROR] ไม่พบรูปภาพใน {dest_dir} กรุณาตรวจสอบ path ที่ระบุ")
        sys.exit(1)
    return image_paths


def load_from_zip_path(zip_path: str, dest_dir: Path = INPUT_DIR) -> List[Path]:
    """แตกไฟล์ .zip จาก path ที่ระบุล่วงหน้า → คัดลอกรูปภาพลง dest_dir"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ขั้นตอนที่ 1 : รับ Input จาก ZIP (Data Ingestion)")
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
                dest = dest_dir / img_path.name
                shutil.copy2(img_path, dest)
                copied += 1
                print(f"     คัดลอก: {img_path.name}")

    print(f"\n  → นำเข้ารูปภาพสำเร็จ {copied} ไฟล์ → {dest_dir}\n")

    image_paths = collect_image_paths(dest_dir)
    if not image_paths:
        print("[ERROR] ไม่พบรูปภาพใน zip กรุณาตรวจสอบไฟล์")
        sys.exit(1)
    return image_paths


# ─── Google Colab Specific Uploaders ──────────────────────────────────────────

def upload_zip(dest_dir: Path = INPUT_DIR) -> List[Path]:
    """Google Colab: อัปโหลด .zip ผ่านอินเทอร์เฟซของ Colab → แตกไฟล์ลง dest_dir"""
    from google.colab import files  # type: ignore[import]

    dest_dir.mkdir(parents=True, exist_ok=True)
    print("กรุณาอัปโหลดไฟล์ .zip ที่รวบรวมไฟล์ภาพของคุณ")
    uploaded_zip = files.upload()

    for zip_name, zip_bytes in uploaded_zip.items():
        if Path(zip_name).suffix.lower() != ".zip":
            print(f"ข้ามไฟล์เนื่องจากไม่ใช่ .zip: {zip_name}")
            continue

        print(f"กำลังดำเนินการกับไฟล์: {zip_name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = Path(temp_dir) / zip_name
            temp_zip_path.write_bytes(zip_bytes)

            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

                count = 0
                for file_path in Path(temp_dir).rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        destination = dest_dir / file_path.name
                        shutil.copy2(str(file_path), str(destination))
                        count += 1

            print(f"แตกไฟล์สำเร็จ: นำเข้า {count} รูปภาพไปยัง {dest_dir.name}")

    return collect_image_paths(dest_dir)


def upload_images(dest_dir: Path = INPUT_DIR) -> List[Path]:
    """Google Colab: อัปโหลดรูปภาพหลายไฟล์โดยตรงผ่านอินเทอร์เฟซของ Colab"""
    from google.colab import files  # type: ignore[import]

    dest_dir.mkdir(parents=True, exist_ok=True)
    print("อัปโหลดไฟล์ JPG/JPEG/PNG ได้หลายไฟล์พร้อมกัน")
    uploaded = files.upload()

    for uploaded_name, file_bytes in uploaded.items():
        safe_name = Path(uploaded_name).name
        if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"ข้ามไฟล์ที่ไม่รองรับ: {safe_name}")
            continue

        destination = dest_dir / safe_name
        source = Path.cwd() / safe_name
        if source.is_file() and source.resolve() != destination.resolve():
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
        else:
            destination.write_bytes(file_bytes)
        print("บันทึก:", destination.name)

    image_paths = collect_image_paths(dest_dir)
    if not image_paths:
        raise ValueError("ไม่พบรูปภาพใน input_images กรุณาอัปโหลดไฟล์ JPG/PNG")
    return image_paths
