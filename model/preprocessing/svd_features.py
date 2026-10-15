# svd_features.py — SVD Feature Extraction, Save Outputs, Cleanup
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from config import (
    SVD_DIR, MATRIX_DIR, INPUT_DIR, GRAY_DIR, CONV_DIR,
    IMAGE_SIZE, N_FEATURES, RANDOM_STATE,
    ALL_DIRS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    processed: list[tuple[str, Path]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """อ่าน Grayscale → resize → normalize → stack เป็น X [N, IMAGE_SIZE²]

    Parameters
    ----------
    processed : รายการ (original_filename, gray_path)

    Returns
    -------
    X         : float32 ndarray shape (N, IMAGE_SIZE²)
    filenames : list[str] ชื่อไฟล์ต้นฉบับ
    """
    filenames: list[str] = []
    vectors:   list[np.ndarray] = []

    for original_filename, gray_path in processed:
        gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            raise ValueError(f'ไฟล์ Grayscale อ่านไม่ได้: {gray_path}')

        resized    = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        vectors.append(normalized.reshape(-1))
        filenames.append(original_filename)

    X = np.stack(vectors, axis=0)  # (N, IMAGE_SIZE²)
    return X, filenames


def run_svd(
    X: np.ndarray,
    n_features: int = N_FEATURES,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centering → Randomized SVD → คืน (X_centered, Z, U, S, Vt)

    Returns
    -------
    mean      : float32 shape (D,)
    Vt        : float32 shape (n_features, D)  — projection matrix
    S         : float32 shape (n_features,)    — singular values
    Z         : float32 shape (N, n_features)  — projected features
    """
    if X.shape[0] < n_features + 1:
        raise ValueError(
            f'มีภาพที่อ่านได้ {X.shape[0]} ภาพ '
            f'แต่ต้องมีอย่างน้อย {n_features + 1} ภาพ '
            f'เพื่อดึง {n_features} SVD features หลัง Centering'
        )

    mean      = X.mean(axis=0, dtype=np.float32)        # (D,)
    X_centered = X - mean[None, :]

    U, S, Vt = randomized_svd(
        X_centered,
        n_components=n_features,
        n_iter=5,
        random_state=random_state,
    )

    Z = (X_centered @ Vt.T).astype(np.float32)          # (N, n_features)

    print('X shape   :', X.shape)
    print('Mean shape:', mean.shape)
    print('Vt shape  :', Vt.shape)
    print('Z shape   :', Z.shape)
    return mean, Vt, S, Z


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_features(
    Z: np.ndarray,
    filenames: list[str],
    n_features: int = N_FEATURES,
) -> pd.DataFrame:
    """บันทึก features.csv และ features.npy ลง SVD_DIR

    Returns
    -------
    df : DataFrame (filename + F001…F064)
    """
    feature_cols = [f'F{i:03d}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(Z, columns=feature_cols)
    df.insert(0, 'filename', filenames)

    df.to_csv(SVD_DIR / 'features.csv', index=False,
              float_format='%.8f', encoding='utf-8-sig')
    np.save(SVD_DIR / 'features.npy', Z)
    return df


def save_matrices(Z: np.ndarray, filenames: list[str]) -> None:
    """บันทึก feature vector ของแต่ละภาพเป็นเมทริกซ์ 8×8 (NPY + CSV)"""
    for filename, feature_vector in zip(filenames, Z):
        matrix_8x8 = feature_vector.reshape(8, 8)
        base = f'{filename}_8x8'
        np.save(MATRIX_DIR / f'{base}.npy', matrix_8x8)
        np.savetxt(MATRIX_DIR / f'{base}.csv', matrix_8x8,
                   fmt='%.8f', delimiter=',')


def save_svd_model(
    mean: np.ndarray,
    Vt: np.ndarray,
    S: np.ndarray,
    image_size: int = IMAGE_SIZE,
    n_features: int = N_FEATURES,
) -> None:
    """บันทึก svd_model.npz พร้อม preprocessing parameters

    โหลดกลับด้วย:
        model = np.load(SVD_DIR / 'svd_model.npz')
        new_z = (new_x - model['mean']) @ model['Vt'].T  # shape (64,)
    """
    np.savez_compressed(
        SVD_DIR / 'svd_model.npz',
        mean=mean,
        Vt=Vt.astype(np.float32),
        S=S.astype(np.float32),
        image_size=np.int32(image_size),
        n_features=np.int32(n_features),
        translation=np.array([50, 30], dtype=np.float32),
        scaling=np.float32(1.2),
        rotation_degrees=np.float32(25.0),
    )


def save_all_outputs(
    Z: np.ndarray,
    filenames: list[str],
    mean: np.ndarray,
    Vt: np.ndarray,
    S: np.ndarray,
) -> pd.DataFrame:
    """Convenience wrapper: บันทึก features, matrices และ SVD model ทั้งหมด"""
    df = save_features(Z, filenames)
    save_matrices(Z, filenames)
    save_svd_model(mean, Vt, S)

    print('\nบันทึกผลลัพธ์เรียบร้อยที่:', SVD_DIR)
    print('- features.csv :', df.shape)
    print('- features.npy :', Z.shape)
    print('- matrices      :', len(filenames), 'ชุด (CSV + NPY)')
    print('- svd_model.npz : mean, Vt, S และค่าตั้งต้นสำหรับ preprocessing')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Project into SVD space (inference helper)
# ─────────────────────────────────────────────────────────────────────────────

def project_new_image(
    gray_flat: np.ndarray,
    model_path: Path = SVD_DIR / 'svd_model.npz',
) -> np.ndarray:
    """ฉายภาพใหม่ไปยัง SVD feature space

    Parameters
    ----------
    gray_flat  : float32 flatten pixel array shape (IMAGE_SIZE²,) ค่า 0-1
    model_path : path ของ svd_model.npz

    Returns
    -------
    z : float32 shape (n_features,)
    """
    model = np.load(str(model_path))
    mean  = model['mean']
    Vt    = model['Vt']
    z     = (gray_flat.astype(np.float32) - mean) @ Vt.T
    return z.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def clear_pipeline_folders() -> None:
    """ลบไฟล์ทั้งหมดในโฟลเดอร์ pipeline (โฟลเดอร์ยังคงอยู่)"""
    deleted = 0
    print('กำลังเริ่มต้นล้างไฟล์ข้อมูลในแต่ละโฟลเดอร์...')
    for folder in ALL_DIRS:
        if not folder.exists():
            continue
        for item in folder.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                    deleted += 1
                except Exception as exc:
                    print(f'ไม่สามารถลบ {item.name}: {exc}')
        print(f'- เคลียร์ไฟล์ในโฟลเดอร์เรียบร้อย: {folder.name}')
    print(f'\nเสร็จสิ้น! ลบไฟล์ทั้งหมด {deleted} ไฟล์ '
          'ระบบพร้อมสำหรับการเริ่มต้นประมวลผลรูปใหม่')
