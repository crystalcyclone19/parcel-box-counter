"""
test_preprocessing.py
Unit tests สำหรับ Preprocessing Pipeline ทั้ง 4 โมดูลตามเอกสาร documents/model_training_pipeline.md
  1. data_ingestion.py        — Data Ingestion & Image Reading
  2. geometric_transform.py   — Geometric Transformation (Translation → Scaling → Rotation)
  3. edge_processing.py       — Grayscale, Downscaling, Gaussian Blur & Edge Convolution
  4. svd_features.py          — SVD Feature Extraction (Resize → Centering → SVD → Matrix 8×8)
  + image_pipeline.py         — Pipeline Orchestrator & Backward-compatible Re-exports

รันด้วย:
    pytest model/preprocessing/test_preprocessing.py -v
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Stub Google Colab ก่อน import modules (ไม่มีใน local env)
# ──────────────────────────────────────────────────────────────────────────────
_colab_stub = types.ModuleType("google.colab")
_colab_stub.files = None  # ฟังก์ชัน upload ถูก mock แยกในแต่ละ test
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.colab", _colab_stub)

# Patch config ให้ใช้ tmp dirs เพื่อไม่ให้กระทบ runtime จริง
import config as _cfg  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dirs(tmp_path, monkeypatch):
    """Override config dirs ให้ชี้ไปที่ tmp_path ทุก test"""
    dirs = {
        "INPUT_DIR":   tmp_path / "input_images",
        "GRAY_DIR":    tmp_path / "Grayscale",
        "RESIZED_DIR": tmp_path / "Resized",
        "BLUR_DIR":    tmp_path / "Blurred",
        "CONV_DIR":    tmp_path / "Convolution",
        "SVD_DIR":     tmp_path / "output_svd",
        "MATRIX_DIR":  tmp_path / "output_svd" / "matrices",
        "BASE_DIR":    tmp_path,
    }
    for attr, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(_cfg, attr, path)

    # แทนที่ ALL_DIRS ด้วย tuple ใหม่ที่ชี้ไป tmp
    monkeypatch.setattr(
        _cfg, "ALL_DIRS",
        tuple(dirs[k] for k in ("INPUT_DIR", "GRAY_DIR", "RESIZED_DIR", "BLUR_DIR", "CONV_DIR", "SVD_DIR", "MATRIX_DIR")),
    )
    # ป้องกันไม่ให้ process_images เด้ง GUI popup ค้างระหว่าง test
    monkeypatch.setattr(_cfg, "MAX_PLOTS", 0)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    # โหลดใหม่เพื่อให้โมดูลใช้ค่าจาก config ที่แพตช์แล้ว
    import data_ingestion      as di
    import geometric_transform as gt
    import edge_processing     as ep
    import image_pipeline      as ip
    import svd_features        as sf

    importlib.reload(di)
    importlib.reload(gt)
    importlib.reload(ep)
    importlib.reload(ip)
    importlib.reload(sf)

    return dirs, di, gt, ep, ip, sf


@pytest.fixture()
def sample_bgr():
    """ภาพสีจำลอง 100×100 (gradient เพื่อให้มี texture)"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(100, dtype=np.uint8), (100, 1))   # B
    img[:, :, 1] = np.tile(np.arange(100, dtype=np.uint8).reshape(-1, 1), (1, 100))  # G
    return img


@pytest.fixture()
def sample_gray():
    """ภาพ Grayscale 100×100"""
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Tests: data_ingestion.py
# ──────────────────────────────────────────────────────────────────────────────

class TestDataIngestion:

    def test_read_image_valid(self, tmp_path, sample_bgr):
        from data_ingestion import read_image
        img_path = tmp_path / "valid.jpg"
        cv2.imwrite(str(img_path), sample_bgr)
        loaded = read_image(img_path)
        assert loaded is not None
        assert loaded.shape == sample_bgr.shape
        assert loaded.dtype == np.uint8

    def test_read_image_nonexistent(self, tmp_path):
        from data_ingestion import read_image
        assert read_image(tmp_path / "nonexistent.jpg") is None

    def test_read_image_corrupt(self, tmp_path):
        from data_ingestion import read_image
        bad = tmp_path / "corrupt.png"
        bad.write_bytes(b"not a valid image format")
        assert read_image(bad) is None

    def test_collect_image_paths(self, tmp_dirs, sample_bgr):
        dirs, di, _, _, _, _ = tmp_dirs
        p1 = dirs["INPUT_DIR"] / "b_img.jpg"
        p2 = dirs["INPUT_DIR"] / "a_img.png"
        dummy_txt = dirs["INPUT_DIR"] / "notes.txt"

        cv2.imwrite(str(p1), sample_bgr)
        cv2.imwrite(str(p2), sample_bgr)
        dummy_txt.write_text("ignore this")

        found = di.collect_image_paths(dirs["INPUT_DIR"])
        assert len(found) == 2
        # ต้องเรียงตามชื่อไฟล์
        assert [f.name for f in found] == ["a_img.png", "b_img.jpg"]

    def test_load_from_local_paths_files_and_dirs(self, tmp_dirs, sample_bgr):
        dirs, di, _, _, _, _ = tmp_dirs
        src_dir = dirs["BASE_DIR"] / "source_folder"
        src_dir.mkdir(parents=True, exist_ok=True)
        img1 = src_dir / "sample1.jpg"
        cv2.imwrite(str(img1), sample_bgr)

        img2 = dirs["BASE_DIR"] / "sample2.png"
        cv2.imwrite(str(img2), sample_bgr)

        loaded = di.load_from_local_paths([str(src_dir), str(img2)], dest_dir=dirs["INPUT_DIR"])
        assert len(loaded) == 2
        dest_names = [p.name for p in loaded]
        assert "sample1.jpg" in dest_names
        assert "sample2.png" in dest_names

    def test_load_from_zip_path(self, tmp_dirs, sample_bgr):
        dirs, di, _, _, _, _ = tmp_dirs
        src_img = dirs["BASE_DIR"] / "inside_zip.jpg"
        cv2.imwrite(str(src_img), sample_bgr)

        zip_p = dirs["BASE_DIR"] / "archive.zip"
        with zipfile.ZipFile(zip_p, "w") as zf:
            zf.write(src_img, arcname="folder/inside_zip.jpg")

        loaded = di.load_from_zip_path(str(zip_p), dest_dir=dirs["INPUT_DIR"])
        assert len(loaded) == 1
        assert loaded[0].name == "inside_zip.jpg"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tests: geometric_transform.py
# ──────────────────────────────────────────────────────────────────────────────

class TestGeometricTransform:

    def test_translate_image(self, sample_bgr):
        from geometric_transform import translate_image
        moved = translate_image(sample_bgr, shift_x=10, shift_y=5)
        assert moved.shape == sample_bgr.shape
        assert moved.dtype == np.uint8
        assert not np.array_equal(moved, sample_bgr)

    def test_scale_image(self, sample_bgr):
        from geometric_transform import scale_image
        scaled = scale_image(sample_bgr, scale_factor=1.1)
        assert scaled.shape == sample_bgr.shape
        assert scaled.dtype == np.uint8

    def test_rotate_image(self, sample_bgr):
        from geometric_transform import rotate_image
        rotated = rotate_image(sample_bgr, angle_deg=15.0)
        assert rotated.shape == sample_bgr.shape
        assert rotated.dtype == np.uint8

    def test_transform_image_shape_and_dtype(self, sample_bgr):
        from geometric_transform import transform_image
        result = transform_image(sample_bgr)
        assert result.shape == sample_bgr.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, sample_bgr)

    def test_single_color_image(self):
        from geometric_transform import transform_image
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        out = transform_image(white)
        assert out.shape == (64, 64, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Tests: edge_processing.py
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeProcessing:

    def test_to_grayscale(self, sample_bgr, sample_gray):
        from edge_processing import to_grayscale
        # แปลงจาก 3-channel
        gray = to_grayscale(sample_bgr)
        assert gray.ndim == 2
        assert gray.shape == sample_bgr.shape[:2]
        assert gray.dtype == np.uint8

        # ภาพที่เป็น grayscale อยู่แล้วต้องไม่เปลี่ยนมิติ
        gray2 = to_grayscale(sample_gray)
        assert gray2.shape == sample_gray.shape

    def test_resize_image_fixed(self, sample_gray):
        from edge_processing import resize_image
        resized = resize_image(sample_gray, target_size=(50, 60))
        assert resized.shape == (60, 50)  # (height, width)
        assert resized.dtype == np.uint8

    def test_resize_image_aspect_ratio(self):
        from edge_processing import resize_image
        img = np.zeros((200, 100), dtype=np.uint8)
        resized = resize_image(img, target_size=(100, 100), keep_aspect_ratio=True)
        assert resized.shape == (100, 50)

    def test_apply_gaussian_blur(self, sample_gray):
        from edge_processing import apply_gaussian_blur
        blurred = apply_gaussian_blur(sample_gray, kernel_size=5)
        assert blurred.shape == sample_gray.shape
        assert blurred.dtype == np.uint8

    def test_apply_gaussian_blur_reduces_noise(self):
        from edge_processing import apply_gaussian_blur
        np.random.seed(42)
        noisy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        blurred = apply_gaussian_blur(noisy, kernel_size=7)
        assert blurred.std() < noisy.std()

    def test_create_edge_map(self, sample_gray):
        from edge_processing import create_edge_map
        result = create_edge_map(sample_gray)
        assert result.shape == sample_gray.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_create_edge_map_uniform_is_zero(self):
        from edge_processing import create_edge_map
        uniform = np.full((50, 50), 128, dtype=np.uint8)
        result = create_edge_map(uniform)
        assert result.max() == 0

    def test_process_image_edges(self, sample_bgr):
        from edge_processing import process_image_edges
        gray, resized, blurred, edge_map = process_image_edges(
            sample_bgr,
            resize_target=(64, 64),
            blur_ksize=5,
            clip_percentile=98.0,
        )
        assert gray.ndim == 2
        assert resized.shape == (64, 64)
        assert blurred.shape == (64, 64)
        assert edge_map.shape == (64, 64)
        assert edge_map.dtype == np.uint8


# ──────────────────────────────────────────────────────────────────────────────
# 4. Tests: image_pipeline.py (Orchestrator & Re-exports)
# ──────────────────────────────────────────────────────────────────────────────

class TestImagePipeline:

    def _write_sample_image(self, path: Path, size: int = 80) -> None:
        img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def test_reexports_availability(self):
        """ตรวจสอบว่า image_pipeline ยังคง re-export ฟังก์ชันหลักตามสัญญา API"""
        import image_pipeline as ip
        assert callable(ip.read_image)
        assert callable(ip.collect_image_paths)
        assert callable(ip.load_from_local_paths)
        assert callable(ip.load_from_zip_path)
        assert callable(ip.transform_image)
        assert callable(ip.translate_image)
        assert callable(ip.scale_image)
        assert callable(ip.rotate_image)
        assert callable(ip.to_grayscale)
        assert callable(ip.resize_image)
        assert callable(ip.apply_gaussian_blur)
        assert callable(ip.create_edge_map)
        assert callable(ip.process_image_edges)
        assert callable(ip.process_images)

    def test_returns_list_of_tuples(self, tmp_dirs):
        dirs, _, _, _, ip, _ = tmp_dirs
        p = dirs["INPUT_DIR"] / "sample.jpg"
        self._write_sample_image(p)
        result = ip.process_images([p])
        assert isinstance(result, list)
        assert len(result) == 1
        filename, gray_path = result[0]
        assert filename == "sample.jpg"
        assert gray_path.exists()

    def test_all_output_files_created(self, tmp_dirs):
        dirs, _, _, _, ip, _ = tmp_dirs
        p = dirs["INPUT_DIR"] / "img_full.jpg"
        self._write_sample_image(p)
        ip.process_images([p], save_intermediate=True)

        gray_files = list(dirs["GRAY_DIR"].iterdir())
        conv_files = list(dirs["CONV_DIR"].iterdir())
        resized_files = list(dirs["RESIZED_DIR"].iterdir())
        blur_files = list(dirs["BLUR_DIR"].iterdir())

        assert len(gray_files) == 1
        assert len(conv_files) == 1
        assert len(resized_files) == 1
        assert len(blur_files) == 1
        assert resized_files[0].name == "resized_img_full.jpg"
        assert blur_files[0].name == "blur_img_full.jpg"

    def test_skip_bad_image(self, tmp_dirs, capsys):
        dirs, _, _, _, ip, _ = tmp_dirs
        bad = dirs["INPUT_DIR"] / "not_an_image.jpg"
        bad.write_bytes(b"not image data")
        result = ip.process_images([bad])
        assert result == []
        captured = capsys.readouterr()
        assert "SKIP" in captured.out

    def test_empty_list(self, tmp_dirs):
        _, _, _, _, ip, _ = tmp_dirs
        result = ip.process_images([])
        assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# 5. Tests: svd_features.py
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def _write_gray_image(self, path: Path, size: int = 100) -> None:
        gray = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        cv2.imwrite(str(path), gray)

    def test_output_shape(self, tmp_dirs):
        dirs, _, _, _, _, sf = tmp_dirs
        n = 5
        processed = []
        for i in range(n):
            p = dirs["GRAY_DIR"] / f"gray_img{i}.jpg"
            self._write_gray_image(p)
            processed.append((f"img{i}.jpg", p))

        X, filenames = sf.build_feature_matrix(processed)
        expected_pixels = _cfg.IMAGE_SIZE ** 2
        assert X.shape == (n, expected_pixels), f"Expected ({n}, {expected_pixels}), got {X.shape}"
        assert len(filenames) == n

    def test_values_normalized_0_to_1(self, tmp_dirs):
        dirs, _, _, _, _, sf = tmp_dirs
        p = dirs["GRAY_DIR"] / "gray_norm.jpg"
        self._write_gray_image(p)
        X, _ = sf.build_feature_matrix([("norm.jpg", p)])
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_dtype_float32(self, tmp_dirs):
        dirs, _, _, _, _, sf = tmp_dirs
        p = dirs["GRAY_DIR"] / "gray_dtype.jpg"
        self._write_gray_image(p)
        X, _ = sf.build_feature_matrix([("dtype.jpg", p)])
        assert X.dtype == np.float32

    def test_raises_on_bad_gray_file(self, tmp_dirs):
        dirs, _, _, _, _, sf = tmp_dirs
        bad = dirs["GRAY_DIR"] / "bad.jpg"
        bad.write_bytes(b"garbage")
        with pytest.raises(ValueError, match="อ่านไม่ได้"):
            sf.build_feature_matrix([("bad.jpg", bad)])


class TestRunSVD:

    def _make_X(self, n=70, d=128**2):
        rng = np.random.default_rng(0)
        return rng.random((n, d)).astype(np.float32)

    def test_output_shapes(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        X = self._make_X(70)
        mean, Vt, S, Z = sf.run_svd(X, n_features=_cfg.N_FEATURES)
        assert mean.shape == (X.shape[1],)
        assert Vt.shape   == (_cfg.N_FEATURES, X.shape[1])
        assert S.shape    == (_cfg.N_FEATURES,)
        assert Z.shape    == (X.shape[0], _cfg.N_FEATURES)

    def test_z_dtype_float32(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        X = self._make_X(70)
        _, _, _, Z = sf.run_svd(X)
        assert Z.dtype == np.float32

    def test_raises_when_too_few_images(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        X = self._make_X(n=10)
        with pytest.raises(ValueError, match="อย่างน้อย"):
            sf.run_svd(X)

    def test_projection_consistency(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        X = self._make_X(70)
        mean, Vt, _, Z = sf.run_svd(X)
        X_centered = X - mean[None, :]
        Z_expected = (X_centered @ Vt.T).astype(np.float32)
        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)


class TestSaveFeatures:

    def _make_data(self, n=10):
        rng  = np.random.default_rng(1)
        Z    = rng.random((n, _cfg.N_FEATURES)).astype(np.float32)
        fnames = [f"img_{i}.jpg" for i in range(n)]
        return Z, fnames

    def test_csv_created(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        Z, fnames = self._make_data()
        sf.save_features(Z, fnames)
        assert (_cfg.SVD_DIR / "features.csv").exists()

    def test_npy_created(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        Z, fnames = self._make_data()
        sf.save_features(Z, fnames)
        assert (_cfg.SVD_DIR / "features.npy").exists()

    def test_csv_row_count(self, tmp_dirs):
        import pandas as pd
        _, _, _, _, _, sf = tmp_dirs
        n = 8
        Z, fnames = self._make_data(n)
        sf.save_features(Z, fnames)
        df = pd.read_csv(_cfg.SVD_DIR / "features.csv")
        assert len(df) == n

    def test_npy_roundtrip(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        Z, fnames = self._make_data()
        sf.save_features(Z, fnames)
        Z_loaded = np.load(_cfg.SVD_DIR / "features.npy")
        np.testing.assert_array_almost_equal(Z, Z_loaded, decimal=6)


class TestSaveMatrices:

    def test_files_created(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        n = 4
        rng = np.random.default_rng(2)
        Z   = rng.random((n, 64)).astype(np.float32)
        fnames = [f"p{i}.jpg" for i in range(n)]
        sf.save_matrices(Z, fnames)
        files = list(_cfg.MATRIX_DIR.iterdir())
        assert len(files) == n * 2

    def test_matrix_shape_8x8(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        Z = np.arange(64, dtype=np.float32).reshape(1, 64)
        sf.save_matrices(Z, ["test.jpg"])
        npy_files = list(_cfg.MATRIX_DIR.glob("*.npy"))
        mat = np.load(str(npy_files[0]))
        assert mat.shape == (8, 8)


class TestSaveSVDModel:

    def test_npz_created(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        rng  = np.random.default_rng(3)
        mean = rng.random(128**2).astype(np.float32)
        Vt   = rng.random((_cfg.N_FEATURES, 128**2)).astype(np.float32)
        S    = rng.random(_cfg.N_FEATURES).astype(np.float32)
        sf.save_svd_model(mean, Vt, S)
        assert (_cfg.SVD_DIR / "svd_model.npz").exists()

    def test_npz_keys(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        rng  = np.random.default_rng(4)
        mean = rng.random(128**2).astype(np.float32)
        Vt   = rng.random((_cfg.N_FEATURES, 128**2)).astype(np.float32)
        S    = rng.random(_cfg.N_FEATURES).astype(np.float32)
        sf.save_svd_model(mean, Vt, S)
        model = np.load(_cfg.SVD_DIR / "svd_model.npz")
        for key in ("mean", "Vt", "S", "image_size", "n_features",
                    "translation", "scaling", "rotation_degrees"):
            assert key in model, f"Missing key: {key}"


class TestProjectNewImage:

    def _save_dummy_model(self, sf, tmp_dirs, n: int = 70):
        rng = np.random.default_rng(5)
        d   = _cfg.IMAGE_SIZE ** 2
        X   = rng.random((n, d)).astype(np.float32)
        mean, Vt, S, _ = sf.run_svd(X)
        sf.save_svd_model(mean, Vt, S)
        return mean, Vt

    def test_output_shape(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        mean, Vt = self._save_dummy_model(sf, tmp_dirs)
        rng      = np.random.default_rng(6)
        gray_flat = rng.random(_cfg.IMAGE_SIZE ** 2).astype(np.float32)
        z = sf.project_new_image(gray_flat, _cfg.SVD_DIR / "svd_model.npz")
        assert z.shape == (_cfg.N_FEATURES,)

    def test_output_dtype_float32(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        self._save_dummy_model(sf, tmp_dirs)
        rng       = np.random.default_rng(7)
        gray_flat = rng.random(_cfg.IMAGE_SIZE ** 2).astype(np.float32)
        z = sf.project_new_image(gray_flat, _cfg.SVD_DIR / "svd_model.npz")
        assert z.dtype == np.float32

    def test_consistent_with_manual_projection(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        mean, Vt = self._save_dummy_model(sf, tmp_dirs)
        rng       = np.random.default_rng(8)
        gray_flat = rng.random(_cfg.IMAGE_SIZE ** 2).astype(np.float32)
        z_func    = sf.project_new_image(gray_flat, _cfg.SVD_DIR / "svd_model.npz")
        z_manual  = ((gray_flat - mean) @ Vt.T).astype(np.float32)
        np.testing.assert_allclose(z_func, z_manual, rtol=1e-5, atol=1e-5)


class TestClearPipelineFolders:

    def test_files_deleted(self, tmp_dirs):
        dirs, _, _, _, _, sf = tmp_dirs
        for folder in _cfg.ALL_DIRS:
            (folder / "dummy.txt").write_text("x")
        sf.clear_pipeline_folders()
        remaining = [f for d in _cfg.ALL_DIRS for f in d.iterdir() if f.is_file()]
        assert remaining == [], f"ยังมีไฟล์เหลืออยู่: {remaining}"

    def test_folders_still_exist(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        sf.clear_pipeline_folders()
        for folder in _cfg.ALL_DIRS:
            assert folder.exists(), f"โฟลเดอร์หายไป: {folder}"

    def test_empty_folders_no_error(self, tmp_dirs):
        _, _, _, _, _, sf = tmp_dirs
        sf.clear_pipeline_folders()
        sf.clear_pipeline_folders()
