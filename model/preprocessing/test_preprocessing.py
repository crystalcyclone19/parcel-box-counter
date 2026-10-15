"""
test_preprocessing.py
Unit tests สำหรับ image_pipeline และ svd_features
(ไม่เกี่ยวข้องกับ mlp_regression)

รันด้วย:
    pytest test_preprocessing.py -v
"""
import sys
import types
import importlib
from pathlib import Path

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
        "INPUT_DIR":  tmp_path / "input_images",
        "GRAY_DIR":   tmp_path / "Grayscale",
        "CONV_DIR":   tmp_path / "Convolution",
        "SVD_DIR":    tmp_path / "output_svd",
        "MATRIX_DIR": tmp_path / "output_svd" / "matrices",
        "BASE_DIR":   tmp_path,
    }
    for attr, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(_cfg, attr, path)

    # แทนที่ ALL_DIRS ด้วย tuple ใหม่ที่ชี้ไป tmp
    monkeypatch.setattr(
        _cfg, "ALL_DIRS",
        tuple(dirs[k] for k in ("INPUT_DIR", "GRAY_DIR", "CONV_DIR", "SVD_DIR", "MATRIX_DIR")),
    )

    # โหลดใหม่เพื่อให้ module ใช้ค่าจาก config ที่แพตช์แล้ว
    import image_pipeline as ip
    import svd_features   as sf
    importlib.reload(ip)
    importlib.reload(sf)
    return dirs, ip, sf


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
# Tests: image_pipeline — transform_image
# ──────────────────────────────────────────────────────────────────────────────

class TestTransformImage:

    def test_output_shape_unchanged(self, sample_bgr):
        """ขนาดภาพต้องไม่เปลี่ยนหลัง transform"""
        from image_pipeline import transform_image
        result = transform_image(sample_bgr)
        assert result.shape == sample_bgr.shape, (
            f"Expected {sample_bgr.shape}, got {result.shape}"
        )

    def test_output_dtype_uint8(self, sample_bgr):
        """dtype ต้องเป็น uint8 (เหมือนต้นฉบับ)"""
        from image_pipeline import transform_image
        result = transform_image(sample_bgr)
        assert result.dtype == np.uint8

    def test_transform_changes_pixels(self, sample_bgr):
        """ภาพหลัง transform ต้องต่างจากต้นฉบับ (ไม่ใช่ identity)"""
        from image_pipeline import transform_image
        result = transform_image(sample_bgr)
        assert not np.array_equal(result, sample_bgr)

    def test_single_color_image(self):
        """ภาพสีเดียวล้วน (ขาว) ต้องไม่ crash"""
        from image_pipeline import transform_image
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        out = transform_image(white)
        assert out.shape == (64, 64, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: image_pipeline — create_edge_map
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateEdgeMap:

    def test_output_shape(self, sample_gray):
        from image_pipeline import create_edge_map
        result = create_edge_map(sample_gray)
        assert result.shape == sample_gray.shape

    def test_output_dtype_uint8(self, sample_gray):
        from image_pipeline import create_edge_map
        result = create_edge_map(sample_gray)
        assert result.dtype == np.uint8

    def test_value_range(self, sample_gray):
        """ทุก pixel ต้องอยู่ใน [0, 255]"""
        from image_pipeline import create_edge_map
        result = create_edge_map(sample_gray)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_image_is_zero(self):
        """ภาพสีเดียวกันหมด → edge = 0 ทุก pixel"""
        from image_pipeline import create_edge_map
        uniform = np.full((50, 50), 128, dtype=np.uint8)
        result = create_edge_map(uniform)
        assert result.max() == 0, "ภาพสีเดียวกันไม่ควรมี edge"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: image_pipeline — process_images
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessImages:

    def _write_sample_image(self, path: Path, size: int = 80) -> None:
        img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def test_returns_list_of_tuples(self, tmp_dirs):
        dirs, ip, _ = tmp_dirs
        p = dirs["INPUT_DIR"] / "sample.jpg"
        self._write_sample_image(p)
        result = ip.process_images([p])
        assert isinstance(result, list)
        assert len(result) == 1
        filename, gray_path = result[0]
        assert filename == "sample.jpg"
        assert gray_path.exists()

    def test_gray_file_created(self, tmp_dirs):
        dirs, ip, _ = tmp_dirs
        p = dirs["INPUT_DIR"] / "img.png"
        self._write_sample_image(p)
        result = ip.process_images([p])
        assert len(result) == 1
        _, gray_path = result[0]
        assert gray_path.is_file()

    def test_conv_file_created(self, tmp_dirs):
        dirs, ip, _ = tmp_dirs
        p = dirs["INPUT_DIR"] / "img2.jpg"
        self._write_sample_image(p)
        ip.process_images([p])
        conv_files = list(dirs["CONV_DIR"].iterdir())
        assert len(conv_files) == 1

    def test_skip_bad_image(self, tmp_dirs, capsys):
        """ไฟล์ที่ไม่ใช่ภาพต้องถูก skip ไม่ crash"""
        dirs, ip, _ = tmp_dirs
        bad = dirs["INPUT_DIR"] / "not_an_image.jpg"
        bad.write_bytes(b"not image data")
        result = ip.process_images([bad])
        assert result == []
        captured = capsys.readouterr()
        assert "SKIP" in captured.out

    def test_empty_list(self, tmp_dirs):
        _, ip, _ = tmp_dirs
        result = ip.process_images([])
        assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# Tests: svd_features — build_feature_matrix
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def _write_gray_image(self, path: Path, size: int = 100) -> None:
        gray = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        cv2.imwrite(str(path), gray)

    def test_output_shape(self, tmp_dirs):
        dirs, _, sf = tmp_dirs
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
        """ค่า pixel ต้องอยู่ใน [0, 1] หลัง /255"""
        dirs, _, sf = tmp_dirs
        p = dirs["GRAY_DIR"] / "gray_norm.jpg"
        self._write_gray_image(p)
        X, _ = sf.build_feature_matrix([("norm.jpg", p)])
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_dtype_float32(self, tmp_dirs):
        dirs, _, sf = tmp_dirs
        p = dirs["GRAY_DIR"] / "gray_dtype.jpg"
        self._write_gray_image(p)
        X, _ = sf.build_feature_matrix([("dtype.jpg", p)])
        assert X.dtype == np.float32

    def test_raises_on_bad_gray_file(self, tmp_dirs):
        dirs, _, sf = tmp_dirs
        bad = dirs["GRAY_DIR"] / "bad.jpg"
        bad.write_bytes(b"garbage")
        with pytest.raises(ValueError, match="อ่านไม่ได้"):
            sf.build_feature_matrix([("bad.jpg", bad)])


# ──────────────────────────────────────────────────────────────────────────────
# Tests: svd_features — run_svd
# ──────────────────────────────────────────────────────────────────────────────

class TestRunSVD:

    def _make_X(self, n=70, d=128**2):
        rng = np.random.default_rng(0)
        return rng.random((n, d)).astype(np.float32)

    def test_output_shapes(self, tmp_dirs):
        _, _, sf = tmp_dirs
        X = self._make_X(70)
        mean, Vt, S, Z = sf.run_svd(X, n_features=_cfg.N_FEATURES)
        assert mean.shape == (X.shape[1],)
        assert Vt.shape   == (_cfg.N_FEATURES, X.shape[1])
        assert S.shape    == (_cfg.N_FEATURES,)
        assert Z.shape    == (X.shape[0], _cfg.N_FEATURES)

    def test_z_dtype_float32(self, tmp_dirs):
        _, _, sf = tmp_dirs
        X = self._make_X(70)
        _, _, _, Z = sf.run_svd(X)
        assert Z.dtype == np.float32

    def test_raises_when_too_few_images(self, tmp_dirs):
        _, _, sf = tmp_dirs
        X = self._make_X(n=10)  # น้อยกว่า N_FEATURES + 1
        with pytest.raises(ValueError, match="อย่างน้อย"):
            sf.run_svd(X)

    def test_projection_consistency(self, tmp_dirs):
        """Z ต้องเท่ากับ X_centered @ Vt.T"""
        _, _, sf = tmp_dirs
        X = self._make_X(70)
        mean, Vt, _, Z = sf.run_svd(X)
        X_centered = X - mean[None, :]
        Z_expected = (X_centered @ Vt.T).astype(np.float32)
        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: svd_features — save helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestSaveFeatures:

    def _make_data(self, n=10):
        rng  = np.random.default_rng(1)
        Z    = rng.random((n, _cfg.N_FEATURES)).astype(np.float32)
        fnames = [f"img_{i}.jpg" for i in range(n)]
        return Z, fnames

    def test_csv_created(self, tmp_dirs):
        _, _, sf = tmp_dirs
        Z, fnames = self._make_data()
        sf.save_features(Z, fnames)
        assert (_cfg.SVD_DIR / "features.csv").exists()

    def test_npy_created(self, tmp_dirs):
        _, _, sf = tmp_dirs
        Z, fnames = self._make_data()
        sf.save_features(Z, fnames)
        assert (_cfg.SVD_DIR / "features.npy").exists()

    def test_csv_row_count(self, tmp_dirs):
        import pandas as pd
        _, _, sf = tmp_dirs
        n = 8
        Z, fnames = self._make_data(n)
        sf.save_features(Z, fnames)
        df = pd.read_csv(_cfg.SVD_DIR / "features.csv")
        assert len(df) == n

    def test_npy_roundtrip(self, tmp_dirs):
        _, _, sf = tmp_dirs
        Z, fnames = self._make_data()
        sf.save_features(Z, fnames)
        Z_loaded = np.load(_cfg.SVD_DIR / "features.npy")
        np.testing.assert_array_almost_equal(Z, Z_loaded, decimal=6)


class TestSaveMatrices:

    def test_files_created(self, tmp_dirs):
        _, _, sf = tmp_dirs
        n = 4
        rng = np.random.default_rng(2)
        Z   = rng.random((n, 64)).astype(np.float32)
        fnames = [f"p{i}.jpg" for i in range(n)]
        sf.save_matrices(Z, fnames)
        # แต่ละภาพต้องมี 2 ไฟล์ (npy + csv)
        files = list(_cfg.MATRIX_DIR.iterdir())
        assert len(files) == n * 2

    def test_matrix_shape_8x8(self, tmp_dirs):
        _, _, sf = tmp_dirs
        Z = np.arange(64, dtype=np.float32).reshape(1, 64)
        sf.save_matrices(Z, ["test.jpg"])
        npy_files = list(_cfg.MATRIX_DIR.glob("*.npy"))
        mat = np.load(str(npy_files[0]))
        assert mat.shape == (8, 8)


class TestSaveSVDModel:

    def test_npz_created(self, tmp_dirs):
        _, _, sf = tmp_dirs
        rng  = np.random.default_rng(3)
        mean = rng.random(128**2).astype(np.float32)
        Vt   = rng.random((_cfg.N_FEATURES, 128**2)).astype(np.float32)
        S    = rng.random(_cfg.N_FEATURES).astype(np.float32)
        sf.save_svd_model(mean, Vt, S)
        assert (_cfg.SVD_DIR / "svd_model.npz").exists()

    def test_npz_keys(self, tmp_dirs):
        _, _, sf = tmp_dirs
        rng  = np.random.default_rng(4)
        mean = rng.random(128**2).astype(np.float32)
        Vt   = rng.random((_cfg.N_FEATURES, 128**2)).astype(np.float32)
        S    = rng.random(_cfg.N_FEATURES).astype(np.float32)
        sf.save_svd_model(mean, Vt, S)
        model = np.load(_cfg.SVD_DIR / "svd_model.npz")
        for key in ("mean", "Vt", "S", "image_size", "n_features",
                    "translation", "scaling", "rotation_degrees"):
            assert key in model, f"Missing key: {key}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: svd_features — project_new_image
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectNewImage:

    def _save_dummy_model(self, sf, tmp_dirs, n: int = 70):
        rng = np.random.default_rng(5)
        d   = _cfg.IMAGE_SIZE ** 2
        X   = rng.random((n, d)).astype(np.float32)
        mean, Vt, S, _ = sf.run_svd(X)
        sf.save_svd_model(mean, Vt, S)
        return mean, Vt

    def test_output_shape(self, tmp_dirs):
        _, _, sf = tmp_dirs
        mean, Vt = self._save_dummy_model(sf, tmp_dirs)
        rng      = np.random.default_rng(6)
        gray_flat = rng.random(_cfg.IMAGE_SIZE ** 2).astype(np.float32)
        z = sf.project_new_image(gray_flat, _cfg.SVD_DIR / "svd_model.npz")
        assert z.shape == (_cfg.N_FEATURES,)

    def test_output_dtype_float32(self, tmp_dirs):
        _, _, sf = tmp_dirs
        self._save_dummy_model(sf, tmp_dirs)
        rng       = np.random.default_rng(7)
        gray_flat = rng.random(_cfg.IMAGE_SIZE ** 2).astype(np.float32)
        z = sf.project_new_image(gray_flat, _cfg.SVD_DIR / "svd_model.npz")
        assert z.dtype == np.float32

    def test_consistent_with_manual_projection(self, tmp_dirs):
        """project_new_image ต้องให้ค่าเดียวกับการคำนวณด้วยมือ"""
        _, _, sf = tmp_dirs
        mean, Vt = self._save_dummy_model(sf, tmp_dirs)
        rng       = np.random.default_rng(8)
        gray_flat = rng.random(_cfg.IMAGE_SIZE ** 2).astype(np.float32)
        z_func    = sf.project_new_image(gray_flat, _cfg.SVD_DIR / "svd_model.npz")
        z_manual  = ((gray_flat - mean) @ Vt.T).astype(np.float32)
        np.testing.assert_allclose(z_func, z_manual, rtol=1e-5, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: svd_features — clear_pipeline_folders
# ──────────────────────────────────────────────────────────────────────────────

class TestClearPipelineFolders:

    def test_files_deleted(self, tmp_dirs):
        dirs, _, sf = tmp_dirs
        # สร้างไฟล์จำลองในทุกโฟลเดอร์
        for folder in _cfg.ALL_DIRS:
            (folder / "dummy.txt").write_text("x")
        sf.clear_pipeline_folders()
        # ต้องไม่มีไฟล์เหลืออยู่
        remaining = [f for d in _cfg.ALL_DIRS for f in d.iterdir() if f.is_file()]
        assert remaining == [], f"ยังมีไฟล์เหลืออยู่: {remaining}"

    def test_folders_still_exist(self, tmp_dirs):
        _, _, sf = tmp_dirs
        sf.clear_pipeline_folders()
        for folder in _cfg.ALL_DIRS:
            assert folder.exists(), f"โฟลเดอร์หายไป: {folder}"

    def test_empty_folders_no_error(self, tmp_dirs):
        """โฟลเดอร์ว่างต้องไม่ทำให้ error"""
        _, _, sf = tmp_dirs
        sf.clear_pipeline_folders()  # ครั้งแรก (อาจไม่มีไฟล์)
        sf.clear_pipeline_folders()  # ครั้งสอง (ว่างแน่นอน) → ต้องไม่ crash
