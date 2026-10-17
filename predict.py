# predict.py
"""
Parcel Box Counter — Inference & Theoretical Analysis Engine
============================================================
สคริปต์สำหรับ:
  1. โหลดรูปภาพจาก Path ที่กำหนด (รองรับ JPG, PNG, HEIC, WEBP, BMP, ฯลฯ)
  2. อ่านและโหลดค่า Parameters จากโฟลเดอร์โมเดล (best_weights.json, svd_model.npz, pipeline_config.json)
     หรือรับค่า Parameter จาก Command-line โดยตรง
  3. คำนวณผลลัพธ์การทำนายจำนวนกล่องพัสดุ (ทั้งค่าต่อเนื่อง และค่าจำนวนเต็ม)
  4. คำนวณและรายงานค่าทางทฤษฎีครบทุกขั้นตอน:
     - ทฤษฎี Spatial & Edge Domain (Laplacian filter, Percentile Clipping threshold, Edge Energy)
     - ทฤษฎี Linear Algebra & SVD Subspace (Mean Centering, Projection Vector, Latent Matrix 8x8,
       Reconstruction Vector, Orthogonal Projection Residual / Loss)
     - ทฤษฎี Neural Activation & Dynamics (Logit z, Logistic Sigmoid, Scaled Output,
       Local Sensitivity dy/dz, Feature Attributions z_j * W_j, Cosine Alignment)

Usage Examples:
---------------
# 1. ทำนายด้วย Champion Session โฟลเดอร์:
python predict.py --image dataset/dataset/1_box/IMG_6510.jpg --model-dir model/trained_models/session_xxxx

# 2. ทำนายโดยให้อ่าน Session ล่าสุดใน model/trained_models/ อัตโนมัติ:
python predict.py --image dataset/dataset/1_box/IMG_6510.jpg

# 3. ระบุไฟล์แยก และกำหนดค่า Parameters อิสระ:
python predict.py --image test.jpg --weights best_weights.json --svd svd_model.npz --k 10.0 --clip-percentile 98.0

# 4. บันทึกผลลัพธ์เป็นไฟล์ JSON สำหรับเชื่อมต่อระบบอื่น (API/Web):
python predict.py --image test.jpg --output-json result.json

# 5. บันทึกภาพขั้นตอนย่อยทางทฤษฎี (Grayscale, Edge, SVD 128x128, Reconstruction):
python predict.py --image test.jpg --save-steps debug_output/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Enable UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# Add local modules to sys.path
_ROOT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _ROOT_DIR / "model"
_PREPROCESSING_DIR = _MODEL_DIR / "preprocessing"
_TRAINING_DIR = _MODEL_DIR / "training"
_MLP_DIR = _MODEL_DIR / "mlp"

for p in (_MODEL_DIR, _PREPROCESSING_DIR, _TRAINING_DIR, _MLP_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image Reader with Broad Format Support
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """อ่านรูปภาพจาก Path พร้อมดึง Metadata พื้นฐาน

    Parameters
    ----------
    image_path : Path หรือ str ของรูปภาพ

    Returns
    -------
    bgr : np.ndarray (H, W, 3) BGR format
    meta : dict รายละเอียดไฟล์รูป
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"[LoadImage] ไม่พบไฟล์ภาพ: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    # Fallback to PIL / pillow-heif for HEIC or unusual formats if OpenCV returns None
    if bgr is None:
        try:
            from PIL import Image  # type: ignore
            try:
                import pillow_heif  # type: ignore
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            pil_img = Image.open(path).convert("RGB")
            rgb_arr = np.array(pil_img)
            bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"[LoadImage] ไม่สามารถอ่านไฟล์ภาพ '{path.name}': {e}")

    if bgr is None or bgr.size == 0:
        raise ValueError(f"[LoadImage] ภาพว่างเปล่าหรือไฟล์เสียหาย: {path}")

    h, w, c = bgr.shape
    meta = {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "height": h,
        "width": w,
        "channels": c,
        "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
    }
    return bgr, meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameters & Model Bundle Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelParameters:
    """พารามิเตอร์ทั้งหมดที่ใช้ในการทำงานของระบบ"""
    # Preprocessing Parameters
    resize_target: Tuple[int, int] = (512, 512)
    blur_ksize: int = 5
    clip_percentile: float = 98.0
    image_size_svd: int = 128

    # SVD Subspace Parameters
    n_svd_features: int = 64
    svd_mean: Optional[np.ndarray] = field(default=None, repr=False)
    svd_vt: Optional[np.ndarray] = field(default=None, repr=False)
    svd_s: Optional[np.ndarray] = field(default=None, repr=False)

    # MLP / Model Parameters
    k: float = 10.0
    W: Optional[np.ndarray] = field(default=None, repr=False)
    b: Optional[np.ndarray] = field(default=None, repr=False)
    weight_init: str = "xavier"

    # Metadata
    session_dir: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None

    @classmethod
    def from_session_dir(cls, session_dir: Union[str, Path]) -> "ModelParameters":
        """โหลด Parameters และ Weights ทั้งหมดจากโฟลเดอร์ session ของ Model Registry"""
        s_dir = Path(session_dir).resolve()
        if not s_dir.is_dir():
            raise FileNotFoundError(f"[ModelParameters] ไม่พบโฟลเดอร์ session: {s_dir}")

        weights_file = s_dir / "best_weights.json"
        svd_file = s_dir / "svd_model.npz"
        config_file = s_dir / "pipeline_config.json"
        manifest_file = s_dir / "manifest.json"

        if not weights_file.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ weights: {weights_file}")
        if not svd_file.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ svd: {svd_file}")

        # 1. Load pipeline_config.json (ถ้ามี)
        cfg: Dict[str, Any] = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        pre_cfg = cfg.get("preprocessing", {})
        mdl_cfg = cfg.get("model", {})

        resize_target = tuple(pre_cfg.get("resize_target", [512, 512]))
        blur_ksize = int(pre_cfg.get("blur_ksize", 5))
        clip_percentile = float(pre_cfg.get("clip_percentile", 98.0))
        image_size_svd = int(pre_cfg.get("image_size_svd", 128))
        n_features = int(mdl_cfg.get("n_svd_features", 64))

        # 2. Load best_weights.json
        with open(weights_file, "r", encoding="utf-8") as f:
            w_data = json.load(f)

        W = np.array(w_data["W"], dtype=np.float32)
        b = np.array(w_data["b"], dtype=np.float32)
        k = float(w_data.get("k", mdl_cfg.get("k", 10.0)))
        arch = w_data.get("architecture", {})
        weight_init = arch.get("weight_init", "xavier")

        # 3. Load svd_model.npz
        svd_npz = np.load(svd_file)
        svd_mean = svd_npz["mean"].astype(np.float32)
        svd_vt = svd_npz["Vt"].astype(np.float32)
        svd_s = svd_npz["S"].astype(np.float32) if "S" in svd_npz else None

        # 4. Load manifest.json (Optional)
        manifest = None
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        return cls(
            resize_target=resize_target,  # type: ignore
            blur_ksize=blur_ksize,
            clip_percentile=clip_percentile,
            image_size_svd=image_size_svd,
            n_svd_features=n_features,
            svd_mean=svd_mean,
            svd_vt=svd_vt,
            svd_s=svd_s,
            k=k,
            W=W,
            b=b,
            weight_init=weight_init,
            session_dir=s_dir,
            manifest=manifest,
        )

    @classmethod
    def find_latest_session(cls, search_root: Optional[Path] = None) -> "ModelParameters":
        """ค้นหาโฟลเดอร์ Session ล่าสุดใน model/trained_models โดยอัตโนมัติ"""
        root = search_root or (_MODEL_DIR / "trained_models")
        if not root.is_dir():
            raise FileNotFoundError(f"ไม่พบไดเรกทอรี trained_models: {root}")

        sessions = sorted(
            [d for d in root.iterdir() if d.is_dir() and (d / "best_weights.json").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not sessions:
            raise FileNotFoundError(f"ไม่พบ session ที่มี best_weights.json ใน: {root}")

        latest = sessions[0]
        print(f"[ModelParameters] โหลด Session ล่าสุดอัตโนมัติ: {latest.name}")
        return cls.from_session_dir(latest)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mathematical & Theoretical Computation Engine
# ─────────────────────────────────────────────────────────────────────────────

# Laplacian 3x3 Edge Kernel
LAPLACIAN_KERNEL_3X3 = np.array(
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]],
    dtype=np.float32,
)

def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard Logistic Sigmoid: sigma(z) = 1 / (1 + exp(-z))"""
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class TheoreticalAnalysis:
    """รวมค่าและสถิติทั้งหมดที่คำนวณได้ตามหลักทฤษฎีในแต่ละช่วงของโมเดล"""

    # ── 1. Spatial Domain & Preprocessing ──
    raw_dimensions: Tuple[int, int, int]
    grayscale_mean: float
    grayscale_std: float
    grayscale_min: int
    grayscale_max: int

    blur_kernel_size: int
    edge_clip_percentile: float
    edge_raw_max_gradient: float
    edge_percentile_threshold: float
    edge_density_pct: float             # % พิกเซลขอบที่มีค่ามากกว่า 0
    edge_mean_intensity: float
    edge_total_energy: float            # sum(E^2)

    # ── 2. Linear Algebra & SVD Subspace Projection ──
    pixel_vector_dim: int               # 128^2 = 16384
    pixel_vector_l1_norm: float
    pixel_vector_l2_norm: float
    centering_shift_norm: float         # ||x - mean||_2 (ระยะห่างจาก Centroid ของชุดข้อมูล)

    n_svd_components: int               # เช่น 64
    svd_latent_vector: List[float]      # z in R^64
    svd_latent_matrix_8x8: List[List[float]]  # representation 8x8
    svd_latent_energy: float            # ||z||_2^2
    svd_singular_values_top: List[float] # Singular values sigma_1 .. sigma_k (ถ้ามี)
    svd_singular_energy_ratio: Optional[float] # sigma^2 ratio

    reconstruction_residual_l2: float   # ||x - x_hat||_2 (Orthogonal projection loss)
    reconstruction_relative_error: float # ||x - x_hat||_2 / ||x||_2
    subspace_fidelity_score: float      # 1 - relative_error

    # ── 3. Neural Activation & Dynamics ──
    weight_l2_norm: float
    bias_value: float
    cosine_similarity_z_w: float        # alignment ระหว่าง latent vector กับ weights

    logit_z: float                      # z = x_svd . W + b
    standard_sigmoid: float             # sigma(z) in (0, 1)
    scaled_sigmoid_k: float             # k factor

    continuous_prediction: float        # y_hat = k * sigma(z)
    theoretical_sensitivity: float      # dy_hat / dz = k * sigma(z) * (1 - sigma(z))
    theoretical_curvature: float        # d^2 y_hat / dz^2

    top_positive_features: List[Dict[str, Any]] # ฟีเจอร์ที่ดันให้จำนวนกล่องเพิ่ม (z_j * W_j > 0)
    top_negative_features: List[Dict[str, Any]] # ฟีเจอร์ที่ดึงให้จำนวนกล่องลดลง (z_j * W_j < 0)

    # ── 4. Decision & Uncertainty ──
    discrete_prediction: int            # round(y_hat)
    prediction_floor: int               # floor(y_hat)
    prediction_ceil: int                # ceil(y_hat)
    rounding_ambiguity: float           # 2 * |y_hat - round(y_hat)| in [0, 1] (0=มั่นใจมาก, 1=ก้ำกึ่ง .5)
    decision_confidence: float          # 1 - ambiguity in [0, 1]


@dataclass
class PredictionOutput:
    """ผลลัพธ์สุดท้ายของการทำนาย"""
    image_path: str
    predicted_box_count: int            # ผลลัพธ์จำนวนกล่องเต็ม (Discrete Answer)
    continuous_box_count: float         # ค่าทำนายต่อเนื่อง (Continuous Output)
    decision_confidence: float          # ความมั่นใจในการตัดสินใจ (0.0 - 1.0)
    theoretical_values: TheoreticalAnalysis

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dict สำหรับบันทึก JSON หรือส่งผ่าน API"""
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 4. BoxPredictor Core Class
# ─────────────────────────────────────────────────────────────────────────────

class BoxPredictor:
    """เครื่องมือประมวลผล ทำนายผล และสกัดค่าทางทฤษฎีจากภาพพัสดุ"""

    def __init__(self, params: ModelParameters):
        self.params = params
        self._validate_params()

    def _validate_params(self):
        p = self.params
        if p.svd_mean is None or p.svd_vt is None:
            raise ValueError("[BoxPredictor] ขาดข้อมูล SVD (mean หรือ Vt) ใน ModelParameters")
        if p.W is None or p.b is None:
            raise ValueError("[BoxPredictor] ขาดค่าน้ำหนัก (W หรือ b) ใน ModelParameters")

    @classmethod
    def from_session(cls, session_dir: Union[str, Path]) -> "BoxPredictor":
        """สร้าง Predictor จาก Session โฟลเดอร์"""
        params = ModelParameters.from_session_dir(session_dir)
        return cls(params)

    @classmethod
    def from_latest_session(cls, search_root: Optional[Path] = None) -> "BoxPredictor":
        """สร้าง Predictor จาก Session ล่าสุดที่พบ"""
        params = ModelParameters.find_latest_session(search_root)
        return cls(params)

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        save_intermediate_dir: Optional[Union[str, Path]] = None,
    ) -> PredictionOutput:
        """ทำนายจำนวนกล่องและคำนวณค่าทางทฤษฎีจากรูปภาพ

        Parameters
        ----------
        image_input : Path, str ของไฟล์ภาพ หรือ np.ndarray BGR
        save_intermediate_dir : โฟลเดอร์บันทึกภาพขั้นตอนย่อย (Optional)

        Returns
        -------
        PredictionOutput
        """
        p = self.params

        # ── Step 1: Read Image & Basic Meta ──
        if isinstance(image_input, (str, Path)):
            bgr, meta = load_image(image_input)
            img_path_str = str(Path(image_input).resolve())
        else:
            bgr = image_input
            img_path_str = "<numpy_array_input>"
            meta = {
                "height": bgr.shape[0],
                "width": bgr.shape[1],
                "channels": bgr.shape[2] if bgr.ndim > 2 else 1,
            }

        # ── Step 2: Grayscale & Geometry Preprocessing ──
        if bgr.ndim == 2:
            gray = bgr.copy()
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        g_mean = float(np.mean(gray))
        g_std = float(np.std(gray))
        g_min = int(np.min(gray))
        g_max = int(np.max(gray))

        # Downscale to resize_target (512x512)
        resized_512 = cv2.resize(gray, p.resize_target, interpolation=cv2.INTER_AREA)

        # ── Step 3: Noise Reduction (Gaussian Blur) ──
        ksize = p.blur_ksize if p.blur_ksize % 2 != 0 else p.blur_ksize + 1
        ksize = max(1, ksize)
        blurred = cv2.GaussianBlur(resized_512, (ksize, ksize), sigmaX=0, sigmaY=0)

        # ── Step 4: Laplacian Edge Convolution & Percentile Clipping ──
        edge_raw = cv2.filter2D(blurred.astype(np.float32), cv2.CV_32F, LAPLACIAN_KERNEL_3X3)
        edge_abs = np.abs(edge_raw)

        raw_max_gradient = float(np.max(edge_abs))
        p_clip_thresh = float(np.percentile(edge_abs, p.clip_percentile))

        if p_clip_thresh > 1e-6:
            scaled_edges = (edge_abs / p_clip_thresh) * 255.0
        else:
            scaled_edges = edge_abs

        edge_map_uint8 = np.clip(scaled_edges, 0, 255).astype(np.uint8)

        # Edge statistics
        edge_density_pct = float(100.0 * np.count_nonzero(edge_map_uint8) / edge_map_uint8.size)
        edge_mean_intensity = float(np.mean(edge_map_uint8))
        edge_total_energy = float(np.sum(edge_map_uint8.astype(np.float64) ** 2))

        # ── Step 5: SVD Space Preparation (Resize 128x128 -> Vector 16384) ──
        svd_img = cv2.resize(
            edge_map_uint8,
            (p.image_size_svd, p.image_size_svd),
            interpolation=cv2.INTER_AREA,
        )
        x_norm = svd_img.astype(np.float32) / 255.0
        x_flat = x_norm.flatten()  # shape (16384,)

        l1_norm = float(np.sum(np.abs(x_flat)))
        l2_norm = float(np.linalg.norm(x_flat))

        # ── Step 6: Mean Centering & SVD Projection ──
        mean_vec = p.svd_mean.reshape(-1)
        x_centered = x_flat - mean_vec
        centering_shift_norm = float(np.linalg.norm(x_centered))

        # Projection: z = (x - mean) . Vt^T
        Vt = p.svd_vt  # shape (n_features, 16384)
        z_vector = np.dot(x_centered, Vt.T).astype(np.float32)  # shape (n_features,)
        latent_energy = float(np.sum(z_vector ** 2))

        # Latent Matrix 8x8 representation
        matrix_8x8 = z_vector.reshape(8, 8).tolist()

        # Orthogonal Reconstruction & Residual: x_hat = mean + z . Vt
        x_reconstructed = mean_vec + np.dot(z_vector, Vt)
        residual_vec = x_flat - x_reconstructed
        residual_l2 = float(np.linalg.norm(residual_vec))
        relative_error = float(residual_l2 / (l2_norm + 1e-8))
        fidelity_score = float(max(0.0, 1.0 - relative_error))

        # Singular value energy ratio (if S is stored)
        singular_energy_ratio = None
        s_top_list = []
        if p.svd_s is not None and len(p.svd_s) > 0:
            s_top_list = [round(float(s), 4) for s in p.svd_s[:10]]
            total_s_energy = float(np.sum(p.svd_s ** 2))
            if total_s_energy > 1e-8:
                singular_energy_ratio = float(np.sum(p.svd_s[:p.n_svd_features] ** 2) / total_s_energy)

        # ── Step 7: Neural Forward Pass (MLP Scaled Sigmoid) ──
        W = p.W.reshape(-1)  # shape (n_features,)
        b = float(p.b.reshape(-1)[0])
        w_norm = float(np.linalg.norm(W))

        # Cosine similarity between latent vector z and weights W
        z_norm = float(np.linalg.norm(z_vector))
        if z_norm > 1e-7 and w_norm > 1e-7:
            cos_sim = float(np.dot(z_vector, W) / (z_norm * w_norm))
        else:
            cos_sim = 0.0

        # Linear Logit: z_logit = z . W + b
        logit = float(np.dot(z_vector, W) + b)

        # Standard Sigmoid: sigma(z)
        sig = float(sigmoid(logit))

        # Scaled Prediction: y_hat = k * sigma(z)
        y_hat = float(p.k * sig)

        # Local Derivatives (Sensitivity & Curvature)
        dy_dz = float(p.k * sig * (1.0 - sig))
        d2y_dz2 = float(p.k * sig * (1.0 - sig) * (1.0 - 2.0 * sig))

        # Feature Attribution: c_j = z_j * W_j
        attributions = z_vector * W
        sorted_indices = np.argsort(attributions)

        # Top 5 positive contributors
        top_pos = []
        for idx in reversed(sorted_indices[-5:]):
            top_pos.append({
                "feature_index": int(idx),
                "z_value": round(float(z_vector[idx]), 5),
                "weight": round(float(W[idx]), 5),
                "contribution": round(float(attributions[idx]), 5),
            })

        # Top 5 negative contributors
        top_neg = []
        for idx in sorted_indices[:5]:
            top_neg.append({
                "feature_index": int(idx),
                "z_value": round(float(z_vector[idx]), 5),
                "weight": round(float(W[idx]), 5),
                "contribution": round(float(attributions[idx]), 5),
            })

        # ── Step 8: Discrete Decision & Uncertainty ──
        discrete_count = int(np.clip(round(y_hat), 0, round(p.k)))
        pred_floor = int(np.floor(y_hat))
        pred_ceil = int(np.ceil(y_hat))

        # Rounding ambiguity (0 = exact integer, 1 = maximum ambiguity at .5)
        ambiguity = float(2.0 * abs(y_hat - round(y_hat)))
        confidence = float(max(0.0, min(1.0, 1.0 - ambiguity)))

        # ── Step 9: Assemble Theoretical Analysis Package ──
        analysis = TheoreticalAnalysis(
            raw_dimensions=(meta["height"], meta["width"], meta.get("channels", 3)),
            grayscale_mean=round(g_mean, 2),
            grayscale_std=round(g_std, 2),
            grayscale_min=g_min,
            grayscale_max=g_max,
            blur_kernel_size=ksize,
            edge_clip_percentile=p.clip_percentile,
            edge_raw_max_gradient=round(raw_max_gradient, 2),
            edge_percentile_threshold=round(p_clip_thresh, 4),
            edge_density_pct=round(edge_density_pct, 2),
            edge_mean_intensity=round(edge_mean_intensity, 2),
            edge_total_energy=round(edge_total_energy, 2),
            pixel_vector_dim=len(x_flat),
            pixel_vector_l1_norm=round(l1_norm, 2),
            pixel_vector_l2_norm=round(l2_norm, 4),
            centering_shift_norm=round(centering_shift_norm, 4),
            n_svd_components=len(z_vector),
            svd_latent_vector=[round(float(v), 5) for v in z_vector],
            svd_latent_matrix_8x8=[[round(float(v), 4) for v in row] for row in matrix_8x8],
            svd_latent_energy=round(latent_energy, 4),
            svd_singular_values_top=s_top_list,
            svd_singular_energy_ratio=round(singular_energy_ratio, 4) if singular_energy_ratio else None,
            reconstruction_residual_l2=round(residual_l2, 4),
            reconstruction_relative_error=round(relative_error, 4),
            subspace_fidelity_score=round(fidelity_score, 4),
            weight_l2_norm=round(w_norm, 4),
            bias_value=round(b, 4),
            cosine_similarity_z_w=round(cos_sim, 4),
            logit_z=round(logit, 5),
            standard_sigmoid=round(sig, 5),
            scaled_sigmoid_k=p.k,
            continuous_prediction=round(y_hat, 4),
            theoretical_sensitivity=round(dy_dz, 5),
            theoretical_curvature=round(d2y_dz2, 5),
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            discrete_prediction=discrete_count,
            prediction_floor=pred_floor,
            prediction_ceil=pred_ceil,
            rounding_ambiguity=round(ambiguity, 4),
            decision_confidence=round(confidence, 4),
        )

        output = PredictionOutput(
            image_path=img_path_str,
            predicted_box_count=discrete_count,
            continuous_box_count=round(y_hat, 4),
            decision_confidence=round(confidence, 4),
            theoretical_values=analysis,
        )

        # ── Step 10: Optionally Save Intermediate Visual Steps ──
        if save_intermediate_dir is not None:
            out_dir = Path(save_intermediate_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_dir / "01_grayscale.png"), gray)
            cv2.imwrite(str(out_dir / "02_blurred_512.png"), blurred)
            cv2.imwrite(str(out_dir / "03_edge_map_512.png"), edge_map_uint8)
            cv2.imwrite(str(out_dir / "04_svd_input_128.png"), svd_img)

            # Reconstructed image back to 128x128
            recon_img = np.clip(x_reconstructed.reshape(128, 128) * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / "05_svd_reconstruction_128.png"), recon_img)

            # Residual map (difference)
            diff_img = np.clip(np.abs(x_flat - x_reconstructed).reshape(128, 128) * 255.0 * 2.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / "06_projection_residual_map.png"), diff_img)
            print(f"  ✓ บันทึกภาพขั้นตอนย่อยทางทฤษฎีลงใน: {out_dir}")

        return output


# ─────────────────────────────────────────────────────────────────────────────
# 5. Formatted Terminal Reporter
# ─────────────────────────────────────────────────────────────────────────────

def print_pretty_report(output: PredictionOutput, params: ModelParameters) -> None:
    """แสดงรายงานผลลัพธ์และค่าทางทฤษฎีบนหน้าจอ Terminal แบบชัดเจนและสวยงาม"""
    t = output.theoretical_values

    print("\n" + "=" * 76)
    print("           📦 PARCEL BOX COUNTER — THEORETICAL INFERENCE REPORT")
    print("=" * 76)

    # ── 1. Answer ──
    print("\n【 1. ผลลัพธ์การทำนาย (FINAL PREDICTION ANSWER) 】")
    print(f"  • รูปภาพเป้าหมาย         : {Path(output.image_path).name}")
    print(f"  • เส้นทางไฟล์ (Path)      : {output.image_path}")
    print(f"  • 🎯 คำตอบจำนวนกล่องพัสดุ:  >>> {output.predicted_box_count} กล่อง <<<  (จำนวนเต็ม)")
    print(f"  • ค่าทำนายต่อเนื่อง (y_hat) : {output.continuous_box_count:.4f} กล่อง (สเกล [0, {t.scaled_sigmoid_k:.1f}])")
    print(f"  • ช่วงจำนวนที่เป็นไปได้   : [{t.prediction_floor}, {t.prediction_ceil}] กล่อง")
    print(f"  • ระดับความมั่นใจ (Conf) : {output.decision_confidence * 100:.1f}% (Ambiguity index = {t.rounding_ambiguity:.3f})")

    # ── 2. Active Parameters ──
    print("\n【 2. พารามิเตอร์ที่ใช้ในการทำงาน (ACTIVE PARAMETERS) 】")
    if params.session_dir:
        print(f"  • โฟลเดอร์ Session       : {params.session_dir.name}")
    print(f"  • Resize Target          : {params.resize_target[0]}×{params.resize_target[1]} px")
    print(f"  • Gaussian Blur Kernel   : {params.blur_ksize}×{params.blur_ksize}")
    print(f"  • Outlier Percentile Clip: {params.clip_percentile:.1f}%")
    print(f"  • SVD Input Resolution   : {params.image_size_svd}×{params.image_size_svd} px (Total {t.pixel_vector_dim} pixels)")
    print(f"  • SVD Latent Dimension   : {params.n_svd_features} Components (Represented as 8×8 Matrix)")
    print(f"  • Scaled Sigmoid Max (k) : {params.k:.1f}")
    print(f"  • Weight Vector L2 Norm  : {t.weight_l2_norm:.4f}")
    print(f"  • Bias Value (b)         : {t.bias_value:.4f}")

    # ── 3. Spatial Domain & Edge Theory ──
    print("\n【 3. ค่าทางทฤษฎีด้านภาพและเส้นขอบ (SPATIAL & EDGE DOMAIN) 】")
    print(f"  • ขนาดมิติต้นฉบับ        : {t.raw_dimensions[0]}×{t.raw_dimensions[1]} (Channels: {t.raw_dimensions[2]})")
    print(f"  • ค่าสว่าง Grayscale     : Mean = {t.grayscale_mean:.1f}, Std = {t.grayscale_std:.1f}, Range = [{t.grayscale_min}, {t.grayscale_max}]")
    print(f"  • ตัวกรองขอบ (Kernel)    : Laplacian 3×3 Convolved with Gaussian Filter")
    print(f"  • Gradient สูงสุดดิบ     : {t.edge_raw_max_gradient:.2f}")
    print(f"  • ขีดจำกัดตัดแสงสะท้อน (T_clip): {t.edge_percentile_threshold:.4f} (ณ Percentile {t.edge_clip_percentile}%)")
    print(f"  • ความหนาแน่นของเส้นขอบ  : {t.edge_density_pct:.2f}% ของพื้นที่ภาพ")
    print(f"  • ความเข้มขอบเฉลี่ย       : {t.edge_mean_intensity:.2f} / 255.0")
    print(f"  • พลังงานเส้นขอบรวม (E²) : {t.edge_total_energy:,.1f}")

    # ── 4. SVD Subspace & Linear Algebra Theory ──
    print("\n【 4. ค่าทางทฤษฎีพีชคณิตเชิงเส้น (LINEAR ALGEBRA & SVD SUBSPACE) 】")
    print(f"  • ระยะห่างจากเซนทรอยด์   : ||x - μ_train||₂ = {t.centering_shift_norm:.4f}")
    print(f"  • Normalization Vector   : L1 Norm = {t.pixel_vector_l1_norm:.1f}, L2 Norm = {t.pixel_vector_l2_norm:.4f}")
    print(f"  • พลังงานเวกเตอร์แฝง (z²): ||z||₂² = {t.svd_latent_energy:.4f}")
    print(f"  • Residual การฉายภาพ     : ||x - x_recon||₂ = {t.reconstruction_residual_l2:.4f}")
    print(f"  • ความคลาดเคลื่อนสัมพัทธ์ : {t.reconstruction_relative_error * 100:.2f}%")
    print(f"  • ความเที่ยงตรง (Fidelity): {t.subspace_fidelity_score * 100:.2f}% (ความสามารถในการสร้างภาพกลับ)")
    if t.svd_singular_energy_ratio:
        print(f"  • อัตราพลังงาน Singular  : {t.svd_singular_energy_ratio * 100:.2f}% ครอบคลุมใน {t.n_svd_components} องค์ประกอบ")

    # ── 5. Neural Activation & Dynamics Theory ──
    print("\n【 5. ค่าทางทฤษฎีประสาทเทียมและการกระตุ้น (NEURAL ACTIVATION DYNAMICS) 】")
    print(f"  • Cosine Alignment (z, W): cos(θ) = {t.cosine_similarity_z_w:.4f}")
    print(f"  • ผลรวมเชิงเส้น (Logit z): z = z·W + b = {t.logit_z:.5f}")
    print(f"  • ค่าฟังก์ชัน Sigmoid (σ): σ(z) = 1/(1+e⁻ᶻ) = {t.standard_sigmoid:.5f}  (ช่วง (0, 1))")
    print(f"  • ค่าทำนายสเกล k·σ(z)    : {t.scaled_sigmoid_k:.1f} × {t.standard_sigmoid:.5f} = {t.continuous_prediction:.4f}")
    print(f"  • อัตราความไวเฉพาะที่    : dy_hat/dz = k·σ(1-σ) = {t.theoretical_sensitivity:.5f} กล่อง/logit")
    print(f"  • ความโค้ง (Curvature)   : d²y_hat/dz² = {t.theoretical_curvature:.5f}")

    # ── 6. Top Feature Attributions ──
    print("\n【 6. การแจกแจงอิทธิพลของแต่ละ SVD Component (TOP ATTRIBUTIONS) 】")
    print("  ▲ องค์ประกอบที่มีผลบวกมากที่สุด (ผลักดันให้จำนวนกล่องเพิ่มขึ้น):")
    for item in t.top_positive_features:
        print(f"    - SVD Component #{item['feature_index']:02d}: z = {item['z_value']:+.4f}, W = {item['weight']:+.4f} → Contribution = {item['contribution']:+.5f}")

    print("  ▼ องค์ประกอบที่มีผลลบมากที่สุด (ดึงให้จำนวนกล่องลดลง):")
    for item in t.top_negative_features:
        print(f"    - SVD Component #{item['feature_index']:02d}: z = {item['z_value']:+.4f}, W = {item['weight']:+.4f} → Contribution = {item['contribution']:+.5f}")

    print("\n" + "=" * 76 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI Argument Parser & Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parcel Box Counter — Inference & Theoretical Analysis Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input Image ──
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Path ไปยังไฟล์รูปภาพที่ต้องการตรวจนับกล่อง",
    )

    # ── Model Session / Parameters ──
    parser.add_argument(
        "--model-dir", "-m", type=Path, default=None,
        help="Path ไปยังโฟลเดอร์ session ของโมเดล (มี best_weights.json, svd_model.npz, pipeline_config.json)",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path ไปยัง best_weights.json (ระบุเฉพาะเจาะจง)",
    )
    parser.add_argument(
        "--svd", type=Path, default=None,
        help="Path ไปยัง svd_model.npz (ระบุเฉพาะเจาะจง)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path ไปยัง pipeline_config.json (ระบุเฉพาะเจาะจง)",
    )

    # ── Parameter Overrides ──
    parser.add_argument("--k", type=float, default=None, help="Override ค่า k (เพดานจำนวนกล่อง)")
    parser.add_argument("--clip-percentile", type=float, default=None, help="Override เปอร์เซ็นไทล์ตัดขอบ")
    parser.add_argument("--blur-ksize", type=int, default=None, help="Override ขนาด blur kernel")

    # ── Output Options ──
    parser.add_argument(
        "--output-json", "-o", type=Path, default=None,
        help="บันทึกผลลัพธ์และค่าทางทฤษฎีทั้งหมดเป็นไฟล์ JSON",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="พิมพ์ผลลัพธ์ออกทาง stdout เป็น JSON รูปแบบดิบสำหรับ API",
    )
    parser.add_argument(
        "--save-steps", type=Path, default=None,
        help="โฟลเดอร์สำหรับบันทึกรูปภาพผลลัพธ์ของแต่ละขั้นตอนย่อยทางทฤษฎี",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # ── 1. Validate Image Path ──
    if not args.image.exists():
        print(f"[ERROR] ไม่พบไฟล์ภาพ: {args.image}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Resolve Model Parameters ──
    try:
        if args.model_dir:
            params = ModelParameters.from_session_dir(args.model_dir)
        elif args.weights and args.svd:
            # Custom individual paths
            with open(args.weights, "r", encoding="utf-8") as f:
                w_data = json.load(f)
            svd_npz = np.load(args.svd)

            cfg = {}
            if args.config and args.config.exists():
                with open(args.config, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            pre_cfg = cfg.get("preprocessing", {})
            mdl_cfg = cfg.get("model", {})

            params = ModelParameters(
                resize_target=tuple(pre_cfg.get("resize_target", [512, 512])),  # type: ignore
                blur_ksize=int(pre_cfg.get("blur_ksize", 5)),
                clip_percentile=float(pre_cfg.get("clip_percentile", 98.0)),
                image_size_svd=int(pre_cfg.get("image_size_svd", 128)),
                n_svd_features=int(mdl_cfg.get("n_svd_features", 64)),
                svd_mean=svd_npz["mean"].astype(np.float32),
                svd_vt=svd_npz["Vt"].astype(np.float32),
                svd_s=svd_npz["S"].astype(np.float32) if "S" in svd_npz else None,
                k=float(w_data.get("k", 10.0)),
                W=np.array(w_data["W"], dtype=np.float32),
                b=np.array(w_data["b"], dtype=np.float32),
            )
        else:
            # Try finding latest session automatically
            params = ModelParameters.find_latest_session()

        # Apply any explicit CLI overrides
        if args.k is not None:
            params.k = args.k
        if args.clip_percentile is not None:
            params.clip_percentile = args.clip_percentile
        if args.blur_ksize is not None:
            params.blur_ksize = args.blur_ksize

    except Exception as e:
        print(f"[ERROR] ไม่สามารถโหลดโมเดลหรือพารามิเตอร์ได้: {e}", file=sys.stderr)
        print("คำแนะนำ: ระบุ --model-dir <โฟลเดอร์ session> หรือ --weights และ --svd", file=sys.stderr)
        sys.exit(1)

    # ── 3. Run Prediction & Theoretical Computation ──
    predictor = BoxPredictor(params)
    output = predictor.predict(
        image_input=args.image,
        save_intermediate_dir=args.save_steps,
    )

    # ── 4. Output Results ──
    if args.json:
        print(json.dumps(output.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_pretty_report(output, params)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✓ บันทึกไฟล์ JSON รายงานผลลัพธ์ลงใน: {args.output_json}")


if __name__ == "__main__":
    main()
