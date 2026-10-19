# predict.py
"""
Parcel Box Counter — Multi-class Classification Inference Engine
================================================================
สคริปต์สำหรับ:
  1. โหลดรูปภาพจาก Path ที่กำหนด (รองรับ JPG, PNG, HEIC, WEBP, BMP, ฯลฯ)
  2. โหลด Champion Model Bundle (best_weights.json, svd_model.npz, pipeline_config.json)
  3. รัน Inference Pipeline:
     Edge Detection → SVD Projection + Stat Features → MLP Softmax → Class Prediction
  4. Output: จำนวนกล่องที่ทำนาย + Softmax Probability Distribution
  5. รายงาน Theoretical Values ทุกขั้นตอน (สำหรับ Debug และ ความเข้าใจ)

Usage:
  # ใช้ Session ล่าสุดอัตโนมัติ:
  python predict.py --image dataset/boxes/IMG_001.jpg

  # ระบุ Session โดยตรง:
  python predict.py --image test.jpg --model-dir model/trained_models/session_xxxx

  # บันทึกผลเป็น JSON:
  python predict.py --image test.jpg --output-json result.json

  # บันทึกภาพขั้นตอนย่อย:
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _ROOT_DIR / "model"
_PREPROCESSING_DIR = _MODEL_DIR / "preprocessing"
_TRAINING_DIR = _MODEL_DIR / "training"
_MLP_DIR = _MODEL_DIR / "mlp"

for p in (_MODEL_DIR, _PREPROCESSING_DIR, _TRAINING_DIR, _MLP_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image Reader
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """อ่านรูปภาพจาก Path พร้อมดึง Metadata พื้นฐาน

    Returns
    -------
    bgr  : np.ndarray (H, W, 3) BGR format
    meta : dict รายละเอียดไฟล์รูป
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"[LoadImage] ไม่พบไฟล์ภาพ: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        try:
            from PIL import Image  # type: ignore
            try:
                import pillow_heif  # type: ignore
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            pil_img = Image.open(path).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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
# 2. Model Parameters Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelParameters:
    """พารามิเตอร์ทั้งหมดที่ใช้ในการทำงานของระบบ (Multi-class Classifier)"""
    # Preprocessing
    resize_target:   Tuple[int, int] = (512, 512)
    blur_ksize:      int             = 5
    clip_percentile: float           = 98.0
    image_size_svd:  int             = 128

    # SVD
    n_svd_features:  int                   = 64
    svd_mean:        Optional[np.ndarray]  = field(default=None, repr=False)
    svd_vt:          Optional[np.ndarray]  = field(default=None, repr=False)
    svd_s:           Optional[np.ndarray]  = field(default=None, repr=False)
    stat_mean:       Optional[np.ndarray]  = field(default=None, repr=False)
    stat_std:        Optional[np.ndarray]  = field(default=None, repr=False)

    # MLP Classifier Weights
    n_classes:       int                   = 9
    hidden_dim:      int                   = 64
    W1:              Optional[np.ndarray]  = field(default=None, repr=False)
    b1:              Optional[np.ndarray]  = field(default=None, repr=False)
    W2:              Optional[np.ndarray]  = field(default=None, repr=False)
    b2:              Optional[np.ndarray]  = field(default=None, repr=False)
    weight_init:     str                   = "xavier"

    # Metadata
    session_dir:     Optional[Path]            = None
    manifest:        Optional[Dict[str, Any]]  = None

    @classmethod
    def from_session_dir(cls, session_dir: Union[str, Path]) -> "ModelParameters":
        """โหลด Parameters และ Weights ทั้งหมดจากโฟลเดอร์ session"""
        s_dir = Path(session_dir).resolve()
        if not s_dir.is_dir():
            raise FileNotFoundError(f"[ModelParameters] ไม่พบโฟลเดอร์ session: {s_dir}")

        weights_file  = s_dir / "best_weights.json"
        svd_file      = s_dir / "svd_model.npz"
        config_file   = s_dir / "pipeline_config.json"
        manifest_file = s_dir / "manifest.json"

        if not weights_file.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ weights: {weights_file}")
        if not svd_file.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ svd: {svd_file}")

        cfg: Dict[str, Any] = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        pre_cfg = cfg.get("preprocessing", {})
        mdl_cfg = cfg.get("model", {})

        resize_target   = tuple(pre_cfg.get("resize_target", [512, 512]))
        blur_ksize      = int(pre_cfg.get("blur_ksize", 5))
        clip_percentile = float(pre_cfg.get("clip_percentile", 98.0))
        image_size_svd  = int(pre_cfg.get("image_size_svd", 128))
        n_svd_features  = int(mdl_cfg.get("n_svd_features", 64))
        n_classes       = int(mdl_cfg.get("n_classes", 9))

        # Load weights
        with open(weights_file, "r", encoding="utf-8") as f:
            w_data = json.load(f)

        arch       = w_data.get("architecture", {})
        hidden_dim = int(arch.get("hidden_dim", 64))
        weight_init = arch.get("weight_init", "xavier")

        # Support both old (W, b) single-layer and new (W1, b1, W2, b2) formats
        if "W1" in w_data:
            W1 = np.array(w_data["W1"], dtype=np.float32)
            b1 = np.array(w_data["b1"], dtype=np.float32)
            W2 = np.array(w_data["W2"], dtype=np.float32)
            b2 = np.array(w_data["b2"], dtype=np.float32)
        else:
            raise ValueError(
                "[ModelParameters] best_weights.json ไม่มี W1/b1/W2/b2 — "
                "กรุณาเทรน Model ใหม่ด้วย Multi-class Pipeline"
            )

        # Load SVD + stat normalizer
        svd_npz  = np.load(svd_file)
        svd_mean = svd_npz["mean"].astype(np.float32)
        svd_vt   = svd_npz["Vt"].astype(np.float32)
        svd_s    = svd_npz["S"].astype(np.float32) if "S" in svd_npz else None
        stat_mean = svd_npz["stat_mean"].astype(np.float32) if "stat_mean" in svd_npz else np.zeros(3, dtype=np.float32)
        stat_std  = svd_npz["stat_std"].astype(np.float32)  if "stat_std"  in svd_npz else np.ones(3,  dtype=np.float32)

        manifest = None
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        return cls(
            resize_target=resize_target,   # type: ignore
            blur_ksize=blur_ksize,
            clip_percentile=clip_percentile,
            image_size_svd=image_size_svd,
            n_svd_features=n_svd_features,
            svd_mean=svd_mean,
            svd_vt=svd_vt,
            svd_s=svd_s,
            stat_mean=stat_mean,
            stat_std=stat_std,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            W1=W1, b1=b1, W2=W2, b2=b2,
            weight_init=weight_init,
            session_dir=s_dir,
            manifest=manifest,
        )

    @classmethod
    def find_latest_session(cls, search_root: Optional[Path] = None) -> "ModelParameters":
        """ค้นหา Session ล่าสุดใน model/trained_models โดยอัตโนมัติ"""
        root = search_root or (_MODEL_DIR / "trained_models")
        if not root.is_dir():
            raise FileNotFoundError(f"ไม่พบ trained_models: {root}")

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
# 3. Prediction Output Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionOutput:
    """ผลลัพธ์การทำนายครบชุด"""
    # Prediction
    predicted_boxes:      int            # จำนวนกล่องที่ทำนาย (1-9)
    predicted_class_idx:  int            # 0-indexed class (0-8)
    confidence:           float          # Softmax probability ของ class ที่ทำนาย
    class_probabilities:  List[float]    # Softmax probabilities ทุกคลาส (1-9)

    # Feature Vector
    svd_latent_vector:    List[float]    # Top-64 SVD components
    stat_features:        List[float]    # [total_edge_area, log_contour_count, aspect_ratio]
    combined_feature_dim: int            # 67 = 64 + 3

    # Theoretical values
    edge_density_pct:     float          # % pixels เป็น edge
    edge_total_energy:    float          # sum(edge^2)
    svd_latent_energy:    float          # ||z_svd||^2
    hidden_activations:   List[float]    # ReLU hidden layer (top 10)
    logits:               List[float]    # Pre-softmax logits (per class)

    # Metadata
    image_meta:           Dict[str, Any]
    model_session:        Optional[str]
    inferred_at:          str

    def summary(self) -> str:
        """สรุปผลการทำนายแบบอ่านง่าย"""
        lines = [
            f"{'─'*60}",
            f"  PARCEL BOX COUNTER — PREDICTION RESULT",
            f"{'─'*60}",
            f"  ภาพ     : {self.image_meta.get('filename', 'N/A')}",
            f"  Session : {self.model_session or 'N/A'}",
            f"",
            f"  ┌─ PREDICTION ──────────────────────────────┐",
            f"  │  จำนวนกล่อง (Predicted) : {self.predicted_boxes} กล่อง       │",
            f"  │  Confidence             : {self.confidence*100:.1f}%            │",
            f"  └───────────────────────────────────────────┘",
            f"",
            f"  Probability Distribution (กล่อง 1-{len(self.class_probabilities)}):",
        ]
        for i, prob in enumerate(self.class_probabilities):
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ◄ Predicted" if (i + 1) == self.predicted_boxes else ""
            lines.append(f"    กล่อง {i+1}: [{bar}] {prob*100:5.1f}%{marker}")

        lines += [
            f"",
            f"  ─ Theoretical Values ─────────────────────────",
            f"  Edge Density      : {self.edge_density_pct:.2f}%",
            f"  Edge Total Energy : {self.edge_total_energy:.2f}",
            f"  SVD Latent Energy : {self.svd_latent_energy:.4f}",
            f"  Stat Features     : EdgeArea={self.stat_features[0]:.4f}, "
            f"LogContours={self.stat_features[1]:.4f}, AR={self.stat_features[2]:.4f}",
            f"  Feature Dim       : {self.combined_feature_dim}  (SVD={len(self.svd_latent_vector)} + Stat=3)",
            f"",
            f"  Pre-Softmax Logits (per class 1-{len(self.logits)}):",
            f"    {', '.join(f'{v:+.3f}' for v in self.logits)}",
            f"{'─'*60}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dict สำหรับ JSON output"""
        return {
            "predicted_boxes":      self.predicted_boxes,
            "predicted_class_idx":  self.predicted_class_idx,
            "confidence":           self.confidence,
            "class_probabilities":  {
                str(i + 1): round(p, 6)
                for i, p in enumerate(self.class_probabilities)
            },
            "theoretical": {
                "edge_density_pct":     round(self.edge_density_pct, 4),
                "edge_total_energy":    round(self.edge_total_energy, 4),
                "svd_latent_energy":    round(self.svd_latent_energy, 6),
                "stat_features": {
                    "total_edge_area":      round(self.stat_features[0], 6),
                    "log_contour_count":    round(self.stat_features[1], 6),
                    "aspect_ratio":         round(self.stat_features[2], 6),
                },
                "svd_latent_vector_top10": [round(v, 6) for v in self.svd_latent_vector[:10]],
                "hidden_activations_top10": [round(v, 6) for v in self.hidden_activations],
                "logits": [round(v, 6) for v in self.logits],
            },
            "image_meta":      self.image_meta,
            "model_session":   self.model_session,
            "inferred_at":     self.inferred_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. BoxPredictor (Main Inference Class)
# ─────────────────────────────────────────────────────────────────────────────

class BoxPredictor:
    """Multi-class Classification Inference Engine

    Pipeline:
      BGR → Edge Map (Laplacian) → SVD Projection + Stat Features
          → 2-Layer MLP → Softmax → Predicted Box Count
    """

    def __init__(self, params: ModelParameters):
        self.params = params
        self._validate()

    def _validate(self):
        p = self.params
        missing = [name for name, val in [
            ("svd_mean", p.svd_mean), ("svd_vt", p.svd_vt),
            ("W1", p.W1), ("b1", p.b1), ("W2", p.W2), ("b2", p.b2),
        ] if val is None]
        if missing:
            raise ValueError(f"[BoxPredictor] ขาดพารามิเตอร์: {missing}")

    def predict(self, bgr: np.ndarray) -> PredictionOutput:
        """รัน Inference Pipeline ครบสำหรับรูปภาพ 1 รูป

        Parameters
        ----------
        bgr : np.ndarray (H, W, 3) BGR

        Returns
        -------
        PredictionOutput : ผลลัพธ์ครบชุด
        """
        from edge_processing import process_image_edges
        from mlp_classifier import forward, softmax

        p = self.params

        # ── Step 1: Edge Map ─────────────────────────────────────────────────
        _, _, _, edge_map = process_image_edges(
            bgr,
            resize_target=p.resize_target,
            blur_ksize=p.blur_ksize,
            clip_percentile=p.clip_percentile,
        )
        # edge_map : uint8 (512, 512)

        # ── Step 2: Statistical Features ─────────────────────────────────────
        binary = (edge_map > 0).astype(np.uint8) * 255
        edge_density_pct  = float(np.count_nonzero(binary)) / float(edge_map.size) * 100.0
        edge_total_energy = float(np.sum(edge_map.astype(np.float32) ** 2))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_contours = len(contours)
        if n_contours > 0:
            all_x = np.concatenate([c[:, 0, 0] for c in contours])
            all_y = np.concatenate([c[:, 0, 1] for c in contours])
            bw = float(all_x.max() - all_x.min() + 1)
            bh = float(all_y.max() - all_y.min() + 1)
            aspect_ratio = bw / (bh + 1e-6)
        else:
            aspect_ratio = 1.0

        stat_raw = np.array([
            float(np.count_nonzero(binary)) / float(edge_map.size),  # total_edge_area
            float(np.log1p(n_contours)),                              # log_contour_count
            aspect_ratio,                                             # aspect_ratio
        ], dtype=np.float32)

        # Normalize using train statistics
        stat_norm = (stat_raw - p.stat_mean) / (p.stat_std + 1e-8)

        # ── Step 3: SVD Projection ────────────────────────────────────────────
        small = cv2.resize(edge_map, (p.image_size_svd, p.image_size_svd), interpolation=cv2.INTER_AREA)
        x_pixel = (small.astype(np.float32) / 255.0).flatten()

        x_centered = x_pixel - p.svd_mean
        z_svd = (x_centered @ p.svd_vt.T).astype(np.float32)

        # ── Step 4: Combine Feature Vector ───────────────────────────────────
        z_combined = np.concatenate([z_svd, stat_norm]).reshape(1, -1)  # (1, 67)

        # ── Step 5: 2-Layer MLP Forward (no dropout) ─────────────────────────
        probs_2d, h, z1, _ = forward(
            z_combined, p.W1, p.b1, p.W2, p.b2,
            dropout_rate=0.0, training=False,
        )
        probs = probs_2d[0]           # (n_classes,)
        logits_raw = h @ p.W2 + p.b2             # (1, n_classes) — pre-softmax logits
        logits = logits_raw[0].tolist()

        predicted_class_idx = int(np.argmax(probs))
        predicted_boxes     = predicted_class_idx + 1
        confidence          = float(probs[predicted_class_idx])

        # Hidden layer (top 10)
        h_vec = h[0]
        top10_h = h_vec[np.argsort(np.abs(h_vec))[::-1][:10]].tolist()

        return PredictionOutput(
            predicted_boxes=predicted_boxes,
            predicted_class_idx=predicted_class_idx,
            confidence=confidence,
            class_probabilities=probs.tolist(),
            svd_latent_vector=z_svd.tolist(),
            stat_features=stat_raw.tolist(),
            combined_feature_dim=z_combined.shape[1],
            edge_density_pct=edge_density_pct,
            edge_total_energy=edge_total_energy,
            svd_latent_energy=float(np.sum(z_svd ** 2)),
            hidden_activations=top10_h,
            logits=logits,
            image_meta={},
            model_session=None,
            inferred_at=datetime.now().isoformat(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Step Image Saver
# ─────────────────────────────────────────────────────────────────────────────

def save_step_images(
    bgr: np.ndarray,
    params: ModelParameters,
    output_dir: Path,
    prefix: str = "step",
) -> List[Path]:
    """บันทึกภาพขั้นตอนย่อยของ Pipeline สำหรับ Debug/Visualization"""
    from edge_processing import process_image_edges

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Original (resized)
    p_path = output_dir / f"{prefix}_1_original.jpg"
    resized = cv2.resize(bgr, params.resize_target)
    cv2.imwrite(str(p_path), resized)
    saved.append(p_path)

    # Grayscale
    gray_path = output_dir / f"{prefix}_2_grayscale.jpg"
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(gray_path), gray)
    saved.append(gray_path)

    # Blurred
    blur_path = output_dir / f"{prefix}_3_blurred.jpg"
    blurred = cv2.GaussianBlur(gray, (params.blur_ksize, params.blur_ksize), 0)
    cv2.imwrite(str(blur_path), blurred)
    saved.append(blur_path)

    # Edge Map
    _, _, _, edge_map = process_image_edges(
        bgr,
        resize_target=params.resize_target,
        blur_ksize=params.blur_ksize,
        clip_percentile=params.clip_percentile,
    )
    edge_path = output_dir / f"{prefix}_4_edge_map.jpg"
    cv2.imwrite(str(edge_path), edge_map)
    saved.append(edge_path)

    # SVD-resized
    svd_size_path = output_dir / f"{prefix}_5_svd_input.jpg"
    small = cv2.resize(edge_map, (params.image_size_svd, params.image_size_svd))
    cv2.imwrite(str(svd_size_path), small)
    saved.append(svd_size_path)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Parcel Box Counter — Multi-class Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Champion model session directory (default: auto-detect latest)")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Save prediction result as JSON to this path")
    parser.add_argument("--save-steps", type=Path, default=None,
                        help="Directory to save intermediate step images")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output (only print predicted count)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load image
    try:
        bgr, meta = load_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not args.quiet:
        print(f"\n[Predict] โหลดภาพสำเร็จ: {meta['filename']} "
              f"({meta['width']}×{meta['height']})")

    # Load model
    try:
        if args.model_dir:
            params = ModelParameters.from_session_dir(args.model_dir)
        else:
            params = ModelParameters.find_latest_session()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] โหลดโมเดลไม่สำเร็จ: {e}")
        sys.exit(1)

    # Run inference
    predictor = BoxPredictor(params)
    result = predictor.predict(bgr)
    result.image_meta = meta
    result.model_session = params.session_dir.name if params.session_dir else None

    # Save step images
    if args.save_steps:
        saved = save_step_images(bgr, params, args.save_steps)
        if not args.quiet:
            print(f"[Predict] บันทึกภาพขั้นตอน {len(saved)} ไฟล์ → {args.save_steps}")

    # Output
    if args.quiet:
        print(result.predicted_boxes)
    else:
        print(result.summary())

    # JSON export
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"[Predict] บันทึก JSON → {args.output_json}")


if __name__ == "__main__":
    main()
