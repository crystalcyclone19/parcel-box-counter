"""Comprehensive Unit & Pipeline Test for Multi-class Classification System."""
import sys
import os
from pathlib import Path

# Add paths
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir.parent / "preprocessing"))
sys.path.insert(0, str(base_dir.parent / "mlp"))
sys.path.insert(0, str(base_dir.parent / "augmentation"))
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir.parent.parent))

import numpy as np

print("Testing imports...")
from dataset import load_dataset, split_dataset, DatasetSplit, extract_edge_features, _extract_stat_features
print("  [OK] dataset.py")
from trainer import Trainer, TrainResult, labels_to_class_idx
print("  [OK] trainer.py")
from evaluator import Evaluator, compute_metrics, MetricsResult
print("  [OK] evaluator.py")
from tuner import GridSearchTuner, build_search_space, DEFAULT_SEARCH_SPACE, _fit_svd, SVDModel
print("  [OK] tuner.py")
from model_registry import ModelRegistry
print("  [OK] model_registry.py")
from mlp_classifier import init_weights_xavier, forward, cross_entropy_loss, backward, train_step, predict as mlp_predict
print("  [OK] mlp_classifier.py")
from predict import ModelParameters, BoxPredictor, PredictionOutput
print("  [OK] predict.py")

print()
print("Testing Search Space...")
sp = build_search_space()
n_trials = (
    len(sp["clip_percentile"])
    * len(sp["learning_rate"])
    * len(sp["batch_size"])
    * len(sp["hidden_dim"])
)
print(f"  Total Trials : {n_trials}")
print(f"  clip_pct     : {sp['clip_percentile']}")
print(f"  learning_rate: {sp['learning_rate']}")
print(f"  batch_size   : {sp['batch_size']}")
print(f"  hidden_dim   : {sp['hidden_dim']}")

print()
print("Testing Xavier Init for Multi-class MLP...")
rng = np.random.default_rng(42)
W1, b1, W2, b2 = init_weights_xavier(input_dim=67, hidden_dim=64, n_classes=9, rng=rng)
print(f"  W1 shape: {W1.shape}, std: {W1.std():.4f}")
print(f"  b1 shape: {b1.shape}")
print(f"  W2 shape: {W2.shape}, std: {W2.std():.4f}")
print(f"  b2 shape: {b2.shape}")
assert W1.shape == (67, 64)
assert b1.shape == (64,)
assert W2.shape == (64, 9)
assert b2.shape == (9,)

print()
print("Testing Forward Pass, Loss, and Backward Pass...")
X_syn = rng.standard_normal((10, 67)).astype(np.float32)
y_syn = rng.integers(0, 9, size=10).astype(np.int32)
probs, h, z1, mask = forward(X_syn, W1, b1, W2, b2, dropout_rate=0.3, training=True, rng=rng)
print(f"  probs shape: {probs.shape}, prob sum per row: {probs.sum(axis=1)[:3]}")
loss = cross_entropy_loss(probs, y_syn)
print(f"  Cross-Entropy Loss: {loss:.4f}")
dW1, db1, dW2, db2 = backward(X_syn, probs, y_syn, W1, W2, h, z1, mask)
print(f"  dW1 shape: {dW1.shape}, dW2 shape: {dW2.shape}")
assert probs.shape == (10, 9)
assert dW1.shape == W1.shape
assert dW2.shape == W2.shape

print()
print("Testing Trainer.fit() with synthetic data...")
n_train, n_val, d = 40, 10, 67
Z_train = rng.standard_normal((n_train, d)).astype(np.float32)
# y values are box counts 1..9
y_train = rng.integers(1, 10, (n_train, 1)).astype(np.float32)
Z_val = rng.standard_normal((n_val, d)).astype(np.float32)
y_val = rng.integers(1, 10, (n_val, 1)).astype(np.float32)

trainer = Trainer(
    n_classes=9,
    hidden_dim=32,
    dropout_rate=0.3,
    max_epochs=20,
    patience=5,
    random_seed=42,
    verbose=0,
)
result = trainer.fit(Z_train, y_train, Z_val, y_val, learning_rate=0.01, batch_size=8)
print(f"  Best Val CE  : {result.best_val_ce:.6f} @ epoch {result.best_epoch}")
print(f"  Stopped Early: {result.stopped_early}")
print(f"  W1 shape     : {result.W1.shape}, W2 shape: {result.W2.shape}")
assert result.W2.shape == (32, 9)

print()
print("Testing compute_metrics()...")
metrics = compute_metrics(
    Z_val, y_val,
    result.W1, result.b1, result.W2, result.b2,
    n_classes=9,
    train_ce=result.train_loss_history[-1],
)
print(metrics.summary())
print("Confusion Matrix:")
print(metrics.confusion_str(9))

print()
print("Testing SVD & Stat Normalizer...")
pixel_syn = rng.standard_normal((100, 128 * 128)).astype(np.float32)
stat_syn = rng.uniform(0.1, 5.0, (100, 3)).astype(np.float32)
svd = _fit_svd(pixel_syn, stat_syn, n_features=64, random_state=42)
z_proj = svd.project(pixel_syn[:5], stat_syn[:5])
print(f"  Projected Z shape: {z_proj.shape}")
assert z_proj.shape == (5, 67)

print()
print("Testing Statistical Edge Feature Extraction...")
fake_edge = np.zeros((512, 512), dtype=np.uint8)
fake_edge[100:150, 100:200] = 255
fake_edge[300:350, 300:400] = 255
stats = _extract_stat_features(fake_edge)
print(f"  Extracted Stats [edge_area, log_contours, aspect_ratio]: {stats}")
assert len(stats) == 3

print()
print("Testing ModelParameters & BoxPredictor Inference...")
mock_params = ModelParameters(
    resize_target=(512, 512),
    blur_ksize=5,
    clip_percentile=98.0,
    image_size_svd=128,
    n_svd_features=64,
    svd_mean=svd.mean,
    svd_vt=svd.Vt,
    svd_s=svd.S,
    stat_mean=svd.stat_mean,
    stat_std=svd.stat_std,
    n_classes=9,
    hidden_dim=32,
    W1=result.W1,
    b1=result.b1,
    W2=result.W2,
    b2=result.b2,
)
predictor = BoxPredictor(mock_params)
fake_img = np.zeros((600, 800, 3), dtype=np.uint8)
fake_img[100:300, 200:500] = [120, 150, 180]
pred_result = predictor.predict(fake_img)
print(f"  Predicted Boxes: {pred_result.predicted_boxes} (confidence: {pred_result.confidence*100:.1f}%)")
print(f"  Class Probs len: {len(pred_result.class_probabilities)}")
print(f"  Logits len     : {len(pred_result.logits)}")
assert 1 <= pred_result.predicted_boxes <= 9
assert len(pred_result.class_probabilities) == 9
assert len(pred_result.logits) == 9
assert np.isclose(sum(pred_result.class_probabilities), 1.0, atol=1e-4)

print()
print("All classification tests passed successfully!")
