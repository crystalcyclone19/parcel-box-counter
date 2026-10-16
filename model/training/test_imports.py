"""Quick import test for all training modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mlp'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

print("Testing imports...")
from dataset import load_dataset, split_dataset, DatasetSplit, extract_edge_features
print("  [OK] dataset.py")
from trainer import Trainer, TrainResult, _init_weights_xavier
print("  [OK] trainer.py")
from evaluator import Evaluator, compute_metrics, MetricsResult
print("  [OK] evaluator.py")
from tuner import GridSearchTuner, build_search_space, DEFAULT_SEARCH_SPACE
print("  [OK] tuner.py")
from model_registry import ModelRegistry
print("  [OK] model_registry.py")

print()
print("Testing Search Space...")
sp = build_search_space()
n_trials = len(sp["clip_percentile"]) * len(sp["learning_rate"]) * len(sp["batch_size"])
print(f"  Total Trials : {n_trials}")
print(f"  clip_pct     : {sp['clip_percentile']}")
print(f"  learning_rate: {sp['learning_rate']}")
print(f"  batch_size   : {sp['batch_size']}")

print()
print("Testing Xavier Init...")
rng = np.random.default_rng(42)
W, b = _init_weights_xavier(64, rng)
print(f"  W shape: {W.shape}, std: {W.std():.4f}")
print(f"  b shape: {b.shape}")

print()
print("Testing Trainer.fit() with synthetic data...")
n, d = 20, 64
Z_train = np.random.randn(n, d).astype(np.float32)
y_train = np.random.randint(1, 6, (n, 1)).astype(np.float32)
Z_val   = np.random.randn(5, d).astype(np.float32)
y_val   = np.random.randint(1, 6, (5, 1)).astype(np.float32)

trainer = Trainer(k=10.0, max_epochs=50, patience=10, random_seed=42, verbose=10)
result = trainer.fit(Z_train, y_train, Z_val, y_val, learning_rate=0.01, batch_size=8)
print(f"  Best Val MSE : {result.best_val_mse:.6f} @ epoch {result.best_epoch}")
print(f"  Stopped Early: {result.stopped_early}")
print(f"  W shape: {result.W.shape}")

print()
print("Testing compute_metrics()...")
metrics = compute_metrics(Z_val, y_val, result.W, result.b, k=10.0, train_mse=result.train_loss_history[-1])
print(metrics.summary())

print()
print("All tests passed!")
