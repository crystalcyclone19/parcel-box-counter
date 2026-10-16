# model/training/__init__.py
"""
Automated Training & Model Selection Package
=============================================
โมดูลสำหรับระบบ End-to-End Automated Training Pipeline

ประกอบด้วย:
  dataset.py       — โหลด Label, จับคู่ภาพ-Label, แบ่ง Train/Val/Test
  trainer.py       — Training Loop, Early Stopping, Validation
  tuner.py         — Hyperparameter Grid Search, Trial Execution
  evaluator.py     — Metrics (MSE, MAE, R², Accuracy±1)
  model_registry.py — Champion Model Export & Leaderboard
"""

from .dataset import load_dataset, split_dataset, DatasetSplit
from .trainer import Trainer, TrainResult
from .tuner import GridSearchTuner, build_search_space
from .evaluator import Evaluator, compute_metrics, MetricsResult
from .model_registry import ModelRegistry

__all__ = [
    "load_dataset",
    "split_dataset",
    "DatasetSplit",
    "Trainer",
    "TrainResult",
    "GridSearchTuner",
    "build_search_space",
    "Evaluator",
    "compute_metrics",
    "MetricsResult",
    "ModelRegistry",
]
