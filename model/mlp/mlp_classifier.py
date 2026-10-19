# model/mlp/mlp_classifier.py
"""
Multi-Layer Perceptron — Multi-class Classifier
================================================
สถาปัตยกรรม 2-Layer MLP สำหรับจำแนกจำนวนกล่องพัสดุ 1–9 กล่อง
(Multi-class Classification ด้วย Softmax + Cross-Entropy Loss)

โครงสร้าง:
  Input → Linear (input_dim → hidden_dim) → ReLU → Dropout →
  Linear (hidden_dim → n_classes) → Softmax

Output:
  - Softmax probabilities : shape (N, n_classes)
  - Predicted class index : argmax + 1  (เพื่อให้ตรงกับจำนวนกล่อง 1-9)
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically-stable Row-wise Softmax"""
    z_shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_weights_xavier(
    input_dim: int,
    hidden_dim: int,
    n_classes: int,
    rng: np.random.Generator,
):
    """Xavier/Glorot Normal Initialization สำหรับ 2-Layer MLP

    Returns (W1, b1, W2, b2)
    """
    std1 = np.sqrt(2.0 / (input_dim + hidden_dim))
    std2 = np.sqrt(2.0 / (hidden_dim + n_classes))

    W1 = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) * std1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.standard_normal((hidden_dim, n_classes)).astype(np.float32) * std2
    b2 = np.zeros(n_classes, dtype=np.float32)

    return W1, b1, W2, b2


# ─────────────────────────────────────────────────────────────────────────────
# Forward Pass
# ─────────────────────────────────────────────────────────────────────────────

def forward(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    dropout_rate: float = 0.0,
    training: bool = True,
    rng=None,
):
    """2-Layer MLP Forward Pass

    Returns (probs, h, z1, mask)
      probs : (N, n_classes) Softmax probabilities
      h     : (N, hidden_dim) post-ReLU hidden layer
      z1    : (N, hidden_dim) pre-ReLU hidden layer
      mask  : (N, hidden_dim) dropout mask
    """
    z1 = X @ W1 + b1
    h  = relu(z1)

    if training and dropout_rate > 0.0 and rng is not None:
        keep_prob = 1.0 - dropout_rate
        mask = (rng.random(h.shape) < keep_prob).astype(np.float32) / keep_prob
    else:
        mask = np.ones_like(h)

    h_drop = h * mask
    z2     = h_drop @ W2 + b2
    probs  = softmax(z2)

    return probs, h, z1, mask


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Categorical Cross-Entropy Loss (average per sample)

    Parameters
    ----------
    probs : (N, C) Softmax probabilities
    y_idx : (N,)   True class indices (0-indexed)
    """
    N = probs.shape[0]
    p_correct = np.clip(probs[np.arange(N), y_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p_correct)))


# ─────────────────────────────────────────────────────────────────────────────
# Backward Pass
# ─────────────────────────────────────────────────────────────────────────────

def backward(
    X: np.ndarray,
    probs: np.ndarray,
    y_idx: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    h: np.ndarray,
    z1: np.ndarray,
    mask: np.ndarray,
):
    """Backpropagation สำหรับ 2-Layer MLP + Cross-Entropy

    Returns (dW1, db1, dW2, db2)
    """
    N = X.shape[0]

    dz2 = probs.copy()
    dz2[np.arange(N), y_idx] -= 1.0
    dz2 /= N

    h_drop = h * mask
    dW2 = h_drop.T @ dz2
    db2 = dz2.sum(axis=0)

    dh_drop = dz2 @ W2.T
    dh      = dh_drop * mask
    dz1     = dh * relu_grad(z1)

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return dW1, db1, dW2, db2


# ─────────────────────────────────────────────────────────────────────────────
# Train Step
# ─────────────────────────────────────────────────────────────────────────────

def train_step(
    X: np.ndarray,
    y_idx: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    learning_rate: float,
    dropout_rate: float = 0.3,
    grad_clip: float = 5.0,
    rng=None,
):
    """1 Mini-batch Forward + Backward + SGD Update

    Returns (W1, b1, W2, b2, loss)
    """
    probs, h, z1, mask = forward(
        X, W1, b1, W2, b2,
        dropout_rate=dropout_rate,
        training=True,
        rng=rng,
    )
    loss = cross_entropy_loss(probs, y_idx)
    dW1, db1, dW2, db2 = backward(X, probs, y_idx, W1, W2, h, z1, mask)

    # Gradient Clipping
    for g in [dW1, db1, dW2, db2]:
        np.clip(g, -grad_clip, grad_clip, out=g)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2, loss


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
):
    """Inference-only Forward Pass (no dropout)

    Returns (probs, class_idx)
      probs     : (N, n_classes) Softmax probabilities
      class_idx : (N,) 0-indexed predicted class
    """
    probs, _, _, _ = forward(X, W1, b1, W2, b2, dropout_rate=0.0, training=False)
    class_idx = np.argmax(probs, axis=1)
    return probs, class_idx
