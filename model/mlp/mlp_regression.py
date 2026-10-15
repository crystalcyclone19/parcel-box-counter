import numpy as np
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_STORE_PATH = os.path.join(os.path.dirname(__file__), "model_store.json")


# ─────────────────────────────────────────────
# MODEL STORE UTILITIES
# ─────────────────────────────────────────────
def load_model_store(path: str) -> dict:
    """Load the model store JSON from disk."""
    with open(path, "r") as f:
        return json.load(f)


def save_model_store(path: str, store: dict) -> None:
    """Persist the model store JSON to disk."""
    with open(path, "w") as f:
        json.dump(store, f, indent=2)
    print(f"\n[ModelStore] Saved to '{path}' (version={store['version']}, epoch={store['epoch']})")


def get_params(store: dict, input_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (W, b) from the store.
    If the store has no saved weights (null), initialize them randomly.
    """
    if store["W"] is None or store["b"] is None:
        print("[ModelStore] No saved weights found — initialising fresh parameters.")
        W = np.random.randn(input_size, 1) * 0.01
        b = np.zeros((1,))
    else:
        W = np.array(store["W"])
        b = np.array(store["b"])
        print(f"[ModelStore] Loaded weights from version {store['version']}, epoch {store['epoch']}.")
    return W, b


def update_store(store: dict, W_new: np.ndarray, b_new: np.ndarray, loss: float) -> dict:
    """
    Bump version counter, record epoch, update W/b and append loss to history.
    """
    store["version"] += 1
    store["epoch"]   += 1
    store["last_loss"] = round(float(loss), 8)
    store["W"] = W_new.tolist()
    store["b"] = b_new.tolist()
    store["history"].append({
        "version":   store["version"],
        "epoch":     store["epoch"],
        "loss":      store["last_loss"],
        "timestamp": datetime.now().isoformat()
    })
    return store


# ─────────────────────────────────────────────
# ACTIVATION
# ─────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ─────────────────────────────────────────────
# MAIN TRAINING STEP
# ─────────────────────────────────────────────
def train_step(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    k: float,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Perform one forward pass + backward pass.

    Returns
    -------
    y_hat        : predictions  shape (batch, 1)
    W_updated    : new weights  shape (input_size, 1)
    b_updated    : new bias     shape (1,)
    loss         : scalar MSE loss
    """
    batch_size = X.shape[0]

    # ── Forward pass ──────────────────────────
    linear_output = np.dot(X, W) + b          # (batch, 1)
    sig_z         = sigmoid(linear_output)
    y_hat         = k * sig_z                 # scaled sigmoid output ∈ [0, k]

    # ── Loss (MSE) ────────────────────────────
    loss = 0.5 * np.mean((y_hat - y) ** 2)

    # ── Backward pass ─────────────────────────
    dL_dy_hat = (y_hat - y) / batch_size                  # ∂L/∂ŷ
    dy_hat_dz = k * sig_z * (1.0 - sig_z)                # ∂ŷ/∂z  (chain through scaled sigmoid)
    dL_dz     = dL_dy_hat * dy_hat_dz                    # ∂L/∂z

    dW = np.dot(X.T, dL_dz)                              # ∂L/∂W
    db = np.sum(dL_dz, axis=0)                           # ∂L/∂b

    # ── Gradient descent update ───────────────
    W_updated = W - learning_rate * dW
    b_updated = b - learning_rate * db

    return y_hat, W_updated, b_updated, loss


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load model store
    store = load_model_store(MODEL_STORE_PATH)

    # Read hyper-parameters from store
    m_dim     = store["architecture"]["input_size"]   # 64 (8×8 flattened)
    k         = store["architecture"]["k"]            # 10.0
    lr        = store["training_config"]["learning_rate"]
    batch_size = store["training_config"]["batch_size"]

    # 2. Build input data (same fixed seed so results are reproducible per session)
    np.random.seed(42)
    X_raw = np.random.rand(batch_size, 8, 8)
    X     = X_raw.reshape(batch_size, m_dim)

    print(f"Raw input batch shape  : {X_raw.shape}")
    print(f"Flattened input shape  : {X.shape}")

    # 3. Load (or initialise) parameters
    W, b = get_params(store, input_size=m_dim)

    # 4. Ground-truth parcel counts for this batch
    y = np.array([[3.0], [7.0], [5.0], [2.0]])

    # 5. One training step
    y_hat, W_new, b_new, loss = train_step(X, y, W, b, k=k, learning_rate=lr)

    # 6. Print results
    print(f"\nGround truth (y)       :\n{y.ravel()}")
    print(f"Predictions  (y_hat)   :\n{y_hat.ravel()}")
    print(f"MSE Loss               : {loss:.6f}")
    print(f"\nGradient-descent update (first 5 W elements):")
    print(f"  W before: {W[:5].ravel()}")
    print(f"  W after : {W_new[:5].ravel()}")
    print(f"  b before: {b}")
    print(f"  b after : {b_new}")

    # 7. Persist updated parameters back to model store
    store = update_store(store, W_new, b_new, loss)
    save_model_store(MODEL_STORE_PATH, store)