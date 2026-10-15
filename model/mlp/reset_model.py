import json
import os

MODEL_STORE_PATH = os.path.join(os.path.dirname(__file__), "model_store.json")

def reset_model_store(path: str = MODEL_STORE_PATH) -> None:
    """
    Reset model_store.json back to default initial state:
    - Clears W and b to None (so mlp_regression.py initializes new weights)
    - Resets version and epoch to 0
    - Clears all training history
    - Clears last_loss
    """
    initial_store = {
        "model_name": "mlp_parcel_counter",
        "description": "Single-layer MLP for counting parcels in 8x8 feature maps",
        "architecture": {
            "input_size": 64,
            "output_size": 1,
            "activation": "scaled_sigmoid",
            "k": 10.0
        },
        "training_config": {
            "learning_rate": 0.1,
            "batch_size": 4,
            "loss_fn": "MSE"
        },
        "version": 0,
        "epoch": 0,
        "last_loss": None,
        "history": [],
        "W": None,
        "b": None
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(initial_store, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("[SUCCESS] Model store has been reset to initial state!")
    print(f"File path : {path}")
    print(" - version   : 0")
    print(" - epoch     : 0")
    print(" - last_loss : None")
    print(" - history   : [] (all past runs cleared)")
    print(" - W, b      : None (will re-initialize on next training run)")
    print("=" * 60)

if __name__ == "__main__":
    reset_model_store()
