# model/predict.py
"""
Wrapper for predict.py located at root
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predict import main, BoxPredictor, ModelParameters, PredictionOutput, load_image

if __name__ == "__main__":
    main()
