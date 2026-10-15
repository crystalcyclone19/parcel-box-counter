# config.py — ตั้งค่าระบบกลาง (Paths, Constants, Hyperparameters)
from pathlib import Path

# ─── Drive / Storage ──────────────────────────────────────────────────────────
SAVE_TO_DRIVE = False  # True = บันทึกถาวรที่ MyDrive/Box_Project

def get_base_dir() -> Path:
    """คืน BASE_DIR ตามค่า SAVE_TO_DRIVE (mount Drive ถ้าจำเป็น)"""
    if SAVE_TO_DRIVE:
        from google.colab import drive  # type: ignore[import]
        drive.mount('/content/drive')
        return Path('/content/drive/MyDrive/Box_Project')
    return Path('/content/Box_Project')


BASE_DIR = get_base_dir()

# ─── Sub-directories ──────────────────────────────────────────────────────────
INPUT_DIR  = BASE_DIR / 'input_images'
GRAY_DIR   = BASE_DIR / 'Grayscale'
CONV_DIR   = BASE_DIR / 'Convolution'
SVD_DIR    = BASE_DIR / 'output_svd'
MATRIX_DIR = SVD_DIR  / 'matrices'

ALL_DIRS = (INPUT_DIR, GRAY_DIR, CONV_DIR, SVD_DIR, MATRIX_DIR)


def ensure_dirs() -> None:
    """สร้างโฟลเดอร์ทั้งหมดถ้ายังไม่มี"""
    for folder in ALL_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


# ─── Image constants ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set = {'.jpg', '.jpeg', '.png'}
IMAGE_SIZE   = 128      # resize เป็น IMAGE_SIZE × IMAGE_SIZE ก่อน flatten
N_FEATURES   = 64       # จำนวน SVD components
MAX_PLOTS    = 5        # None = แสดงทุกรูป
RANDOM_STATE = 42

# ─── Geometric transform parameters ──────────────────────────────────────────
TRANSLATE_X: float = 50.0   # pixel shift แกน X
TRANSLATE_Y: float = 30.0   # pixel shift แกน Y
SCALE_FACTOR: float = 1.2   # อัตราส่วนขยาย
ROTATION_DEG: float = 25.0  # องศาหมุน (ทวนเข็มนาฬิกา)
