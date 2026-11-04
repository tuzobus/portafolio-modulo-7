from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SEED = 67

RAW_DATA_DIR = BASE_DIR / "data" / "Mushrooms"

SPLIT_DIR = BASE_DIR / "data" / "split"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

BASE_MODEL_PATH = MODELS_DIR / "base_cnn.keras"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25

CLASS_NAMES = [
    "Agaricus",
    "Amanita",
    "Boletus",
    "Cortinarius",
    "Entoloma",
    "Hygrocybe",
    "Lactarius",
    "Russula",
    "Suillus",
]
