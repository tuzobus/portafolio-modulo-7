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
REGULARIZED_MODEL_PATH = MODELS_DIR / "regularized_cnn.keras"
IMPROVED_1_MODEL_PATH = MODELS_DIR / "improved_1_efficientnet.keras"
IMPROVED_2_MODEL_PATH = MODELS_DIR / "improved_2_finetuned.keras"

# modelo base y CNN regularizada
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25

# CNN regularizada: máximo de epochs (early stopping puede terminar antes)
REG_EPOCHS = 50
REG_PATIENCE = 10

# transfer learning con EfficientNetB0 (tamaño nativo de 224x224)
TL_IMG_SIZE = (224, 224)

TL_EPOCHS = 30
TL_LEARNING_RATE = 1e-3

FINE_TUNE_EPOCHS = 20
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_FROM_LAYER = "block6a_expand_conv"

PATIENCE = 5

# modelos disponibles en predict.py y en la app
# img_size y rescale indican cómo preparar una imagen para cada modelo:
# - las CNN propias reciben píxeles escalados a 0-1 (rescale = 1/255)
# - EfficientNet trae su propia normalización dentro del modelo,
#   así que recibe los píxeles originales en 0-255 (rescale = None)
MODELS = {
    "base": {
        "name": "CNN base",
        "path": BASE_MODEL_PATH,
        "img_size": IMG_SIZE,
        "rescale": 1.0 / 255,
    },
    "regularized": {
        "name": "CNN regularizada",
        "path": REGULARIZED_MODEL_PATH,
        "img_size": IMG_SIZE,
        "rescale": 1.0 / 255,
    },
    "improved_1": {
        "name": "EfficientNetB0 (transfer learning)",
        "path": IMPROVED_1_MODEL_PATH,
        "img_size": TL_IMG_SIZE,
        "rescale": None,
    },
    "improved_2": {
        "name": "EfficientNetB0 + fine-tuning",
        "path": IMPROVED_2_MODEL_PATH,
        "img_size": TL_IMG_SIZE,
        "rescale": None,
    },
}

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
