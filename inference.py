from functools import cache
from pathlib import Path

import numpy as np

from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

from config import CLASS_NAMES, IMG_SIZE, MODELS

# FUNCIONES DE PREDICCION para predict.py y app.py


# redimensiona a IMG_SIZE, convierte a arreglo, agrega la dimensión
# del lote y escala a 0-1
def load_and_preprocess(img_source):
    img = image.load_img(img_source, target_size=IMG_SIZE)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor


# obtiene clase de imagen a partir de carpeta
def real_class_of(img_path):
    folder = Path(img_path).parent.name
    return folder if folder in CLASS_NAMES else None


# modelos de config.MODELS cuyo archivo ya existe
def available_models():
    return {
        model_id: info for model_id, info in MODELS.items() if info["path"].exists()
    }


@cache
def get_model(model_id):
    return models.load_model(MODELS[model_id]["path"])
