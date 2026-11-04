import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

from config import BASE_MODEL_PATH, TEST_DIR, RESULTS_DIR, IMG_SIZE, CLASS_NAMES

# PREDICCIONES CON MODELO ENTRENADO
# uv run predict.py - predice 8 imágenes al azar del set de test

OUTPUT_DIR = RESULTS_DIR / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo en {BASE_MODEL_PATH}. Ejecuta primero train_base.py"
    )

model = models.load_model(BASE_MODEL_PATH)


# redimensiona a IMG_SIZE, convierte a arreglo, agrega la dimensión
# del lote y escala a 0-1
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor


# obtiene clase de imagen a partir de carpeta
def real_class_of(img_path):
    folder = Path(img_path).parent.name
    return folder if folder in CLASS_NAMES else None


def predict_image(img_path):
    img_tensor = load_and_preprocess(img_path)

    # probabilidad de cada una de las 9 clases
    confidence = model.predict(img_tensor, verbose=0)[0]
    predicted_index = int(np.argmax(confidence))
    top3 = np.argsort(confidence)[::-1][:3]

    real_class = real_class_of(img_path)

    print(f"\nImagen: {Path(img_path).name}")
    if real_class is not None:
        print(f"Clase real: {real_class}")
    print(
        f"Clase predicha: {CLASS_NAMES[predicted_index]} "
        f"(confianza {confidence[predicted_index] * 100:.2f} %)"
    )
    print("Top 3:")
    for k in top3:
        print(f"  {CLASS_NAMES[k]:<22} {confidence[k] * 100:6.2f} %")

    return img_tensor[0], confidence, predicted_index, real_class


# elige 8 imágenes al azar distintas de test en cada ejecución
all_test_images = sorted(p for p in TEST_DIR.rglob("*") if p.is_file())
chosen = np.random.choice(len(all_test_images), size=8, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(12, 7.5))
n_correct = 0

for ax, idx in zip(axes.ravel(), chosen):
    img_path = all_test_images[idx]
    img, confidence, predicted_index, real_class = predict_image(img_path)

    is_correct = CLASS_NAMES[predicted_index] == real_class
    n_correct += int(is_correct)

    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"Real: {real_class}\nPred: {CLASS_NAMES[predicted_index]} "
        f"({confidence[predicted_index] * 100:.1f} %)",
        fontsize=9,
        color="green" if is_correct else "red",
    )

print(f"\nAciertos: {n_correct} de {len(chosen)}")

plt.suptitle("Predicciones sobre imágenes al azar del set de test")
plt.tight_layout()
out_file = OUTPUT_DIR / "random_test_predictions.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Figura guardada en: {out_file}")

plt.show()
