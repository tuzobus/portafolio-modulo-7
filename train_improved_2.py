import time

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    RESULTS_DIR,
    IMPROVED_1_MODEL_PATH,
    IMPROVED_2_MODEL_PATH,
    TL_IMG_SIZE,
    BATCH_SIZE,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_FROM_LAYER,
    PATIENCE,
    CLASS_NAMES,
    SEED,
)
from evaluation import save_history, plot_training_curves, full_evaluation

# MODELO MEJORADO 2. FINE TUNING A PARTIR DEL MODELO MEJORADO 1
# Se carga el modelo mejorado 1 con su clasificador ya entrenado y se agrega:
# 1. Fine tuning descongelando los últimos bloques de EfficientNetB0
# y reentrenandolos con un learning rate menor
# 2. Pesos por clase

OUTPUT_DIR = RESULTS_DIR / "improved_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

if not IMPROVED_1_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró {IMPROVED_1_MODEL_PATH}. Ejecutar antes train_improved_1.py"
    )

if IMPROVED_2_MODEL_PATH.exists():
    IMPROVED_2_MODEL_PATH.unlink()


# GENERADORES DE DATOS
# Iguales que en el modelo mejorado 1
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
eval_datagen = ImageDataGenerator()

# Generador para train
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=TL_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=True,
    seed=SEED,
)

# Generadores para evaluar train, validation y test
train_eval_generator = eval_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=TL_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=False,
)

val_generator = eval_datagen.flow_from_directory(
    VAL_DIR,
    target_size=TL_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=False,
)

test_generator = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=TL_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=False,
)


# PESOS POR CLASE
# peso_c = n_total / (num_clases * n_c)
# las clases con menos imágenes reciben un peso mayor
class_counts = np.bincount(train_generator.classes, minlength=len(CLASS_NAMES))
n_total = class_counts.sum()
class_weight = {
    i: n_total / (len(CLASS_NAMES) * class_counts[i]) for i in range(len(CLASS_NAMES))
}

print("\nPesos por clase:")
for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"  {class_name:<12} {class_counts[i]:>4} imágenes -> peso {class_weight[i]:.3f}"
    )


# MODELO: SE CARGA EL MEJORADO 1 Y SE DESCONGELAN SUS ÚLTIMOS BLOQUES
model = models.load_model(IMPROVED_1_MODEL_PATH)
conv_base = model.layers[0]  # la efficientnet b0 del Sequential

conv_base.trainable = True
unfreeze = False
for layer in conv_base.layers:
    if layer.name == FINE_TUNE_FROM_LAYER:
        unfreeze = True
    # BatchNormalization siempre congelada
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = unfreeze

n_trainable_layers = sum(1 for layer in conv_base.layers if layer.trainable)
print(
    f"\nCapas de EfficientNet B0 descongeladas: {n_trainable_layers} "
    f"de {len(conv_base.layers)} (desde {FINE_TUNE_FROM_LAYER})"
)

model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Punto de partida del modelo mejorado 1
start_loss, start_acc = model.evaluate(val_generator, verbose=0)
print(
    f"\nPunto de partida (Mejorado 1) -> val_loss: {start_loss:.4f}, val_accuracy: {start_acc:.4f}"
)


# CALLBACKS
# initial_value_threshold solo guarda un modelo si mejora el val_loss del
# modelo Mejorado 1. Si el fine tuning no lo mejora entonces no se guarda nada
training_callbacks = [
    callbacks.ModelCheckpoint(
        IMPROVED_2_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        initial_value_threshold=start_loss,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ENTRENAMIENTO
start_time = time.time()

history = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=training_callbacks,
    verbose=2,
)

training_minutes = (time.time() - start_time) / 60
print(f"\nTiempo de entrenamiento: {training_minutes:.2f} min")

save_history(history.history, OUTPUT_DIR)
plot_training_curves(history.history, OUTPUT_DIR, "Mejorado 2 (fine-tuning)")

if not IMPROVED_2_MODEL_PATH.exists():
    print(
        "\nEl fine-tuning no mejoró el val_loss del Mejorado 1, "
        "así que no se guardó un Mejorado 2."
    )
else:
    best_model = models.load_model(IMPROVED_2_MODEL_PATH)
    print(f"\nMejor modelo cargado desde: {IMPROVED_2_MODEL_PATH}")

    full_evaluation(
        best_model,
        train_eval_generator,
        val_generator,
        test_generator,
        OUTPUT_DIR,
        "mejorado 2",
    )
