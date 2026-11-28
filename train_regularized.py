import time

import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    REGULARIZED_MODEL_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    REG_EPOCHS,
    REG_PATIENCE,
    CLASS_NAMES,
    SEED,
)
from evaluation import save_history, plot_training_curves, full_evaluation

# CNN REGULARIZADA
# CNN entrenada desde cero, 150x150, Adam. Incluye cambios para
# combatir el overfitting diagnosticado en el modelo base:
# 1. Un cuarto bloque convolucional (128 filtros)
# 2. Dropout de 0.5 antes de la capa densa
# 3. Data augmentation sobre las imágenes de entrenamiento
# 4. ModelCheckpoint y EarlyStopping

OUTPUT_DIR = RESULTS_DIR / "regularized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# GENERADORES DE DATOS
# Data augmentation para train y sin augmentation para eval
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generador para train
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=True,
    seed=SEED,
)

# Generadores para evaluar train, validation y test
train_eval_generator = eval_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=False,
)

val_generator = eval_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=False,
)

test_generator = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASS_NAMES,
    shuffle=False,
)


# ARQUITECTURA
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

model.summary()

# tabla de capas
layer_rows = []
for layer in model.layers:
    layer_rows.append(
        {
            "layer": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(tuple(layer.output.shape)),
            "params": layer.count_params(),
        }
    )

pd.DataFrame(layer_rows).to_csv(OUTPUT_DIR / "model_layers.csv", index=False)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# CALLBACKS
# ModelCheckpoint guarda el modelo solo cuando val_loss mejora
# EarlyStopping detiene el entrenamiento si val_loss no mejora en 10 epochs
training_callbacks = [
    callbacks.ModelCheckpoint(
        REGULARIZED_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=REG_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ENTRENAMIENTO
start_time = time.time()

history = model.fit(
    train_generator,
    epochs=REG_EPOCHS,
    validation_data=val_generator,
    callbacks=training_callbacks,
    verbose=2,
)

training_minutes = (time.time() - start_time) / 60
print(f"\nTiempo de entrenamiento: {training_minutes:.2f} min")

save_history(history.history, OUTPUT_DIR)
plot_training_curves(history.history, OUTPUT_DIR, "CNN regularizada")


# EVALUACIÓN DEL MEJOR MODELO GUARDADO: menor val_loss
best_model = models.load_model(REGULARIZED_MODEL_PATH)
print(f"\nMejor modelo cargado desde: {REGULARIZED_MODEL_PATH}")

full_evaluation(
    best_model,
    train_eval_generator,
    val_generator,
    test_generator,
    OUTPUT_DIR,
    "CNN regularizada",
)
