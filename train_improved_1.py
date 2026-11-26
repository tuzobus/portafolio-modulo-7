import time

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    IMPROVED_1_MODEL_PATH,
    TL_IMG_SIZE,
    BATCH_SIZE,
    TL_EPOCHS,
    TL_LEARNING_RATE,
    PATIENCE,
    CLASS_NAMES,
    SEED,
)
from evaluation import save_history, plot_training_curves, full_evaluation

# MODELO MEJORADO 1. TRANSFER LEARNING CON EFFICIENTNET B0 CONGELADA
# Se aplicaron cambios respecto al modelo base para combatir el overfitting
# 1. Base convolucional EfficientNet B0 preentrenada con ImageNet y congelada
# 2. GlobalAveragePooling2D en vez de Flatten
# 3. Dropout de 0.3 antes de la capa de salida
# 4. Data augmentation sobre las imágenes de entrenamiento
# 5. ModelCheckpoint y EarlyStopping

# Pesos preentrenados de ImageNet
WEIGHTS = "imagenet"

OUTPUT_DIR = RESULTS_DIR / "improved_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# GENERADORES DE DATOS
# EfficientNetB0 ya incluye capas de Rescaling y Normalization,
# así que espera los píxeles originales en el rango 0-255.

# Data augmentation para train y sin augmentation para eval
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


# ARQUITECTURA
conv_base = EfficientNetB0(
    weights=WEIGHTS,
    include_top=False,  # quita clasificador original de ImageNet
    input_shape=(TL_IMG_SIZE[0], TL_IMG_SIZE[1], 3),
)

# se congela toda la base y solo se entrena el clasificador nuevo
conv_base.trainable = False

model = models.Sequential(
    [
        layers.Input(shape=(TL_IMG_SIZE[0], TL_IMG_SIZE[1], 3)),
        conv_base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=TL_LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# CALLBACKS
# ModelCheckpoint guarda el modelo solo cuando val_loss mejora
# EarlyStopping detiene el entrenamiento si val_loss no mejora en 5 epochs
training_callbacks = [
    callbacks.ModelCheckpoint(
        IMPROVED_1_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
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
    epochs=TL_EPOCHS,
    validation_data=val_generator,
    callbacks=training_callbacks,
    verbose=2,
)

training_minutes = (time.time() - start_time) / 60
print(f"\nTiempo de entrenamiento: {training_minutes:.2f} min")

save_history(history.history, OUTPUT_DIR)
plot_training_curves(history.history, OUTPUT_DIR, "Mejorado 1")


# EVALUACIÓN DEL MEJOR MODELO GUARDADO: menor val_loss
best_model = models.load_model(IMPROVED_1_MODEL_PATH)
print(f"\nMejor modelo cargado desde: {IMPROVED_1_MODEL_PATH}")

full_evaluation(
    best_model,
    train_eval_generator,
    val_generator,
    test_generator,
    OUTPUT_DIR,
    "mejorado 1",
)
