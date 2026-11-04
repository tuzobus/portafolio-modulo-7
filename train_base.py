import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    BASE_MODEL_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    CLASS_NAMES,
    SEED,
)

# ENTRENAMIENTO DE MODELO BASE
# CNN entrenada desde cero

OUTPUT_DIR = RESULTS_DIR / "base"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# GENERADORES DE DATOS
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generador para train
# Hace shuffle de las imágenes de train en cada epoch
# y escala los pixeles a 0-1
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
# No hacen shuffle para poder calcular la matriz de confusión
# y el reporte de clasificación correctamente
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

print("\nIndice asignado a cada clase:")
print(train_generator.class_indices)


# ARQUITECTURA DEL MODELO BASE
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

model.summary()

# Tabla de capas
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

# se usa sparse_categorical_crossentropy porque las etiquetas son enteros
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ENTRENAMIENTO
start_time = time.time()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=2,
)

training_minutes = (time.time() - start_time) / 60
print(f"\nTiempo de entrenamiento: {training_minutes:.2f} min")

model.save(BASE_MODEL_PATH)
print(f"Modelo guardado en: {BASE_MODEL_PATH}")

history_df = pd.DataFrame(history.history)
history_df.index = history_df.index + 1
history_df.index.name = "epoch"
history_df.to_csv(OUTPUT_DIR / "history.csv")


# CURVAS DE ENTRENAMIENTO
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(epochs, acc, "bo", label="train accuracy")
axes[0].plot(epochs, val_acc, "b", label="validation accuracy")
axes[0].set_title("Accuracy de entrenamiento vs validación")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(epochs, loss, "bo", label="training loss")
axes[1].plot(epochs, val_loss, "b", label="validation loss")
axes[1].set_title("Loss de entrenamiento vs validación")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss (entropía cruzada)")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches="tight")
plt.close()

# epoch con menor loss de validación
best_loss_epoch = int(np.argmin(val_loss)) + 1
best_acc_epoch = int(np.argmax(val_acc)) + 1

print("\nDiagnóstico de las curvas:")
print(f"Menor val_loss: {min(val_loss):.4f} en la epoch {best_loss_epoch}")
print(f"val_loss en la última epoch: {val_loss[-1]:.4f}")
print(f"Mayor val_accuracy: {max(val_acc):.4f} en la epoch {best_acc_epoch}")
print(f"train accuracy en la última epoch: {acc[-1]:.4f}")


# EVALUACIÓN EN TRAIN , VALIDATION Y TEST
def plot_confusion_matrix(cm, title, file_name):
    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    # Etiquetas del eje x inclinadas y alineadas a la derecha para que
    # los nombres largos no se encimen
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_split(generator, split_name):
    split_loss, _ = model.evaluate(generator, verbose=0)

    probabilities = model.predict(generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = generator.classes

    split_accuracy = accuracy_score(y_true, y_pred)
    split_macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{split_name.upper()}")
    print(f"Loss: {split_loss:.4f}")
    print(f"Accuracy: {split_accuracy:.4f}")
    print(f"Macro F1: {split_macro_f1:.4f}")

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().round(4).to_csv(
        OUTPUT_DIR / f"classification_report_{split_name}.csv"
    )

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / f"confusion_matrix_{split_name}.csv"
    )

    return {
        "split": split_name,
        "loss": split_loss,
        "accuracy": split_accuracy,
        "macro_f1": split_macro_f1,
    }, cm


print("\n==========")
print("EVALUACIÓN DEL MODELO BASE")
print("==========")

train_metrics, train_cm = evaluate_split(train_eval_generator, "train")
val_metrics, val_cm = evaluate_split(val_generator, "validation")

# el set de test solo se reporta como referencia
print("\n(Test de referencia)")
test_metrics, test_cm = evaluate_split(test_generator, "test")

plot_confusion_matrix(
    val_cm, "Matriz de confusión - Validación", "confusion_matrix_validation.png"
)
plot_confusion_matrix(
    test_cm, "Matriz de confusión - Test", "confusion_matrix_test.png"
)

metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
metrics_df.round(4).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

print("\nResumen de métricas:")
print(metrics_df.round(4).to_string(index=False))


# DIAGNÓSTICO DE SESGO/VARIANZA
gap_f1 = train_metrics["macro_f1"] - val_metrics["macro_f1"]
gap_acc = train_metrics["accuracy"] - val_metrics["accuracy"]
gap_loss = val_metrics["loss"] - train_metrics["loss"]

print("\nBrecha de generalización (train - validation):")
print(f"Macro F1: {gap_f1:.4f}")
print(f"Accuracy: {gap_acc:.4f}")
print(f"Loss: {gap_loss:.4f}")

# pares de clases que más se confunden en val
confusions = []
for i, real_class in enumerate(CLASS_NAMES):
    for j, pred_class in enumerate(CLASS_NAMES):
        if i != j and val_cm[i, j] > 0:
            confusions.append((val_cm[i, j], real_class, pred_class))

confusions.sort(reverse=True)

print("\nConfusiones más frecuentes en validación :")
for count, real_class, pred_class in confusions[:8]:
    print(f"Real: {real_class}, predicha: {pred_class} ({count} veces)")

print(f"\nResultados guardados en: {OUTPUT_DIR}")
