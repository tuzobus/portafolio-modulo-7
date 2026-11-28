import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import CLASS_NAMES

# FUNCIONES DE EVALUACIÓN PARA MODELOS MEJORADOS
# Iguales que las de train_base.py pero reciben el modelo y la carpeta
# de resultados como parámetros para reutilizarlas en modelos mejorados


def save_history(history_dict, output_dir):
    history_df = pd.DataFrame(history_dict)
    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_df.to_csv(output_dir / "history.csv")


def plot_training_curves(history_dict, output_dir, title):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, acc, "bo", label="train accuracy")
    axes[0].plot(epochs, val_acc, "b", label="validation accuracy")
    axes[0].set_title(f"Accuracy - {title}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, loss, "bo", label="training loss")
    axes[1].plot(epochs, val_loss, "b", label="validation loss")
    axes[1].set_title(f"Loss - {title}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (entropía cruzada)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    best_loss_epoch = int(np.argmin(val_loss)) + 1
    best_acc_epoch = int(np.argmax(val_acc)) + 1

    print("\nDiagnóstico de las curvas:")
    print(f"Epochs entrenadas: {len(acc)}")
    print(f"Menor val_loss: {min(val_loss):.4f} en la epoch {best_loss_epoch}")
    print(f"val_loss en la última epoch: {val_loss[-1]:.4f}")
    print(f"Mayor val_accuracy: {max(val_acc):.4f} en la epoch {best_acc_epoch}")
    print(f"train accuracy en la última epoch: {acc[-1]:.4f}")

    return best_loss_epoch


def plot_confusion_matrix(cm, title, file_path):
    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_split(model, generator, split_name, output_dir):
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
        output_dir / f"classification_report_{split_name}.csv"
    )

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / f"confusion_matrix_{split_name}.csv"
    )

    return {
        "split": split_name,
        "loss": split_loss,
        "accuracy": split_accuracy,
        "macro_f1": split_macro_f1,
    }, cm


def full_evaluation(model, train_eval_gen, val_gen, test_gen, output_dir, model_name):
    print("\n==========")
    print(f"EVALUACIÓN DEL MODELO {model_name.upper()}")
    print("==========")

    train_metrics, _ = evaluate_split(model, train_eval_gen, "train", output_dir)
    val_metrics, val_cm = evaluate_split(model, val_gen, "validation", output_dir)

    # el set de test solo se reporta como referencia
    print("\n(Test de referencia)")
    test_metrics, test_cm = evaluate_split(model, test_gen, "test", output_dir)

    plot_confusion_matrix(
        val_cm,
        f"Matriz de confusión - Validación ({model_name})",
        output_dir / "confusion_matrix_validation.png",
    )
    plot_confusion_matrix(
        test_cm,
        f"Matriz de confusión - Test ({model_name})",
        output_dir / "confusion_matrix_test.png",
    )

    metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    metrics_df.round(4).to_csv(output_dir / "metrics.csv", index=False)

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

    print("\nConfusiones más frecuentes en validación:")
    for count, real_class, pred_class in confusions[:8]:
        print(f"Real: {real_class}, predicha: {pred_class} ({count} veces)")

    print(f"\nResultados guardados en: {output_dir}")
