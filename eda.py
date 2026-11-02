import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from config import RAW_DATA_DIR, RESULTS_DIR, CLASS_NAMES, SEED

EDA_RESULTS_DIR = RESULTS_DIR / "eda"
EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# cada imagen se reduce a 64x64 para obtener imagen promedio y estadística de color
MEAN_SIZE = (64, 64)


# EDA Y REVISIÓN DE CALIDAD DE LOS DATOS
# Se recorre cada imagen para contar cuántas tiene cada clase,
# detectar archivos corruptos, revisar tamaño y modo de color,
# calcular hash md5 para detectar duplicados,
# y acumular la imagen promedio y la media / desviación de cada canal por clase.

records = []
corrupt_files = []
non_jpg_files = []
mean_images = {}
channel_stats = []

for class_name in CLASS_NAMES:
    class_dir = RAW_DATA_DIR / class_name
    files = sorted(p for p in class_dir.iterdir() if p.is_file())
    print(f"Revisando clase {class_name}")

    pixel_sum = np.zeros((MEAN_SIZE[1], MEAN_SIZE[0], 3), dtype=np.float64)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_valid = 0

    for path in files:
        if path.suffix.lower() not in [".jpg", ".jpeg"]:
            non_jpg_files.append(str(path))
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                arr = np.asarray(img.convert("RGB"), dtype=np.float64)
        except Exception:
            corrupt_files.append(str(path))
            continue

        md5 = hashlib.md5(path.read_bytes()).hexdigest()

        # Escala de grises con luminancia perceptual para brillo y contraste
        gray = arr @ np.array([0.299, 0.587, 0.114])

        records.append(
            {
                "class": class_name,
                "file": path.name,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "mode": mode,
                "md5": md5,
                "brightness": gray.mean() / 255.0,
                "contrast": gray.std() / 255.0,
            }
        )

        # Se reduce a 64x64 para el promedio y la estadística de color
        arr = np.asarray(
            Image.fromarray(arr.astype(np.uint8)).resize(MEAN_SIZE),
            dtype=np.float64,
        )

        pixel_sum += arr
        channel_sum += arr.reshape(-1, 3).sum(axis=0)
        channel_sq_sum += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        n_valid += 1

    n_pixels = n_valid * MEAN_SIZE[0] * MEAN_SIZE[1]
    channel_mean = channel_sum / n_pixels
    channel_std = np.sqrt(channel_sq_sum / n_pixels - channel_mean**2)

    mean_images[class_name] = pixel_sum / n_valid
    channel_stats.append(
        {
            "class": class_name,
            "n_images": n_valid,
            "mean_R": channel_mean[0],
            "mean_G": channel_mean[1],
            "mean_B": channel_mean[2],
            "std_R": channel_std[0],
            "std_G": channel_std[1],
            "std_B": channel_std[2],
        }
    )

df = pd.DataFrame(records)
stats_df = pd.DataFrame(channel_stats)

print("\nInfo del dataset:")
df.info()

print("\nTotal de imágenes válidas:")
print(len(df))

print("\nArchivos no JPG:")
print(len(non_jpg_files), non_jpg_files[:10])

print("\nImágenes corruptas:")
print(len(corrupt_files))
for path in corrupt_files:
    print(path)

# se resume la distribución de ancho, alto y proporción de las imágenes
print("\nTamaños de imagen distintos:")
print(df.groupby(["width", "height"]).ngroups)

print("\nResumen de ancho, alto y proporción:")
print(df[["width", "height", "aspect_ratio"]].describe().round(2))

print("\nImágenes más pequeñas que 150px en ancho o alto:")
print(int(((df["width"] < 150) | (df["height"] < 150)).sum()))

# se muestra qué tan variadas son las resoluciones y proporciones
# que se deformarán al redimensionar a 1:1
plt.figure(figsize=(7, 6))
plt.scatter(df["width"], df["height"], s=6, alpha=0.3)
max_side = max(df["width"].max(), df["height"].max())
plt.plot([0, max_side], [0, max_side], "k--", linewidth=1, label="imagen 1:1")
plt.xlabel("Ancho px")
plt.ylabel("Alto px")
plt.title("Resolución de las imágenes")
plt.legend()
plt.tight_layout()
plt.savefig(EDA_RESULTS_DIR / "image_sizes.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nModos de color:")
print(df["mode"].value_counts())

# Imagenes duplicadas
duplicated_mask = df.duplicated(subset="md5", keep=False)
n_duplicate_groups = df[duplicated_mask]["md5"].nunique()
print("\nImágenes duplicadas por hash MD5:")
print(int(duplicated_mask.sum()), "imágenes en", n_duplicate_groups, "grupos")

classes_per_hash = df[duplicated_mask].groupby("md5")["class"].nunique()
print("Grupos de duplicados que aparecen en más de una clase:")
print(int((classes_per_hash > 1).sum()))

if duplicated_mask.any():
    print(df[duplicated_mask].sort_values("md5")[["class", "file", "md5"]].head(10))

# Número de imágenes por clase
class_counts = df["class"].value_counts().reindex(CLASS_NAMES)
print("\nDistribución de clases:")
print(class_counts)

print("\nDistribución porcentual de clases:")
print((class_counts / class_counts.sum() * 100).round(2))

print("\nRazón entre clase más grande y más pequeña:")
print(round(class_counts.max() / class_counts.min(), 2))


# Plot para visualizar balance entre clases
plt.figure(figsize=(10, 5))
plt.ylim(0, class_counts.max())

bars = plt.bar(class_counts.index, class_counts.values)

plt.xlabel("Clase")
plt.ylabel("Número de imágenes")
plt.title("Distribución de clases")
plt.xticks(rotation=45, ha="right")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(EDA_RESULTS_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# Ejemplos de 4 imagenes por clase
rng = np.random.RandomState(SEED)
n_examples = 4

fig, axes = plt.subplots(n_examples, len(CLASS_NAMES), figsize=(15, 6.5))

for col, class_name in enumerate(CLASS_NAMES):
    class_files = df[df["class"] == class_name]["file"].values
    chosen = rng.choice(class_files, size=n_examples, replace=False)

    for row, file_name in enumerate(chosen):
        img = Image.open(RAW_DATA_DIR / class_name / file_name).convert("RGB")
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

    axes[0, col].set_title(class_name, fontsize=10)

plt.suptitle("Ejemplos aleatorios por clase")
plt.tight_layout()
plt.savefig(EDA_RESULTS_DIR / "sample_images.png", dpi=300, bbox_inches="tight")
plt.close()


# Imagen promedio por clase
fig, axes = plt.subplots(3, 3, figsize=(9, 9.5))

for ax, class_name in zip(axes.ravel(), CLASS_NAMES):
    ax.imshow(mean_images[class_name].astype(np.uint8))
    ax.set_title(class_name, fontsize=10)
    ax.axis("off")

plt.suptitle("Imagen promedio por clase")
plt.tight_layout()
plt.savefig(EDA_RESULTS_DIR / "mean_images.png", dpi=300, bbox_inches="tight")
plt.close()


# Estadística de color por clase
# Media y desviación estándar de cada canal RGB (0 a 255)
print("\nEstadística de color por clase (0 a 255):")
print(stats_df.round(4).to_string(index=False))

x = np.arange(len(CLASS_NAMES))
width = 0.30

plt.figure(figsize=(11, 5))

plt.bar(
    x - width,
    stats_df["mean_R"],
    width,
    yerr=stats_df["std_R"],
    color="tab:red",
    alpha=0.8,
    capsize=2,
    label="R",
)
plt.bar(
    x,
    stats_df["mean_G"],
    width,
    yerr=stats_df["std_G"],
    color="tab:green",
    alpha=0.8,
    capsize=2,
    label="G",
)
plt.bar(
    x + width,
    stats_df["mean_B"],
    width,
    yerr=stats_df["std_B"],
    color="tab:blue",
    alpha=0.8,
    capsize=2,
    label="B",
)

plt.xticks(x, CLASS_NAMES, rotation=45, ha="right")
plt.xlabel("Clase")
plt.ylabel("Intensidad media del canal (0 a 255)")
plt.title("Media y desviación estándar de cada canal de color por clase")
plt.legend()
plt.tight_layout()
plt.savefig(EDA_RESULTS_DIR / "class_color_stats.png", dpi=300, bbox_inches="tight")
plt.close()

# Estadística global del dataset en escala 0 a 1
total_images = stats_df["n_images"].sum()
global_mean = (
    stats_df[["mean_R", "mean_G", "mean_B"]].mul(stats_df["n_images"], axis=0).sum()
    / total_images
    / 255.0
)

print("\nMedia global por canal RGB en escala 0 a 1:")
print(global_mean.round(4).values)

# Brillo y contraste por imagen
# El brillo es la media de la escala de grises
# El contraste es la desviación estándar (0 a 1)
print("\nBrillo y contraste por clase (0 a 1):")
print(
    df.groupby("class")[["brightness", "contrast"]]
    .describe()
    .reindex(CLASS_NAMES)
    .round(4)
    .to_string()
)

bc_stats = (
    df.groupby("class")[["brightness", "contrast"]]
    .agg(["mean", "std"])
    .reindex(CLASS_NAMES)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, label, color in zip(
    axes,
    ["brightness", "contrast"],
    ["Brillo", "Contraste"],
    ["tab:gray", "tab:purple"],
):
    mean = bc_stats[(col, "mean")]
    std = bc_stats[(col, "std")]
    ax.bar(
        x,
        mean,
        yerr=[np.minimum(std, mean), std],
        color=color,
        alpha=0.8,
        capsize=2,
    )
    ax.set_xticks(x, CLASS_NAMES, rotation=45, ha="right")
    ax.set_xlabel("Clase")
    ax.set_ylabel(f"{label} (0 a 1)")
    ax.set_title(f"{label} por clase")

plt.suptitle("Media y desviación estándar de brillo y contraste por clase")
plt.tight_layout()
plt.savefig(
    EDA_RESULTS_DIR / "class_brightness_contrast.png", dpi=300, bbox_inches="tight"
)
plt.close()

print("\nImágenes más oscuras:")
print(
    df.nsmallest(5, "brightness")[["class", "file", "brightness"]].to_string(
        index=False
    )
)

print("\nImágenes más claras:")
print(
    df.nlargest(5, "brightness")[["class", "file", "brightness"]].to_string(index=False)
)

print(f"\n Resultados guardados en: {EDA_RESULTS_DIR}")
