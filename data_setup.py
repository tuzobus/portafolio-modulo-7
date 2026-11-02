import hashlib
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from config import (
    RAW_DATA_DIR,
    SPLIT_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    RESULTS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    CLASS_NAMES,
    SEED,
)

# SETUP DE LOS DATOS
# Las imágenes se copian a las carpetas train / validation / test con el ratio
# 60/20/20 por clase

OUTPUT_DIR = RESULTS_DIR / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Partición estratificada 60/20/20 por clase para que las tres
# particiones conserven la misma proporción
def stratified_split_files(files, rng):
    files = np.array(files)
    rng.shuffle(files)

    n = len(files)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    return files[:train_end], files[train_end:val_end], files[val_end:]


# Se borran particiones anteriores para no mezclar archivos
if SPLIT_DIR.exists():
    print(f"Borrando partición anterior en {SPLIT_DIR}")
    shutil.rmtree(SPLIT_DIR)

# REVISIÓN DE IMÁGENES
# si una imagen no se puede abrir se descarta, y también se calcula
# hash md5 para encontrar duplicados
records = []
corrupt_files = []

for class_name in CLASS_NAMES:
    class_dir = RAW_DATA_DIR / class_name

    # se hace sorted para que la partición con la seed siempre sea la misma
    all_files = sorted(
        p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]
    )

    for path in all_files:
        try:
            with Image.open(path) as img:
                img.convert("RGB")
        except Exception:
            corrupt_files.append(str(path))
            continue

        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        records.append({"class": class_name, "file": path.name, "md5": md5})

files_df = pd.DataFrame(records)

# si una misma imagen sale en dos clases se descartan todas sus copias
# si se repite dentro de misma clase se conserva solo la primera copia
classes_per_hash = files_df.groupby("md5")["class"].nunique()
conflicting_hashes = classes_per_hash[classes_per_hash > 1].index
is_conflicting = files_df["md5"].isin(conflicting_hashes)
is_repeated = files_df.duplicated(subset="md5", keep="first")

clean_df = files_df[~is_conflicting & ~is_repeated]

print("\nImágenes corruptas descartadas:")
print(len(corrupt_files))
for path in corrupt_files:
    print(path)

print("\nImágenes repetidas en clases distintas descartadas:")
print(int(is_conflicting.sum()))

print("\nImágenes repetidas dentro de misma clase descartadas:")
print(int((is_repeated & ~is_conflicting).sum()))

print("\nImágenes válidas y únicas:")
print(len(clean_df), "de", len(files_df) + len(corrupt_files))


# PARTICIÓN Y COPIA A LAS CARPETAS
rng = np.random.RandomState(SEED)
split_rows = []

for class_name in CLASS_NAMES:
    class_dir = RAW_DATA_DIR / class_name
    unique_files = clean_df[clean_df["class"] == class_name]["file"].tolist()

    train_files, val_files, test_files = stratified_split_files(unique_files, rng)

    for split_dir, split_files in [
        (TRAIN_DIR, train_files),
        (VAL_DIR, val_files),
        (TEST_DIR, test_files),
    ]:
        target_dir = split_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_name in split_files:
            shutil.copy2(class_dir / file_name, target_dir / file_name)

    split_rows.append(
        {
            "class": class_name,
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        }
    )

split_df = pd.DataFrame(split_rows)
split_df["total"] = split_df[["train", "validation", "test"]].sum(axis=1)


print("\nImágenes por clase en cada partición:")
print(split_df.to_string(index=False))

totals = split_df[["train", "validation", "test", "total"]].sum()
print("\nTotales:")
print(totals)

print("\nPorcentaje de cada partición:")
print((totals[["train", "validation", "test"]] / totals["total"] * 100).round(2))


# Verificación de que no hay archivos repetidos entre particiones
def files_in(split_dir):
    return {f"{p.parent.name}/{p.name}" for p in split_dir.rglob("*") if p.is_file()}


train_set = files_in(TRAIN_DIR)
val_set = files_in(VAL_DIR)
test_set = files_in(TEST_DIR)

print("\nArchivos repetidos entre particiones:")
print("train and validation:", len(train_set & val_set))
print("train and test:", len(train_set & test_set))
print("validation and test:", len(val_set & test_set))

split_df.to_csv(OUTPUT_DIR / "split_counts.csv", index=False)
print(f"Particiones creadas en: {SPLIT_DIR}")
