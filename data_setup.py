import hashlib
import shutil

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent / "data" / "EuroSAT_RGB"
SPLIT_DIR = Path(__file__).resolve().parent / "data" / "split"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

SEED = 67


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

rng = np.random.RandomState(SEED)
split_rows = []
seen_hashes = set()
n_skipped_duplicates = 0

for class_name in CLASS_NAMES:
    class_dir = RAW_DATA_DIR / class_name

    # se hace sorted para que la partición con la seed siempre sea la misma
    all_files = sorted(
        p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]
    )

    # se descartan duplicados por hash md5
    unique_files = []
    for path in all_files:
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if md5 in seen_hashes:
            n_skipped_duplicates += 1
            continue
        seen_hashes.add(md5)
        unique_files.append(path.name)

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

print("\nDuplicados exactos descartados:")
print(n_skipped_duplicates)

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
print(f"\nConteos guardados en: {OUTPUT_DIR / 'split_counts.csv'}")
print(f"Partición creada en: {SPLIT_DIR}")
