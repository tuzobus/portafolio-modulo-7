# portafolio-modulo-7
Repositorio para el Momento de Retroalimentación: Módulo 7 Implementación de un modelo de deep learning. (Portafolio Implementación)

## Dataset
Se utiliza el [Mushroom Classification Dataset](https://www.kaggle.com/datasets/lizhecheng/mushroom-classification) de Zhecheng Li publicado en Kaggle. Contiene fotografías de hongos de 9 géneros: Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula y Suillus.

La carpeta `Mushrooms` contiene 6,714 imágenes JPG. `data_setup.py` descarta las imágenes corruptas, vacías o duplicadas, y conserva 6,643 repartidas en train (3,983), validation (1,326) y test (1,334).

### Cómo acomodar los datos
La descarga de Kaggle incluye dos carpetas: `archive/Mushrooms` y `Transfer data`. Este proyecto solo usa `Mushrooms`:

1. Descargar el dataset desde Kaggle y descomprimirlo.
2. Crear la carpeta `data/` en la raíz del repositorio (git ya la ignora).
3. Mover la carpeta `archive/Mushrooms` a `data/`, conservando el nombre `Mushrooms` con mayúscula inicial.

El resultado debe quedar así:

```text
portafolio-modulo-7/
└── data/
    └── Mushrooms/
        ├── Agaricus/
        ├── Amanita/
        ├── Boletus/
        ├── Cortinarius/
        ├── Entoloma/
        ├── Hygrocybe/
        ├── Lactarius/
        ├── Russula/
        └── Suillus/
```

## Instalación
El proyecto usa Python 3.13 y [uv](https://docs.astral.sh/uv/). Para instalar las dependencias, se ejecuta desde la raíz del repositorio:

```bash
uv sync
```

## Ejecución
Con el dataset ya acomodado en `data/Mushrooms/`, los scripts se ejecutan en este orden desde la raíz del repositorio:

```bash
uv run eda.py          # 1. EDA -> results/eda/
uv run data_setup.py   # 2. Limpieza y partición -> data/split/
uv run train_base.py   # 3. CNN base -> results/base/
uv run train_regularized.py  # 4. CNN regularizada -> results/regularized/
uv run train_improved_1.py   # 5. EfficientNetB0 congelada -> results/improved_1/
uv run train_improved_2.py   # 6. Fine-tuning del modelo 5 -> results/improved_2/
```

`data_setup.py` usa la seed 67, así que la partición siempre es la misma. El paso 6 parte del modelo que guarda el paso 5, por lo que deben ejecutarse en ese orden. La primera vez que se ejecuta el paso 5 se descargan los pesos de EfficientNetB0 preentrenada en ImageNet (16 MB).

| Modelo | Script | Archivo en `models/` | Tiempo aprox. (CPU MacBook Air M1) |
|---|---|---|---|
| CNN base | `train_base.py` | `base_cnn.keras` | 18 min |
| CNN regularizada | `train_regularized.py` | `regularized_cnn.keras` | PENDIENTE |
| EfficientNetB0 (transfer learning) | `train_improved_1.py` | `improved_1_efficientnet.keras` | 39 min |
| EfficientNetB0 + fine-tuning | `train_improved_2.py` | `improved_2_finetuned.keras` | 36 min |

Los modelos ya entrenados están incluidos en `models/`, por lo que los pasos 3 a 6 son opcionales si solo se quieren hacer predicciones. Las predicciones con imágenes aleatorias de test sí requieren haber ejecutado `data_setup.py`.

## Predicciones
### Consola
`predict.py` usa el modelo base por defecto; con `--model` se elige otro (`base`, `regularized`, `improved_1` o `improved_2`):

```bash
uv run predict.py --model improved_2                        # 8 imágenes al azar de test
uv run predict.py --model improved_2 ruta/a/imagen.jpg      # una imagen local
uv run predict.py --model improved_2 https://.../imagen.jpg # una imagen de internet
```

Las figuras se guardan en `results/predictions/`.

### App web
Para hacer predicciones con los modelos se implementó una app web ligera con Flask, la cual se puede iniciar con el siguiente comando:

```bash
uv run app.py
```

Después de ejecutar el comando, la app estará disponible en http://127.0.0.1:5001 y ahí se podrá predecir con una imagen seleccionada, un enlace a una imagen o 8 imágenes aleatorias del set de test.

En la app se puede elegir cualquiera de los modelos entrenados. Para agregar un nuevo modelo al selector, se debe registrar en `MODELS` dentro de `config.py` su nombre, su ruta, el tamaño de imagen que recibe (`img_size`) y su escalado (`rescale`: `1/255` para las CNN propias y `None` para EfficientNet).