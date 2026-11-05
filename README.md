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
uv run train_base.py   # 3. Entrenamiento y evaluación -> models/ y results/base/
```

`data_setup.py` usa la seed 67, así que la partición siempre es la misma. El entrenamiento del modelo base tarda alrededor de 18 minutos en el CPU de una MacBook Air M1.

El modelo base ya entrenado está incluido en `models/base_cnn.keras`, por lo que el paso 3 es opcional si solo se quiere usar la app. Las predicciones con imágenes aleatorias de test sí requieren haber ejecutado `data_setup.py`.

## Predicciones
Para hacer predicciones con los modelos se implementó una app web ligera con Flask, la cual se puede iniciar con el siguiente comando:

```bash
uv run app.py
```

Después de ejecutar el comando, la app estará disponible en http://127.0.0.1:5001 y ahí se podrá predecir con una imagen seleccionada, un enlace a una imagen o 8 imágenes aleatorias del set de test.

Para agregar un nuevo modelo al selector, se debe registrar su nombre y ruta en `MODELS` dentro de `config.py`.
