# portafolio-modulo-7
Repositorio para el Momento de Retroalimentación: Módulo 7 Implementación de un modelo de deep learning. (Portafolio Implementación)

## Predicciones
Para hacer predicciones con los modelos se implementó una app web ligera con Flask, la cual se puede iniciar con el siguiente comando:

```bash
uv run app.py
```

Después de ejecutar el comando, la app estará disponible en http://127.0.0.1:5001 y ahí se podrá predecir con una imagen seleccionada, un enlace a una imagen o 8 imágenes aleatorias del set de test.

Para agregar un nuevo modelo al selector, se debe registrar su nombre y ruta en `MODELS` dentro de `config.py`.
