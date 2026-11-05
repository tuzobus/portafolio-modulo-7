import base64
import http.client
import io
import ipaddress
import socket
import urllib.parse
import urllib.request

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from config import CLASS_NAMES, TEST_DIR
from inference import available_models, get_model, load_and_preprocess, real_class_of

# APP WEB PARA PREDECIR CON LOS MODELOS ENTRENADOS
# uv run app.py abre la app en http://127.0.0.1:5001

MAX_IMAGE_BYTES = 10 * 1024 * 1024
THUMBNAIL_SIZE = (480, 480)
N_RANDOM_IMAGES = 8
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


# error con un mensaje que se le puede mostrar al usuario
class PredictionError(Exception):
    pass


# rechaza URLs que no sean http(s) o que apunten a la red local
def check_public_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PredictionError("La URL debe empezar con http:// o https://")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except (socket.gaierror, UnicodeError) as error:
        raise PredictionError("No se encontró el servidor de la URL") from error

    if not all(ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise PredictionError("La URL debe apuntar a un sitio público de internet")


# revisa también las URLs a las que redirige el servidor
class PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_image(url):
    check_public_url(url)

    opener = urllib.request.build_opener(PublicRedirectHandler)
    download = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with opener.open(download, timeout=30) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise PredictionError("No se pudo descargar la imagen de la URL") from error

    if len(data) > MAX_IMAGE_BYTES:
        raise PredictionError("La imagen pesa más de 10 MB")
    return data


# vista previa en JPEG para mostrarla directamente en la página
def make_thumbnail(data):
    with Image.open(io.BytesIO(data)) as img:
        thumbnail = img.convert("RGB")
    thumbnail.thumbnail(THUMBNAIL_SIZE)

    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# prepara lote de imágenes y predice con modelo seleccionado
def predict_images(model_id, images):
    model = get_model(model_id)

    try:
        batch = np.concatenate(
            [load_and_preprocess(io.BytesIO(data)) for _, data, _ in images]
        )
        thumbnails = [make_thumbnail(data) for _, data, _ in images]
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise PredictionError(
            "No se pudo leer la imagen. Si usaste un enlace, "
            "revisa que abra directamente la imagen (.jpg, .png)"
        ) from error

    # probabilidad de cada una de las 9 clases para cada imagen
    confidences = model.predict(batch, verbose=0)

    results = []
    for (name, _, real_class), thumbnail, confidence in zip(
        images, thumbnails, confidences
    ):
        predicted = CLASS_NAMES[int(np.argmax(confidence))]
        results.append(
            {
                "name": name,
                "image": thumbnail,
                "real_class": real_class,
                "predicted_class": predicted,
                "confidence": float(np.max(confidence)),
                "probabilities": [
                    {"class_name": CLASS_NAMES[k], "probability": float(confidence[k])}
                    for k in np.argsort(confidence)[::-1]
                ],
            }
        )

    return jsonify(model=available_models()[model_id]["name"], results=results)


def selected_model():
    model_id = request.form.get("model", "")
    if model_id not in available_models():
        raise PredictionError("Selecciona un modelo válido")
    return model_id


@app.errorhandler(PredictionError)
def handle_prediction_error(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(413)
def handle_too_large(error):
    return jsonify(error="La imagen pesa más de 10 MB"), 413


@app.get("/")
def index():
    return render_template("index.html", models=available_models())


@app.post("/predict/upload")
def predict_upload():
    model_id = selected_model()
    file = request.files.get("image")
    if file is None or file.filename == "":
        raise PredictionError("Selecciona una imagen")
    return predict_images(model_id, [(file.filename, file.read(), None)])


@app.post("/predict/url")
def predict_url():
    model_id = selected_model()
    url = request.form.get("url", "").strip()
    name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1] or url
    return predict_images(model_id, [(name, download_image(url), None)])


# elige imágenes al azar distintas de test en cada petición
@app.post("/predict/random")
def predict_random():
    model_id = selected_model()
    test_images = sorted(
        p for p in TEST_DIR.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(test_images) < N_RANDOM_IMAGES:
        raise PredictionError(
            f"No hay suficientes imágenes en {TEST_DIR.name}. Ejecuta data_setup.py"
        )

    chosen = np.random.choice(len(test_images), size=N_RANDOM_IMAGES, replace=False)
    images = [
        (
            test_images[i].name,
            test_images[i].read_bytes(),
            real_class_of(test_images[i]),
        )
        for i in chosen
    ]
    return predict_images(model_id, images)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, threaded=False)
