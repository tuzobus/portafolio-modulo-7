const modelSelect = document.getElementById("model");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file");
const urlForm = document.getElementById("url-form");
const urlInput = document.getElementById("url");
const randomButton = document.getElementById("random");
const statusText = document.getElementById("status");
const resultsSection = document.getElementById("results");
const panel = document.querySelector(".panel");

const hasModels = !modelSelect.disabled;
let busy = false;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.hidden = message === "";
  statusText.classList.toggle("error", isError);
}

function setBusy(value) {
  busy = value;
  panel.classList.toggle("busy", value);
  for (const control of panel.querySelectorAll("select, input, button")) {
    control.disabled = value || !hasModels;
  }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function percent(value) {
  return `${(value * 100).toFixed(1)} %`;
}

function resultImage(result) {
  const img = element("img");
  img.src = result.image;
  img.alt = `Imagen analizada: ${result.name}`;
  return img;
}

// una sola imagen: imagen a la izquierda y distribución de probabilidad a la derecha
function renderSingle(result, modelName) {
  const container = element("article", "single-result");
  const details = element("div");

  details.append(
    element("p", "prediction-name", result.predicted_class),
    element(
      "p",
      "prediction-detail",
      `Confianza ${percent(result.confidence)} · ${modelName} · ${result.name}`,
    ),
  );

  result.probabilities.forEach((item, index) => {
    const row = element("div", index === 0 ? "bar-row top" : "bar-row");
    const track = element("div", "bar-track");
    const fill = element("div", "bar-fill");
    fill.style.width = `${item.probability * 100}%`;
    track.append(fill);
    row.append(
      element("span", "", item.class_name),
      track,
      element("output", "", percent(item.probability)),
    );
    details.append(row);
  });

  container.append(resultImage(result), details);
  resultsSection.replaceChildren(container);
}

// imágenes de test: se conoce la clase real, así que se marca cada acierto
function renderGrid(results, modelName) {
  const nCorrect = results.filter(
    (result) => result.predicted_class === result.real_class,
  ).length;
  const grid = element("div", "grid");

  for (const result of results) {
    const isCorrect = result.predicted_class === result.real_class;
    const card = element("article", "card");
    const body = element("div", "card-body");
    body.append(
      element(
        "p",
        `verdict ${isCorrect ? "correct" : "wrong"}`,
        isCorrect ? "Correcto" : "Incorrecto",
      ),
      element("p", "", `Real: ${result.real_class}`),
      element("p", "", `Pred: ${result.predicted_class} (${percent(result.confidence)})`),
    );
    card.append(resultImage(result), body);
    grid.append(card);
  }

  resultsSection.replaceChildren(
    element(
      "p",
      "results-summary",
      `Aciertos: ${nCorrect} de ${results.length} · ${modelName}`,
    ),
    grid,
  );
}

async function requestPrediction(path, fields, statusMessage) {
  if (busy || !hasModels) return;

  const body = new FormData();
  body.append("model", modelSelect.value);
  for (const [name, value] of Object.entries(fields)) {
    body.append(name, value);
  }

  setBusy(true);
  setStatus(statusMessage);

  try {
    const response = await fetch(path, { method: "POST", body });
    const contentType = response.headers.get("content-type") ?? "";
    const data = contentType.includes("json") ? await response.json() : {};
    if (!response.ok) {
      throw new Error(data.error ?? "El servidor no pudo procesar la petición");
    }

    if (data.results.length === 1) {
      renderSingle(data.results[0], data.model);
    } else {
      renderGrid(data.results, data.model);
    }
    setStatus("");
  } catch (error) {
    const message =
      error instanceof TypeError ? "No se pudo conectar con el servidor" : error.message;
    resultsSection.replaceChildren();
    setStatus(message, true);
  } finally {
    setBusy(false);
  }
}

function predictFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Solo se permiten archivos de imagen", true);
    return;
  }
  requestPrediction("/predict/upload", { image: file }, "Analizando imagen...");
}

fileInput.addEventListener("change", () => {
  predictFile(fileInput.files[0]);
  // permite volver a elegir el mismo archivo
  fileInput.value = "";
});

// evita que el navegador abra el archivo si se suelta fuera de la caja
for (const eventName of ["dragover", "drop"]) {
  window.addEventListener(eventName, (event) => event.preventDefault());
}

dropzone.addEventListener("dragover", () => {
  if (!busy) dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", (event) => {
  dropzone.classList.remove("dragging");
  predictFile(event.dataTransfer.files[0]);
});

urlForm.addEventListener("submit", (event) => {
  event.preventDefault();
  requestPrediction("/predict/url", { url: urlInput.value }, "Descargando y analizando imagen...");
});

randomButton.addEventListener("click", () => {
  requestPrediction("/predict/random", {}, "Analizando 8 imágenes del set de test...");
});

setBusy(false);
