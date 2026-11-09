const fileInput = document.querySelector("#file");
const sampleButton = document.querySelector("#sample");
const fileName = document.querySelector("#fileName");
const runButton = document.querySelector("#run");
const showOriginal = document.querySelector("#showOriginal");
const showCharacters = document.querySelector("#showCharacters");
const canvas = document.querySelector("#canvas");
const placeholder = document.querySelector("#placeholder");
const drop = document.querySelector("#drop");
const output = document.querySelector("#output");
const curl = document.querySelector("#curl");
const status = document.querySelector("#status");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

let currentFile = null;
let previewUrl = null;
let image = null;
let lastResult = null;

function selectedModel() {
  return document.querySelector('input[name="model"]:checked').value;
}

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function updateCurl() {
  const origin = window.location.origin;
  const model = selectedModel();
  curl.textContent =
    `curl -X POST ${origin}/api/ocr \\\n` +
    `  -F "file=@image.jpg" \\\n` +
    `  -F "model=${model}" \\\n` +
    `  -F "characters=true"\n\n` +
    `# characters 可选。不传则响应里没有单字框。\n\n` +
    `GET ${origin}/api/ocr/models\n` +
    `GET ${origin}/scalar`;
}

function setBusy(busy) {
  runButton.disabled = busy || !currentFile;
  sampleButton.disabled = busy;
  fileInput.disabled = busy;
  document.querySelectorAll('input[name="model"]').forEach((el) => { el.disabled = busy; });
}

function revokePreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function drawPreview(result) {
  if (!image) return;
  canvas.hidden = false;
  placeholder.hidden = true;
  OcrOverlay.draw(ctx, image, result?.lines, {
    showOriginal: showOriginal.checked,
    showCharacters: showCharacters.checked,
  });
}

function redraw() {
  drawPreview(lastResult);
}

async function loadFile(file, name) {
  currentFile = file;
  lastResult = null;
  fileName.value = name || file.name || "image";
  runButton.disabled = false;
  output.value = "";
  revokePreview();
  previewUrl = URL.createObjectURL(file);
  image = new Image();
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("无法预览该图片"));
    image.src = previewUrl;
  });
  drawPreview();
  setStatus(`图片：${fileName.value}    ${image.naturalWidth}×${image.naturalHeight}`);
  updateCurl();
}

async function runOcr() {
  if (!currentFile) return;
  const model = selectedModel();
  const form = new FormData();
  form.append("file", currentFile, currentFile.name || "image.jpg");
  form.append("model", model);
  form.append("characters", "true");
  setBusy(true);
  setStatus("正在运行 OCR…");
  try {
    const response = await fetch("/api/ocr", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    lastResult = data;
    output.value = data.text || "";
    showOriginal.checked = false;
    drawPreview(data);
    const elapsed = data.elapsedMs;
    setStatus(
      `完成：${data.detectedCount} 行，总耗时 ${elapsed.total.toFixed(1)} ms` +
      `（解码 ${elapsed.decode.toFixed(1)} ms，OCR ${elapsed.ocr.toFixed(1)} ms）。` +
      `当前为 ${data.buildConfiguration}，模型 ${data.model}。再次点击“运行 OCR”可重复运行。`
    );
  } catch (err) {
    setStatus(err.message || "OCR 执行失败", true);
    output.value = err.message || "OCR 执行失败";
  } finally {
    setBusy(false);
  }
}

async function loadRemoteImage(url, name, missingMessage) {
  setBusy(true);
  setStatus(`正在加载 ${name}…`);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(missingMessage);
    const blob = await response.blob();
    await loadFile(new File([blob], name, { type: blob.type || "image/png" }), name);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (file) await loadFile(file);
});

sampleButton.addEventListener("click", () => loadRemoteImage(
  "/sample.jpg",
  "sample.jpg",
  "未找到示例图 examples/sample.jpg"));

runButton.addEventListener("click", runOcr);
document.querySelectorAll('input[name="model"]').forEach((el) => el.addEventListener("change", updateCurl));
showOriginal.addEventListener("change", redraw);
showCharacters.addEventListener("change", redraw);

["dragenter", "dragover"].forEach((eventName) => {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.remove("drag");
  });
});
drop.addEventListener("drop", async (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) await loadFile(file);
});

updateCurl();
