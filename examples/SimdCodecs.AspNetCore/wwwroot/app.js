const fileInput = document.querySelector("#file");
const sampleButton = document.querySelector("#sample");
const rotateButton = document.querySelector("#rotate");
const fileName = document.querySelector("#fileName");
const runButton = document.querySelector("#run");
const documentOrientation = document.querySelector("#documentOrientation");
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
let baseFileName = "";
let manualRotation = 0;
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
  const docOriLine = documentOrientation.checked
    ? `  -F "documentOrientation=true" \\\n`
    : "";
  curl.textContent =
    `curl -X POST ${origin}/api/ocr \\\n` +
    `  -F "file=@image.jpg" \\\n` +
    `  -F "model=${model}" \\\n` +
    docOriLine +
    `  -F "characters=true"\n\n` +
    `# 独立单阶段 API（共享已加载模型权重）：\n` +
    `POST ${origin}/api/ocr/orientation  # 整页 0°/90°/180°/270° 方向分类\n` +
    `POST ${origin}/api/ocr/detect       # 仅文本检测 (DET)\n` +
    `POST ${origin}/api/ocr/classify     # 仅文本行方向分类 (CLS, 0°/180°)\n` +
    `POST ${origin}/api/ocr/recognize    # 仅单行识别 (REC)\n` +
    `GET  ${origin}/scalar`;
}

function setBusy(busy) {
  runButton.disabled = busy || !currentFile;
  rotateButton.disabled = busy || !image;
  sampleButton.disabled = busy;
  fileInput.disabled = busy;
  documentOrientation.disabled = busy;
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
    rotationDegrees: result?.documentOrientation?.orientationDegrees ?? 0,
  });
}

function redraw() {
  drawPreview(lastResult);
}

async function loadFile(file, name, resetRotation = true) {
  currentFile = file;
  lastResult = null;
  if (resetRotation) {
    baseFileName = name || file.name || "image";
    manualRotation = 0;
  }
  fileName.value = manualRotation
    ? `${baseFileName}（已旋转 ${manualRotation}°）`
    : baseFileName;
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
  rotateButton.disabled = false;
  drawPreview();
  setStatus(`图片：${fileName.value}    ${image.naturalWidth}×${image.naturalHeight}`);
  updateCurl();
}

async function rotateCurrentImage() {
  if (!image) return;
  setBusy(true);
  try {
    const srcW = image.naturalWidth || image.width;
    const srcH = image.naturalHeight || image.height;
    const offscreen = document.createElement("canvas");
    offscreen.width = srcH;
    offscreen.height = srcW;
    const offCtx = offscreen.getContext("2d");
    offCtx.translate(srcH, 0);
    offCtx.rotate(Math.PI / 2);
    offCtx.drawImage(image, 0, 0);

    const blob = await new Promise((resolve, reject) => {
      offscreen.toBlob((b) => (b ? resolve(b) : reject(new Error("旋转图片失败"))), "image/jpeg", 0.95);
    });
    manualRotation = (manualRotation + 90) % 360;
    const rotatedName = baseFileName.replace(/\.[^.]+$/, "") + ".jpg";
    await loadFile(new File([blob], rotatedName, { type: "image/jpeg" }), rotatedName, false);
  } catch (err) {
    setStatus(err.message || "旋转图片失败", true);
  } finally {
    setBusy(false);
  }
}

async function runOcr() {
  if (!currentFile) return;
  const model = selectedModel();
  const form = new FormData();
  form.append("file", currentFile, currentFile.name || "image.jpg");
  form.append("model", model);
  form.append("characters", "true");
  if (documentOrientation.checked) {
    form.append("documentOrientation", "true");
  }
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
    const ori = data.documentOrientation;
    const oriTiming = ori && typeof elapsed.orientation === "number"
      ? `，方向 ${elapsed.orientation.toFixed(1)} ms`
      : "";
    const oriSummary = ori
      ? `整页方向 ${ori.orientationDegrees}°（${(ori.score * 100).toFixed(1)}%${ori.orientationDegrees ? "，已自动扶正" : ""}），`
      : "";
    setStatus(
      `完成：${oriSummary}${data.detectedCount} 行，总耗时 ${elapsed.total.toFixed(1)} ms` +
      `（解码 ${elapsed.decode.toFixed(1)} ms${oriTiming}，OCR ${elapsed.ocr.toFixed(1)} ms）。` +
      `当前为 ${data.buildConfiguration}，模型 ${data.model}。`
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

rotateButton.addEventListener("click", rotateCurrentImage);
runButton.addEventListener("click", runOcr);
documentOrientation.addEventListener("change", updateCurl);
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
