const fileInput = document.querySelector("#file");
const sampleButton = document.querySelector("#sample");
const fileName = document.querySelector("#fileName");
const runButton = document.querySelector("#run");
const canvas = document.querySelector("#canvas");
const placeholder = document.querySelector("#placeholder");
const drop = document.querySelector("#drop");
const output = document.querySelector("#output");
const curl = document.querySelector("#curl");
const status = document.querySelector("#status");
const ctx = canvas.getContext("2d");

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
    `  -F "model=${model}"\n\n` +
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
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  canvas.hidden = false;
  placeholder.hidden = true;
  ctx.drawImage(image, 0, 0);
  if (!result?.lines?.length) return;

  const fontSize = Math.min(40, Math.max(14, Math.min(canvas.width, canvas.height) / 45));
  ctx.lineWidth = Math.max(2, fontSize / 8);
  ctx.strokeStyle = "#32cd32";
  ctx.fillStyle = "#ff0000";
  ctx.font = `bold ${fontSize}px "Microsoft YaHei UI", "Segoe UI", sans-serif`;
  ctx.textBaseline = "bottom";

  for (const line of result.lines) {
    const box = line.box;
    if (!box || box.length !== 4) continue;
    ctx.beginPath();
    ctx.moveTo(box[0][0], box[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(box[i][0], box[i][1]);
    ctx.closePath();
    ctx.stroke();
    const label = line.text ?? "";
    if (label) ctx.fillText(label, box[0][0], Math.max(fontSize, box[0][1] - 2));
  }
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
  setBusy(true);
  setStatus("正在运行 OCR…");
  try {
    const response = await fetch("/api/ocr", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    lastResult = data;
    output.value = data.text || "";
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

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (file) await loadFile(file);
});

sampleButton.addEventListener("click", async () => {
  setBusy(true);
  setStatus("正在加载示例图…");
  try {
    const response = await fetch("/sample.jpg");
    if (!response.ok) throw new Error("未找到示例图 examples/sample.jpg");
    const blob = await response.blob();
    await loadFile(new File([blob], "sample.jpg", { type: "image/jpeg" }), "sample.jpg");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
});

runButton.addEventListener("click", runOcr);
document.querySelectorAll('input[name="model"]').forEach((el) => el.addEventListener("change", updateCurl));

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
