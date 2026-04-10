const MODEL_URL = './models/ssd_int8_with_pre_post.onnx';
const CLASS_NAMES = ['biker', 'car', 'pedestrian', 'trafficLight', 'truck'];
const INPUT_H = 300;
const INPUT_W = 300;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const fileInput = document.getElementById('fileInput');
const runBtn = document.getElementById('runBtn');
const scoreThreshInput = document.getElementById('scoreThresh');
const statusEl = document.getElementById('status');
const errorBox = document.getElementById('errorBox');
const originalCanvas = document.getElementById('originalCanvas');
const predictionCanvas = document.getElementById('predictionCanvas');
const originalCtx = originalCanvas.getContext('2d');
const predictionCtx = predictionCanvas.getContext('2d');

let session = null;
let currentImage = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}

function clearError() {
  errorBox.textContent = '';
  errorBox.classList.add('hidden');
}

function canvasFit(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
}

function drawImage(canvas, ctx, img) {
  canvasFit(canvas, img);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
}

function isImageFile(file) {
  return !!file && (!file.type || file.type.startsWith('image/'));
}

function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load image.'));
    img.src = URL.createObjectURL(file);
  });
}

function preprocessImage(img) {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = INPUT_W;
  tempCanvas.height = INPUT_H;
  const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
  tempCtx.drawImage(img, 0, 0, INPUT_W, INPUT_H);

  const rgba = tempCtx.getImageData(0, 0, INPUT_W, INPUT_H).data;
  const chw = new Float32Array(1 * 3 * INPUT_H * INPUT_W);
  const planeSize = INPUT_H * INPUT_W;

  for (let y = 0; y < INPUT_H; y++) {
    for (let x = 0; x < INPUT_W; x++) {
      const pixelIdx = y * INPUT_W + x;
      const rgbaIdx = pixelIdx * 4;

      const r = rgba[rgbaIdx] / 255.0;
      const g = rgba[rgbaIdx + 1] / 255.0;
      const b = rgba[rgbaIdx + 2] / 255.0;

      chw[pixelIdx] = (r - MEAN[0]) / STD[0];
      chw[planeSize + pixelIdx] = (g - MEAN[1]) / STD[1];
      chw[2 * planeSize + pixelIdx] = (b - MEAN[2]) / STD[2];
    }
  }

  return new ort.Tensor('float32', chw, [1, 3, INPUT_H, INPUT_W]);
}

function detectOutput(outputs) {
  const keys = Object.keys(outputs);

  const boxesKey = keys.find(k => /box/i.test(k)) || keys[0];
  const scoresKey = keys.find(k => /score/i.test(k));
  const labelsKey = keys.find(k => /label/i.test(k));

  if (!boxesKey || !scoresKey || !labelsKey) {
    throw new Error(`Could not identify model outputs. Found: ${keys.join(', ')}`);
  }

  return {
    boxes: outputs[boxesKey].data,
    scores: outputs[scoresKey].data,
    labels: outputs[labelsKey].data,
  };
}

function toLabelName(rawLabel) {
  const i = Number(rawLabel);
  if (Number.isNaN(i)) return 'unknown';

  if (i >= 0 && i < CLASS_NAMES.length) return CLASS_NAMES[i];
  if (i - 1 >= 0 && i - 1 < CLASS_NAMES.length) return CLASS_NAMES[i - 1];
  return `cls=${i}`;
}

function detectBoxMode(boxes, imgW, imgH) {
  if (!boxes || boxes.length === 0) return 'original';

  let maxAbs = 0;
  let maxX = 0;
  let maxY = 0;

  for (let i = 0; i < boxes.length; i++) {
    const v = Math.abs(Number(boxes[i]));
    if (v > maxAbs) maxAbs = v;
  }

  if (maxAbs <= 1.5) {
    return 'normalized';
  }

  for (let i = 0; i + 3 < boxes.length; i += 4) {
    const x1 = Number(boxes[i]);
    const y1 = Number(boxes[i + 1]);
    const x2 = Number(boxes[i + 2]);
    const y2 = Number(boxes[i + 3]);

    maxX = Math.max(maxX, x1, x2);
    maxY = Math.max(maxY, y1, y2);
  }

  const fitsModelInput = maxX <= INPUT_W * 1.1 && maxY <= INPUT_H * 1.1;
  const imageIsLargerThanModel = imgW > INPUT_W * 1.1 || imgH > INPUT_H * 1.1;

  if (fitsModelInput && imageIsLargerThanModel) {
    return 'model_input';
  }

  return 'original';
}

function scaleBoxToImage(x1, y1, x2, y2, mode, imgW, imgH) {
  if (mode === 'normalized') {
    return [x1 * imgW, y1 * imgH, x2 * imgW, y2 * imgH];
  }

  if (mode === 'model_input') {
    const sx = imgW / INPUT_W;
    const sy = imgH / INPUT_H;
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy];
  }

  return [x1, y1, x2, y2];
}

function clamp(value, lo, hi) {
  return Math.min(Math.max(value, lo), hi);
}

function getOverlayScale(canvas, img) {
  const rect = canvas.getBoundingClientRect();

  if (!rect || rect.width <= 0 || rect.height <= 0) {
    return 1;
  }

  const cssScaleX = canvas.width / rect.width;
  const cssScaleY = canvas.height / rect.height;
  const cssScale = Math.max(cssScaleX, cssScaleY, 1);

  const imageScale = Math.max(Math.min(img.width, img.height) / 900, 1);
  return Math.max(cssScale, imageScale);
}

function drawCaption(ctx, text, x, y, fontSize, overlayScale, imgW, imgH) {
  const padX = Math.max(6 * overlayScale, 8);
  const padY = Math.max(3 * overlayScale, 4);
  const textWidth = ctx.measureText(text).width;
  const boxW = textWidth + 2 * padX;
  const boxH = fontSize + 2 * padY;

  let boxX = clamp(x, 0, Math.max(0, imgW - boxW));
  let boxY = y - boxH - 4 * overlayScale;

  if (boxY < 0) {
    boxY = clamp(y + 4 * overlayScale, 0, Math.max(0, imgH - boxH));
  }

  ctx.fillStyle = 'rgba(220, 38, 38, 0.95)';
  ctx.fillRect(boxX, boxY, boxW, boxH);

  ctx.fillStyle = 'white';
  ctx.fillText(text, boxX + padX, boxY + padY);
}

function drawPredictions(img, boxes, scores, labels, scoreThresh) {
  drawImage(predictionCanvas, predictionCtx, img);

  const mode = detectBoxMode(boxes, img.width, img.height);
  const overlayScale = getOverlayScale(predictionCanvas, img);
  const lineWidth = Math.max(3 * overlayScale, Math.min(img.width, img.height) / 250);
  const fontSize = Math.max(18 * overlayScale, Math.min(img.width, img.height) / 35);

  predictionCtx.lineWidth = lineWidth;
  predictionCtx.font = `bold ${fontSize}px Arial`;
  predictionCtx.textBaseline = 'top';
  predictionCtx.strokeStyle = 'rgb(220, 38, 38)';

  for (let i = 0; i < scores.length; i++) {
    const score = Number(scores[i]);
    if (score < scoreThresh) continue;

    const base = i * 4;
    if (base + 3 >= boxes.length) break;

    let x1 = Number(boxes[base]);
    let y1 = Number(boxes[base + 1]);
    let x2 = Number(boxes[base + 2]);
    let y2 = Number(boxes[base + 3]);

    [x1, y1, x2, y2] = scaleBoxToImage(x1, y1, x2, y2, mode, img.width, img.height);

    x1 = clamp(x1, 0, img.width);
    y1 = clamp(y1, 0, img.height);
    x2 = clamp(x2, 0, img.width);
    y2 = clamp(y2, 0, img.height);

    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    const right = Math.max(x1, x2);
    const bottom = Math.max(y1, y2);
    const w = right - left;
    const h = bottom - top;

    if (w <= 1 || h <= 1) continue;

    predictionCtx.strokeRect(left, top, w, h);

    const label = toLabelName(labels[i]);
    const caption = `${label} ${score.toFixed(2)}`;
    drawCaption(predictionCtx, caption, left, top, fontSize, overlayScale, img.width, img.height);
  }
}

async function loadModel() {
  setStatus('Loading model…');
  clearError();

  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm']
    });
    setStatus('Model loaded.');
  } catch (err) {
    console.error(err);
    showError('Could not load the ONNX model. Make sure ./models/ssd_int8_with_pre_post.onnx exists.');
    setStatus('Model load failed.');
  }
}

fileInput.addEventListener('change', async (event) => {
  clearError();
  const file = event.target.files?.[0];

  if (!isImageFile(file)) {
    showError('Invalid file type. Please upload an image.');
    return;
  }

  try {
    currentImage = await loadImageFile(file);
    drawImage(originalCanvas, originalCtx, currentImage);
    drawImage(predictionCanvas, predictionCtx, currentImage);
    setStatus('Image loaded. Ready to run detection.');
  } catch (err) {
    console.error(err);
    showError('Could not read that image file.');
  }
});

runBtn.addEventListener('click', async () => {
  clearError();

  if (!session) {
    await loadModel();
    if (!session) return;
  }

  if (!currentImage) {
    showError('Choose an image first.');
    return;
  }

  try {
    setStatus('Running inference locally…');
    const scoreThresh = Number(scoreThreshInput.value || 0.2);
    const inputName = session.inputNames[0];
    const inputTensor = preprocessImage(currentImage);
    const outputs = await session.run({ [inputName]: inputTensor });
    const { boxes, scores, labels } = detectOutput(outputs);
    drawPredictions(currentImage, boxes, scores, labels, scoreThresh);
    setStatus('Done.');
  } catch (err) {
    console.error(err);
    showError(`Inference failed: ${err.message}`);
    setStatus('Inference failed.');
  }
});

loadModel();