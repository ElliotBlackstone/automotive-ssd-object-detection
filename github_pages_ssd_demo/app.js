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
      const pixelIdx = (y * INPUT_W + x);
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

function boxesLookNormalized(boxes) {
  if (!boxes || boxes.length === 0) return false;
  let maxAbs = 0;
  for (let i = 0; i < boxes.length; i++) {
    const v = Math.abs(Number(boxes[i]));
    if (v > maxAbs) maxAbs = v;
  }
  return maxAbs <= 1.5;
}

function drawPredictions(img, boxes, scores, labels, scoreThresh) {
  drawImage(predictionCanvas, predictionCtx, img);

  const normalized = boxesLookNormalized(boxes);
  const sx = normalized ? img.width : 1;
  const sy = normalized ? img.height : 1;

  predictionCtx.lineWidth = 2;
  predictionCtx.font = '16px Arial';

  for (let i = 0; i < scores.length; i++) {
    const score = Number(scores[i]);
    if (score < scoreThresh) continue;

    const base = i * 4;
    if (base + 3 >= boxes.length) break;

    let x1 = Number(boxes[base]) * sx;
    let y1 = Number(boxes[base + 1]) * sy;
    let x2 = Number(boxes[base + 2]) * sx;
    let y2 = Number(boxes[base + 3]) * sy;

    const w = x2 - x1;
    const h = y2 - y1;
    const label = toLabelName(labels[i]);
    const caption = `${label} ${score.toFixed(2)}`;

    predictionCtx.strokeStyle = 'red';
    predictionCtx.strokeRect(x1, y1, w, h);

    const textWidth = predictionCtx.measureText(caption).width;
    const textY = Math.max(18, y1 - 6);
    predictionCtx.fillStyle = 'red';
    predictionCtx.fillRect(x1, textY - 16, textWidth + 10, 20);
    predictionCtx.fillStyle = 'white';
    predictionCtx.fillText(caption, x1 + 5, textY);
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
