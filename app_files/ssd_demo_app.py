# ssd_demo_app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
import torch
from SSD_from_scratch import mySSD
import os
import uvicorn
from pathlib import Path


app = FastAPI()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- MODEL LOADING ---
ssd_model = mySSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                  in_channels=3,
                  variances=(0.1, 0.2))

ssd_model.to(device='cpu')

WEIGHTS_PATH = BASE_DIR / "saved_models" / "noZoomOut_Bootstrap.pth"

state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
ssd_model.load_state_dict(state_dict, strict=False)
ssd_model.eval()

# --- FRONTEND ---
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>SSD Demo</title>
        <style>
            body {
                font-family: sans-serif;
                padding: 20px;
                margin: 0;              /* no global max-width here */
            }
            #result-container {
                margin-top: 20px;
            }
            /* Only style the prediction image, not all <img> tags */
            #predictionImage {
                width: 100%;            /* fill the available width */
                max-width: none;        /* don't cap at 900px */
                height: auto;
                border: 2px solid #333;
                display: none;
            }
        </style>
      </head>
      <body>
        <h1>Single shot detector Demo</h1>
        <p>
            To use, click the "Choose File" button, select an image from your device, and click the "Run detection" button. 
            The model will detect cyclists, cars, pedestrians, traffic lights, and commercial trucks, then put a red box around the detected objects with the class prediction.
        </p>

        <h2>About this project</h2>
        <ul>
          <li>
            <strong>Model</strong>:
            Custom Single Shot Detector (SSD) implemented from scratch in PyTorch
            (no torchvision detection wrappers).
          </li>
          <li>
            <strong>Data</strong>:
            Daytime dashcam images with 5 classes
            (biker, car, pedestrian, traffic light, truck (commercial)).
          </li>
          <li>
            <strong>Training</strong>:
            Custom augmentation pipeline (including IoU-based crops to emphasize small objects),
            hard negative mining, and a warm-up + cosine learning-rate schedule.
          </li>
          <li>
            <strong>Evaluation &amp; deployment</strong>:
            Achieved 0.53 mAP@0.5 on a test set of ~10k images; packaged in Docker and deployed on Google Cloud Run as this interactive demo.
          </li>
          <li>
            <strong>Stack</strong>:
            Achieved PyTorch, FastAPI, Uvicorn, Docker, Google Cloud Run.
          </li>
          <li>
            <strong>Model card</strong>:
            <a href="/model-card">View detailed model card</a>.
          </li>
          <li>
            <strong>Examples</strong>:
            <a href="/examples">View example predictions</a>.
          </li>
        </ul>

        <p>
          Links:
          <a href="https://github.com/ElliotBlackstone/automotive-ssd-object-detection" target="_blank">Project code & technical details (GitHub)</a>,
          <a href="https://www.linkedin.com/in/elliot-blackstone-eblackstone/" target="_blank">LinkedIn</a>
        </p>

        
        <form id="uploadForm">
          <input name="file" type="file" accept="image/*" required>
          <input type="submit" value="Run detection">
        </form>

        <div id="result-container">
            <h3>Prediction Result:</h3>
            <img id="predictionImage" src="" alt="Predicted Image" />
        </div>

        <script>
          const form = document.getElementById('uploadForm');
          const resultImg = document.getElementById('predictionImage');

          form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const formData = new FormData(form);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    
                    resultImg.src = imageUrl;
                    resultImg.style.display = 'block';
                } else {
                    alert('Error processing image');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred.');
            }
          });
        </script>

        <hr>
        <p><strong>For recruiters / hiring managers</strong></p>
        <ul>
          <li>I design and deploy end-to-end ML systems: custom object-detection models, training pipelines, and production APIs on cloud infrastructure.</li>
          <li>This demo reflects full-stack ownership: data preparation, SSD architecture, training/evaluation, Dockerization, and serving on Google Cloud Run.</li>
          <li>I'm interested in Data Scientist / ML Engineer / Computer Vision Engineer roles. For a technical walkthrough or to discuss fit, contact me via <a href="https://www.linkedin.com/in/elliot-blackstone-eblackstone/" target="_blank">LinkedIn</a>.</li>
        </ul>


      </body>
    </html>
    """



@app.get("/model-card", response_class=HTMLResponse)
def model_card():
    return """
    <html>
      <head>
        <title>SSD Demo - Model card</title>
        <style>
            body { font-family: sans-serif; padding: 20px; max-width: 900px; margin: 0 auto; }
        </style>
      </head>
      <body>
        <h1>Model card</h1>

        <ul>
          <li>
            <strong>Intended use</strong>:
            Demo model for detecting vehicles and vulnerable road users in daytime dashcam footage.
            Suitable for educational / portfolio purposes and qualitative exploration of SSD behavior.
          </li>
          <li>
            <strong>Data</strong>:
            Trained on a limited dataset of daytime automotive scenes with 5 classes
            (biker, car, pedestrian, traffic light, truck/commercial). Distribution is skewed toward
            clear-weather, forward-facing dashcam views.
          </li>
          <li>
            <strong>Limitations</strong>:
            Not designed for nighttime, severe weather, non-dashcam viewpoints, or arbitrarily
            diverse environments. May miss small or heavily occluded pedestrians and cyclists,
            and may produce false positives on unusual objects or reflections.
          </li>
          <li>
            <strong>Safety / deployment note</strong>:
            This model is not validated for real-world safety-critical use (e.g., ADAS, collision
            avoidance) and should not be used to make driving decisions. It is provided strictly
            as a research and demonstration tool.
          </li>
        </ul>

        <p><a href="/">Back to demo</a></p>
      </body>
    </html>
    """




@app.get("/examples", response_class=HTMLResponse)
def examples():
    return """
    <html>
      <head>
        <title>SSD Demo - Examples</title>
        <style>
            body {
                font-family: sans-serif;
                padding: 20px;
                margin: 0;              /* no max-width, no centering */
            }
            .img-row {
                margin-bottom: 30px;
            }
            .img-row h3 {
                margin-bottom: 10px;
            }
            /* Only style images on this page */
            .example-image {
                width: 100%;            /* fill available width */
                max-width: none;        /* don't cap at 900px */
                height: auto;
                border: 2px solid #333;
                display: block;
                margin: 0 auto;
            }
        </style>
      </head>
      <body>
        <h1>Example predictions</h1>
        <p>
          These images illustrate typical outputs from the SSD model on daytime dashcam scenes. 
          Each example shows the original user image on the left and the model's predictions on the right (precomputed for this demo). 
          None of these example images were present in the model train/validation set.
        </p>

        <div class="img-row">
          <h3>Example 1</h3>
          <img src="/static/good1.png" alt="SSD prediction example 1">
        </div>

        <div class="img-row">
          <h3>Example 2</h3>
          <img src="/static/good2_from_test_set.png" alt="SSD prediction example 2">
        </div>

        <div class="img-row">
          <h3>Example 3</h3>
          <img src="/static/good3_from_test_set.png" alt="SSD prediction example 3">
        </div>

        <div class="img-row">
          <h3>Example 4</h3>
          <img src="/static/good4_from_test_set.png" alt="SSD prediction example 4">
        </div>

        <p><a href="/">Back to demo</a></p>
      </body>
    </html>
    """



# --- PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) Read uploaded image
    data = await file.read()
    pil_img = Image.open(BytesIO(data)).convert("RGB")

    # 2) Run your existing side-by-side code
    out_img = ssd_model.show_prediction_side_by_side(image_path=None,
                                                     pil_img=pil_img,
                                                     score_thresh=0.2,
                                                     nms_thresh=0.3,
                                                     max_per_img=100,
                                                     class_agnostic=False,
                                                     target_height=512)

    # 3) Encode to PNG bytes and return as image/png
    buf = BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")