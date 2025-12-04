# ssd_demo_app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from io import BytesIO
from PIL import Image
import torch
from SSD_from_scratch import mySSD
import os
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

app = FastAPI()

# --- MODEL LOADING ---
ssd_model = mySSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                  in_channels=3,
                  variances=(0.1, 0.2))

ssd_model.to(device='cpu')

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "saved_models" / "last_11_26_2025_mAP_467_noZoomOut_weight_only.pth"

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
            body { font-family: sans-serif; padding: 20px; }
            #result-container { margin-top: 20px; }
            img { max-width: 100%; height: auto; border: 2px solid #333; display: none; }
        </style>
      </head>
      <body>
        <h1>Single shot detector Demo</h1>
        <p>
            The model was trained on images from dashcam footage. The model will detect 5 classes: biker, car, pedestrian, traffic light, and truck (commercial). </br>
            Links: <a href="https://github.com/ElliotBlackstone" target="_blank">GitHub</a>, <a href="https://www.linkedin.com/in/elliot-blackstone-eblackstone/" target="_blank">LinkedIn</a>
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
            // Prevent the default form submission (which causes page reload)
            event.preventDefault();

            const formData = new FormData(form);

            // Send data to your existing endpoint
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    // Convert the response to a Blob (binary large object)
                    const blob = await response.blob();
                    // Create a URL for that blob
                    const imageUrl = URL.createObjectURL(blob);
                    
                    // Set the image source and make it visible
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
                                                     target_width=512,
                                                     target_height=512)

    # 3) Encode to PNG bytes and return as image/png
    buf = BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")