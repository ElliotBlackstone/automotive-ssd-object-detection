# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from io import BytesIO
from PIL import Image
import torch

from SSD_from_scratch import mySSD

app = FastAPI()

# Load once at startup
ssd_model = mySSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                  in_channels=3,
                  variances=(0.1, 0.2))

ssd_model.to(device='cpu')

WEIGHTS_PATH = r"C:\Users\eblac\OneDrive\Documents\GitHub\self-driving-car\saved_models\weight_only_mAP_432_11_15_2025.pth"
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
ssd_model.load_state_dict(state_dict)
ssd_model.eval()



@app.get("/", response_class=HTMLResponse)
def index():
    # Very simple inline HTML to start. Later you can move this to a real frontend.
    return """
    <html>
      <body>
        <h1>Single shot detector Demo</h1>
        <p> The model was trained on images from dashcam footage. The model will detect 5 classes: biker, car, pedestrian, traffic light, and truck. </p>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input name="file" type="file" accept="image/*">
          <input type="submit" value="Run detection">
        </form>
      </body>
    </html>
    """

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
