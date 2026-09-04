# End-to-end image prediction benchmark

| Model | Device | N | Mean (ms/image) | p50 (ms/image) | p95 (ms/image) | Min (ms) | Max (ms) | FPS at p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SSD (FP32) | CPU | 50 | 191.178 | 191.042 | 196.217 | 184.726 | 200.883 | 5.23 |
| Transformer (FP32) | CPU | 50 | 42.169 | 42.390 | 45.617 | 38.011 | 46.396 | 23.59 |
| SSD (FP32) | NVIDIA GeForce RTX 3060 Ti | 50 | 13.037 | 12.705 | 14.760 | 11.995 | 16.829 | 78.71 |
| Transformer (FP32) | NVIDIA GeForce RTX 3060 Ti | 50 | 16.249 | 13.331 | 25.397 | 12.177 | 26.765 | 75.01 |

## Configuration

- Timestamp: 2026-09-03T16:36:06-07:00
- Image: `C:\Udacity_car_data\data\test\1478019952686311006_jpg.rf.JLSB3LP2Q4RuGHYKqfF6.jpg`
- Input/batch size: 300 x 300, batch size 1
- Precision: FP32 (automatic mixed precision disabled)
- Requested devices: cpu, cuda
- Warm-up iterations: 10 per model/device
- Measured iterations: 50 per model/device
- Python executable: `C:\Users\eblac\Documents\GitHub\.venv-win-test\Scripts\python.exe`
- PyTorch: 2.10.0+cu128
- CUDA build used by PyTorch: 12.8
- gen_nms: `C:\Users\eblac\Documents\GitHub\.venv-win-test\Lib\site-packages\gen_nms\__init__.py`
- CPU: Intel64 Family 6 Model 167 Stepping 1, GenuineIntel
- CPU threads used by PyTorch: 8
- SSD checkpoint: `C:\Users\eblac\Documents\GitHub\self-driving-car\v2\saved_models\DIoU_mAP_551_iou_thresh_45_max_img_per_det_200.pth`
- Transformer checkpoint: `C:\Users\eblac\Documents\GitHub\self-driving-car\myTransformer\saved_models\epoch290_mAP_on_val_7230.ckpt`
- SSD thresholds: score=0.2, DIoU-NMS=0.45, max detections=200
- Transformer confidence threshold: 0.4

## Measurement boundary

Each sample starts before `cv2.imread` and ends after the prediction tensors have
been copied to CPU memory. It therefore includes disk image read/decode,
preprocessing, host-to-device transfer where applicable, FP32 model inference,
and model postprocessing. Checkpoint loading and warm-up are excluded. Repeated
reads normally benefit from the operating-system file cache, so these numbers do
not represent cold-storage latency.
