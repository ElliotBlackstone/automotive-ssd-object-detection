r"""
Benchmark SSD pipeline stage timings for:
1) Pure PyTorch
2) Raw ONNX forward + PyTorch postprocess
3) ONNX with decode/NMS baked into the graph

Expected pipeline output:
{"labels": [...], "scores": [...], "boxes": [...]}

Assumptions:
- Your PyTorch SSD has:
      model(x) -> (loc_all, conf_all)
      model.predict(x, pre_loc_all=..., pre_conf_all=..., ...) -> [dict] or dict
- The raw ONNX model outputs loc_all and conf_all
- The stitched ONNX model outputs labels, scores, boxes (in any order;
  the script infers which is which from shape/dtype)
- Input is one frame (e.g. from webcam), benchmarked repeatedly

Usage examples:

# From one image:
python benchmarking/benchmark_ssd_pipeline.py \
    --pytorch-pth /home/eblackstone/repos/automotive-ssd-object-detection/v2/saved_models/DIoU_mAP_551_iou_thresh_45_max_img_per_det_200.pth \
    --onnx-raw /home/eblackstone/repos/automotive-ssd-object-detection/PTQ_testing/ssd_int8_v2.onnx \
    --onnx-e2e /home/eblackstone/repos/automotive-ssd-object-detection/PTQ_testing/ssd_int8_with_pre_post.onnx \
    --image /home/eblackstone/datasets/Udacity_car_data/data/test/1478019975685727611_jpg.rf.6c5fc5c2d37cd11484ca1631067c0e23.jpg \
    --torch-device cpu \
    --ort-provider cpu \
    --pyt-model-version 2

# windows
python "benchmarking\\benchmark_ssd_pipeline.py" `
    --pytorch-pth "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\v2\\saved_models\\DIoU_mAP_551_iou_thresh_45_max_img_per_det_200.pth" `
    --onnx-raw "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\PTQ_testing\\ssd_int8_v2.onnx" `
    --onnx-e2e "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\PTQ_testing\\ssd_int8_with_pre_post.onnx" `
    --image "C:\\Udacity_car_data\data\\test\\1478020441702436005_jpg.rf.2tMFzQOxSFdtoIPC2DaC.jpg" `
    --torch-device cuda `
    --ort-provider cuda `
    --pyt-model-version 2


python benchmarking/benchmark_ssd_pipeline.py \
    --pytorch-pth /home/eblackstone/repos/automotive-ssd-object-detection/app_files/saved_models/noZoomOut_Bootstrap.pth \
    --onnx-raw /home/eblackstone/repos/automotive-ssd-object-detection/PTQ_testing/ssd_int8.onnx \
    --onnx-e2e /home/eblackstone/repos/automotive-ssd-object-detection/PTQ_testing/ssd_int8_with_pre_post.onnx \
    --image /home/eblackstone/datasets/Udacity_car_data/data/test/1478019975685727611_jpg.rf.6c5fc5c2d37cd11484ca1631067c0e23.jpg \
    --torch-device cpu \
    --ort-provider cpu \
    --pyt-model-version 1

# From webcam:
(windows)
python benchmarking\\benchmark_ssd_pipeline.py `
    --pytorch-pth "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\v2\\saved_models\\DIoU_mAP_551_iou_thresh_45_max_img_per_det_200.pth" `
    --onnx-raw "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\PTQ_testing\\ssd_int8_v2.onnx" `
    --onnx-e2e "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\PTQ_testing\\ssd_int8_with_pre_post.onnx" `
    --camera 0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from v1.SSD_from_scratch import mySSD as oldSSD
from v2.model_files.SSD_from_scratch import mySSD as newSSD

# ----------------------------
# Constants
# ----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------
# Utilities
# ----------------------------
def make_torch_sync(device: torch.device):
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return _sync


def timed_call(fn, sync_before=None, sync_after=None):
    if sync_before is not None:
        sync_before()
    t0 = time.perf_counter()
    out = fn()
    if sync_after is not None:
        sync_after()
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


def summarize_records(records: List[Dict[str, float]], fields: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for field in fields:
        vals = np.array(
            [r[field] for r in records if field in r and np.isfinite(r[field])],
            dtype=np.float64,
        )
        if vals.size == 0:
            out[field] = {"mean": math.nan, "median": math.nan, "p95": math.nan}
        else:
            out[field] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.quantile(vals, 0.95)),
            }
    return out


def print_summary(name: str, summary: Dict[str, Dict[str, float]]):
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in summary.items():
        mean = v["mean"]
        med = v["median"]
        p95 = v["p95"]
        if np.isfinite(mean):
            print(f"{k:18s} mean={mean:9.3f} ms   median={med:9.3f} ms   p95={p95:9.3f} ms")
        else:
            print(f"{k:18s} mean=      nan      median=      nan      p95=      nan")


def convert_pred_to_python(pred: Any) -> Dict[str, Any]:
    """
    Normalize output to:
    {"labels": [...], "scores": [...], "boxes": [[...], ...]}
    """
    if isinstance(pred, list):
        if len(pred) == 0:
            return {"labels": [], "scores": [], "boxes": []}
        pred = pred[0]

    out = {}
    for key in ("labels", "scores", "boxes"):
        value = pred[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        out[key] = value.tolist()
    return out


def maybe_rescale_boxes_to_original(boxes: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    """
    If boxes look normalized (roughly in [0, 1]), rescale to original image size.
    Otherwise leave unchanged.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes

    mx = float(np.max(np.abs(boxes)))
    if mx <= 1.5:
        h, w = orig_hw
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
    return boxes


def infer_stitched_outputs(raw_outputs: List[np.ndarray], orig_hw: Tuple[int, int]) -> Dict[str, Any]:
    """
    Infer which output is labels/scores/boxes from dtype/shape.
    """
    boxes = None
    scores = None
    labels = None

    for arr in raw_outputs:
        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.ndim >= 1 and arr.shape[-1] == 4 and np.issubdtype(arr.dtype, np.floating):
            boxes = arr.reshape(-1, 4)
        elif np.issubdtype(arr.dtype, np.integer):
            labels = arr.reshape(-1)
        elif np.issubdtype(arr.dtype, np.floating):
            scores = arr.reshape(-1)

    if boxes is None or scores is None or labels is None:
        raise RuntimeError(
            "Could not infer stitched ONNX outputs as labels/scores/boxes. "
            "Inspect session.get_outputs() and adjust infer_stitched_outputs()."
        )

    n = min(len(labels), len(scores), len(boxes))
    labels = labels[:n]
    scores = scores[:n]
    boxes = boxes[:n]

    valid = np.isfinite(scores)
    if np.issubdtype(labels.dtype, np.integer):
        valid &= (labels >= 0)
    valid &= np.all(np.isfinite(boxes), axis=1)

    labels = labels[valid]
    scores = scores[valid]
    boxes = boxes[valid]

    boxes = maybe_rescale_boxes_to_original(boxes, orig_hw)

    return {
        "labels": labels.tolist(),
        "scores": scores.astype(np.float32).tolist(),
        "boxes": boxes.astype(np.float32).tolist(),
    }


# ----------------------------
# Input preprocessing
# ----------------------------
def preprocess_numpy(frame_bgr: np.ndarray, input_wh: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns NCHW float32 numpy array suitable for ONNXRuntime.
    frame_bgr: HWC uint8 BGR from cv2
    input_wh: (width, height)
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, input_wh, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None, ...].copy()  # [1,3,H,W]
    meta = {
        "orig_hw": (orig_h, orig_w),
        "input_wh": input_wh,
    }
    return x, meta


def preprocess_torch(frame_bgr: np.ndarray, input_wh: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x_np, meta = preprocess_numpy(frame_bgr, input_wh)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    return x, meta


# ----------------------------
# Model loading
# ----------------------------
def load_pytorch_model(pth_path: str, device: torch.device, version: int = 2):
    """
    Loads PyTorch model from .pth file.
    """
    if version == 1:
        model = oldSSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                    in_channels=3,
                    variances=(0.1, 0.2))
    else:
        model = newSSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                    in_channels=3,
                    variances=(0.1, 0.2))
    # WEIGHTS_PATH = r"/mnt/c/Users/eblac/Documents/GitHub/self-driving-car/app_files/saved_models/noZoomOut_Bootstrap.pth"
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    return model


def make_ort_session(model_path: str, provider: str) -> ort.InferenceSession:
    avail = ort.get_available_providers()

    if provider == "cuda":
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(f"Requested ORT CUDA provider, but available providers are: {avail}")
    elif provider == "tensorrt":
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

    sess = ort.InferenceSession(model_path, providers=providers)
    return sess


# ----------------------------
# Frame source
# ----------------------------
def get_frame(args) -> np.ndarray:
    if args.image is not None:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return frame

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    # let camera settle a bit
    for _ in range(args.camera_warmup):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Could not grab a frame from webcam")

    return frame


# ----------------------------
# Benchmark runners
# ----------------------------
def benchmark_pytorch(
    model,
    frame_bgr: np.ndarray,
    input_wh: Tuple[int, int],
    device: torch.device,
    runs: int,
    warmup: int,
    predict_kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    sync = make_torch_sync(device)
    records = []
    last_pred = None

    with torch.inference_mode():
        for i in range(warmup + runs):
            (x, _meta), t_pre = timed_call(
                lambda: preprocess_torch(frame_bgr, input_wh, device),
                sync_before=sync,
                sync_after=sync,
            )

            (loc_conf, t_fwd) = timed_call(
                lambda: model(x),
                sync_before=sync,
                sync_after=sync,
            )
            loc_all, conf_all = loc_conf

            (pred, t_pred) = timed_call(
                lambda: model.predict(
                    x,
                    pre_loc_all=loc_all,
                    pre_conf_all=conf_all,
                    **predict_kwargs,
                ),
                sync_before=sync,
                sync_after=sync,
            )

            total_ms = t_pre + t_fwd + t_pred
            last_pred = convert_pred_to_python(pred)

            if i >= warmup:
                records.append(
                    {
                        "preprocess_ms": t_pre,
                        "forward_ms": t_fwd,
                        "predict_ms": t_pred,
                        "graph_run_ms": math.nan,
                        "end_to_end_ms": total_ms,
                    }
                )

    return records, last_pred


def benchmark_onnx_raw_plus_torch_post(
    ort_sess: ort.InferenceSession,
    postprocess_model,
    frame_bgr: np.ndarray,
    input_wh: Tuple[int, int],
    torch_device: torch.device,
    runs: int,
    warmup: int,
    predict_kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    sync = make_torch_sync(torch_device)
    input_name = ort_sess.get_inputs()[0].name
    output_names = [o.name for o in ort_sess.get_outputs()]
    records = []
    last_pred = None

    with torch.inference_mode():
        for i in range(warmup + runs):
            (prep_out, t_pre) = timed_call(
                lambda: preprocess_numpy(frame_bgr, input_wh)
            )
            x_np, _meta = prep_out

            (ort_outs, t_fwd) = timed_call(
                lambda: ort_sess.run(output_names, {input_name: x_np})
            )

            if len(ort_outs) != 2:
                raise RuntimeError(
                    f"Raw ONNX model is expected to output 2 tensors (loc_all, conf_all). Got {len(ort_outs)}."
                )

            loc_np, conf_np = ort_outs

            def _torch_post():
                # This conversion cost is part of the real mixed-runtime pipeline.
                x_t = torch.from_numpy(x_np).to(device=torch_device, dtype=torch.float32)
                loc_t = torch.from_numpy(np.asarray(loc_np)).to(device=torch_device, dtype=torch.float32)
                conf_t = torch.from_numpy(np.asarray(conf_np)).to(device=torch_device, dtype=torch.float32)

                return postprocess_model.predict(
                    x_t,
                    pre_loc_all=loc_t,
                    pre_conf_all=conf_t,
                    **predict_kwargs,
                )

            pred, t_pred = timed_call(
                _torch_post,
                sync_before=sync,
                sync_after=sync,
            )

            total_ms = t_pre + t_fwd + t_pred
            last_pred = convert_pred_to_python(pred)

            if i >= warmup:
                records.append(
                    {
                        "preprocess_ms": t_pre,
                        "forward_ms": t_fwd,
                        "predict_ms": t_pred,
                        "graph_run_ms": math.nan,
                        "end_to_end_ms": total_ms,
                    }
                )

    return records, last_pred


def benchmark_onnx_stitched(
    ort_sess: ort.InferenceSession,
    frame_bgr: np.ndarray,
    input_wh: Tuple[int, int],
    runs: int,
    warmup: int,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    input_name = ort_sess.get_inputs()[0].name
    output_names = [o.name for o in ort_sess.get_outputs()]
    records = []
    last_pred = None

    for i in range(warmup + runs):
        (prep_out, t_pre) = timed_call(
            lambda: preprocess_numpy(frame_bgr, input_wh)
        )
        x_np, meta = prep_out

        (ort_outs, t_graph) = timed_call(
            lambda: ort_sess.run(output_names, {input_name: x_np})
        )

        pred = infer_stitched_outputs(ort_outs, meta["orig_hw"])
        total_ms = t_pre + t_graph
        last_pred = pred

        if i >= warmup:
            records.append(
                {
                    "preprocess_ms": t_pre,
                    "forward_ms": math.nan,     # not externally separable
                    "predict_ms": math.nan,     # not externally separable
                    "graph_run_ms": t_graph,    # forward + decode + NMS in ONNX
                    "end_to_end_ms": total_ms,
                }
            )

    return records, last_pred


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-pth", type=str, required=True)
    parser.add_argument("--onnx-raw", type=str, required=True)
    parser.add_argument("--onnx-e2e", type=str, required=True)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", type=str, default=None)
    group.add_argument("--camera", type=int, default=0)

    parser.add_argument("--camera-warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)

    parser.add_argument("--input-width", type=int, default=300)
    parser.add_argument("--input-height", type=int, default=300)

    parser.add_argument("--torch-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ort-provider", type=str, default="cpu", choices=["cpu", "cuda", "tensorrt"])
    parser.add_argument("--pyt-model-version", type=int, default=2)

    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--max-per-img", type=int, default=100)
    parser.add_argument("--iou-variant", type=str, default="DIoU")

    args = parser.parse_args()

    if args.torch_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested torch cuda, but torch.cuda.is_available() is False")

    torch_device = torch.device(args.torch_device)
    input_wh = (args.input_width, args.input_height)

    frame = get_frame(args)

    print(f"Frame shape: {frame.shape}")
    print(f"Torch device: {torch_device}")
    print(f"ORT providers available: {ort.get_available_providers()}")

    if args.pyt_model_version == 1:
        predict_kwargs = dict(
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            max_per_img=args.max_per_img,
        )
    else:
        predict_kwargs = dict(
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            iou_variant=args.iou_variant,
            max_per_img=args.max_per_img,
        )

    # Load models
    pytorch_model = load_pytorch_model(
        pth_path=args.pytorch_pth,
        device=torch_device,
        version=args.pyt_model_version,
    )
    ort_raw = make_ort_session(args.onnx_raw, provider=args.ort_provider)
    ort_e2e = make_ort_session(args.onnx_e2e, provider=args.ort_provider)

    # Run benchmarks
    pt_records, pt_pred = benchmark_pytorch(
        model=pytorch_model,
        frame_bgr=frame,
        input_wh=input_wh,
        device=torch_device,
        runs=args.runs,
        warmup=args.warmup,
        predict_kwargs=predict_kwargs,
    )

    raw_records, raw_pred = benchmark_onnx_raw_plus_torch_post(
        ort_sess=ort_raw,
        postprocess_model=pytorch_model,
        frame_bgr=frame,
        input_wh=input_wh,
        torch_device=torch_device,
        runs=args.runs,
        warmup=args.warmup,
        predict_kwargs=predict_kwargs,
    )

    e2e_records, e2e_pred = benchmark_onnx_stitched(
        ort_sess=ort_e2e,
        frame_bgr=frame,
        input_wh=input_wh,
        runs=args.runs,
        warmup=args.warmup,
    )

    fields = ["preprocess_ms", "forward_ms", "predict_ms", "graph_run_ms", "end_to_end_ms"]

    print_summary("Pure PyTorch", summarize_records(pt_records, fields))
    print_summary("Raw ONNX + PyTorch postprocess", summarize_records(raw_records, fields))
    print_summary("Stitched ONNX graph", summarize_records(e2e_records, fields))

    # Print one sample output from each pipeline so you can verify format
    print("\nSample output shapes / counts")
    print("-----------------------------")
    print(f"PyTorch: labels={len(pt_pred['labels'])}, scores={len(pt_pred['scores'])}, boxes={len(pt_pred['boxes'])}")
    print(f"Raw ONNX + torch post: labels={len(raw_pred['labels'])}, scores={len(raw_pred['scores'])}, boxes={len(raw_pred['boxes'])}")
    print(f"Stitched ONNX: labels={len(e2e_pred['labels'])}, scores={len(e2e_pred['scores'])}, boxes={len(e2e_pred['boxes'])}")


if __name__ == "__main__":
    main()