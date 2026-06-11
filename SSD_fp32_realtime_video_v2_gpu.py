"""
Real-time webcam/video inference for the standard PyTorch FP32 SSD model.

This is the PyTorch FP32 analogue of SSD_int8_realtime_video_v2_gpu.py.

Example usage:
    (windows)
    python SSD_fp32_realtime_video_v2_gpu.py `
        --model "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\v2\\saved_models\\DIoU_mAP_551_iou_thresh_45_max_img_per_det_200.pth" `
        --device cuda `
        --pyt-model-version 2 `
        --show-fps `
        --batch-size 1

    python SSD_fp32_realtime_video_v2_gpu.py `
        --model "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\app_files\\saved_models\\noZoomOut_Bootstrap.pth" `
        --device cuda `
        --pyt-model-version 1 `
        --show-fps `
        --batch-size 1

    python SSD_fp32_realtime_video_v2_gpu.py \
        --model /path/to/noZoomOut_Bootstrap.pth \
        --device cpu \
        --pyt-model-version 1 \
        --save-video \
        --out-video ssd_fp32_demo.mp4
"""

from __future__ import annotations

import argparse
import inspect
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch



THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    from v1.SSD_from_scratch import mySSD as SSDv1
except Exception:  # pragma: no cover - depends on local repo layout
    SSDv1 = None

try:
    from v2.model_files.SSD_from_scratch import mySSD as SSDv2
except Exception:  # pragma: no cover - depends on local repo layout
    SSDv2 = None


CLASS_TO_IDX: Dict[str, int] = {
    "biker": 0,
    "car": 1,
    "pedestrian": 2,
    "trafficLight": 3,
    "truck": 4,
}


@dataclass(frozen=True)
class PreprocessConfig:
    input_color: str = "bgr"  # "bgr" for cv2 frames; "rgb" if arrays are already RGB
    resize_hw: Tuple[int, int] = (300, 300)  # (H, W)
    mean_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std_rgb: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def resolve_torch_device(device: str) -> torch.device:
    """Resolve cpu/cuda/cuda:N and fail loudly for unavailable CUDA."""
    dev_str = str(device).strip().lower()
    if dev_str == "cpu":
        return torch.device("cpu")

    if dev_str in {"cuda", "gpu"} or dev_str.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device={device!r}, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda" if dev_str in {"cuda", "gpu"} else dev_str)

    raise ValueError(f"Unsupported --device={device!r}. Use 'cpu', 'cuda', or 'cuda:N'.")


def load_state_dict_compatible(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load either a plain state_dict or a checkpoint containing a state_dict/model key.
    Handles older torch versions that do not support weights_only.
    """
    path = Path(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                obj = value
                break

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected {path} to contain a state_dict or checkpoint dict, got {type(obj)!r}."
        )

    # Strip DataParallel/DistributedDataParallel prefix if present.
    if any(str(k).startswith("module.") for k in obj.keys()):
        obj = {str(k).removeprefix("module."): v for k, v in obj.items()}

    return obj


def build_ssd_model(version: int):
    if version == 1:
        if SSDv1 is None:
            raise ImportError(
                "Could not import v1.SSD_from_scratch.mySSD. "
                "Put this script in the repo or add the repo root to PYTHONPATH."
            )
        model_cls = SSDv1
    elif version == 2:
        if SSDv2 is None:
            raise ImportError(
                "Could not import v2.model_files.SSD_from_scratch.mySSD. "
                "Put this script in the repo or add the repo root to PYTHONPATH."
            )
        model_cls = SSDv2
    else:
        raise ValueError(f"Unsupported --pyt-model-version={version}. Use 1 or 2.")

    return model_cls(
        class_to_idx_dict=CLASS_TO_IDX,
        in_channels=3,
        variances=(0.1, 0.2),
    )


class SSDPyTorchFP32Predictor:
    """
    Thin predictor wrapper around the PyTorch SSD model.

    Returns the same external format as the ONNX INT8 real-time script:
        {"labels": list[str], "scores": list[float], "boxes": np.ndarray[K,4]}

    The model's predict(...) method is assumed to return boxes in 300x300 model-input
    pixel coordinates. This wrapper rescales them back to the original frame size.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        pyt_model_version: int = 2,
        preprocess_cfg: PreprocessConfig = PreprocessConfig(),
        score_thresh: float = 0.20,
        nms_thresh: float = 0.30,
        iou_variant: str = "DIoU",
        max_per_img: int = 100,
        class_agnostic: bool = False,
        use_amp: bool = False,
    ):
        self.device = torch.device(device)
        self.pre_cfg = preprocess_cfg
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.iou_variant = str(iou_variant)
        self.max_per_img = int(max_per_img)
        self.class_agnostic = bool(class_agnostic)
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        self.model = build_ssd_model(pyt_model_version)
        state_dict = load_state_dict_compatible(model_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device, dtype=torch.float32)

        self.idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
        self.class_names_fg = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self._mean = np.array(self.pre_cfg.mean_rgb, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self.pre_cfg.std_rgb, dtype=np.float32).reshape(1, 1, 3)

        if missing:
            print(f"[warn] load_state_dict missing keys: {len(missing)}")
        if unexpected:
            print(f"[warn] load_state_dict unexpected keys: {len(unexpected)}")

    def __call__(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        return self.predict_one(image)

    def predict_one(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Sequence[Union[str, np.ndarray]]) -> List[Dict[str, Any]]:
        if len(images) == 0:
            raise ValueError("predict_batch(...) requires at least one image.")

        x_t, orig_sizes = self.preprocess_batch(images)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loc_all, conf_all = self.model(x_t)
            else:
                loc_all, conf_all = self.model(x_t)

            pred = self._call_model_predict(x_t, loc_all, conf_all)

        pred_list = self._normalize_predict_output_batch(pred, expected_batch_size=len(images))
        return [
            self._postprocess_single_result(pred_i, orig_w=orig_w, orig_h=orig_h)
            for pred_i, (orig_w, orig_h) in zip(pred_list, orig_sizes)
        ]

    def preprocess_batch(
        self,
        images: Sequence[Union[str, np.ndarray]],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        arrays: List[np.ndarray] = []
        orig_sizes: List[Tuple[int, int]] = []

        for image in images:
            arr = self._load_to_numpy(image)
            orig_h, orig_w = arr.shape[:2]
            arrays.append(self._preprocess_numpy(arr))
            orig_sizes.append((orig_w, orig_h))

        x_np = np.stack(arrays, axis=0).astype(np.float32, copy=False)  # [B,3,H,W]
        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=(self.device.type == "cuda"),
        )
        return x_t, orig_sizes

    def _load_to_numpy(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            bgr = cv2.imread(image, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cv2.imread failed for: {image}")
            return bgr

        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(f"Expected HxWx3(/4) image array, got shape {image.shape}")
            if image.shape[2] == 4:
                image = image[:, :, :3]
            return np.ascontiguousarray(image)

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _preprocess_numpy(self, img_hwc: np.ndarray) -> np.ndarray:
        color = self.pre_cfg.input_color.lower()
        if color == "bgr":
            img = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB)
        elif color == "rgb":
            img = img_hwc
        else:
            raise ValueError(f"input_color must be 'bgr' or 'rgb', got {self.pre_cfg.input_color!r}")

        resize_h, resize_w = self.pre_cfg.resize_hw
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        return np.transpose(img, (2, 0, 1)).copy()  # [3,H,W]

    def _call_model_predict(
        self,
        x_t: torch.Tensor,
        loc_all: torch.Tensor,
        conf_all: torch.Tensor,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "images": x_t,
            "score_thresh": self.score_thresh,
            "nms_thresh": self.nms_thresh,
            "max_per_img": self.max_per_img,
            "class_agnostic": self.class_agnostic,
            "pre_loc_all": loc_all,
            "pre_conf_all": conf_all,
        }

        # v2 supports iou_variant. Some earlier v1 code does not.
        sig = inspect.signature(self.model.predict)
        if "iou_variant" in sig.parameters:
            kwargs["iou_variant"] = self.iou_variant

        return self.model.predict(**kwargs)

    @staticmethod
    def _normalize_predict_output_batch(pred: Any, expected_batch_size: int) -> List[Dict[str, Any]]:
        if isinstance(pred, dict):
            pred_list = [pred]
        elif isinstance(pred, list):
            pred_list = pred
        elif isinstance(pred, tuple):
            pred_list = list(pred)
        else:
            raise TypeError(f"Unsupported predict(...) output type: {type(pred)!r}")

        if len(pred_list) != expected_batch_size:
            raise RuntimeError(
                f"predict(...) returned {len(pred_list)} predictions for {expected_batch_size} images."
            )

        for i, pred_i in enumerate(pred_list):
            if not isinstance(pred_i, dict):
                raise TypeError(f"Prediction {i} is not a dict: {type(pred_i)!r}")
            for key in ("labels", "scores", "boxes"):
                if key not in pred_i:
                    raise KeyError(f"Prediction {i} is missing key {key!r}")
        return pred_list

    def _postprocess_single_result(self, pred: Dict[str, Any], orig_w: int, orig_h: int) -> Dict[str, Any]:
        labels_raw = pred["labels"]
        scores = pred["scores"]
        boxes = pred["boxes"]

        labels_np = self._to_numpy(labels_raw, dtype=np.int64).reshape(-1)
        scores_np = self._to_numpy(scores, dtype=np.float32).reshape(-1)
        boxes_np = self._to_numpy(boxes, dtype=np.float32).reshape(-1, 4)

        if boxes_np.size == 0 or labels_np.size == 0:
            return self._empty_result()

        n = min(len(labels_np), len(scores_np), len(boxes_np))
        labels_np = labels_np[:n]
        scores_np = scores_np[:n]
        boxes_np = boxes_np[:n]

        boxes_np = self._scale_boxes_to_original(boxes_np, orig_w=orig_w, orig_h=orig_h)
        labels_str = self._map_labels(labels_np)

        return {
            "labels": labels_str,
            "scores": [float(s) for s in scores_np.tolist()],
            "boxes": boxes_np.astype(np.float32, copy=False),
        }

    @staticmethod
    def _to_numpy(value: Any, dtype: Any) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().to("cpu")
        return np.asarray(value, dtype=dtype)

    def _scale_boxes_to_original(self, boxes: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if boxes.size == 0:
            return boxes

        # model.predict normally returns pixel coordinates for the 300x300 model input.
        resize_h, resize_w = self.pre_cfg.resize_hw
        scaled = boxes.copy()

        # If a future model returns normalized boxes, handle that too.
        max_abs = float(np.nanmax(np.abs(scaled))) if scaled.size else 0.0
        if max_abs <= 1.5:
            scaled[:, [0, 2]] *= float(orig_w)
            scaled[:, [1, 3]] *= float(orig_h)
        else:
            scaled[:, [0, 2]] *= float(orig_w) / float(resize_w)
            scaled[:, [1, 3]] *= float(orig_h) / float(resize_h)

        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0.0, float(orig_w - 1))
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0.0, float(orig_h - 1))
        return scaled

    def _map_labels(self, labels: np.ndarray) -> List[str]:
        out: List[str] = []
        for label in labels.astype(np.int64).tolist():
            out.append(self.idx_to_class.get(int(label), f"class_{int(label)}"))
        return out

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "labels": [],
            "scores": [],
            "boxes": np.zeros((0, 4), dtype=np.float32),
        }


def draw_predictions_bgr(
    frame_bgr: np.ndarray,
    pred: dict,
    show_labels: bool = True,
    score_fmt: str = "{:.2f}",
) -> np.ndarray:
    """
    pred = {"labels": [str], "scores": [float], "boxes": np.ndarray (K,4) xyxy in frame pixels}
    """
    out = frame_bgr
    H, W = out.shape[:2]

    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]

    if boxes is None or len(labels) == 0:
        return out

    boxes = np.asarray(boxes, dtype=np.float32)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if show_labels:
            txt = f"{labels[i]}:{score_fmt.format(scores[i])}"
            cv2.putText(
                out,
                txt,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    return out


def fourcc_to_str(fourcc_float: float) -> str:
    fourcc = int(fourcc_float)
    return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))


def open_camera(
    source,
    backend: str = "auto",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    fourcc: str = "MJPG",
):
    backend_map = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
    }

    sysname = platform.system().lower()

    if backend != "auto":
        if backend not in backend_map:
            raise ValueError(f"Unknown backend: {backend!r}")
        trial = [backend]
    else:
        if sysname == "windows":
            trial = ["dshow", "msmf", "any"]
        elif sysname == "linux":
            trial = ["v4l2", "any"]
        else:
            trial = ["any"]

    last_err = None

    for b in trial:
        cap = cv2.VideoCapture(source, backend_map[b])

        if not cap.isOpened():
            last_err = b
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        for _ in range(5):
            cap.read()

        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        print(
            f"Opened camera source={source!r} backend={b} "
            f"requested={width}x{height}@{fps} {fourcc}, "
            f"actual={actual_width:.0f}x{actual_height:.0f}@{actual_fps:.2f} {actual_fourcc!r}"
        )

        return cap, b

    raise RuntimeError(
        f"Could not open camera source={source!r} with backends={trial} "
        f"(last tried: {last_err})"
    )


def open_video_writer(out_path: str, fps: float, frame_size: Tuple[int, int]):
    """Open a writer. Prefer mp4v for .mp4, otherwise fall back to XVID .avi."""
    out_path = str(out_path)
    suffix = Path(out_path).suffix.lower()

    candidates = []
    if suffix == ".mp4":
        candidates.append((out_path, cv2.VideoWriter_fourcc(*"mp4v")))
    candidates.append((out_path, cv2.VideoWriter_fourcc(*"XVID")))
    candidates.append((str(Path(out_path).with_suffix(".avi")), cv2.VideoWriter_fourcc(*"XVID")))

    for candidate_path, fourcc in candidates:
        writer = cv2.VideoWriter(candidate_path, fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer, candidate_path
        writer.release()

    raise RuntimeError("VideoWriter failed to open for all attempted codecs/paths.")


def estimate_fps_from_timestamps(timestamps: List[float], fallback_fps: float) -> float:
    if len(timestamps) < 2:
        return float(fallback_fps)

    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return float(fallback_fps)

    est = (len(timestamps) - 1) / elapsed
    return max(1.0, est)


def collect_frame_batch(cap, batch_size: int) -> Tuple[List[np.ndarray], bool]:
    frames: List[np.ndarray] = []
    stream_ended = False

    for _ in range(batch_size):
        ok, frame_bgr = cap.read()
        if not ok:
            stream_ended = True
            break
        frames.append(frame_bgr)

    return frames, stream_ended


def maybe_open_writer(
    writer,
    out_path: str,
    record_fps,
    fps_smoothed: float,
    requested_fps: float,
    buffered_frames: List[np.ndarray],
    buffered_timestamps: List[float],
):
    if writer is not None:
        return writer, out_path, record_fps

    if record_fps is None:
        record_fps = estimate_fps_from_timestamps(
            buffered_timestamps,
            fallback_fps=max(1.0, fps_smoothed, float(requested_fps)),
        )

    H0, W0 = buffered_frames[0].shape[:2]
    writer, out_path = open_video_writer(out_path, record_fps, (W0, H0))
    print(f"[info] saving video at {record_fps:.2f} FPS -> {out_path}")

    for fr in buffered_frames:
        writer.write(fr)

    buffered_frames.clear()
    buffered_timestamps.clear()
    return writer, out_path, record_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="Path to PyTorch FP32 .pth model")
    ap.add_argument(
        "--pyt-model-version",
        default=2,
        type=int,
        choices=[1, 2],
        help="Which mySSD implementation to instantiate: v1 or v2.",
    )
    ap.add_argument("--camera", default=0, type=int, help="Webcam index")
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=720, type=int)
    ap.add_argument("--fps", default=30, type=int, help="Requested camera FPS")
    ap.add_argument("--batch-size", default=1, type=int, help="Number of frames to run in one inference call")
    ap.add_argument(
        "--record-fps",
        default=0.0,
        type=float,
        help="Saved video FPS. Use 0 for auto-estimate from actual processed frame rate.",
    )
    ap.add_argument(
        "--record-init-frames",
        default=30,
        type=int,
        help="Number of processed frames to observe before auto-selecting saved video FPS.",
    )
    ap.add_argument("--score-thresh", default=0.30, type=float)
    ap.add_argument("--nms-thresh", default=0.30, type=float)
    ap.add_argument(
        "--iou-variant",
        default="DIoU",
        choices=["IoU", "GIoU", "DIoU", "CIoU"],
        help="Used only when the selected model.predict(...) supports iou_variant.",
    )
    ap.add_argument("--max-per-img", default=100, type=int)
    ap.add_argument("--class-agnostic", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--save-video", action="store_true", help="Save annotated output to a video file")
    ap.add_argument("--out-video", default="ssd_fp32_demo.mp4", type=str, help="Output video path")
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "any", "dshow", "msmf", "v4l2", "gstreamer"],
        help="VideoCapture backend. Use 'auto' for OS-specific fallback.",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="PyTorch inference device: cpu, cuda, or cuda:N.",
    )
    ap.add_argument(
        "--camera-device",
        default=None,
        help="Optional camera device path (Linux), e.g. /dev/video2. If set, overrides --camera.",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast for the model forward pass. Disabled automatically on CPU.",
    )
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = resolve_torch_device(args.device)
    print(f"[info] requested compute device={args.device!r} -> torch device={device}")
    print(f"[info] PyTorch version={torch.__version__} | cuda_available={torch.cuda.is_available()}")

    predictor = SSDPyTorchFP32Predictor(
        model_path=args.model,
        device=device,
        pyt_model_version=args.pyt_model_version,
        preprocess_cfg=PreprocessConfig(input_color="bgr"),
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        iou_variant=args.iou_variant,
        max_per_img=args.max_per_img,
        class_agnostic=args.class_agnostic,
        use_amp=args.amp,
    )
    print(f"[info] loaded PyTorch FP32 model={args.model!r}")
    print(f"[info] model version=v{args.pyt_model_version} | amp={predictor.use_amp}")

    source = args.camera_device if args.camera_device is not None else args.camera
    cap, backend_used = open_camera(source, args.backend, width=args.width, height=args.height, fps=args.fps)
    print(f"[info] opened camera source={source!r} using backend={backend_used}")
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {source!r}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Failed to read initial frame.")

    warmup_batch = [frame_bgr.copy() for _ in range(args.batch_size)]
    _ = predictor.predict_batch(warmup_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    writer = None
    out_path = args.out_video
    buffered_frames: List[np.ndarray] = []
    buffered_timestamps: List[float] = []
    record_fps = float(args.record_fps) if args.record_fps > 0 else None

    fps_smoothed = 0.0
    last_print = time.perf_counter()
    should_exit = False

    try:
        while not should_exit:
            loop_t0 = time.perf_counter()

            frames_batch, stream_ended = collect_frame_batch(cap, args.batch_size)
            if not frames_batch:
                break

            preds_batch = predictor.predict_batch(frames_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            if len(preds_batch) != len(frames_batch):
                raise RuntimeError(
                    f"predict_batch returned {len(preds_batch)} predictions for {len(frames_batch)} frames."
                )

            dt = time.perf_counter() - loop_t0
            inst_fps = (len(frames_batch) / dt) if dt > 0 else 0.0
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps

            total_dets = 0
            vis_batch: List[np.ndarray] = []
            for frame_bgr_i, pred_i in zip(frames_batch, preds_batch):
                total_dets += len(pred_i["labels"])
                vis = draw_predictions_bgr(frame_bgr_i, pred_i, show_labels=not args.no_labels)

                if args.show_fps:
                    cv2.putText(
                        vis,
                        f"FPS: {fps_smoothed:.1f} | B: {len(frames_batch)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )

                vis_batch.append(vis)

            if args.save_video:
                if writer is None:
                    for vis in vis_batch:
                        buffered_frames.append(vis.copy())
                        buffered_timestamps.append(time.perf_counter())

                    ready_to_open = record_fps is not None or len(buffered_frames) >= max(2, args.record_init_frames)
                    if ready_to_open:
                        writer, out_path, record_fps = maybe_open_writer(
                            writer=writer,
                            out_path=out_path,
                            record_fps=record_fps,
                            fps_smoothed=fps_smoothed,
                            requested_fps=float(args.fps),
                            buffered_frames=buffered_frames,
                            buffered_timestamps=buffered_timestamps,
                        )
                else:
                    for vis in vis_batch:
                        writer.write(vis)

            for vis in vis_batch:
                cv2.imshow("SSD PyTorch FP32", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    should_exit = True
                    break

            now = time.perf_counter()
            if now - last_print > 5.0:
                print(
                    f"[info] smoothed FPS ~ {fps_smoothed:.1f} | "
                    f"batch={len(frames_batch)} | total dets={total_dets}"
                )
                last_print = now

            if stream_ended:
                break

    finally:
        if args.save_video and writer is None and buffered_frames:
            writer, out_path, record_fps = maybe_open_writer(
                writer=writer,
                out_path=out_path,
                record_fps=record_fps,
                fps_smoothed=fps_smoothed,
                requested_fps=float(args.fps),
                buffered_frames=buffered_frames,
                buffered_timestamps=buffered_timestamps,
            )

        if writer is not None:
            writer.release()
            print(f"Saved video: {out_path}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
