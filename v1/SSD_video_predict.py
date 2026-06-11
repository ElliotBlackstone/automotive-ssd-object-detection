from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


# ImageNet normalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def preprocess_frame_bgr_to_tensor(frame_bgr: np.ndarray,
                                   device: torch.device,
                                   size: Tuple[int, int] = (300, 300),  # (W,H)
                                   ) -> torch.Tensor:
    """
    Matches:
      ToDtype(float32, scale=True) -> Resize((300,300)) -> Normalize(ImageNet mean/std)
    Input: BGR uint8 (H,W,3) from OpenCV
    Output: float32 tensor (3,H,W) normalized
    """
    
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input
    out_w, out_h = size
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # HWC uint8 -> CHW float32 in [0,1]
    t = torch.from_numpy(rgb).to(device=device)
    t = t.permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0

    # Normalize
    mean = _IMAGENET_MEAN.to(device=device)
    std = _IMAGENET_STD.to(device=device)
    t = (t - mean) / std
    return t


def draw_detections_bgr(frame_bgr: np.ndarray,
                        boxes_xyxy: np.ndarray,
                        labels: np.ndarray,
                        scores: np.ndarray,
                        label_map: Optional[Dict[int, str]] = None,
                        thickness: int = 2,
                        ) -> np.ndarray:
    """
    Draws on a BGR frame. boxes_xyxy must be in ORIGINAL frame pixel coords.
    All boxes (and caption backgrounds) are red.
    """

    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    red = (0, 0, 255)  # BGR

    for (x1, y1, x2, y2), cls, sc in zip(boxes_xyxy, labels, scores):
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        cv2.rectangle(out, (x1, y1), (x2, y2), red, thickness)

        name = label_map.get(int(cls), str(int(cls))) if label_map else str(int(cls))
        caption = f"{name} {float(sc):.2f}"

        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(y1 - th - baseline - 4, 0)

        # Caption background in red
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y_top + th + baseline + 6), red, -1)

        cv2.putText(
            out,
            caption,
            (x1 + 3, y_top + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return out




def annotate_video_file(model,
                        in_path: str,
                        out_path: str,
                        *,
                        device: str = "cuda",
                        model_input_size: Tuple[int, int] = (300, 300),  # (W,H)
                        score_thresh: float = 0.2,
                        nms_thresh: float = 0.5,
                        max_per_img: int = 100,
                        class_agnostic: bool = False,
                        label_map: Optional[Dict[int, str]] = None,
                        batch_size: int = 1,
                        max_frames: Optional[int] = None,  # set for quick tests
                        ) -> None:
    
    device_t = torch.device(device)
    model.eval()

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not open output video for writing: {out_path}")

    frames_bgr = []
    frame_count = 0
    in_w, in_h = model_input_size  # (300,300)

    def flush_batch():
        nonlocal frames_bgr
        if not frames_bgr:
            return

        # preprocess
        batch = torch.stack(
            [preprocess_frame_bgr_to_tensor(fr, device_t, model_input_size) for fr in frames_bgr],
            dim=0
        )  # [B,3,300,300]

        with torch.no_grad():
            preds = model.predict(
                images=batch,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                max_per_img=max_per_img,
                class_agnostic=class_agnostic,
            )

        for fr_bgr, pred in zip(frames_bgr, preds):
            boxes = pred["boxes"].detach().cpu().numpy().astype(np.float32)
            labels = pred["labels"].detach().cpu().numpy().astype(np.int32)
            scores = pred["scores"].detach().cpu().numpy().astype(np.float32)

            # map from (300,300) coords back to original frame coords
            oh, ow = fr_bgr.shape[:2]
            sx = ow / float(in_w)
            sy = oh / float(in_h)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

            annotated = draw_detections_bgr(fr_bgr, boxes, labels, scores, label_map=label_map)
            writer.write(annotated)

        frames_bgr = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames_bgr.append(frame)
        frame_count += 1

        if len(frames_bgr) >= max(1, batch_size):
            flush_batch()

        if max_frames is not None and frame_count >= max_frames:
            break

    flush_batch()
    cap.release()
    writer.release()

