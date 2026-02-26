from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, Optional

import numpy as np
from PIL import Image, ImageOps
import cv2

import torch
import torchvision.transforms.v2 as v2

import onnxruntime as ort





@dataclass(frozen=True)
class PreprocessConfig:
    input_color: str = "bgr"   # "bgr" if you pass cv2 frames; "rgb" if you already converted
    resize_hw: Tuple[int, int] = (300, 300)  # (H, W)
    mean_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std_rgb: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class SSDInt8ONNXPredictor:
    """
    Predictor for an ONNX SSD model with postprocess baked into the graph.

    Expected model I/O:
      input:  images  (1,3,300,300) float32
      output: boxes_out  (N,4) float32  normalized xyxy in [0,1]
              scores_out (N,)  float32
              labels_out (N,)  int64     foreground label ids 0..C-2
    """

    def __init__(
        self,
        onnx_model_path: str,
        class_to_idx: Dict[str, int],
        providers: Optional[Sequence[str]] = None,
        preprocess_cfg: PreprocessConfig = PreprocessConfig(),
        output_names: Tuple[str, str, str] = ("boxes_out", "scores_out", "labels_out"),
    ):
        self.pre_cfg = preprocess_cfg

        # Build foreground class names list in index order 0..C-2
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        n = len(idx_to_class)
        self.class_names_fg = [idx_to_class[i] for i in range(n)]

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(onnx_model_path, providers=list(providers))

        ins = self.sess.get_inputs()
        if len(ins) != 1:
            raise ValueError(f"Expected 1 input, got {len(ins)} inputs: {[i.name for i in ins]}")
        self.input_name = ins[0].name

        self.out_boxes, self.out_scores, self.out_labels = output_names

        outs = {o.name for o in self.sess.get_outputs()}
        missing = [n for n in output_names if n not in outs]
        if missing:
            raise ValueError(
                f"Model missing outputs {missing}. Found outputs: {sorted(outs)}. "
                f"Are you loading the stitched model?"
            )

        # Precompute mean/std for numpy path (RGB order), broadcastable to HWC
        self._mean = np.array(self.pre_cfg.mean_rgb, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self.pre_cfg.std_rgb, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        image:
          - np.ndarray HxWx3 (uint8/float), either BGR or RGB per preprocess_cfg.input_color
          - or str path (read via cv2.imread -> BGR)

        returns:
          {
            "labels": list[str],
            "scores": list[float],
            "boxes":  np.ndarray (K,4) float32 xyxy in ORIGINAL image pixels
          }
        """
        x, (orig_w, orig_h) = self.preprocess(image)

        boxes_norm, scores, labels0 = self.sess.run(
            [self.out_boxes, self.out_scores, self.out_labels],
            {self.input_name: x},
        )

        # Defensive empty handling
        if boxes_norm is None:
            return {"labels": [], "scores": [], "boxes": np.zeros((0, 4), dtype=np.float32)}

        boxes_norm = np.asarray(boxes_norm, dtype=np.float32)
        if boxes_norm.size == 0:
            return {"labels": [], "scores": [], "boxes": np.zeros((0, 4), dtype=np.float32)}

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels0 = np.asarray(labels0, dtype=np.int64).reshape(-1)

        # Scale normalized xyxy -> original pixel coords
        boxes = boxes_norm.copy()
        boxes[:, [0, 2]] *= float(orig_w)
        boxes[:, [1, 3]] *= float(orig_h)

        # Map labels
        labels_str = []
        for i in labels0.tolist():
            if 0 <= i < len(self.class_names_fg):
                labels_str.append(self.class_names_fg[i])
            else:
                labels_str.append("unknown")

        return {
            "labels": labels_str,
            "scores": [float(s) for s in scores.tolist()],
            "boxes": boxes.astype(np.float32, copy=False),
        }

    def preprocess(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Returns:
          x: float32 (1,3,300,300)
          (orig_w, orig_h) from the *input* image
        """
        arr = self._load_to_numpy(image)  # HWC BGR or RGB per cfg.input_color

        orig_h, orig_w = arr.shape[:2]
        x = self._preprocess_numpy(arr)  # (1,3,300,300)
        return x, (orig_w, orig_h)

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
        """
        NumPy/OpenCV-only equivalent of:
          - ToDtype(float32, scale=True)
          - Resize(300,300)
          - Normalize(ImageNet mean/std, RGB)
        Output: NCHW float32 with batch.
        """
        img = img_hwc

        # Convert to RGB if needed
        color = self.pre_cfg.input_color.lower()
        if color == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color == "rgb":
            pass
        else:
            raise ValueError(f"preprocess_cfg.input_color must be 'bgr' or 'rgb', got {self.pre_cfg.input_color}")

        # float32 in [0,1]
        if img.dtype == np.uint8:
            x = img.astype(np.float32) / 255.0
        else:
            x = img.astype(np.float32)
            # If it looks like [0,255], scale down; else assume already [0,1]
            mx = float(np.nanmax(x))
            if mx > 1.5:
                x = x / 255.0
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        # Resize to 300x300
        Ht, Wt = self.pre_cfg.resize_hw
        h, w = x.shape[:2]
        if (h, w) != (Ht, Wt):
            interp = cv2.INTER_AREA if (h > Ht or w > Wt) else cv2.INTER_LINEAR
            x = cv2.resize(x, (Wt, Ht), interpolation=interp)

        # Normalize (RGB)
        x = (x - self._mean) / self._std  # HWC

        # HWC -> NCHW
        x = np.transpose(x, (2, 0, 1))        # (3,H,W)
        x = np.expand_dims(x, axis=0)         # (1,3,H,W)
        return x.astype(np.float32, copy=False)
