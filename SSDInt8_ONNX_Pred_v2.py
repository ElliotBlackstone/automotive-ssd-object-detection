from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch

from v2.model_files.SSD_from_scratch import mySSD


@dataclass(frozen=True)
class PreprocessConfig:
    input_color: str = "bgr"   # "bgr" if you pass cv2 frames; "rgb" if already converted
    resize_hw: Tuple[int, int] = (300, 300)  # (H, W)
    mean_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std_rgb: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class SSDInt8ONNXPredictorRaw:
    """
    Predictor for a raw int8 ONNX SSD model where postprocessing is done by a
    PyTorch SSD model's predict(...) method.

    External behavior matches the stitched predictor:
      input:
        - np.ndarray HxWx3 (uint8/float), either BGR or RGB per preprocess_cfg.input_color
        - or str path (read via cv2.imread -> BGR)

      returns:
        {
          "labels": list[str],
          "scores": list[float],
          "boxes":  np.ndarray (K,4) float32 xyxy in ORIGINAL image pixels
        }

    Assumptions:
      1) The ONNX model has one input and two outputs: loc_all, conf_all.
      2) pytorch_post_model.predict(...) accepts:
            x, pre_loc_all=..., pre_conf_all=...,
            score_thresh=..., nms_thresh=..., iou_variant=..., max_per_img=...
      3) pytorch_post_model.predict(...) returns either:
            {"labels": Tensor[K], "scores": Tensor[K], "boxes": Tensor[K,4]}
         or
            [ {"labels": ..., "scores": ..., "boxes": ...} ]
      4) The boxes returned by predict(...) are in model-input pixel coordinates
         (for example 300x300), so this class rescales them back to ORIGINAL image pixels.
    """

    def __init__(
        self,
        onnx_model_path: str,
        class_to_idx: Dict[str, int],
        providers: Optional[Sequence[str]] = None,
        preprocess_cfg: PreprocessConfig = PreprocessConfig(),
        output_names: Optional[Tuple[str, str]] = None,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.3,
        iou_variant: str = "DIoU",
        max_per_img: int = 100,
        postprocess_device: Optional[Union[str, torch.device]] = None,
        pred_labels_are_one_based: bool = False,
    ):
        self.pre_cfg = preprocess_cfg
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.iou_variant = str(iou_variant)
        self.max_per_img = int(max_per_img)
        self.pred_labels_are_one_based = bool(pred_labels_are_one_based)

        self.pytorch_post_model = mySSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                                        in_channels=3,
                                        variances=(0.1, 0.2)).eval()

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

        sess_outs = self.sess.get_outputs()
        sess_out_names = [o.name for o in sess_outs]

        if output_names is None:
            if len(sess_out_names) != 2:
                raise ValueError(
                    f"Expected 2 raw ONNX outputs when output_names is None, found {len(sess_out_names)}: "
                    f"{sess_out_names}"
                )
            self.out_loc, self.out_conf = sess_out_names[0], sess_out_names[1]
        else:
            self.out_loc, self.out_conf = output_names
            missing = [n for n in output_names if n not in set(sess_out_names)]
            if missing:
                raise ValueError(
                    f"Model missing outputs {missing}. Found outputs: {sorted(sess_out_names)}"
                )

        if postprocess_device is None:
            self.postprocess_device = self._infer_torch_device(self.pytorch_post_model)
        else:
            self.postprocess_device = torch.device(postprocess_device)

        self.pytorch_post_model.to(self.postprocess_device)

        self._mean = np.array(self.pre_cfg.mean_rgb, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self.pre_cfg.std_rgb, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        x_np, (orig_w, orig_h) = self.preprocess(image)

        loc_all_np, conf_all_np = self.sess.run(
            [self.out_loc, self.out_conf],
            {self.input_name: x_np},
        )

        if loc_all_np is None or conf_all_np is None:
            return self._empty_result()

        loc_all_np = np.asarray(loc_all_np, dtype=np.float32)
        conf_all_np = np.asarray(conf_all_np, dtype=np.float32)

        if loc_all_np.size == 0 or conf_all_np.size == 0:
            return self._empty_result()

        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(
            device=self.postprocess_device, dtype=torch.float32
        )
        loc_all_t = torch.from_numpy(np.ascontiguousarray(loc_all_np)).to(
            device=self.postprocess_device, dtype=torch.float32
        )
        conf_all_t = torch.from_numpy(np.ascontiguousarray(conf_all_np)).to(
            device=self.postprocess_device, dtype=torch.float32
        )

        with torch.inference_mode():
            pred = self.pytorch_post_model.predict(
                x_t,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh,
                iou_variant=self.iou_variant,
                max_per_img=self.max_per_img,
                pre_loc_all=loc_all_t,
                pre_conf_all=conf_all_t,
            )

        pred = self._normalize_predict_output(pred)

        labels_raw = pred["labels"]
        scores = pred["scores"]
        boxes = pred["boxes"]

        if boxes.numel() == 0:
            return self._empty_result()

        labels_np = labels_raw.detach().to("cpu", dtype=torch.int64).numpy().reshape(-1)
        scores_np = scores.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
        boxes_np = boxes.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1, 4)

        boxes_np = self._scale_boxes_to_original(boxes_np, orig_w=orig_w, orig_h=orig_h)
        labels_str = self._map_labels(labels_np)

        return {
            "labels": labels_str,
            "scores": [float(s) for s in scores_np.tolist()],
            "boxes": boxes_np.astype(np.float32, copy=False),
        }

    def preprocess(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Returns:
          x: float32 (1,3,H,W)
          (orig_w, orig_h) from the input image
        """
        arr = self._load_to_numpy(image)
        orig_h, orig_w = arr.shape[:2]
        x = self._preprocess_numpy(arr)
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
        img = img_hwc

        color = self.pre_cfg.input_color.lower()
        if color == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color == "rgb":
            pass
        else:
            raise ValueError(
                f"preprocess_cfg.input_color must be 'bgr' or 'rgb', got {self.pre_cfg.input_color}"
            )

        if img.dtype == np.uint8:
            x = img.astype(np.float32) / 255.0
        else:
            x = img.astype(np.float32)
            mx = float(np.nanmax(x))
            if mx > 1.5:
                x = x / 255.0
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        Ht, Wt = self.pre_cfg.resize_hw
        h, w = x.shape[:2]
        if (h, w) != (Ht, Wt):
            interp = cv2.INTER_AREA if (h > Ht or w > Wt) else cv2.INTER_LINEAR
            x = cv2.resize(x, (Wt, Ht), interpolation=interp)

        x = (x - self._mean) / self._std
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        return x.astype(np.float32, copy=False)

    def _normalize_predict_output(self, pred: Any) -> Dict[str, torch.Tensor]:
        if isinstance(pred, list):
            if len(pred) == 0:
                return {
                    "labels": torch.empty((0,), dtype=torch.int64),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                }
            pred = pred[0]

        if not isinstance(pred, dict):
            raise TypeError(f"Expected predict(...) to return dict or list[dict], got {type(pred)}")

        req = {"labels", "scores", "boxes"}
        missing = req - set(pred.keys())
        if missing:
            raise ValueError(f"predict(...) output missing keys: {sorted(missing)}")

        labels = pred["labels"]
        scores = pred["scores"]
        boxes = pred["boxes"]

        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.int64)
        if not torch.is_tensor(scores):
            scores = torch.as_tensor(scores, dtype=torch.float32)
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        return {
            "labels": labels,
            "scores": scores,
            "boxes": boxes,
        }

    def _scale_boxes_to_original(self, boxes_xyxy: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """
        Assumes predict(...) boxes are in resized model-input pixel coordinates.
        Rescales them to original image pixel coordinates.
        """
        if boxes_xyxy.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        Ht, Wt = self.pre_cfg.resize_hw
        sx = float(orig_w) / float(Wt)
        sy = float(orig_h) / float(Ht)

        boxes = boxes_xyxy.astype(np.float32, copy=True)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        return boxes

    def _map_labels(self, labels_np: np.ndarray) -> list[str]:
        labels_str: list[str] = []
        for raw_i in labels_np.tolist():
            i = int(raw_i)
            if self.pred_labels_are_one_based:
                i = i - 1

            if 0 <= i < len(self.class_names_fg):
                labels_str.append(self.class_names_fg[i])
            else:
                labels_str.append("unknown")
        return labels_str

    @staticmethod
    def _infer_torch_device(model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "labels": [],
            "scores": [],
            "boxes": np.zeros((0, 4), dtype=np.float32),
        }