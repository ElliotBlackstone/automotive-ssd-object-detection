from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, Optional

import numpy as np
from PIL import Image, ImageOps
import cv2

import torch
import torchvision.transforms.v2 as v2

import onnxruntime as ort



# @dataclass(frozen=True)
# class SSDPostprocessConfig:
#     score_thresh: float = 0.30
#     nms_thresh: float = 0.50
#     max_per_img: int = 100
#     class_agnostic: bool = False
#     variances: Tuple[float, float] = (0.1, 0.2)  # (v_c, v_s)


# class SSDInt8ONNXPredictor:
#     """
#     Pipeline:
#       1) preprocess: RGB -> float32 -> resize 300x300 -> ImageNet normalize -> NCHW
#       2) inference: ORT session.run -> (loc_all, conf_all)
#       3) postprocess: mimic mySSD.predict() logic (threshold-before-decode, decode_ssd, iou_nms)
#     """

#     def __init__(
#         self,
#         onnx_model_path: str,
#         class_names_fg: Sequence[str],
#         providers: Sequence[str] | None = None,
#         #post_cfg: SSDPostprocessConfig = SSDPostprocessConfig(),
#         preprocess_backend: str = "numpy",  # "numpy" or "torchvision"
#     ):
#         """
#         class_names_fg: foreground class names in the SAME order as your training labels
#                         (length = C-1, where C includes background).
#                         Example: ["car", "truck", "pedestrian", ...]
#         """
#         if len(class_names_fg) == 0:
#             raise ValueError("class_names_fg must be non-empty (foreground classes only).")

#         self.class_names_fg = list(class_names_fg)
#         self.post_cfg = post_cfg

#         if providers is None:
#             providers = ["CPUExecutionProvider"]

#         sess_opts = ort.SessionOptions()
#         self.sess = ort.InferenceSession(onnx_model_path, sess_options=sess_opts, providers=list(providers))

#         # Input / output names
#         inputs = self.sess.get_inputs()
#         if len(inputs) != 1:
#             raise ValueError(f"Expected 1 input, got {len(inputs)}.")
#         self.input_name = inputs[0].name

#         out_names = [o.name for o in self.sess.get_outputs()]
#         if "loc" not in out_names or "conf" not in out_names:
#             raise ValueError(f"Expected outputs named 'loc' and 'conf'. Got: {out_names}")
#         self.loc_name = "loc"
#         self.conf_name = "conf"

#         # Priors (8732,4) in normalized cxcywh, as in your model.
#         # create_default_boxes() is your canonical SSD300 prior generator.
#         self.priors_cxcywh = mySSD.create_default_boxes().to(dtype=torch.float32, device="cpu")  # (8732,4)

#         self.preprocess_backend = preprocess_backend

#         # Cache mean/std for numpy path (RGB order)
#         self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
#         self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

#         # Preprocess exactly like your show_prediction_side_by_side pipeline:
#         # ToImage -> ToDtype(float32, scale=True) -> Resize(300,300, antialias=True) -> Normalize(ImageNet)
#         self.preprocess_tfms = v2.Compose([
#             v2.ToImage(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Resize((300, 300), antialias=True),
#             v2.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
#         ])

#     def preprocess(self, image):
#         """
#         Returns:
#         x: float32 numpy array shaped (1,3,300,300)
#         (orig_w, orig_h)
#         """
#         # If user wants numpy backend, funnel everything into a numpy RGB array first.
#         if self.preprocess_backend == "numpy":
#             rgb = self._to_numpy_rgb(image)     # HWC RGB
#             orig_h, orig_w = rgb.shape[:2]
#             x = self.preprocess_numpy(rgb)      # (1,3,300,300)
#             return x, (orig_w, orig_h)

#         # Otherwise use your original torchvision/PIL path
#         pil = self._to_pil_rgb(image)
#         pil = ImageOps.exif_transpose(pil)
#         orig_w, orig_h = pil.size
#         x_t = self.preprocess_tfms(pil).unsqueeze(0).contiguous()
#         x = x_t.cpu().numpy().astype(np.float32)
#         return x, (orig_w, orig_h)

#     def preprocess_numpy(self, rgb: np.ndarray) -> np.ndarray:
#         """
#         NumPy/OpenCV-only preprocessing:
#         - expects HWC RGB
#         - converts to float32 in [0,1]
#         - resize to 300x300
#         - ImageNet normalize
#         - output NCHW with batch: (1,3,300,300) float32
#         """
#         if not isinstance(rgb, np.ndarray):
#             raise TypeError(f"preprocess_numpy expects np.ndarray, got {type(rgb)}")
#         if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
#             raise ValueError(f"Expected HxWx3(/4) RGB array, got shape {rgb.shape}")

#         # Drop alpha if present
#         if rgb.shape[2] == 4:
#             rgb = rgb[:, :, :3]

#         # Ensure contiguous
#         rgb = np.ascontiguousarray(rgb)

#         # Convert to float32 [0,1]
#         if rgb.dtype == np.uint8:
#             x = rgb.astype(np.float32) / 255.0
#         else:
#             x = rgb.astype(np.float32)
#             # If it looks like [0,255], scale down; else assume already [0,1]
#             mx = float(np.nanmax(x))
#             if mx > 1.5:
#                 x = x / 255.0
#             # Replace NaNs/Infs defensively
#             x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

#         # Resize to 300x300
#         h, w = x.shape[:2]
#         if (h, w) != (300, 300):
#             # INTER_AREA is usually better for downsampling; INTER_LINEAR for upsampling
#             interp = cv2.INTER_AREA if (h > 300 or w > 300) else cv2.INTER_LINEAR
#             x = cv2.resize(x, (300, 300), interpolation=interp)

#         # ImageNet normalize (RGB)
#         x = (x - self._mean) / self._std  # still HWC

#         # HWC -> CHW, add batch
#         x = np.transpose(x, (2, 0, 1))          # (3,300,300)
#         x = np.expand_dims(x, axis=0)           # (1,3,300,300)
#         return x.astype(np.float32, copy=False)

#     @staticmethod
#     def _to_numpy_rgb(image):
#         """
#         Convert input to HWC RGB numpy array.
#         - If you pass a cv2 frame (BGR), convert before calling predictor: frame[..., ::-1]
#         """
#         if isinstance(image, str):
#             pil = Image.open(image).convert("RGB")
#             pil = ImageOps.exif_transpose(pil)
#             return np.asarray(pil)

#         if isinstance(image, Image.Image):
#             pil = image.convert("RGB")
#             pil = ImageOps.exif_transpose(pil)
#             return np.asarray(pil)

#         if isinstance(image, np.ndarray):
#             arr = image
#             if arr.ndim != 3:
#                 raise ValueError(f"Expected HxWxC array, got shape {arr.shape}")
#             if arr.shape[2] not in (3, 4):
#                 raise ValueError(f"Expected 3 or 4 channels, got shape {arr.shape}")
#             return arr

#         raise TypeError(f"Unsupported image type: {type(image)}")

#     def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         x: (1,3,300,300) float32
#         returns:
#           loc_all: (1,8732,4) float32
#           conf_all: (1,8732,C) float32 logits
#         """
#         loc_all, conf_all = self.sess.run([self.loc_name, self.conf_name], {self.input_name: x})
#         return loc_all, conf_all

#     @torch.no_grad()
#     def postprocess(
#         self,
#         loc_all: np.ndarray,
#         conf_all: np.ndarray,
#         orig_size: Tuple[int, int],
#     ) -> Dict[str, Any]:
#         """
#         Returns dict:
#           labels: list[str]
#           scores: list[float]
#           boxes : np.ndarray (K,4) xyxy in ORIGINAL image pixel coords
#         """
#         orig_w, orig_h = orig_size
#         cfg = self.post_cfg

#         # Squeeze batch
#         if loc_all.ndim == 3:
#             loc = loc_all[0]
#         else:
#             raise ValueError(f"loc_all must have shape (1,P,4). Got {loc_all.shape}")

#         if conf_all.ndim == 3:
#             conf = conf_all[0]
#         else:
#             raise ValueError(f"conf_all must have shape (1,P,C). Got {conf_all.shape}")

#         # To torch CPU
#         loc_t = torch.from_numpy(loc).to(dtype=torch.float32, device="cpu")    # (P,4)
#         conf_t = torch.from_numpy(conf).to(dtype=torch.float32, device="cpu")  # (P,C)

#         P, C = conf_t.shape
#         if self.priors_cxcywh.shape[0] != P:
#             raise ValueError(f"Prior count mismatch: priors={self.priors_cxcywh.shape[0]} vs outputs P={P}")

#         num_fg = C - 1
#         if len(self.class_names_fg) != num_fg:
#             raise ValueError(
#                 f"class_names_fg length must be C-1={num_fg}. "
#                 f"Got {len(self.class_names_fg)}."
#             )

#         # This mirrors your mySSD.predict:
#         # scores_all = softmax(conf)[..., 1:] and threshold-before-decode. :contentReference[oaicite:5]{index=5}
#         scores_fg = conf_t.softmax(dim=-1)[..., 1:]  # (P, C-1)

#         keep_mask = scores_fg > cfg.score_thresh
#         if not keep_mask.any():
#             return {"labels": [], "scores": [], "boxes": np.zeros((0, 4), dtype=np.float32)}

#         pri_idx, cls0_idx = keep_mask.nonzero(as_tuple=True)  # (M,), (M,)
#         loc_sel = loc_t[pri_idx]                              # (M,4)
#         pri_sel = self.priors_cxcywh[pri_idx]                 # (M,4)

#         # decode_ssd produces normalized cxcywh. :contentReference[oaicite:6]{index=6}
#         boxes_cxcywh = mySSD.decode_ssd(loc=loc_sel, priors=pri_sel, variances=cfg.variances)  # (M,4)

#         # Convert to xyxy in 300x300 model pixel coords exactly as your predict does. :contentReference[oaicite:7]{index=7}
#         cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
#         x1 = (cx - 0.5 * w).clamp(0, 1) * 300.0
#         y1 = (cy - 0.5 * h).clamp(0, 1) * 300.0
#         x2 = (cx + 0.5 * w).clamp(0, 1) * 300.0
#         y2 = (cy + 0.5 * h).clamp(0, 1) * 300.0
#         sel_boxes_300 = torch.stack([x1, y1, x2, y2], dim=1)  # (M,4)

#         sel_scores = scores_fg[pri_idx, cls0_idx]   # (M,)
#         sel_labels0 = cls0_idx.to(torch.int64)      # (M,) 0-based foreground labels

#         # NMS identical to your predict logic (class-agnostic optional). :contentReference[oaicite:8]{index=8}
#         if cfg.class_agnostic:
#             keep = mySSD.iou_nms(sel_boxes_300, sel_scores, iou_threshold=cfg.nms_thresh)
#             keep = keep[sel_scores[keep].argsort(descending=True)]
#         else:
#             order = torch.argsort(sel_labels0)
#             boxes = sel_boxes_300[order]
#             scores = sel_scores[order]
#             labels = sel_labels0[order]

#             kept = []
#             i = 0
#             N = labels.numel()
#             while i < N:
#                 c = labels[i].item()
#                 j = i + 1
#                 while j < N and labels[j].item() == c:
#                     j += 1
#                 local_keep = mySSD.iou_nms(boxes[i:j], scores[i:j], iou_threshold=cfg.nms_thresh)
#                 kept.append(torch.arange(i, j, device=boxes.device)[local_keep])
#                 i = j

#             keep = torch.cat(kept, dim=0)
#             keep = order[keep]
#             keep = keep[sel_scores[keep].argsort(descending=True)]

#         keep = keep[: cfg.max_per_img]

#         boxes_300 = sel_boxes_300[keep]   # (K,4) in 300x300 coords
#         scores_out = sel_scores[keep]
#         labels0_out = sel_labels0[keep]

#         # Scale boxes from 300x300 to ORIGINAL image pixel coords
#         sx = float(orig_w) / 300.0
#         sy = float(orig_h) / 300.0
#         boxes_orig = boxes_300.clone()
#         boxes_orig[:, [0, 2]] *= sx
#         boxes_orig[:, [1, 3]] *= sy

#         # Output format
#         labels_str = [self.class_names_fg[int(i)] for i in labels0_out.tolist()]
#         scores_list = [float(s) for s in scores_out.tolist()]
#         boxes_np = boxes_orig.cpu().numpy().astype(np.float32)

#         return {"labels": labels_str, "scores": scores_list, "boxes": boxes_np}

#     def __call__(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
#         x, orig_size = self.preprocess(image)
#         loc_all, conf_all = self.infer(x)
#         return self.postprocess(loc_all, conf_all, orig_size)

#     @staticmethod
#     def _to_pil_rgb(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
#         if isinstance(image, str):
#             return Image.open(image).convert("RGB")
#         if isinstance(image, Image.Image):
#             return image.convert("RGB")
#         if isinstance(image, np.ndarray):
#             arr = image
#             if arr.ndim != 3 or arr.shape[2] != 3:
#                 raise ValueError(f"Expected HxWx3 numpy image, got shape {arr.shape}")
#             # Assume it's already RGB; if you pass BGR (cv2), convert before calling:
#             # arr = arr[..., ::-1]
#             return Image.fromarray(arr.astype(np.uint8), mode="RGB")
#         raise TypeError(f"Unsupported image type: {type(image)}")





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
