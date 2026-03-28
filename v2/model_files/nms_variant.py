# nms_variant.py

import torch
import gen_nms


def run_nms_by_variant(boxes: torch.Tensor,
                       scores: torch.Tensor,
                       nms_thresh: float,
                       variant: str = "IoU",
                       class_agnostic: bool = True,
                       idxs: torch.Tensor | None = None,
                       ) -> torch.Tensor:
    """
    Select and run the appropriate NMS function based on IoU variant.

    Args:
        boxes: Tensor[N, 4] in xyxy format
        scores: Tensor[N]
        nms_thresh: suppression threshold
        variant: one of {"IoU", "GIoU", "DIoU", "CIoU"}
        class_agnostic: if True, run class-agnostic NMS
        idxs: Tensor[N] of class ids, required if class_agnostic=False

    Returns:
        keep indices as a Tensor
    """
    if not isinstance(variant, str):
        raise TypeError("variant must be a string")

    key = variant.strip().upper()

    if not class_agnostic and idxs is None:
        raise ValueError("idxs must be provided when class_agnostic=False")

    class_agnostic_fns = {
                          "IOU":  lambda b, s, t: gen_nms.ops.iou_nms(boxes=b, scores=s, iou_threshold=t),
                          "GIOU": lambda b, s, t: gen_nms.ops.giou_nms(boxes=b, scores=s, giou_threshold=t),
                          "DIOU": lambda b, s, t: gen_nms.ops.diou_nms(boxes=b, scores=s, diou_threshold=t),
                          "CIOU": lambda b, s, t: gen_nms.ops.ciou_nms(boxes=b, scores=s, ciou_threshold=t),
                         }

    batched_fns = {
                   "IOU":  lambda b, s, i, t: gen_nms.ops.batched_iou_nms(boxes=b, scores=s, idxs=i, iou_threshold=t),
                   "GIOU": lambda b, s, i, t: gen_nms.ops.batched_giou_nms(boxes=b, scores=s, idxs=i, giou_threshold=t),
                   "DIOU": lambda b, s, i, t: gen_nms.ops.batched_diou_nms(boxes=b, scores=s, idxs=i, diou_threshold=t),
                   "CIOU": lambda b, s, i, t: gen_nms.ops.batched_ciou_nms(boxes=b, scores=s, idxs=i, ciou_threshold=t),
                  }

    allowed = tuple(class_agnostic_fns.keys())
    if key not in class_agnostic_fns:
        raise ValueError(f"Unsupported variant {variant!r}. Expected one of {allowed}")

    if class_agnostic:
        return class_agnostic_fns[key](boxes, scores, nms_thresh)
    else:
        return batched_fns[key](boxes, scores, idxs, nms_thresh)