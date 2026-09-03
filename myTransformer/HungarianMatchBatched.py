import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from typing import Sequence


@torch.no_grad()
def hungarian_match_batched(
    pred_class_logits: torch.Tensor,
    pred_bbox: torch.Tensor,
    gt_classes: Sequence[torch.Tensor],
    gt_bbox: Sequence[torch.Tensor],
    lambda_CE: float = 1.0,
    lambda_L1: float = 5.0,
    lambda_GIoU: float = 2.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Batched version of Hungarian matching.

    The CE, L1, and GIoU cost matrices are constructed for the whole batch
    at once. Only the final discrete Hungarian assignment is performed
    image-by-image because scipy.optimize.linear_sum_assignment accepts a
    single 2-D cost matrix.

    Parameters
    ----------
    pred_class_logits:
        Tensor of shape (B, Q, K+1). Raw class logits.

    pred_bbox:
        Tensor of shape (B, Q, 4). Predicted normalized boxes in
        (cx, cy, w, h) format.

    gt_classes:
        Sequence of length B. gt_classes[i] has shape (M_i,) and contains
        integer class labels in {0, ..., K-1}.

    gt_bbox:
        Sequence of length B. gt_bbox[i] has shape (M_i, 4) and contains
        normalized ground-truth boxes in (x1, y1, x2, y2) format.

    Returns
    -------
    matches:
        List of length B. Each element is (pred_indices, gt_indices), where
        both tensors are on the same device as pred_bbox.
    """
    if pred_class_logits.ndim != 3:
        raise ValueError("pred_class_logits must have shape (B, Q, K+1)")
    if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
        raise ValueError("pred_bbox must have shape (B, Q, 4)")

    B, Q, _ = pred_class_logits.shape
    if pred_bbox.shape[:2] != (B, Q):
        raise ValueError("pred_class_logits and pred_bbox must agree in B and Q")
    if len(gt_classes) != B or len(gt_bbox) != B:
        raise ValueError("gt_classes and gt_bbox must each contain B elements")

    device = pred_bbox.device
    target_sizes = [boxes.shape[0] for boxes in gt_bbox]

    for i, (classes_i, boxes_i) in enumerate(zip(gt_classes, gt_bbox)):
        if classes_i.ndim != 1:
            raise ValueError(f"gt_classes[{i}] must have shape (M_i,)")
        if boxes_i.ndim != 2 or boxes_i.shape[-1] != 4:
            raise ValueError(f"gt_bbox[{i}] must have shape (M_i, 4)")
        if classes_i.shape[0] != boxes_i.shape[0]:
            raise ValueError(f"gt_classes[{i}] and gt_bbox[{i}] have different M_i")
        if classes_i.device != device or boxes_i.device != device:
            raise ValueError("All predictions and targets must be on the same device")

    M_max = max(target_sizes, default=0)

    # Entire batch contains no objects.
    if M_max == 0:
        return [
            (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
            for _ in range(B)
        ]

    # ------------------------------------------------------------------
    # Pad variable-length ground-truth data to M_max.
    # Only this inexpensive packing step loops over the batch.
    # ------------------------------------------------------------------
    padded_classes = torch.zeros((B, M_max), dtype=torch.long, device=device)
    # Use a valid dummy box for padding so GIoU remains finite in padded
    # columns. Those columns are discarded before assignment.
    padded_bbox_xyxy = torch.zeros(
        (B, M_max, 4), dtype=pred_bbox.dtype, device=device
    )
    padded_bbox_xyxy[..., 2:] = 1.0

    for i, M_i in enumerate(target_sizes):
        if M_i == 0:
            continue
        padded_classes[i, :M_i] = gt_classes[i]
        padded_bbox_xyxy[i, :M_i] = gt_bbox[i].to(dtype=pred_bbox.dtype)

    # ------------------------------------------------------------------
    # 1. Batched classification cost: (B, Q, M_max)
    # ------------------------------------------------------------------
    pred_probs = torch.softmax(pred_class_logits, dim=-1)
    class_index = padded_classes[:, None, :].expand(B, Q, M_max)
    class_cost = -torch.gather(pred_probs, dim=2, index=class_index)

    # ------------------------------------------------------------------
    # 2. Batched L1 cost: (B, Q, M_max)
    # ------------------------------------------------------------------
    padded_bbox_cxcywh = torchvision.ops.box_convert(
        padded_bbox_xyxy,
        in_fmt="xyxy",
        out_fmt="cxcywh",
    )
    L1_cost = torch.cdist(
        x1=pred_bbox,
        x2=padded_bbox_cxcywh,
        p=1,
    )

    # ------------------------------------------------------------------
    # 3. Batched GIoU cost: (B, Q, M_max)
    # ------------------------------------------------------------------
    pred_bbox_xyxy = torchvision.ops.box_convert(
        pred_bbox,
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )
    giou = batched_generalized_box_iou(
        pred_bbox_xyxy,
        padded_bbox_xyxy,
    )
    GIoU_cost = -giou

    # ------------------------------------------------------------------
    # 4. Total batched matching cost: (B, Q, M_max)
    # ------------------------------------------------------------------
    total_cost = (
        lambda_CE * class_cost
        + lambda_L1 * L1_cost
        + lambda_GIoU * GIoU_cost
    )

    # One device -> CPU transfer for the whole padded batch.
    total_cost_cpu = total_cost.cpu()

    # ------------------------------------------------------------------
    # 5. Hungarian assignment remains per image because SciPy is 2-D only.
    # ------------------------------------------------------------------
    matches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, M_i in enumerate(target_sizes):
        if M_i == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            matches.append((empty, empty))
            continue

        cost_i = total_cost_cpu[i, :, :M_i].numpy()
        pred_indices_np, gt_indices_np = linear_sum_assignment(cost_i)

        pred_indices = torch.as_tensor(
            pred_indices_np, dtype=torch.long, device=device
        )
        gt_indices = torch.as_tensor(
            gt_indices_np, dtype=torch.long, device=device
        )
        matches.append((pred_indices, gt_indices))

    return matches







def batched_generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    boxes1: [B, N, 4], xyxy
    boxes2: [B, M, 4], xyxy

    returns:
        [B, N, M]
    """

    # [B, N, 1, 2]
    boxes1_lt = boxes1[:, :, None, :2]
    boxes1_rb = boxes1[:, :, None, 2:]

    # [B, 1, M, 2]
    boxes2_lt = boxes2[:, None, :, :2]
    boxes2_rb = boxes2[:, None, :, 2:]

    # Intersection
    inter_lt = torch.maximum(boxes1_lt, boxes2_lt)
    inter_rb = torch.minimum(boxes1_rb, boxes2_rb)

    inter_wh = (inter_rb - inter_lt).clamp(min=0)

    # [B, N, M]
    intersection = inter_wh[..., 0] * inter_wh[..., 1]

    # Individual box areas
    area1 = (
        (boxes1[..., 2] - boxes1[..., 0])
        * (boxes1[..., 3] - boxes1[..., 1])
    )  # [B, N]

    area2 = (
        (boxes2[..., 2] - boxes2[..., 0])
        * (boxes2[..., 3] - boxes2[..., 1])
    )  # [B, M]

    # [B, N, M]
    union = (
        area1[:, :, None]
        + area2[:, None, :]
        - intersection
    )

    iou = intersection / union

    # Smallest enclosing boxes
    enclosing_lt = torch.minimum(boxes1_lt, boxes2_lt)
    enclosing_rb = torch.maximum(boxes1_rb, boxes2_rb)

    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)

    enclosing_area = (
        enclosing_wh[..., 0]
        * enclosing_wh[..., 1]
    )

    giou = iou - (enclosing_area - union) / enclosing_area

    return giou