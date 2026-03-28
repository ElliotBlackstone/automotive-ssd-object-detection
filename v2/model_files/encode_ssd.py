# encode_ssd.py
import torch
from torchvision.ops import complete_box_iou, box_iou, distance_box_iou, generalized_box_iou
from typing import Tuple


def encode_ssd(priors_cxcywh: torch.Tensor,
               priors_xyxy: torch.Tensor,
               gt_boxes_xyxy: torch.Tensor,
               gt_labels: torch.Tensor,
               iou_thresh: float = 0.5,
               background_class: int = 0,
               variances: Tuple[float, float] = (0.1, 0.2),
               iou_variant: str = "IoU",
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SSD target encoding for one image.

    Inputs
    priors_cxcywh: [P, 4] priors in cxcywh format
    priors_xyxy:   [P, 4] priors in xyxy format
    gt_boxes_xyxy: [G, 4] normalized GT boxes in xyxy format
    gt_labels:     [G]    GT labels in {0, ..., num_foreground_classes - 1}
    iou_thresh: matching threshold
    background_class: must be 0
    variances: (center_var, size_var)
    iou_variant: string that must be on of "IoU", "GIoU", "DIoU", "CIoU"

    Returns
    loc_pos:   [N_pos, 4] encoded offsets only for positive priors
    cls_target:[P]        class targets, with 0 reserved for background
    pos_mask:  [P]        boolean mask of positives
    """
    if background_class != 0:
        raise ValueError(f"background_class must be 0, got {background_class}")
    if not (0.0 < iou_thresh < 1.0):
        raise ValueError(f"iou_thresh must be in (0, 1), got {iou_thresh}")

    device = priors_cxcywh.device
    dtype = priors_cxcywh.dtype
    P = priors_cxcywh.shape[0]
    G = gt_boxes_xyxy.shape[0]

    gt_labels = gt_labels.to(dtype=torch.long)

    # Empty image: all priors are background, no localization targets.
    if G == 0:
        cls_target = torch.zeros((P,), dtype=torch.long, device=device)
        pos_mask = torch.zeros((P,), dtype=torch.bool, device=device)
        loc_pos = torch.zeros((0, 4), dtype=dtype, device=device)
        return loc_pos, cls_target, pos_mask

    # Standard SSD matching uses plain IoU.
    iou = compute_box_metric(priors_xyxy, gt_boxes_xyxy, iou_variant)  # [P, G]

    # Force bipartite matches so every GT gets at least one prior.
    best_prior_per_gt = iou.argmax(dim=0)  # [G]
    iou[best_prior_per_gt, torch.arange(G, device=device)] = 2.0

    # Best GT for each prior.
    best_iou_per_prior, best_gt_per_prior = iou.max(dim=1)  # both [P]
    pos_mask = best_iou_per_prior >= iou_thresh

    # Classification targets for all priors.
    matched_labels = gt_labels[best_gt_per_prior]  # [P]
    cls_target = torch.zeros((P,), dtype=torch.long, device=device)
    cls_target[pos_mask] = matched_labels[pos_mask] + 1  # reserve 0 for background

    # Localization targets only for positives.
    pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)  # [N_pos]
    priors_pos = priors_cxcywh[pos_idx]                    # [N_pos, 4]
    gt_xyxy_pos = gt_boxes_xyxy[best_gt_per_prior[pos_idx]]  # [N_pos, 4]

    gt_cxy = 0.5 * (gt_xyxy_pos[:, :2] + gt_xyxy_pos[:, 2:])
    gt_wh = gt_xyxy_pos[:, 2:] - gt_xyxy_pos[:, :2]

    v_c, v_s = variances
    t_xy = (gt_cxy - priors_pos[:, :2]) / priors_pos[:, 2:] / v_c
    t_wh = torch.log((gt_wh / priors_pos[:, 2:]).clamp_min(1e-12)) / v_s
    loc_pos = torch.cat((t_xy, t_wh), dim=1)  # [N_pos, 4]

    return loc_pos, cls_target, pos_mask


def compute_box_metric(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    variant: str = "IoU",
) -> torch.Tensor:
    """
    Compute pairwise box metric between boxes1 and boxes2.

    Args:
        boxes1: Tensor[N, 4] in xyxy format
        boxes2: Tensor[M, 4] in xyxy format
        variant: one of {"IoU", "GIoU", "DIoU", "CIoU"}

    Returns:
        Tensor[N, M] of pairwise scores
    """

    key = variant.strip().upper()

    metric_fns = {
        "IOU": box_iou,
        "GIOU": generalized_box_iou,
        "DIOU": distance_box_iou,
        "CIOU": complete_box_iou,
    }

    return metric_fns[key](boxes1, boxes2)