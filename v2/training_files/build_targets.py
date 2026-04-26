# build_targets.py
import torch
from torchvision.ops import complete_box_iou, box_iou, distance_box_iou, generalized_box_iou
from typing import Tuple, Dict, List



def move_targets_to_device(targets, device, non_blocking=True):
    moved = []
    for t in targets:
        moved.append({
            k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
            for k, v in t.items()
        })
    return moved





def build_targets(priors_cxcywh: torch.Tensor,
                  priors_xyxy: torch.Tensor,
                  targets: List[Dict],
                  H: int = 300,
                  W: int = 300,
                  iou_thresh: float = 0.50,
                  variances: Tuple[float, float] = (0.1, 0.2),
                  iou_variant: str = "IoU",
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs
    priors_cxcywh: [P, 4] priors in cxcywh format
    priors_xyxy:   [P, 4] priors in xyxy format
    targets: list of length B, each with keys 'boxes' and 'labels'
    H, W: image size used for normalization
    iou_thresh: matching threshold
    variances: SSD encoding variances
    iou_variant: string that must be on of "IoU", "GIoU", "DIoU", "CIoU"

    Returns
    pos_mask: [B, P] boolean positive mask
    loc_t_pm: [N_pos_total, 4] localization targets for positives only
    cls_t:    [B, P] class targets
    """
    device = priors_cxcywh.device
    dtype  = priors_cxcywh.dtype
    B = len(targets)
    P = priors_cxcywh.shape[0]
    norm = priors_cxcywh.new_tensor((W, H, W, H))

    gt_boxes_list  = [t["boxes"].to(dtype=dtype, device=device) / norm for t in targets]
    gt_labels_list = [t["labels"].to(dtype=torch.long, device=device) for t in targets]

    gt_counts = [g.shape[0] for g in gt_boxes_list]
    max_G = max(gt_counts, default=0)

    cls_t    = torch.zeros((B, P), dtype=torch.long, device=device)
    pos_mask = torch.zeros((B, P), dtype=torch.bool, device=device)

    if max_G == 0:
        return pos_mask, priors_cxcywh.new_zeros((0, 4)), cls_t

    gt_boxes_pad  = priors_cxcywh.new_zeros((B, max_G, 4))
    gt_labels_pad = torch.zeros((B, max_G), dtype=torch.long, device=device)
    gt_valid_mask = torch.zeros((B, max_G), dtype=torch.bool, device=device)

    for i, (boxes, labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
        G = boxes.shape[0]
        if G > 0:
            gt_boxes_pad[i, :G] = boxes
            gt_labels_pad[i, :G] = labels
            gt_valid_mask[i, :G] = True

    priors_batch = priors_xyxy.unsqueeze(0).expand(B, P, 4)
    iou = compute_box_metric(priors_batch, gt_boxes_pad, iou_variant)
    iou.masked_fill_(~gt_valid_mask[:, None, :], -1.0)

    best_prior_per_gt = iou.argmax(dim=1)

    b_idx = torch.arange(B, device=device)[:, None].expand(B, max_G)
    g_idx = torch.arange(max_G, device=device)[None, :].expand(B, max_G)
    valid_pairs = gt_valid_mask
    iou[b_idx[valid_pairs], best_prior_per_gt[valid_pairs], g_idx[valid_pairs]] = 2.0

    best_iou_per_prior, best_gt_per_prior = iou.max(dim=2)
    pos_mask = best_iou_per_prior >= iou_thresh

    matched_labels = gt_labels_pad.gather(1, best_gt_per_prior)
    cls_t[pos_mask] = matched_labels[pos_mask] + 1


    # Encode only positive priors.
    # This preserves the same ordering as the old loc_all[pos_mask]:
    # batch-major, then prior-major.
    pos_idx = pos_mask.nonzero(as_tuple=False)   # [N_pos_total, 2]

    if pos_idx.numel() == 0:
        return pos_mask, priors_cxcywh.new_zeros((0, 4)), cls_t

    b_pos = pos_idx[:, 0]                        # [N_pos_total]
    p_pos = pos_idx[:, 1]                        # [N_pos_total]

    matched_gt_pos = best_gt_per_prior[b_pos, p_pos]   # [N_pos_total]
    gt_boxes_pos   = gt_boxes_pad[b_pos, matched_gt_pos]  # [N_pos_total, 4]
    priors_pos     = priors_cxcywh[p_pos]               # [N_pos_total, 4]

    v_c, v_s = variances
    gt_cxy = 0.5 * (gt_boxes_pos[:, :2] + gt_boxes_pos[:, 2:])
    gt_wh  = gt_boxes_pos[:, 2:] - gt_boxes_pos[:, :2]

    t_xy = (gt_cxy - priors_pos[:, :2]) / priors_pos[:, 2:] / v_c
    t_wh = torch.log((gt_wh / priors_pos[:, 2:]).clamp_min(1e-12)) / v_s

    loc_t_pm = torch.cat((t_xy, t_wh), dim=-1)

    return pos_mask, loc_t_pm, cls_t



def compute_box_metric(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    variant: str = "IoU",
) -> torch.Tensor:
    """
    Compute pairwise box metric between boxes1 and boxes2.

    Args:
        boxes1: Tensor[B, N, 4] in xyxy format
        boxes2: Tensor[B, M, 4] in xyxy format
        variant: one of {"IoU", "GIoU", "DIoU", "CIoU"}

    Returns:
        Tensor[B, N, M] of pairwise scores
    """

    key = variant.strip().upper()

    metric_fns = {
        "IOU": box_iou,
        "GIOU": generalized_box_iou,
        "DIOU": distance_box_iou,
        "CIOU": complete_box_iou,
    }

    return metric_fns[key](boxes1, boxes2)