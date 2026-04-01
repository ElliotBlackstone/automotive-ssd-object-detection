# build_targets.py
import torch
from torchvision.ops import complete_box_iou, box_iou, distance_box_iou, generalized_box_iou
from typing import Tuple, Dict, List

from ..model_files.encode_ssd import encode_ssd



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
    if not (0.0 < iou_thresh < 1.0):
        raise ValueError(f"iou_thresh must be in (0, 1), got {iou_thresh}")

    device = priors_cxcywh.device
    dtype = priors_cxcywh.dtype
    B = len(targets)
    P = priors_cxcywh.shape[0]

    norm = priors_cxcywh.new_tensor((W, H, W, H))

    cls_t = torch.empty((B, P), dtype=torch.long, device=device)
    pos_mask = torch.empty((B, P), dtype=torch.bool, device=device)
    loc_pos_list = []

    for i, t in enumerate(targets):
        gt_xyxy_px = t["boxes"].to(dtype=dtype)
        gt_labels = t["labels"].to(dtype=torch.long)

        if gt_xyxy_px.shape[0] == 0:
            gt_xyxy = gt_xyxy_px.new_zeros((0, 4))
        else:
            gt_xyxy = gt_xyxy_px / norm

        loc_pos_i, cls_t_i, pos_mask_i = encode_ssd(priors_cxcywh=priors_cxcywh,
                                                    priors_xyxy=priors_xyxy,
                                                    gt_boxes_xyxy=gt_xyxy,
                                                    gt_labels=gt_labels,
                                                    iou_thresh=iou_thresh,
                                                    background_class=0,
                                                    variances=variances,
                                                    iou_variant=iou_variant)

        cls_t[i] = cls_t_i
        pos_mask[i] = pos_mask_i
        loc_pos_list.append(loc_pos_i)

    if loc_pos_list:
        loc_t_pm = torch.cat(loc_pos_list, dim=0)
    else:
        loc_t_pm = priors_cxcywh.new_zeros((0, 4))

    return pos_mask, loc_t_pm, cls_t



def build_targets_2(priors_cxcywh, priors_xyxy, targets, H=300, W=300,
                    iou_thresh=0.50, variances=(0.1, 0.2), iou_variant="IoU"):
    device = priors_cxcywh.device
    dtype  = priors_cxcywh.dtype
    B = len(targets)
    P = priors_cxcywh.shape[0]
    norm = priors_cxcywh.new_tensor((W, H, W, H))

    gt_boxes_list  = [t["boxes"].to(dtype=dtype, device=device) / norm for t in targets]
    gt_labels_list = [t["labels"].to(dtype=torch.long, device=device) for t in targets]

    gt_counts = torch.tensor([g.shape[0] for g in gt_boxes_list], device=device, dtype=torch.long)
    max_G = int(gt_counts.max().item()) if B > 0 else 0

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

    gt_boxes_matched = gt_boxes_pad.gather(1, best_gt_per_prior.unsqueeze(-1).expand(B, P, 4))
    priors_exp = priors_cxcywh.unsqueeze(0).expand(B, P, 4)

    v_c, v_s = variances
    gt_cxy = 0.5 * (gt_boxes_matched[..., :2] + gt_boxes_matched[..., 2:])
    gt_wh  = gt_boxes_matched[..., 2:] - gt_boxes_matched[..., :2]

    t_xy = (gt_cxy - priors_exp[..., :2]) / priors_exp[..., 2:] / v_c
    t_wh = torch.log((gt_wh / priors_exp[..., 2:]).clamp_min(1e-12)) / v_s

    loc_all = torch.cat((t_xy, t_wh), dim=-1)
    loc_t_pm = loc_all[pos_mask]

    return pos_mask, loc_t_pm, cls_t



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