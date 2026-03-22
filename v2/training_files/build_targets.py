# build_targets.py
import torch
from typing import Tuple, Dict, List

from ..model_files.encode_ssd import encode_ssd


def build_targets(
    priors_cxcywh: torch.Tensor,
    priors_xyxy: torch.Tensor,
    targets: List[Dict],
    H: int = 300,
    W: int = 300,
    iou_thresh: float = 0.50,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs
    priors_cxcywh: [P, 4] priors in cxcywh format
    priors_xyxy:   [P, 4] priors in xyxy format
    targets: list of length B, each with keys 'boxes' and 'labels'
    H, W: image size used for normalization
    iou_thresh: matching threshold
    variances: SSD encoding variances

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
                                                    variances=variances,)

        cls_t[i] = cls_t_i
        pos_mask[i] = pos_mask_i
        loc_pos_list.append(loc_pos_i)

    if loc_pos_list:
        loc_t_pm = torch.cat(loc_pos_list, dim=0)
    else:
        loc_t_pm = priors_cxcywh.new_zeros((0, 4))

    return pos_mask, loc_t_pm, cls_t