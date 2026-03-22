import torch
from torchvision.ops import box_convert, complete_box_iou
from typing import Tuple


def encode_ssd(priors_cxcywh: torch.Tensor,
               priors_xyxy: torch.Tensor,
               gt_boxes_xyxy: torch.Tensor,
               gt_labels: torch.Tensor,
               iou_thresh: float = 0.5,
               background_class: int = 0,
               variances: Tuple[float, float] = (0.1, 0.2),
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs
        priors_cxcywh: priors in 'cxcywh' format (8732 in total)
        priors_xyxy: priors in 'xyxy' format (8732 in total)
        variances: (center, size) (both should be positive)
        gt_boxes_xyxy: Ground truth (GT) bounding boxes tensor in 'xyxy' format
        gt_labels: Tensor containing labels (0, 1, ..., C-2, where C is the total
                   number of classes, including background) corresponding to GT boxes
        background_class: integer denoting background class, must be 0

        Returns:
        loc_target: [P,4] (tx,ty,tw,th) per prior (positives encoded, negatives filled too)
        cls_target: [P]   background for negatives, matched GT label for positives
        pos_mask:   [P]   boolean positives
        matched_gt_cxcywh: [P,4] GT boxes matched to each prior (cxcywh, normalized)
        (P is the number of priors, 8732)
        """

        # only works if background_class = 0
        if background_class != 0:
            raise ValueError(f"Background should be 0, recieved {background_class}.")
        
        device = priors_cxcywh.device
        dtype  = priors_cxcywh.dtype

        G = gt_boxes_xyxy.shape[0]
        P = priors_cxcywh.shape[0]

        # Edge case: no GT in the image
        if G == 0:
            cls_target = torch.full((P,), background_class, dtype=gt_labels.dtype, device=device)
            loc_target = torch.zeros((P, 4), dtype=dtype, device=device)
            pos_mask   = torch.zeros((P,), dtype=torch.bool, device=device)
            matched_gt_cxcywh = torch.zeros((P, 4), dtype=dtype, device=device)
            return loc_target, cls_target, pos_mask, matched_gt_cxcywh

        # IoU and matching
        iou = complete_box_iou(priors_xyxy, gt_boxes_xyxy)           # [P,G]
        # Force bipartite matches: each GT gets its best prior
        best_prior_per_gt = iou.argmax(dim=0)                # [G]
        iou[best_prior_per_gt, torch.arange(G, device=device)] = 2.0

        best_gt_per_prior  = iou.argmax(dim=1)               # [P]
        best_iou_per_prior = iou.gather(1, best_gt_per_prior.view(-1,1)).squeeze(1)
        pos_mask = best_iou_per_prior >= iou_thresh

        # matched_gt_xyxy  = gt_boxes_xyxy[best_gt_per_prior]  # [P,4]
        gt_boxes_cxcywh = box_convert(boxes=gt_boxes_xyxy, in_fmt='xyxy', out_fmt='cxcywh')
        matched_gt_cxcywh = gt_boxes_cxcywh[best_gt_per_prior]  # [P,4]

        # Encode offsets (inverse of SSD decode)
        v_c, v_s = variances
        t_xy = (matched_gt_cxcywh[:, :2] - priors_cxcywh[:, :2]) / priors_cxcywh[:, 2:] / v_c
        t_wh = torch.log(
            (matched_gt_cxcywh[:, 2:] / priors_cxcywh[:, 2:]).clamp(min=1e-12)
        ) / v_s

        loc_target = torch.empty_like(priors_cxcywh)
        loc_target[:, :2] = t_xy
        loc_target[:, 2:] = t_wh

        # Class targets
        matched_labels = gt_labels[best_gt_per_prior]        # [P]
        cls_target = torch.full((P,), background_class, dtype=matched_labels.dtype, device=device)
        cls_target[pos_mask] = matched_labels[pos_mask] + 1  # shift by 1 because 0 is reserved for 'background'

        return loc_target, cls_target, pos_mask, matched_gt_cxcywh