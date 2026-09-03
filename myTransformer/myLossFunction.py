import torch
import torch.nn as nn
import torchvision



def myLossFunction(pred_class_logits,
                   pred_bbox,
                   gt_class,
                   gt_bbox,
                   lambda_class=0.4,
                   lambda_L1=0.2,
                   lambda_GIoU=0.4):
    """
    Computes linear combination of smooth L1, cross entropy, and GIoU loss.

    Inputs:
     - pred_class_logits: Tensor of shape (K+1) holding predicted class logits for a query.
     - pred_bbox: Tensor of shape (4) holding predicted bounding boxes for a query,
                    in normalized (c_x, c_y, w, h) coordinates.
     - gt_class_logits: Tensor of shape (K+1) holding ground truth class logits for a query.
     - gt_bbox: Tensor of shape (4) holding ground truth bounding boxes for a query,
                  in normalized (c_x, c_y, w, h) coordinates.
     - lambda_class: Float for classification loss weight.
     - lambda_L1: Float for smooth L1 loss weight.
     - lambda_IoU: Float for IoU loss weight.
    """

    # Classification loss
    class_loss = nn.functional.cross_entropy(
        pred_class_logits,
        gt_class
    )

    # Bounding-box L1 loss in (cx, cy, w, h)
    L1_loss = nn.functional.l1_loss(
        pred_bbox,
        gt_bbox,
        reduction='mean'
    )

    # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
    pred_xyxy = torch.stack([
        pred_bbox[0] - pred_bbox[2] / 2,
        pred_bbox[1] - pred_bbox[3] / 2,
        pred_bbox[0] + pred_bbox[2] / 2,
        pred_bbox[1] + pred_bbox[3] / 2
    ])

    gt_xyxy = torch.stack([
        gt_bbox[0] - gt_bbox[2] / 2,
        gt_bbox[1] - gt_bbox[3] / 2,
        gt_bbox[0] + gt_bbox[2] / 2,
        gt_bbox[1] + gt_bbox[3] / 2
    ])

    GIoU_loss = torchvision.ops.generalized_box_iou_loss(
        pred_xyxy,
        gt_xyxy,
        reduction='mean'
    )

    return (
        lambda_class * class_loss
        + lambda_L1 * L1_loss
        + lambda_GIoU * GIoU_loss
    )