import torch
import torchvision
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match(pred_class_logits: torch.Tensor,
                    pred_bbox: torch.Tensor,
                    gt_classes: torch.Tensor,
                    gt_bbox: torch.Tensor,
                    lambda_CE: float = 1.0,
                    lambda_L1: float = 5.0,
                    lambda_GIoU: float = 2.0,
                    ) -> torch.Tensor:
    """
    Match predicted object queries to ground-truth objects using
    Hungarian bipartite matching.

    Inputs
    ------
    pred_class_logits:
        Tensor of shape (Q, K+1).
        Raw class logits for each predicted query.

    pred_bbox:
        Tensor of shape (Q, 4).
        Predicted boxes in normalized (cx, cy, w, h) format.

    gt_classes:
        Tensor of shape (M,).
        Integer class labels for the M ground-truth objects.
        Labels should be in {0, ..., K-1}.

    gt_bbox:
        Tensor of shape (M, 4).
        Ground-truth boxes in normalized (x1, y1, x2, y2) format.

    lambda_CE:
        Weight of classification matching cost.

    lambda_L1:
        Weight of L1 bounding-box matching cost.

    lambda_GIoU:
        Weight of generalized-IoU matching cost.

    Returns
    -------
    pred_indices:
        Long tensor containing indices of matched predictions.

    gt_indices:
        Long tensor containing the corresponding ground-truth indices.

    If M == 0, both returned tensors are empty.
    """

    Q = pred_bbox.shape[0]
    M = gt_bbox.shape[0]

    # ------------------------------------------------------------
    # No ground-truth objects
    # ------------------------------------------------------------

    if M == 0:
        empty = torch.empty(
            0,
            dtype=torch.long,
            device=pred_bbox.device
        )

        return empty, empty

    # ------------------------------------------------------------
    # 1. Classification cost
    #
    # pred_probs: (Q, K+1)
    #
    # For every ground-truth object m, select the probability
    # that each query assigns to the true class gt_classes[m].
    #
    # Result: (Q, M)
    # ------------------------------------------------------------

    pred_probs = torch.softmax(pred_class_logits, dim=-1)

    class_cost = -pred_probs[:, gt_classes]

    # ------------------------------------------------------------
    # 2. L1 bounding-box cost
    #
    # Computes all pairwise L1 distances:
    #
    #   (Q, 4) versus (M, 4) -> (Q, M)
    #
    # Both boxes remain in (cx, cy, w, h) format here.
    # ------------------------------------------------------------

    gt_bbox_cxcywh = torchvision.ops.box_convert(
            gt_bbox,
            in_fmt="xyxy",
            out_fmt="cxcywh"
        )
    
    L1_cost = torch.cdist(x1=pred_bbox,
                          x2=gt_bbox_cxcywh,
                          p=1,
                          )

    # ------------------------------------------------------------
    # 3. Generalized-IoU cost
    #
    # torchvision.ops.generalized_box_iou expects xyxy boxes,
    # so first convert:
    #
    #   (cx, cy, w, h) -> (x1, y1, x2, y2)
    # ------------------------------------------------------------

    pred_bbox_xyxy = torchvision.ops.box_convert(
        pred_bbox,
        in_fmt="cxcywh",
        out_fmt="xyxy"
    )

    # Shape: (Q, M)
    giou = torchvision.ops.generalized_box_iou(
        pred_bbox_xyxy,
        gt_bbox
    )

    # High GIoU should mean low cost.
    GIoU_cost = -giou

    # ------------------------------------------------------------
    # 4. Total matching cost
    #
    # Shape: (Q, M)
    # ------------------------------------------------------------

    total_cost = (
        lambda_CE * class_cost
        + lambda_L1 * L1_cost
        + lambda_GIoU * GIoU_cost
    )

    # ------------------------------------------------------------
    # 5. Hungarian assignment
    #
    # scipy operates on CPU / NumPy arrays.
    # The matching operation is deliberately outside autograd.
    # ------------------------------------------------------------

    cost_numpy = total_cost.cpu().numpy()

    pred_indices, gt_indices = linear_sum_assignment(
        cost_numpy
    )

    # Convert results back to PyTorch tensors.
    pred_indices = torch.as_tensor(
        pred_indices,
        dtype=torch.long,
        device=pred_bbox.device
    )

    gt_indices = torch.as_tensor(
        gt_indices,
        dtype=torch.long,
        device=pred_bbox.device
    )

    return pred_indices, gt_indices