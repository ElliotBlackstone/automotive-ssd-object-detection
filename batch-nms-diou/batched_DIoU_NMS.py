import torch




def batched_diou_nms(boxes: torch.Tensor,
                     scores: torch.Tensor,
                     idxs: torch.Tensor,
                     diou_threshold: float) -> torch.Tensor:
    
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    # identical to torchvision.ops.batched_nms
    max_coord = boxes.max()
    offsets = idxs.to(boxes.dtype) * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]

    # call your compiled op:
    return diou_nms_ext.diou_nms(boxes_for_nms, scores, diou_threshold)