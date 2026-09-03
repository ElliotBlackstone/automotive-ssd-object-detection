# myTestStep.py
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Dict
import time

from myViT import VisionTransformer
from HungarianMatch import hungarian_match
from HungarianMatchBatched import hungarian_match_batched



def move_targets_to_device(targets, device, non_blocking=True):
    moved = []
    for t in targets:
        moved.append({
            k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
            for k, v in t.items()
        })
    return moved


def myTestStep(model: VisionTransformer,
               dataloader: torch.utils.data.DataLoader,
               lambda_CE: float = 0.4,
               lambda_L1: float = 0.2,
               lambda_GIoU: float = 0.4,
               lambda_CE_HM: float = 0.4,
               lambda_L1_HM: float = 0.2,
               lambda_GIoU_HM: float = 0.4,
               device: str = 'cpu',
               timing: bool = False,
               compute_mAP: bool = False,
               bg_weight: float = 0.1,
               aux_loss_weight: float = 1.0,
               ) -> Dict:
    """
    Inputs:
     - model: The VisionTransformer model to be tested.
     - dataloader: Data on which the model is to be tested.
     - lambda_CE: Classification loss weight.
     - lambda_L1: L1 loss weight.
     - lambda_GIoU: GIoU loss weight.
     - device: cpu or gpu
     - timing: enable (True) or disable (False) timing
     - compute_mAP: True - compute mAP, False - skip

    Output:
     Dictionary with keys "testing loss", "localization loss", "classification loss", "GIoU loss", "mAP", "timing".
    """
    # put model in eval mode
    model.eval()
    device = torch.device(device)
    use_amp = (device.type == "cuda")

    # initialize loss
    test_loss = torch.tensor(0.0, device=device)
    L1_loss = torch.tensor(0.0, device=device)
    CE_loss = torch.tensor(0.0, device=device)
    GIoU_loss = torch.tensor(0.0, device=device)

    # timing
    batch_count = 0
    time_device = 0.0
    time_match = 0.0
    time_forward = 0.0
    time_loss = 0.0
    time_pred = 0.0
    time_mAP = 0.0

    # number of queries, classes
    K = model.num_classes
    Q = model.num_queries

    if compute_mAP:
        map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.50], class_metrics=True).to(device)
        map_metric.reset()

    # turn on inference mode
    with torch.inference_mode():
        # loop through dataloader batches
        for images, targets in dataloader:
            # get batch size
            B = images.shape[0]


            # ----- move to device -----
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_to_device = time.perf_counter()

            # move images to device
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets=targets, device=device, non_blocking=True)

            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_to_device = time.perf_counter()
                time_device += (t1_to_device - t0_to_device)
            # ----- end move to device -----

            # ----- forward pass -----
            with torch.autocast(device_type="cuda", enabled=use_amp):
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0_forward = time.perf_counter()
    
                # shapes (L, B, Q, K+1), (L, B, Q, 4)
                pred_class_logits, pred_bboxes = model(images)
    
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1_forward = time.perf_counter()
                    time_forward += (t1_forward - t0_forward)
            # ----- end forward pass -----
                
            
            # ------------------------------------------------------------
            # Ground-truth targets used by every decoder layer
            # ------------------------------------------------------------
            gt_labels_batch = []
            gt_boxes_xyxy_batch = []
    
            for i in range(B):
                gt_labels_batch.append(targets[i]["labels"])
                gt_boxes_xyxy_batch.append(
                    targets[i]["boxes"].float() / model.img_size
                )
    
            # Model output shapes:
            #   pred_class_logits: (L, B, Q, K+1)
            #   pred_bboxes:       (L, B, Q, 4)
            num_decoder_layers = pred_class_logits.shape[0]
    
            # Store one loss of each type for every decoder layer.
            layer_L1_losses = []
            layer_CE_losses = []
            layer_GIoU_losses = []
    
            class_weights = torch.ones(
                K + 1,
                device=device,
                dtype=torch.float32,
            )
            class_weights[K] = bg_weight
    
            # ------------------------------------------------------------
            # Hungarian matching + detection loss for every decoder layer
            # ------------------------------------------------------------
            for layer_idx in range(num_decoder_layers):
    
                layer_class_logits = pred_class_logits[layer_idx]
                layer_bboxes = pred_bboxes[layer_idx]
    
                # ----- Hungarian matching for this decoder layer -----
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0_match = time.perf_counter()
    
                matches = hungarian_match_batched(
                    pred_class_logits=layer_class_logits.float(),
                    pred_bbox=layer_bboxes.float(),
                    gt_classes=gt_labels_batch,
                    gt_bbox=gt_boxes_xyxy_batch,
                    lambda_CE=lambda_CE_HM,
                    lambda_L1=lambda_L1_HM,
                    lambda_GIoU=lambda_GIoU_HM,
                )
    
                matched_pred_boxes_cxcywh = []
                matched_gt_boxes_xyxy = []
    
                # Unmatched queries are assigned to background class K.
                class_target = torch.full(
                    (B, Q),
                    fill_value=K,
                    dtype=torch.long,
                    device=device,
                )
    
                for i, (pred_indices, gt_indices) in enumerate(matches):
    
                    gt_labels = gt_labels_batch[i]
                    gt_boxes_xyxy = gt_boxes_xyxy_batch[i]
    
                    class_target[i, pred_indices] = gt_labels[gt_indices]
    
                    if pred_indices.numel() == 0:
                        continue
    
                    matched_pred_boxes_cxcywh.append(
                        layer_bboxes[i, pred_indices]
                    )
                    matched_gt_boxes_xyxy.append(
                        gt_boxes_xyxy[gt_indices]
                    )
    
                if len(matched_pred_boxes_cxcywh) > 0:
                    matched_pred_boxes_cxcywh = torch.cat(
                        matched_pred_boxes_cxcywh,
                        dim=0,
                    )
                    matched_gt_boxes_xyxy = torch.cat(
                        matched_gt_boxes_xyxy,
                        dim=0,
                    )
    
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1_match = time.perf_counter()
                    time_match += (t1_match - t0_match)
    
                # ----- Loss for this decoder layer -----
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0_loss = time.perf_counter()
    
                # L1 box loss
                if len(matched_pred_boxes_cxcywh) > 0:
                    matched_gt_boxes_cxcywh = torchvision.ops.box_convert(
                        matched_gt_boxes_xyxy,
                        in_fmt="xyxy",
                        out_fmt="cxcywh",
                    )
    
                    layer_L1_loss = torch.nn.functional.l1_loss(
                        matched_pred_boxes_cxcywh,
                        matched_gt_boxes_cxcywh,
                        reduction="mean",
                    )
                else:
                    layer_L1_loss = layer_bboxes.sum().float() * 0.0
    
                # Classification loss
                layer_CE_loss = torch.nn.functional.cross_entropy(
                    input=layer_class_logits.float().reshape(B * Q, K + 1),
                    target=class_target.reshape(B * Q),
                    weight=class_weights,
                    reduction="mean",
                )
    
                # GIoU box loss
                if len(matched_pred_boxes_cxcywh) > 0:
                    matched_pred_boxes_xyxy = torchvision.ops.box_convert(
                        matched_pred_boxes_cxcywh,
                        in_fmt="cxcywh",
                        out_fmt="xyxy",
                    )
    
                    layer_GIoU_loss = (
                        torchvision.ops.generalized_box_iou_loss(
                            matched_pred_boxes_xyxy,
                            matched_gt_boxes_xyxy,
                            reduction="mean",
                        )
                    )
                else:
                    layer_GIoU_loss = layer_bboxes.sum().float() * 0.0
    
                layer_L1_losses.append(layer_L1_loss)
                layer_CE_losses.append(layer_CE_loss)
                layer_GIoU_losses.append(layer_GIoU_loss)
    
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1_loss = time.perf_counter()
                    time_loss += (t1_loss - t0_loss)
    
            # ------------------------------------------------------------
            # Combine final-layer and auxiliary losses
            #
            # Using the mean of the auxiliary losses avoids making the loss
            # scale grow linearly with the number of decoder layers:
            #
            #   L = L_final + aux_loss_weight * mean(L_aux)
            # ------------------------------------------------------------
            batch_L1_loss = layer_L1_losses[-1]
            batch_CE_loss = layer_CE_losses[-1]
            batch_GIoU_loss = layer_GIoU_losses[-1]
    
            if num_decoder_layers > 1:
                batch_L1_loss = (
                    batch_L1_loss
                    + aux_loss_weight
                    * torch.stack(layer_L1_losses[:-1]).mean()
                )
                batch_CE_loss = (
                    batch_CE_loss
                    + aux_loss_weight
                    * torch.stack(layer_CE_losses[:-1]).mean()
                )
                batch_GIoU_loss = (
                    batch_GIoU_loss
                    + aux_loss_weight
                    * torch.stack(layer_GIoU_losses[:-1]).mean()
                )
    
            batch_loss = (
                lambda_L1 * batch_L1_loss
                + lambda_CE * batch_CE_loss
                + lambda_GIoU * batch_GIoU_loss
            )
            

            # ----- model predict -----
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_pred = time.perf_counter()

            if compute_mAP:
                preds = model.predict(images=images,
                                      pre_class_logits=pred_class_logits[-1],
                                      pre_bboxes=pred_bboxes[-1])

            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_pred = time.perf_counter()
                time_pred += (t1_pred - t0_pred)
            # ----- end model predict -----

            # ----- mAP -----
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_mAP_update = time.perf_counter()

            if compute_mAP:
                map_metric.update(preds=preds, target=targets)

            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_mAP_update = time.perf_counter()
                time_mAP += (t1_mAP_update - t0_mAP_update)
            # ----- end mAP -----

            L1_loss += batch_L1_loss.detach()
            CE_loss += batch_CE_loss.detach()
            GIoU_loss += batch_GIoU_loss.detach()
            test_loss += batch_loss.detach()
            batch_count += 1


    test_loss = (test_loss / len(dataloader)).item()
    L1_loss = (L1_loss / len(dataloader)).item()
    CE_loss = (CE_loss / len(dataloader)).item()
    GIoU_loss = (GIoU_loss / len(dataloader)).item()

    if timing:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0_mAP = time.perf_counter()
    
    if compute_mAP:
        mAP = map_metric.compute()

    if timing:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1_mAP = time.perf_counter()
        time_mAP += (t1_mAP - t0_mAP)

    time_dict = {"to device": time_device/batch_count,
                 "model forward": time_forward/batch_count,
                 "matching": time_match/batch_count,
                 "compute loss": time_loss/batch_count,
                 "model.predict": time_pred/batch_count,
                 "mAP time": time_mAP if compute_mAP else 0.0,}

    return {"testing loss": test_loss,
            "localization loss": L1_loss,
            "classification loss": CE_loss,
            "GIoU loss": GIoU_loss,
            "timing": time_dict,
            "mAP": mAP if compute_mAP else 0.0}