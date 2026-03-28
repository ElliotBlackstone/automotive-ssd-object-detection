# test_step.py
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Dict
import time

from ..model_files.SSD_from_scratch import mySSD
from .build_targets import build_targets
from .CELoss_w_neg_mining import CELoss_w_neg_mining


def SSD_test_step(model: mySSD,
                  dataloader: torch.utils.data.DataLoader,
                  iou_thresh: float = 0.5,
                  iou_variant: str = "IoU",
                  neg_pos_ratio: float = 3.0,
                  score_thresh: float = 0.05,
                  nms_thresh: float = 0.5,
                  max_detections_per_img: int = 100,
                  device: str = 'cpu',
                  timing: bool = False,
                  compute_mAP: bool = False,
                  ) -> Dict:
    """
    Inputs
    model: mySSD class model to be tested
    dataloader: Data on which the model is to be tested
    iou_thresh: IoU threshold for prior/ground truth overlap, float between 0 and 1.
    iou_variant: string that must be on of "IoU", "GIoU", "DIoU", "CIoU"
    neg_pos_ration: Negative to positive ratio for hard negative mining, float greater than 0.
    score_thresh:
    nms_thresh:
    max_detections_per_img:
    device: 'cpu' or 'cuda'
    timing: Boolean for enabling/disabling timing
    compute_mAP: True - compute mAP, False - skip

    Outputs
    Dictonary with localization loss, classification loss, total loss (sum of loc+cls loss), timing results
    (P - number of priors (8732))
    """
    # put model in eval mode
    model.eval()
    device = torch.device(device)
    use_amp = (device.type == "cuda")

    # initialize loss
    conf_loss = torch.tensor(0.0, device=device)
    loc_loss = torch.tensor(0.0, device=device)
    test_loss = torch.tensor(0.0, device=device)

    # timing
    batch_count = 0
    time_device = 0.0
    time_build_tar = 0.0
    time_forward = 0.0
    time_loss = 0.0
    time_pred = 0.0
    time_mAP = 0.0

    if compute_mAP:
        map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.50], class_metrics=True).to(device)
        map_metric.reset()

    # turn on inference mode
    with torch.inference_mode():
        # loop through dataloader batches
        for _, (images, targets) in enumerate(dataloader):
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_to_device = time.perf_counter()

            # move images, targets to device
            images = images.to(device, non_blocking=True)
            targets = [{
                        "boxes": t["boxes"].to(device, non_blocking=True),
                        "labels": t["labels"].to(device, non_blocking=True),
                        }
                        for t in targets]

            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_to_device = time.perf_counter()
                time_device += (t1_to_device - t0_to_device)

            

            # ---------- Build targets ----------
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_build_tar = time.perf_counter()

            pos_mask, loc_t_pm, cls_t = build_targets(priors_cxcywh=model.priors,
                                                      priors_xyxy=model.priors_xyxy,
                                                      targets=targets,
                                                      H=images.shape[-2],
                                                      W=images.shape[-1],
                                                      iou_thresh=iou_thresh,
                                                      variances=(model.variance_center, model.variance_size),
                                                      iou_variant=iou_variant)
            
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_build_tar = time.perf_counter()
                time_build_tar += (t1_build_tar - t0_build_tar)
            
        
            # number of positives per image (avoid zero division)
            total_pos = pos_mask.sum(dim=1).sum().clamp_min(1)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0_forward = time.perf_counter()

                loc_all, conf_all = model(images)

                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1_forward = time.perf_counter()
                    time_forward += (t1_forward - t0_forward)
                


                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0_loss = time.perf_counter()
                # ---------- Losses (no backward) ----------
                # Localization: SmoothL1 on positives only
                batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction="sum") / total_pos.to(loc_all.dtype)

                # Classification: cross-entropy with hard-negative mining
                batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                                      cls_t=cls_t,
                                                      pos_mask=pos_mask,
                                                      neg_pos_ratio=neg_pos_ratio)
                
                if timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1_loss = time.perf_counter()
                    time_loss += (t1_loss - t0_loss)
            
            batch_total_loss = batch_loc_loss + batch_conf_loss

            loc_loss += batch_loc_loss.detach()
            conf_loss += batch_conf_loss.detach()
            test_loss += batch_total_loss.detach()

            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0_pred = time.perf_counter()

            preds = model.predict(images=images,
                                  score_thresh=score_thresh,
                                  nms_thresh=nms_thresh,
                                  iou_variant=iou_variant,
                                  max_per_img=max_detections_per_img,
                                  class_agnostic=False,
                                  pre_loc_all=loc_all,
                                  pre_conf_all=conf_all)

            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_pred = time.perf_counter()
                time_pred += (t1_pred - t0_pred)

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

            batch_count += 1


    test_loss = (test_loss / len(dataloader)).item()
    loc_loss = (loc_loss / len(dataloader)).item()
    conf_loss = (conf_loss / len(dataloader)).item()

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
                 "build targets": time_build_tar/batch_count,
                 "compute loss": time_loss/batch_count,
                 "model.predict": time_pred/batch_count,
                 "mAP time": time_mAP if compute_mAP else 0.0,}

    return {"testing loss": test_loss,
            "localization loss": loc_loss,
            "classification loss": conf_loss,
            "mAP": mAP if compute_mAP else 0.0,
            "timing": time_dict}