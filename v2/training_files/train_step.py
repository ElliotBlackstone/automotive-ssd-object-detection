# train_step.py
import torch
from typing import Dict
import time

from ..model_files.SSD_from_scratch import mySSD
from .build_targets import build_targets, build_targets_2
from .CELoss_w_neg_mining import CELoss_w_neg_mining


def SSD_train_step(model: mySSD,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   iou_thresh: float = 0.5,
                   neg_pos_ratio: float = 3.0,
                   device: str = 'cpu',
                   timing: bool = False,
                   scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                   scaler: torch.amp.GradScaler | None = None,
                   iou_variant: str = "IoU",
                   ) -> Dict:
    """
    Inputs
    model: SSD model to be trained
    dataloader: Data on which the model is to be trained
    optimizer: Optimizer, e.g. SGD, Adam, etc.
    iou_thresh: IoU threshold for prior/ground truth overlap, float between 0 and 1.
    neg_pos_ratio: Negative to positive ratio for hard negative mining, float greater than 0.
    device: 'cpu' or 'cuda'
    timing: Boolean for enabling/disabling timing
    scheduler:
    scaler:
    iou_variant: string that must be on of "IoU", "GIoU", "DIoU", "CIoU"

    Outputs
    Dictonary with localization loss, classification loss, total loss (sum of loc+cls loss), timing results
    (P - number of priors (8732))
    """

    # put model in train mode
    model.train()
    device = torch.device(device)
    use_amp = (device.type == "cuda")

    # initialize loss
    train_loss = torch.tensor(0.0, device=device)
    loc_loss = torch.tensor(0.0, device=device)
    conf_loss = torch.tensor(0.0, device=device)

    # timing
    batch_count = 0
    time_device = 0.0
    time_forward = 0.0
    time_build_tar = 0.0
    time_loss = 0.0
    

    # loop through data loader batches
    for images, targets in dataloader:
        # Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        if timing:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0_to_device = time.perf_counter()

        # move images to device
        images = images.to(device, non_blocking=True)

        if timing:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1_to_device = time.perf_counter()
            time_device += (t1_to_device - t0_to_device)


        # -------- 1) Build per-batch targets --------
        if timing:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0_build_tar = time.perf_counter()
        
        pos_mask, loc_t_pm, cls_t = build_targets_2(priors_cxcywh=model.priors,
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
        num_pos_per_img = pos_mask.sum(dim=1)
        total_pos = num_pos_per_img.sum().clamp_min(1).to(images.dtype)



        # forward pass
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
            # -------- 2) Localization loss (positives only) --------
            batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction='sum') / total_pos


            # -------- 3) Classification loss with hard-negative mining --------
            batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                                  cls_t=cls_t,
                                                  pos_mask=pos_mask,
                                                  neg_pos_ratio=neg_pos_ratio,)
            
            if timing:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1_loss = time.perf_counter()
                time_loss += (t1_loss - t0_loss)

        # loss
        batch_loss = batch_loc_loss + batch_conf_loss

        if use_amp:
            scaler.scale(batch_loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()

            if scheduler is not None and new_scale >= old_scale:
                scheduler.step()
        else:
            batch_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()


        loc_loss += batch_loc_loss.detach()
        conf_loss += batch_conf_loss.detach()
        train_loss += batch_loss.detach()
        batch_count += 1


    train_loss = (train_loss / len(dataloader)).item()
    loc_loss = (loc_loss / len(dataloader)).item()
    conf_loss = (conf_loss / len(dataloader)).item()

    time_dict = {"to device": time_device/batch_count,
                 "model forward": time_forward/batch_count,
                 "build targets": time_build_tar/batch_count,
                 "compute loss": time_loss/batch_count,}
    

    return {"training loss": train_loss, "localization loss": loc_loss, "classification loss": conf_loss, "timing": time_dict}