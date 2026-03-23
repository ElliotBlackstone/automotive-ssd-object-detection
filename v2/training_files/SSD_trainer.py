import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from pathlib import Path
from typing import Dict
import numpy as np

from tqdm.auto import tqdm
import time

from ..model_files.SSD_from_scratch import mySSD
from .build_targets import build_targets
from .CELoss_w_neg_mining import CELoss_w_neg_mining
from .save_load_ckpt import save_checkpoint
from .merge_dicts import merge_dicts_preserve_order



def SSD_train_step(model: mySSD,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   iou_thresh: float = 0.5,
                   neg_pos_ratio: float = 3.0,
                   device: str = 'cpu',
                   timing: bool = False,
                   scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                   scaler: torch.amp.GradScaler | None = None,
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
    for _, (images, targets) in enumerate(dataloader):
        # Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

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


        # -------- 1) Build per-image targets via encode() --------
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
                                                  variances=(model.variance_center, model.variance_size))
        
        if timing:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1_build_tar = time.perf_counter()
            time_build_tar += (t1_build_tar - t0_build_tar)
        
        # number of positives per image (avoid zero division)
        num_pos_per_img = pos_mask.sum(dim=1)                    # [B]
        total_pos = num_pos_per_img.sum().clamp_min(1).to(images.dtype)   # scalar


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

        # scaled backward / optimizer step
        scaler.scale(batch_loss).backward()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        # optimizer step
        optimizer.step()

        if scheduler is not None:
            if (not use_amp) or (new_scale >= old_scale):
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




def SSD_test_step(model: mySSD,
                  dataloader: torch.utils.data.DataLoader,
                  iou_thresh: float = 0.5,
                  neg_pos_ratio: float = 3.0,
                  score_thresh: float = 0.05,
                  nms_thresh: float = 0.5,
                  max_detections_per_img: int = 100,
                  device: str = 'cpu',
                  timing: bool = False,
                  ) -> Dict:
    """
    Inputs
    model: mySSD class model to be tested
    dataloader: Data on which the model is to be tested
    iou_thresh: IoU threshold for prior/ground truth overlap, float between 0 and 1.
    neg_pos_ration: Negative to positive ratio for hard negative mining, float greater than 0.
    score_thresh:
    nms_thresh:
    max_detections_per_img:
    device: 'cpu' or 'cuda'
    timing: Boolean for enabling/disabling timing

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
                                                      variances=(model.variance_center, model.variance_size))
            
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
                 "mAP time": time_mAP,}

    return {"testing loss": test_loss, "localization loss": loc_loss, "classification loss": conf_loss, "mAP": mAP, "timing": time_dict}





def SSD_train(model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
              scaler: torch.amp.GradScaler | None = None,
              sched_step_w_opt: bool = False,
              iou_thresh: float = 0.5,
              neg_pos_ratio: float = 3.0,
              score_thresh: float = 0.05,
              nms_thresh: float = 0.5,
              max_detections_per_img: int = 100,
              epochs: int = 5,
              early_stopping_rounds: int | None = None,
              device: str | torch.device = 'cpu',
              save_model: bool = False,
              save_best_model: bool = True,
              epoch_save_interval: int | None = None,
              SAVE_DIR: Path | None = None,
              timing: bool = False,
              past_train_dict: Dict | None = None,
              ) -> Dict:
    """
    Inputs
    model: mySSD model to be trained/tested
    train_dataloader: Data on which the model is to be trained
    test_dataloader: Data on which the model is to be tested
    optimizer: Optimizer, e.g. SGD, Adam, etc.
    scheduler:
    scaler: 
    sched_step_w_opt: 
    iou_thresh: IoU threshold for prior/ground truth overlap, float between 0 and 1.
    neg_pos_ratio: Negative to positive ratio for hard negative mining, float greater than 0.
    score_thresh: 
    nms_thresh: 
    max_detections_per_img: 
    epochs: Integer number (>0) of train/test cycles
    early_stopping_rounds: Integer or None. Stop the train/test cycle if the testing score
                           has not gone down in the past 'early_stopping_rounds' cycles.
                           None by default (disabled).
    device: 'cpu' or 'cuda'
    save_model: Boolean, True to save model
    save_best_model: Boolean, True to save best model during the train/test cycles
    epoch_save_interval: Integer or None.  If int, save model every 'epoch_save_interval' cycles.
    SAVE_DIR: File path to save location
    timing: Boolean for enabling/disabling timing
    past_train_dict: Dictionary or None.  If not None, dictionary of past training results.

    Outputs
    Dictonary with train+test localization loss, train+test classification loss,
    train+test total loss (sum of loc+cls loss), test mAP, epcohs, train+test timing results
    (P - number of priors (8732))
    """
    device = torch.device(device)
    # device check
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested, but CUDA is not available.")

    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device type: {device.type}")
    
    if save_model and SAVE_DIR is None:
        raise TypeError("If the model is to be saved, SAVE_DIR must be specified.")
    
    best_metric = None
    conseq_rounds = 0
    
    if past_train_dict is not None:
        past_epochs = past_train_dict['epochs'][0]
    else:
        past_epochs = 0
    
    # create results dictionary
    results = {"train_loss": [],
               "train_loss_loc": [],
               "train_loss_conf": [],
               "test_loss": [],
               "test_loss_loc": [],
               "test_loss_conf": [],
               "mAP": [],
               "epochs": [past_epochs],
               "training timing": [],
               "testing timing": [],}
    
    for epoch in tqdm(range(epochs)):
        current_epoch = past_epochs + epoch + 1

        train_dict = SSD_train_step(model=model,
                                    dataloader=train_dataloader,
                                    optimizer=optimizer,
                                    iou_thresh=iou_thresh,
                                    neg_pos_ratio=neg_pos_ratio,
                                    device=device,
                                    timing=timing,
                                    scheduler=scheduler if sched_step_w_opt else None,
                                    scaler=scaler,)

        
        test_dict = SSD_test_step(model=model,
                                  dataloader=test_dataloader,
                                  iou_thresh=iou_thresh,
                                  neg_pos_ratio=neg_pos_ratio,
                                  nms_thresh=nms_thresh,
                                  score_thresh=score_thresh,
                                  max_detections_per_img=max_detections_per_img,
                                  device=device,
                                  timing=timing)
        
        val_err = test_dict["testing loss"]
        
        if scheduler is not None and not sched_step_w_opt:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_err)
            else:
                scheduler.step()
        
        print(f"Epoch: {current_epoch}  |  mAP: {test_dict['mAP']['map_50']:.4f}  |  Train loc loss: {train_dict['localization loss']:.4f}  |  Train class loss: {train_dict['classification loss']:.4f}  |  Test loc loss: {test_dict['localization loss']:.4f}  |  Test class loss: {test_dict['classification loss']:.4f}")

        # update results dictionary
        results['train_loss'].append(train_dict['training loss'])
        results['train_loss_loc'].append(train_dict['localization loss'])
        results['train_loss_conf'].append(train_dict['classification loss'])
        results['test_loss'].append(test_dict['testing loss'])
        results['test_loss_loc'].append(test_dict['localization loss'])
        results['test_loss_conf'].append(test_dict['classification loss'])
        results['mAP'].append(test_dict['mAP'])
        results['training timing'].append(train_dict['timing'])
        results['testing timing'].append(test_dict['timing'])
        results["epochs"][0] = current_epoch


        val_metric = test_dict["mAP"]["map_50"]
        is_better = (best_metric is None) or (val_metric > best_metric)
        if is_better:
                best_metric = val_metric

        # Early stopping rounds
        if early_stopping_rounds is not None:
            if is_better:
                conseq_rounds = 0

            else:
                conseq_rounds += 1
                if conseq_rounds >= early_stopping_rounds:
                    print(f"Early stopping after {early_stopping_rounds} rounds without improvement.")
                    results["epochs"][0] = current_epoch
                    if save_model:
                        loss_dict = (merge_dicts_preserve_order(past_train_dict, results)
                            if past_train_dict is not None else results)
                        save_checkpoint(epoch=current_epoch,
                                        model=model,
                                        loss_dict=loss_dict,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        scaler=scaler,
                                        best_metric=best_metric,
                                        outdir=SAVE_DIR,
                                        tag="last",)
                    break

                
        if save_model:
            # build loss_dict only if we're going to save something this epoch
            will_save_last   = (epoch_save_interval is None)
            will_save_period = (epoch_save_interval is not None
                                and current_epoch % epoch_save_interval == 0)
            will_save_best   = (save_best_model and is_better)

            if will_save_last or will_save_period or will_save_best:
                loss_dict = (merge_dicts_preserve_order(past_train_dict, results) if past_train_dict is not None else results)

            # rolling "last" snapshot
            if will_save_last:
                save_checkpoint(epoch=current_epoch,  # choose 1-based consistently
                                model=model,
                                loss_dict=loss_dict,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                best_metric=val_metric,   # metric at this epoch
                                outdir=SAVE_DIR,
                                tag="last",)

            # periodic labeled checkpoints
            if will_save_period:
                save_checkpoint(epoch=current_epoch,
                                model=model,
                                loss_dict=loss_dict,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                best_metric=val_metric,   # metric at this epoch
                                outdir=SAVE_DIR,
                                tag=f"epoch_{current_epoch:03d}",)

            # separate "best" snapshot
            if will_save_best:
                save_checkpoint(epoch=current_epoch,
                                model=model,
                                loss_dict=loss_dict,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                best_metric=best_metric,  # global best so far
                                outdir=SAVE_DIR,
                                tag="best",)



    # return results
    return merge_dicts_preserve_order(past_train_dict, results) if past_train_dict is not None else results



def collate_detection(batch):
    # batch: list of (img, target) tuples
    imgs  = [img for img, _ in batch]
    tgts  = [tgt for _, tgt in batch]

    # imgs are already float32 CxHxW tensors (or tv_tensors.Image),
    # so stacking is enough
    return torch.stack(imgs, dim=0), tgts
