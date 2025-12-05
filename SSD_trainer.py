import torch
from torchvision.transforms import v2
from torchvision.ops import box_convert, nms, clip_boxes_to_image, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.tv_tensors import BoundingBoxes as TVBoxes

from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

from SSD_from_scratch import mySSD




def SSD_train_step(model: mySSD,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   iou_thresh: float = 0.5,
                   neg_pos_ratio: float = 3.0,
                   device: str = 'cpu',
                   timing: bool = False,
                   scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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

    # initialize loss
    train_loss = 0
    loc_loss = 0
    conf_loss = 0

    # timing
    batch_count = 0
    time_device = 0
    time_forward = 0
    time_build_tar = 0
    

    # loop through data loader batches
    for batch, (images, targets) in enumerate(dataloader):
        # move images, targets to device
        if timing:
            t0_to_device = time.perf_counter()

        images = images.to(device, non_blocking=True)
        for i in range(len(targets)):
            for key in targets[i]:
                targets[i][key] = targets[i][key].to(device=device, non_blocking=True)
        
        if timing:
            t1_to_device = time.perf_counter()
            time_device += t1_to_device - t0_to_device

        
        # forward pass
        if timing:
            t0_forward = time.perf_counter()

        loc_all, conf_all = model(images)

        if timing:
            t1_forward = time.perf_counter()
            time_forward += t1_forward - t0_forward
        
        B, P, C = conf_all.shape          # B - batch size, P - number of priors (8732), C - number of classes

        # -------- 1) Build per-image targets via encode() --------
        if timing:
            t0_build_tar = time.perf_counter()
        
        pos_mask, loc_t_pm, cls_t = build_targets(model=model,
                                                  targets=targets,
                                                  H=images.shape[-2],
                                                  W=images.shape[-1],
                                                  iou_thresh=iou_thresh,
                                                  device=device)
        
        if timing:
            t1_build_tar = time.perf_counter()
            time_build_tar += t1_build_tar - t0_build_tar
        
        # number of positives per image (avoid zero division)
        num_pos_per_img = pos_mask.sum(dim=1)                    # [B]
        total_pos = num_pos_per_img.sum().clamp_min(1).float()   # scalar

        # -------- 2) Localization loss (positives only) --------
        batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction='sum') / total_pos


        # -------- 3) Classification loss with hard-negative mining --------
        batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                              cls_t=cls_t,
                                              pos_mask=pos_mask,
                                              num_pos_per_img=num_pos_per_img,
                                              total_pos=total_pos,
                                              neg_pos_ratio=neg_pos_ratio)

        # loss
        batch_loss = batch_loc_loss + batch_conf_loss
        
        loc_loss += batch_loc_loss.item()
        conf_loss += batch_conf_loss.item()
        train_loss += batch_loss.item()

        # Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # loss backward
        batch_loss.backward()

        # optimizer step
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_count += 1


    train_loss = train_loss / len(dataloader)
    loc_loss = loc_loss / len(dataloader)
    conf_loss = conf_loss / len(dataloader)

    time_dict = {"to device": time_device/batch_count,
                 "model forward": time_forward/batch_count,
                 "build targets": time_build_tar/batch_count,}
    

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
                  ):
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

    # initialize loss
    conf_loss = 0
    loc_loss = 0
    test_loss = 0
    outputs = []

    # timing
    batch_count = 0
    time_pred = 0
    time_mAP = 0
    time_build_tar = 0

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.50], class_metrics=True).to(device)
    map_metric.reset()

    # turn on inference mode
    with torch.inference_mode():
        # loop through dataloader batches
        for batch, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            for i in range(len(targets)):
                for key in targets[i]:
                    targets[i][key] = targets[i][key].to(device=device, non_blocking=True)

            loc_all, conf_all = model(images)

            # ---------- Build targets (same as train) ----------
            if timing:
                t0_build_tar = time.perf_counter()

            pos_mask, loc_t_pm, cls_t = build_targets(model=model,
                                                      targets=targets,
                                                      H=images.shape[-2],
                                                      W=images.shape[-1],
                                                      iou_thresh=iou_thresh,
                                                      device=device)
            
            if timing:
                t1_build_tar = time.perf_counter()
                time_build_tar += t1_build_tar - t0_build_tar
        
            # number of positives per image (avoid zero division)
            num_pos_per_img = pos_mask.sum(dim=1)                    # [N]
            total_pos = num_pos_per_img.sum().clamp_min(1).float()   # scalar

            # ---------- Losses (no backward) ----------
            # Localization: SmoothL1 on positives only
            batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction="sum") / total_pos

            # Classification: cross-entropy with hard-negative mining
            batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                                  cls_t=cls_t,
                                                  pos_mask=pos_mask,
                                                  num_pos_per_img=num_pos_per_img,
                                                  total_pos=total_pos,
                                                  neg_pos_ratio=neg_pos_ratio)
            
            batch_total_loss = batch_loc_loss + batch_conf_loss

            loc_loss += batch_loc_loss.item()
            conf_loss += batch_conf_loss.item()
            test_loss += batch_total_loss.item()

            if timing:
                t0_pred = time.perf_counter()

            preds = model.predict(images=images,
                                  score_thresh=score_thresh,
                                  nms_thresh=nms_thresh,
                                  max_per_img=max_detections_per_img,
                                  class_agnostic=False,
                                  pre_loc_all=loc_all,
                                  pre_conf_all=conf_all)

            if timing:
                t1_pred = time.perf_counter()
                time_pred += t1_pred - t0_pred

            map_metric.update(preds=preds, target=targets)
            batch_count += 1



    conf_loss = conf_loss / len(dataloader)
    loc_loss = loc_loss / len(dataloader)
    test_loss = test_loss / len(dataloader)

    if timing:
        t0_mAP = time.perf_counter()
    
    mAP = map_metric.compute()

    if timing:
        t1_mAP = time.perf_counter()
        time_mAP += t1_mAP - t0_mAP

    time_dict = {"model prediction": time_pred/batch_count,
                 "mAP time": time_mAP,
                 "build targets": time_build_tar/batch_count,}

    return {"testing loss": test_loss, "localization loss": loc_loss, "classification loss": conf_loss, "mAP": mAP, "timing": time_dict} #, outputs





def SSD_train(model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
              sched_step_w_opt: bool = False,
              iou_thresh: float = 0.5,
              neg_pos_ratio: float = 3.0,
              score_thresh: float = 0.05,
              nms_thresh: float = 0.5,
              max_detections_per_img: int = 100,
              epochs: int = 5,
              early_stopping_rounds: int | None = None,
              device: str = 'cpu',
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
    iou_thresh: IoU threshold for prior/ground truth overlap, float between 0 and 1.
    neg_pos_ratio: Negative to positive ratio for hard negative mining, float greater than 0.
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
    # device check
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', recieved {device}.")
    
    if save_model and SAVE_DIR is None:
        raise TypeError("If the model is to be saved, SAVE_DIR must be specified.")
    
    best_err = None  # will be set on first epoch if save_model is True
    
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
               "epochs": [epochs + past_epochs],
               "training timing": [],
               "testing timing": [],}
    
    for epoch in tqdm(range(epochs)):
        train_dict = SSD_train_step(model=model,
                                    dataloader=train_dataloader,
                                    optimizer=optimizer,
                                    iou_thresh=iou_thresh,
                                    neg_pos_ratio=neg_pos_ratio,
                                    device=device,
                                    timing=timing,
                                    scheduler=scheduler if sched_step_w_opt else None)

        
        test_dict = SSD_test_step(model=model,
                                  dataloader=test_dataloader,
                                  iou_thresh=iou_thresh,
                                  neg_pos_ratio=neg_pos_ratio,
                                  nms_thresh=nms_thresh,
                                  score_thresh=score_thresh,
                                  max_detections_per_img=max_detections_per_img,
                                  device=device,
                                  timing=timing)
        
        if (scheduler is not None) & (sched_step_w_opt == False):
            scheduler.step(test_dict['testing loss'])
        
        print(f"Epoch: {epoch+past_epochs}  |  mAP: {test_dict['mAP']['map_50']:.4f}  |  Train loc loss: {train_dict['localization loss']:.4f}  |  Train class loss: {train_dict['classification loss']:.4f}  |  Test loc loss: {test_dict['localization loss']:.4f}  |  Test class loss: {test_dict['classification loss']:.4f}")

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


        # Early stopping rounds
        if early_stopping_rounds is not None:
            val_metric = test_dict["mAP"]["map_50"]

            if epoch == 0:
                best_metric = val_metric
                conseq_rounds = 0
            else:
                if val_metric <= best_metric:
                    best_metric = val_metric
                    conseq_rounds = 0
                else:
                    conseq_rounds += 1
                    if conseq_rounds >= early_stopping_rounds:
                        print(f"Early stopping after {early_stopping_rounds} rounds without improvement.")
                        results["epochs"][0] = epoch + past_epochs
                        if save_model:
                            loss_dict = (merge_dicts_preserve_order(past_train_dict, results) if past_train_dict is not None else results)
                            save_checkpoint(epoch=epoch + past_epochs + 1,
                                            model=model,
                                            loss_dict=loss_dict,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            scaler=None,
                                            best_metric=val_err,
                                            outdir=SAVE_DIR,
                                            tag="last",)

                        break
                
        if save_model:
            val_err = test_dict["testing loss"]

            # initialize best_err on first saving epoch
            if best_err is None:
                best_err = val_err

            # build loss_dict only if we're going to save something this epoch
            will_save_last   = (epoch_save_interval is None)
            will_save_period = (epoch_save_interval is not None
                                and (epoch + 1) % epoch_save_interval == 0)
            will_save_best   = (save_best_model and (val_err < best_err))

            if will_save_last or will_save_period or will_save_best:
                loss_dict = (merge_dicts_preserve_order(past_train_dict, results) if past_train_dict is not None else results)

            # rolling "last" snapshot
            if will_save_last:
                save_checkpoint(epoch=epoch + past_epochs + 1,  # choose 1-based consistently
                                model=model,
                                loss_dict=loss_dict,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=None,
                                best_metric=val_err,   # metric at this epoch
                                outdir=SAVE_DIR,
                                tag="last",)

            # periodic labeled checkpoints
            if will_save_period:
                save_checkpoint(epoch=epoch + past_epochs + 1,
                                model=model,
                                loss_dict=loss_dict,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=None,
                                best_metric=val_err,   # metric at this epoch
                                outdir=SAVE_DIR,
                                tag=f"epoch_{epoch + past_epochs + 1:03d}",)

            # separate "best" snapshot
            if will_save_best:
                best_err = val_err
                save_checkpoint(epoch=epoch + past_epochs + 1,
                                model=model,
                                loss_dict=loss_dict,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=None,
                                best_metric=best_err,  # global best so far
                                outdir=SAVE_DIR,
                                tag="best",)



    # return results
    return merge_dicts_preserve_order(past_train_dict, results) if past_train_dict is not None else results




def build_targets(model: mySSD,
                  targets: List[Dict],
                  H: int = 300,
                  W: int = 300,
                  iou_thresh: float = 0.50,
                  device: str = 'cpu',
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs
    targets: List of length B, where each element is a dictionary containing keys 'boxes', 'labels'
    H: Integer, should be 300
    W: Integer, should be 300
    iou_thresh: Float between 0 and 1 denoting the IoU threshold
    device: String, should be 'cpu' or 'cuda'

    Output
    Tuple containing
    1) Boolean tensor of size [B, P] denoting priors boxes that
       have sufficient (determined by iou_thresh) overlap with GT boxes
    2) Tensor containing offsets (tx, ty, tw, th) corresponding to matched GT boxes
    3) Tensor of size [B, P] where entries (per batch) are labels 0, ..., C-1,
       with 0 corresponding to 'background'
    (B - batch size, P - number of priors (8732), C - number of classes (including background))
    """
    # protect IoU threshold
    if not (0.0 < iou_thresh < 1.0):
            raise ValueError(f"Score threshold should be greater than 0 and less than 1, recieved {iou_thresh}.")
    
    norm = torch.tensor([W, H, W, H], device=device, dtype=torch.float32)

    loc_t_list   = []
    cls_t_list   = []
    pos_mask_lst = []

    for i in range(len(targets)):
        # normalize GT to [0,1] and convert to cxcywh
        gt_xyxy_px = targets[i]['boxes']
        gt_labels  = targets[i]['labels']
        if gt_xyxy_px.numel() == 0:
            gt_xyxy = gt_xyxy_px.new_zeros((0,4))
        else:
            gt_xyxy = gt_xyxy_px / norm

        loc_t, cls_t, pos_mask, _ = model.encode_ssd(gt_boxes_xyxy=gt_xyxy,
                                                     gt_labels=gt_labels,
                                                     iou_thresh=iou_thresh,
                                                     background_class=0)

        loc_t_list.append(loc_t)        # [P, 4]
        cls_t_list.append(cls_t)        # [P]
        pos_mask_lst.append(pos_mask)   # [P]

    loc_t   = torch.stack(loc_t_list, dim=0).to(device)      # [B,P,4]   B = len(targets) aka batch size
    cls_t   = torch.stack(cls_t_list, dim=0).to(device)      # [B,P]
    pos_mask = torch.stack(pos_mask_lst, dim=0).to(device)   # [B,P] bool

    return pos_mask, loc_t[pos_mask], cls_t



def CELoss_w_neg_mining(conf_all: torch.Tensor,
                        cls_t: torch.Tensor,
                        pos_mask: torch.Tensor,
                        num_pos_per_img: torch.Tensor,
                        total_pos: int,
                        neg_pos_ratio: float = 3.0,
                        ) -> torch.Tensor:
    """
    Inputs
    conf_all: Tensor of size [B, P, C] containing class logits for each prior
    cls_t: Tensor of size [B, P] where entries (per batch) are labels 0, ..., C-1,
           with 0 corresponding to 'background'
    pos_mask: Boolean tensor of size [B, P] denoting priors boxes that have
              sufficient overlap with GT boxes
    num_pos_per_img: Tensor of size [B] denoting number of positive priors per image
    total_pos: Integer, sum of num_pos_per_img, at minimum 1
    neg_pos_ratio: Float, positive number giving the ratio of negative to positive priors

    Output
    Float; cross entropy loss with hard negative mining
    (B - batch size, P - number of priors (8732), C - number of classes (including background))
    """
    # cross-entropy per prior (no reduction)
    B, P, C = conf_all.shape     # B - batch size, P - number of priors (8732), C - number of classes
    device = conf_all.device

    ce = torch.nn.functional.cross_entropy(conf_all.view(-1, C), cls_t.view(-1), reduction='none').view(B, P)  # [B, P]

    # keep CE on positives always
    ce_pos = ce[pos_mask].sum()

    # select hardest negatives per image at ratio R:1 w.r.t positives
    ce_neg_sum = torch.zeros((), device=device)
    for i in range(B):
        n_pos = int(num_pos_per_img[i].item())
        if n_pos == 0:
            # still allow some negatives to contribute (common trick: pretend 1 positive)
            max_negs = int(neg_pos_ratio)
        else:
            max_negs = int(neg_pos_ratio * n_pos)

        ce_neg_i = ce[i].masked_select(~pos_mask[i])         # [#neg_i]
        if ce_neg_i.numel() == 0 or max_negs == 0:
            continue  # continue means stop here and proceed to the next iteration

        k = min(max_negs, ce_neg_i.numel())
        topk_vals, _ = torch.topk(ce_neg_i, k, largest=True, sorted=False)
        ce_neg_sum += topk_vals.sum()

    return (ce_pos + ce_neg_sum) / total_pos



def plot_losses(losses: Dict, figsize=(10, 8)) -> None:
    """
    Plots train/test loss results

    Inputs
    losses: Dictionary with keys (all required): 
      "train_loss", "train_loss_loc", "train_loss_conf",
      "test_loss",  "test_loss_loc",  "test_loss_conf", "mAP"
    Values: lists of floats (except for mAP), all the same length.

    Output
    Produces a 2x2 matplotlib figure:
      (1) train_loss vs epoch and test_loss vs epoch
      (2) train_loss_conf vs epoch and test_loss_conf vs epoch
      (3) train_loss_loc  vs epoch and test_loss_loc  vs epoch
      (4) mAP vs epoch
    """
    required = [
        "train_loss", "train_loss_loc", "train_loss_conf",
        "test_loss",  "test_loss_loc",  "test_loss_conf", "mAP"
    ]
    # Key check
    missing = [k for k in required if k not in losses]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    # Type/length checks
    lens = []
    for k in ["train_loss", "train_loss_loc", "train_loss_conf",
              "test_loss",  "test_loss_loc",  "test_loss_conf"]:
        v = losses[k]
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"Value for '{k}' must be a list/tuple of floats.")
        lens.append(len(v))
        if any((not isinstance(x, (int, float)) or np.isnan(float(x)) or np.isinf(float(x))) for x in v):
            raise ValueError(f"Non-finite numeric in '{k}'.")
    if len(set(lens)) != 1:
        raise ValueError(f"All lists must have the same length. Got lengths: {dict(zip(required, lens))}")

    n = lens[0]
    x = list(range(n))

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # (1) total loss
    ax = axes[0,0]
    ax.plot(x, losses["train_loss"], label="train")
    ax.plot(x, losses["test_loss"],  label="validation")
    ax.set_title("Total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # (2) mAP
    mAP = []
    for i in range(len(losses['mAP'])):
        mAP.append(losses['mAP'][i]['map_50'])
    
    ax = axes[0,1]
    ax.plot(x, mAP, label="mAP")
    ax.set_title("mAP")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # (3) classification loss
    ax = axes[1,0]
    ax.plot(x, losses["train_loss_conf"], label="train")
    ax.plot(x, losses["test_loss_conf"],  label="validation")
    ax.set_title("Classification loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # (4) localization loss
    ax = axes[1,1]
    ax.plot(x, losses["train_loss_loc"], label="train")
    ax.plot(x, losses["test_loss_loc"],  label="validation")
    ax.set_title("Localization loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.show()




def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # atomic on the same filesystem



def save_checkpoint(epoch: int,
                    model: torch.nn.Module,
                    loss_dict: Dict,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                    scaler = None,
                    best_metric: float | None = None,
                    outdir: str | Path = "checkpoints",
                    tag: str = "last",               # "last", "best", "epoch_010", etc.
                    ):
    """
    Saves essential model information in a .ckpt file.

    Inputs
    epoch: Integer number of rounds trained
    model: SSD model
    loss_dict: Dictionary containing train/test loss information (per epoch)
    optimizer: Optimizer used to train the model
    scheduler: Scheduler used to train the model
    scaler: Scaler used to train the model
    best_metric: Float denoting the best training metric
    outdir: Folder location to save model
    tag: String, name of save file
    """
    outdir = Path(outdir)
    # Handle DataParallel/Distributed
    model_to_save = model.module if hasattr(model, "module") else model

    ckpt = {
        "epoch": epoch,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_metric": best_metric,
        # RNG states (optional but helps reproducibility)
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "loss_dict": loss_dict,
    }
    _atomic_save(ckpt, outdir / f"{tag}.ckpt")



def load_checkpoint(path: str | Path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                    scaler = None,
                    map_location: str = "cpu",
                    ):
    """
    Load model information saved by the 'save_checkpoint' function.

    Inputs
    path: File path of model to load
    model: New instance of saved model
    optimizer: New instance of the same optimizer used to train the model
    scheduler: New instance of the same scheduler used to train the model
    scaler: New instance of the same scaler used to train the model
    map_location: 'cpu' or 'cuda'

    Outputs
    model/optimizer/scheduler/scaler are all adjusted to be the same as the
    saved checkpoint.
    Returns start_epoch (if training is to be resumed), best_metric, loss_dict
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # Model (handle DataParallel)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state"])

    # Opt / sched / scaler (if present and provided)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    # Restore RNG (optional)
    rng = ckpt.get("rng_state")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    start_epoch = int(ckpt["epoch"]) + 1  # resume at next epoch
    best_metric = ckpt.get("best_metric")
    loss_dict = ckpt.get("loss_dict")

    return start_epoch, best_metric, loss_dict



def collate_detection(batch):
    # batch: list of (img, target) tuples
    imgs  = [img for img, _ in batch]
    tgts  = [tgt for _, tgt in batch]

    # imgs are already float32 CxHxW tensors (or tv_tensors.Image),
    # so stacking is enough
    return torch.stack(imgs, dim=0), tgts



def merge_dicts_preserve_order(d1: dict, d2: dict) -> dict:
    """
    Merge two dictionaries with identical keys while preserving order.

    Inputs
    d1: Dictionary
    d2: Dictionary with same keys as d1

    Output
    Merged dictionary
    Example:
    d1 = {"a": [1, 2], "b": ["python", 8]}
    d2 = {"a": [3, "alpha"], "b": [2]}
    merge_dicts_preserve_order(d1, d2) -> {"a": [1, 2, 3, "alpha"], "b": ["python", 8, 2]}
    merge_dicts_preserve_order(d2, d1) -> {"a": [3, "alpha", 1, 2], "b": [2, "python", 8]}
    """
    if set(d1.keys()) != set(d2.keys()):
        raise KeyError("Dicts must have identical key sets.")

    out = {}
    for k in d1.keys():  # preserves key order from d1
        v1, v2 = d1[k], d2[k]

        # torch tensors
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            out[k] = torch.cat([v1, v2], dim=0)
            continue

        # numpy arrays
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            out[k] = np.concatenate([v1, v2], axis=0)
            continue

        # lists / tuples
        if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if k == 'epochs':
                out[k] = list(v2)
            else:
                seq = list(v1) + list(v2)  # v1-order then v2-order
                out[k] = type(v1)(seq) if type(v1) is type(v2) else seq
            continue

        # sets are unordered
        if isinstance(v1, set) and isinstance(v2, set):
            out[k] = list(v1) + [x for x in v2 if x not in v1]  # insertion-style, no dups
            continue

        # fallback: keep both values
        out[k] = (v1, v2)

    return out




class ConditionalIoUCrop(torch.nn.Module):
    """
    Size-aware IoU-based random cropping for object detection.

    This module wraps two `torchvision.transforms.v2.RandomIoUCrop` instances and
    chooses between them based on the relative area of the ground-truth boxes
    in the current image.

    The logic is:

        - Compute the area fraction of each box: area / (H * W).
        - If at least one box has area fraction >= `min_area_frac`
          ("large" object present), use `iou_crop_large`.
        - Otherwise (all boxes are small), use `iou_crop_small`, which is
          typically configured to zoom in more aggressively and/or use looser
          IoU constraints.

    This is intended for datasets with many small objects, where standard
    IoU-based cropping often fails to generate useful zoomed-in views of
    small targets.

    Parameters
    ----------
    min_area_frac : float, optional (default: 0.002)
        Threshold on box area fraction (box_area / (H * W)) used to decide
        whether the image contains any "large" objects. If any box has
        area_frac >= min_area_frac, the "large-object" crop policy is used;
        otherwise the "small-object" crop policy is used.

    small_min_scale : float, optional (default: 0.3)
        `min_scale` passed to the small-object `RandomIoUCrop`. Controls the
        minimum relative size of the sampled crop when all boxes are small.
        Smaller values allow tighter crops (stronger zoom-in).

    large_min_scale : float, optional (default: 0.6)
        `min_scale` passed to the large-object `RandomIoUCrop`. Controls the
        minimum relative size of the sampled crop when at least one box is
        considered large.

    max_scale : float, optional (default: 1.0)
        `max_scale` passed to both `RandomIoUCrop` instances. Controls the
        maximum relative size of the sampled crop.

    min_aspect_ratio : float, optional (default: 0.75)
        Lower bound on the aspect ratio (w / h) for sampled crops, passed to
        both `RandomIoUCrop` instances.

    max_aspect_ratio : float, optional (default: 1.33)
        Upper bound on the aspect ratio (w / h) for sampled crops, passed to
        both `RandomIoUCrop` instances.

    small_sampler_options : sequence of float or None, optional
        `sampler_options` for the small-object `RandomIoUCrop`. Each entry is
        interpreted as a minimum IoU threshold that the sampled crop must
        satisfy with at least one box (or `None` to allow unconstrained crops).
        Including low values and `None` is useful when all boxes are tiny.

    large_sampler_options : sequence of float or None, optional
        `sampler_options` for the large-object `RandomIoUCrop`, used when at
        least one box is larger than `min_area_frac`.

    trials : int, optional (default: 10)
        Number of attempts for each `RandomIoUCrop` to find a valid crop
        before falling back to returning the original image and boxes.

    Forward
    -------
    forward(img, target) -> Tuple[Tensor, Dict[str, Any]]

    Parameters
    ----------
    img : Tensor
        Image tensor of shape (C, H, W).

    target : dict
        Target dictionary expected to contain the key `"boxes"` with a
        tensor of shape (N, 4) in (x1, y1, x2, y2) format, in the same
        coordinate system as `img`. If `"boxes"` is missing or empty,
        the transform returns `(img, target)` unchanged.

    Returns
    -------
    img_out : Tensor
        Cropped (or original) image tensor.

    target_out : dict
        Updated target dictionary with boxes and any other box-dependent
        fields transformed consistently with the image. The exact behavior
        follows `torchvision.transforms.v2.RandomIoUCrop`.

    Notes
    -----
    - This transform does not change images with no boxes.
    - When all objects are small (no box exceeds `min_area_frac`), the
      "small-object" crop policy is applied explicitly instead of skipping
      cropping. This is intended to produce more training examples where
      small targets occupy a larger fraction of the crop.
    """
    def __init__(
        self,
        *,
        min_area_frac: float = 0.02,          # threshold separating "has big box" vs "all small"
        small_min_scale: float = 0.3,         # more aggressive zoom-in for small-only images
        large_min_scale: float = 0.6,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.75,
        max_aspect_ratio: float = 1.33,
        small_sampler_options = (0.0, 0.05, 0.1, 2.0),
        large_sampler_options = (0.05, 0.1, 0.3, 2.0),
        trials: int = 10,
    ):
        super().__init__()
        self.min_area_frac = float(min_area_frac)

        # crop mode when at least one reasonably large box exists
        self.iou_crop_large = v2.RandomIoUCrop(
            min_scale=large_min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=list(large_sampler_options),
            trials=trials,
        )

        # crop mode when all boxes are small: zoom in more, looser IoU constraints
        self.iou_crop_small = v2.RandomIoUCrop(
            min_scale=small_min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=list(small_sampler_options),
            trials=trials,
        )

    @torch.no_grad()
    def forward(self, img, target):
        H, W = img.shape[-2], img.shape[-1]
        boxes = target.get("boxes", None)
        if boxes is None or boxes.numel() == 0:
            return img, target

        b = torch.as_tensor(boxes)
        area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        area_frac = area / float(H * W)

        has_large = (area_frac >= self.min_area_frac).any()

        if has_large:
            img_out, tgt_out = self.iou_crop_large(img, target)
        else:
            # all boxes small → aggressively zoom-in around them
            img_out, tgt_out = self.iou_crop_small(img, target)

        return img_out, tgt_out




def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    min_lr: float = 0.0,
                                    last_epoch: int = -1,
                                    ) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine decay with linear warmup.

    LR(t) = base_lr * f(t), where f(t) is:
      - warmup: linearly from 0 -> 1 over [0, num_warmup_steps)
      - cosine: from 1 -> (min_lr / base_lr) over [num_warmup_steps, num_training_steps]

    Arguments
    ---------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    num_warmup_steps : int
        Number of steps for linear warmup.
    num_training_steps : int
        Total number of training steps (epochs * steps_per_epoch).
    min_lr : float, default 0.0
        Absolute minimum learning rate. Implemented as a ratio of base_lr.
    last_epoch : int, default -1
        See PyTorch docs for LambdaLR (use -1 when creating scheduler).

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler to be stepped *once per optimizer step*.
    """
    # we implement min_lr by enforcing a minimum multiplicative factor
    # relative to base_lr; per param group the ratio may differ.
    # -> for each param group, factor(t) in [min_ratio, 1]
    # where min_ratio = min_lr / base_lr_group
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    # sanity check assumptions
    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be >= 0")
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")
    if num_warmup_steps > num_training_steps:
        raise ValueError("num_warmup_steps cannot exceed num_training_steps")

    def lr_lambda(current_step: int):
        # this returns one factor per param group
        factors = []
        for base_lr in base_lrs:
            if min_lr > base_lr:
                raise ValueError("min_lr cannot be larger than base_lr")

            min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

            if current_step < num_warmup_steps and num_warmup_steps > 0:
                # linear warmup: 0 -> 1
                warmup_frac = float(current_step) / float(max(1, num_warmup_steps))
                factor = warmup_frac  # in [0,1]
            else:
                # cosine phase
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)  # clamp numerically

                # pure cosine from 1 -> 0
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # in [0,1]

                # rescale to [min_ratio, 1]
                factor = min_ratio + (1.0 - min_ratio) * cosine

            factors.append(factor)

        # LambdaLR expects a scalar or list; we give a list per param group
        # via this hack: return factors for the *first* call, then overwrite
        # param_group["lr"] manually. But LambdaLR actually supports returning
        # a scalar that applies to all groups. To support per-group ratios,
        # we instead pass a scalar and rely on equal base_lrs.
        #
        # If you want per-group LR, tighten the design. For now, assume all
        # param groups share the same base_lr.
        return factors[0]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_optimizer_and_scheduler(model: torch.nn.Module,
                                  train_dataloader: torch.utils.data.DataLoader,
                                  max_epochs: int = 120,
                                  warmup_epochs: int = 5,
                                  base_lr: float = 3e-3,
                                  min_lr: float = 1e-5,
                                  momentum: float = 0.9,
                                  weight_decay: float = 5e-4):
    """
    Create SGD optimizer and cosine-with-warmup scheduler
    for an SSD-style detector.

    Arguments
    ---------
    model : nn.Module
        Your SSD model.
    train_dataloader : DataLoader
        Only used to infer steps_per_epoch.
    max_epochs : int
        Total number of epochs you plan to train.
    warmup_epochs : int
        Number of warmup epochs (linear LR increase).
    base_lr : float
        Peak learning rate after warmup.
    min_lr : float
        Minimum LR at the end of cosine decay.
    momentum : float
        Momentum for SGD.
    weight_decay : float
        L2 weight decay.

    Returns
    -------
    optimizer : torch.optim.SGD
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Must be stepped *once per optimizer step*.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    steps_per_epoch = len(train_dataloader)
    num_training_steps = max_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
    )

    return optimizer, scheduler