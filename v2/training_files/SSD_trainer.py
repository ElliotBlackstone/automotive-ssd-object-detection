import torch
from pathlib import Path
from typing import Dict
from tqdm.auto import tqdm

from .train_step import SSD_train_step
from .test_step import SSD_test_step
from .save_load_ckpt import save_checkpoint
from .merge_dicts import merge_dicts_preserve_order



def SSD_train(model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
              scaler: torch.amp.GradScaler | None = None,
              sched_step_w_opt: bool = False,
              iou_thresh: float = 0.5,
              iou_variant: str = "IoU",
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
              compute_mAP: bool = False,
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
    iou_variant: string that must be on of "IoU", "GIoU", "DIoU", "CIoU"
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
    compute_mAP: True - compute mAP, False - skip

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
    
    if not isinstance(iou_variant, str):
        raise TypeError("variant must be a string")
    if iou_variant not in ("IoU", "GIoU", "DIoU", "CIoU"):
        raise ValueError(f"iou_variant must be one of (IoU, GIoU, DIoU, CIoU) but recieved: {iou_variant}")
    
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
                                    iou_variant=iou_variant,
                                    neg_pos_ratio=neg_pos_ratio,
                                    device=device,
                                    timing=timing,
                                    scheduler=scheduler if sched_step_w_opt else None,
                                    scaler=scaler,)

        
        test_dict = SSD_test_step(model=model,
                                  dataloader=test_dataloader,
                                  iou_thresh=iou_thresh,
                                  iou_variant=iou_variant,
                                  neg_pos_ratio=neg_pos_ratio,
                                  nms_thresh=nms_thresh,
                                  score_thresh=score_thresh,
                                  max_detections_per_img=max_detections_per_img,
                                  device=device,
                                  timing=timing,
                                  compute_mAP=compute_mAP,)
        
        val_err = test_dict["testing loss"]
        
        if scheduler is not None and not sched_step_w_opt:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_err)
            else:
                scheduler.step()
        
        mAP_score = test_dict['mAP']['map_50'] if compute_mAP else {"no mAP": 0.0}
        
        print(f"Epoch: {current_epoch}  |  mAP: {mAP_score:.4f}  |  Train loc loss: {train_dict['localization loss']:.4f}  |  Train class loss: {train_dict['classification loss']:.4f}  |  Test loc loss: {test_dict['localization loss']:.4f}  |  Test class loss: {test_dict['classification loss']:.4f}")

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


        val_metric = mAP_score
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