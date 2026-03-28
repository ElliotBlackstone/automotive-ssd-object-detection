import torch
from torch.profiler import profile, ProfilerActivity, record_function

from .build_targets import build_targets, build_targets_2
from .CELoss_w_neg_mining import CELoss_w_neg_mining



def profile_SSD_train(model: torch.nn.Module,
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
              device: str | torch.device = 'cpu',):

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    device = torch.device(device)
    use_amp = (device.type == "cuda")

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=4, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as prof:

        for epoch in range(epochs):
            for _, (images, targets) in enumerate(train_dataloader):
                with record_function("train_iter"):
                    with record_function("data_to_device"):
                        images = images.to(device, non_blocking=True)
                        targets = [{
                                    "boxes": t["boxes"].to(device, non_blocking=True),
                                    "labels": t["labels"].to(device, non_blocking=True),
                                    }
                                    for t in targets]
                    
                    with record_function("build_targets"):
                        pos_mask, loc_t_pm, cls_t = build_targets_2(priors_cxcywh=model.priors,
                                                  priors_xyxy=model.priors_xyxy,
                                                  targets=targets,
                                                  H=images.shape[-2],
                                                  W=images.shape[-1],
                                                  iou_thresh=iou_thresh,
                                                  variances=(model.variance_center, model.variance_size),
                                                  iou_variant=iou_variant)
                        # number of positives per image (avoid zero division)
                        num_pos_per_img = pos_mask.sum(dim=1)                    # [B]
                        total_pos = num_pos_per_img.sum().clamp_min(1).to(images.dtype)   # scalar

                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        with record_function("forward"):
                            loc_all, conf_all = model(images)

                        with record_function("loss"):
                            # -------- 2) Localization loss (positives only) --------
                            batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction='sum') / total_pos


                            # -------- 3) Classification loss with hard-negative mining --------
                            batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                                                cls_t=cls_t,
                                                                pos_mask=pos_mask,
                                                                neg_pos_ratio=neg_pos_ratio,)
                            batch_loss = batch_loc_loss + batch_conf_loss

                    
                    if use_amp:
                        with record_function("backward"):
                            optimizer.zero_grad(set_to_none=True)
                            scaler.scale(batch_loss).backward()
                        old_scale = scaler.get_scale()
                        with record_function("optimizer_step"):
                            scaler.step(optimizer)
                        scaler.update()
                        new_scale = scaler.get_scale()

                        if scheduler is not None and new_scale >= old_scale:
                            scheduler.step()
                    else:
                        with record_function("backward"):
                            optimizer.zero_grad(set_to_none=True)
                            batch_loss.backward()
                        with record_function("optimizer_step"):
                            optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                prof.step()

        print(prof.key_averages().table(
            sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",))