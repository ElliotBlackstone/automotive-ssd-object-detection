import torch
from torch.profiler import profile, ProfilerActivity, record_function
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .build_targets import build_targets, build_targets_2
from .CELoss_w_neg_mining import CELoss_w_neg_mining



def profile_SSD_test(model: torch.nn.Module,
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

    # put model in eval mode
    model.eval()
    device = torch.device(device)
    use_amp = (device.type == "cuda")

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.50], class_metrics=True).to(device)
    map_metric.reset()

    count = 0

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=50, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=False,
    ) as prof:

        for epoch in range(epochs):
            for _, (images, targets) in enumerate(test_dataloader):
                with record_function("test_iter"):
                    with record_function("data_to_device"):
                        images = images.to(device, non_blocking=True)
                        targets = [{
                                    "boxes": t["boxes"].to(device, non_blocking=True),
                                    "labels": t["labels"].to(device, non_blocking=True),
                                    }
                                    for t in targets]
                    
                    with record_function("build_targets"):
                        # ---------- Build targets ----------
                        pos_mask, loc_t_pm, cls_t = build_targets(priors_cxcywh=model.priors,
                                                                priors_xyxy=model.priors_xyxy,
                                                                targets=targets,
                                                                H=images.shape[-2],
                                                                W=images.shape[-1],
                                                                iou_thresh=iou_thresh,
                                                                variances=(model.variance_center, model.variance_size),
                                                                iou_variant=iou_variant)
                        # number of positives per image (avoid zero division)
                        total_pos = pos_mask.sum(dim=1).sum().clamp_min(1)

                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        with record_function("forward"):
                            loc_all, conf_all = model(images)

                        with record_function("loss"):
                            batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction="sum") / total_pos.to(loc_all.dtype)

                            # Classification: cross-entropy with hard-negative mining
                            batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                                                cls_t=cls_t,
                                                                pos_mask=pos_mask,
                                                                neg_pos_ratio=neg_pos_ratio)

                    
                    

                    with record_function(".predict"):
                        preds = model.predict(images=images,
                                    score_thresh=score_thresh,
                                    nms_thresh=nms_thresh,
                                    iou_variant=iou_variant,
                                    max_per_img=max_detections_per_img,
                                    class_agnostic=False,
                                    pre_loc_all=loc_all,
                                    pre_conf_all=conf_all)
                    
                    with record_function("mAP"):
                        map_metric.update(preds=preds, target=targets)

                prof.step()
                count += 1
                if count >= 55:
                    break

        print(prof.key_averages().table(
            sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",))





def profile_SSD_test_2(model, test_dataloader, optimizer,
                      scheduler=None, scaler=None, sched_step_w_opt=False,
                      iou_thresh=0.5, iou_variant="IoU", neg_pos_ratio=3.0,
                      score_thresh=0.05, nms_thresh=0.5,
                      max_detections_per_img=100, epochs=5,
                      device='cpu'):

    device = torch.device(device)
    use_cuda = (device.type == "cuda")
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
    use_amp = (device.type == "cuda")

    priors_cxcywh = model.priors.to(device)
    priors_xyxy   = model.priors_xyxy.to(device)
    # model = torch.compile(model, mode="reduce-overhead")

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.50], class_metrics=True).to(device)
    map_metric.reset()

    count = 0

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=50, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=False,
    ) as prof:

        for epoch in range(epochs):
            for images, targets in test_dataloader:
                with record_function("train_iter"):
                    with record_function("data_to_device"):
                        images = images.to(device, non_blocking=True)

                    with record_function("build_targets"):
                        pos_mask, loc_t_pm, cls_t = build_targets_2(
                            priors_cxcywh=priors_cxcywh,
                            priors_xyxy=priors_xyxy,
                            targets=targets,
                            H=images.shape[-2],
                            W=images.shape[-1],
                            iou_thresh=iou_thresh,
                            variances=(model.variance_center, model.variance_size),
                            iou_variant=iou_variant,
                        )
                        num_pos_per_img = pos_mask.sum(dim=1)
                        total_pos = num_pos_per_img.sum().clamp_min(1).to(images.dtype)

                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        with record_function("forward"):
                            loc_all, conf_all = model(images)

                        with record_function("loss"):
                            batch_loc_loss = torch.nn.functional.smooth_l1_loss(
                                loc_all[pos_mask], loc_t_pm, reduction='sum'
                            ) / total_pos
                            batch_conf_loss = CELoss_w_neg_mining(
                                conf_all=conf_all,
                                cls_t=cls_t,
                                pos_mask=pos_mask,
                                neg_pos_ratio=neg_pos_ratio,
                            )
                            
                    with record_function(".predict"):
                        preds = model.predict(images=images,
                                    score_thresh=score_thresh,
                                    nms_thresh=nms_thresh,
                                    iou_variant=iou_variant,
                                    max_per_img=max_detections_per_img,
                                    class_agnostic=False,
                                    pre_loc_all=loc_all,
                                    pre_conf_all=conf_all)
                    
                    with record_function("mAP"):
                        map_metric.update(preds=preds, target=targets)
                    

                prof.step()

                count += 1
                if count >= 55:
                    break

        print(prof.key_averages().table(
            sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        ))