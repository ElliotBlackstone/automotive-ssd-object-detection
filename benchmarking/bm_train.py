# bm_train.py

import time
import statistics
import torch
from torch.utils.data import DataLoader

from v2.training_files.build_targets import build_targets, build_targets_2
from v2.training_files.CELoss_w_neg_mining import CELoss_w_neg_mining



def move_targets_to_device(targets, device, non_blocking=True):
    moved = []
    for t in targets:
        moved_dict = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                moved_dict[k] = v.to(device, non_blocking=non_blocking)
            else:
                moved_dict[k] = v
        moved.append(moved_dict)
    return moved

def benchmark_train_loop(
    model,
    dataset,
    collate_fn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device="cuda",
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    multiprocessing_context=None,
    shuffle=True,
    warmup_steps=20,
    measure_steps=100,
):
    device = torch.device(device)
    model = model.to(device).train()

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context

    loader = DataLoader(**kwargs)

    it = iter(loader)

    fetch_times = []
    h2d_times = []
    compute_times = []
    step_times = []
    n_samples = 0

    for step in range(warmup_steps + measure_steps):
        step_t0 = time.perf_counter()

        # batch fetch
        t0 = time.perf_counter()
        try:
            images, targets = next(it)
        except StopIteration:
            it = iter(loader)
            images, targets = next(it)
        t1 = time.perf_counter()

        # host -> device
        if isinstance(images, (list, tuple)):
            images = torch.stack(images, dim=0)

        t2 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        # targets = move_targets_to_device(targets, device, non_blocking=True)
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()

        # compute
        t4 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        

        pos_mask, loc_t_pm, cls_t = build_targets_2(priors_cxcywh=model.priors,
                                                  priors_xyxy=model.priors_xyxy,
                                                  targets=targets,
                                                  H=images.shape[-2],
                                                  W=images.shape[-1],
                                                  iou_thresh=0.5,
                                                  variances=(model.variance_center, model.variance_size))
        

        
        # number of positives per image (avoid zero division)
        num_pos_per_img = pos_mask.sum(dim=1)                    # [B]
        total_pos = num_pos_per_img.sum().clamp_min(1).to(images.dtype)   # scalar


        # forward pass
        with torch.autocast(device_type="cuda", enabled=True):
            loc_all, conf_all = model(images)

            # -------- 2) Localization loss (positives only) --------
            batch_loc_loss = torch.nn.functional.smooth_l1_loss(loc_all[pos_mask], loc_t_pm, reduction='sum') / total_pos


            # -------- 3) Classification loss with hard-negative mining --------
            batch_conf_loss = CELoss_w_neg_mining(conf_all=conf_all,
                                                  cls_t=cls_t,
                                                  pos_mask=pos_mask,
                                                  neg_pos_ratio=3,)

        # loss
        batch_loss = batch_loc_loss + batch_conf_loss



        scaler.scale(batch_loss).backward()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if scheduler is not None and new_scale >= old_scale:
                scheduler.step()
        torch.cuda.synchronize(device)
        t5 = time.perf_counter()

        step_t1 = time.perf_counter()

        if step >= warmup_steps:
            fetch_times.append(t1 - t0)
            h2d_times.append(t3 - t2)
            compute_times.append(t5 - t4)
            step_times.append(step_t1 - step_t0)
            n_samples += images.shape[0]

    total_time = sum(step_times)

    def stats(xs):
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "p95": sorted(xs)[int(0.95 * (len(xs) - 1))],
        }

    return {
        "fetch_time_s": stats(fetch_times),
        "h2d_time_s": stats(h2d_times),
        "compute_time_s": stats(compute_times),
        "step_time_s": stats(step_times),
        "samples_per_sec": n_samples / total_time,
        "batches_per_sec": len(step_times) / total_time,
    }