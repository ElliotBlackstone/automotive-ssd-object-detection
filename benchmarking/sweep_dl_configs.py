# sweep_dl_configs.py

from itertools import product
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bm_train import benchmark_train_loop


def sweep_loader_configs(
    model,
    dataset,
    collate_fn,
    optimizer_ctor,
    scheduler_ctor,
    scaler_ctor,
    batch_size=8,
    device="cuda",
):
    configs = []

    num_workers_list = [2, 4, 8, 12]
    pin_memory_list = [True]
    prefetch_list = [2, 4, 6]
    persistent_list = [False]
    mp_context_list = [None] # [None, "spawn"]

    for nw in num_workers_list:
        for pm in pin_memory_list:
            for ctx in mp_context_list:
                if nw == 0:
                    cfg = dict(
                        num_workers=0,
                        pin_memory=pm,
                        persistent_workers=False,
                        prefetch_factor=None,
                        multiprocessing_context=None,
                    )
                    configs.append(cfg)
                else:
                    for pf, pw in product(prefetch_list, persistent_list):
                        cfg = dict(
                            num_workers=nw,
                            pin_memory=pm,
                            persistent_workers=pw,
                            prefetch_factor=pf,
                            multiprocessing_context=ctx,
                        )
                        configs.append(cfg)

    results = []

    base_model = copy.deepcopy(model)


    for cfg in configs:
        model_copy = copy.deepcopy(base_model)
        optimizer = optimizer_ctor(model_copy.parameters())
        scheduler = scheduler_ctor(optimizer)
        scaler = scaler_ctor()

        out = benchmark_train_loop(
            model=model_copy,
            dataset=dataset,
            collate_fn=collate_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            batch_size=batch_size,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            persistent_workers=cfg["persistent_workers"],
            prefetch_factor=cfg["prefetch_factor"] if cfg["num_workers"] > 0 else 2,
            multiprocessing_context=cfg["multiprocessing_context"],
            warmup_steps=20,
            measure_steps=100,
        )

        results.append({
            **cfg,
            "samples_per_sec": out["samples_per_sec"],
            "median_step_time_s": out["step_time_s"]["median"],
            "median_fetch_time_s": out["fetch_time_s"]["median"],
            "median_h2d_time_s": out["h2d_time_s"]["median"],
            "median_compute_time_s": out["compute_time_s"]["median"],
        })

    results.sort(key=lambda x: x["samples_per_sec"], reverse=True)
    return results



if __name__ == "__main__":
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2
    from v1.CarImageClass import ImageClass
    from v1.SSD_trainer import ConditionalIoUCrop, collate_detection
    from v2.model_files.SSD_from_scratch import mySSD

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # desktop, laptop, ubuntu
    machine = 'ubuntu'

    # Setup path to data folder
    if machine == 'laptop':
        folder_path = Path(r"C:\self-driving-car\data")
    elif machine == 'desktop':
        folder_path = Path(r"C:\Udacity_car_data\data")
    elif machine == 'ubuntu':
        folder_path = Path(r"/mnt/c/Udacity_car_data/data")

    train_path = folder_path / "train"
    test_path = folder_path / "test"

    # transforms
    train_tfms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.RandomZoomOut(fill=0, p=0.5),       # Zoom out hurts model performance

        ConditionalIoUCrop(min_area_frac=0.02,   # threshold between "large" and "small"
                        small_min_scale=0.4,
                        large_min_scale=0.7,
                        max_scale=1.0,
                        min_aspect_ratio=0.75,
                        max_aspect_ratio=1.33,
                        small_sampler_options=(0.0, 0.05, 0.1, 2.0),
                        large_sampler_options=(0.05, 0.1, 0.3, 2.0),
                        trials=10),

        v2.SanitizeBoundingBoxes(min_size=1.0),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomPhotometricDistort(p=0.5),
        v2.Resize((300, 300), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 993 images
    train_data = ImageClass(targ_dir=train_path, transform=train_tfms, file_pct=0.05, rand_seed=724, include_area=False)


    model = mySSD(class_to_idx_dict=train_data.class_to_idx, in_channels=3, variances=(0.1, 0.2))

    BATCH_SIZE = 16

    steps_per_epoch = len(DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_detection,
    ))

    def scheduler_ctor(optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=150,
            steps_per_epoch=steps_per_epoch,
        )

    def optimizer_ctor(params):
        return torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.005)

    def scaler_ctor():
        return torch.amp.GradScaler("cuda", enabled=True)

    
    results = sweep_loader_configs(model=model,
                        dataset=train_data,
                        collate_fn=collate_detection,
                        optimizer_ctor=optimizer_ctor,
                        scheduler_ctor=scheduler_ctor,
                        scaler_ctor=scaler_ctor,
                        batch_size=BATCH_SIZE,
                        device=device)
    
    print(f"Batch Size: {BATCH_SIZE}")
    print()
    for k in results:
        print(k)

# batch size 4 winner:
# {'num_workers': 2, 'pin_memory': True, 'persistent_workers': False, 'prefetch_factor': 4, 'multiprocessing_context': None,
# 'samples_per_sec': 63.92698063891477, 'median_step_time_s': 0.06252471799962223, 'median_fetch_time_s': 0.0002491125001142791,
# 'median_h2d_time_s': 0.0005203460000302584, 'median_compute_time_s': 0.061636413499854825}

# batch size 8 winner:
# {'num_workers': 12, 'pin_memory': True, 'persistent_workers': False, 'prefetch_factor': 4, 'multiprocessing_context': None,
# 'samples_per_sec': 77.16178648058349, 'median_step_time_s': 0.10330096950019652, 'median_fetch_time_s': 0.00027631399962047,
# 'median_h2d_time_s': 0.0008355845002370188, 'median_compute_time_s': 0.10164135200011515}

# batch size 16 winner:
# {'num_workers': 4, 'pin_memory': True, 'persistent_workers': False, 'prefetch_factor': 4, 'multiprocessing_context': None,
# 'samples_per_sec': 81.16432461856895, 'median_step_time_s': 0.18809445600027175, 'median_fetch_time_s': 0.0004448585000318417,
# 'median_h2d_time_s': 0.0016056390004450805, 'median_compute_time_s': 0.18600801900038277}

# batch size 32 winner:
# {'num_workers': 4, 'pin_memory': True, 'persistent_workers': False, 'prefetch_factor': 2, 'multiprocessing_context': None,
# 'samples_per_sec': 80.9101945561321, 'median_step_time_s': 0.3546846935000758, 'median_fetch_time_s': 0.00045518099977925885,
# 'median_h2d_time_s': 0.0029837854999641422, 'median_compute_time_s': 0.3509303290002208}