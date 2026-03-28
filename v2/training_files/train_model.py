# train_model.py

import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from v2.model_files.SSD_from_scratch import mySSD
from v2.training_files.SSD_trainer import SSD_train
from v2.training_files.CosSched import build_optimizer_and_scheduler
from v2.training_files.build_dataloaders import build_train_dl
from v2.training_files.save_load_ckpt import load_checkpoint
from v2.training_files.plot_losses import plot_losses


device = "cuda" if torch.cuda.is_available() else "cpu"

# desktop, laptop, ubuntu
machine = 'desktop'

# Setup path to data folder
if machine == 'laptop':
    folder_path = Path(r"C:\self-driving-car\data")
elif machine == 'desktop':
    folder_path = Path(r"C:\Udacity_car_data\data")
elif machine == 'ubuntu':
    folder_path = Path(r"/mnt/c/Udacity_car_data/data")

train_path = folder_path / "train"




ssdmodel = mySSD(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                 in_channels=3,
                 variances=(0.1, 0.2),
                 ).to(device)

train_dataloader, val_dataloader = build_train_dl(train_path=train_path)

optimizer, scheduler = build_optimizer_and_scheduler(model=ssdmodel,
                                                     train_dataloader=train_dataloader,
                                                     max_epochs=150,
                                                     warmup_epochs=5,
                                                     base_lr=0.003,
                                                     min_lr=1e-6,
                                                     momentum=0.9,
                                                     weight_decay=0.005)

scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

resume_path = Path(r"C:\Users\eblac\Documents\GitHub\self-driving-car\v2\saved_models") / "last.ckpt"
if resume_path.exists():
    start_epoch, best_map, loss_dict = load_checkpoint(
        resume_path, ssdmodel, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, map_location="cpu"  # safe load, then move to device
    )
    ssdmodel.to(device)
else:
    start_epoch, best_map, loss_dict = 0, None, None

if __name__ == "__main__":
    results = SSD_train(model=ssdmodel,
                        train_dataloader=train_dataloader,
                        test_dataloader=val_dataloader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        sched_step_w_opt=True,
                        iou_thresh=0.5,
                        iou_variant="IoU",
                        neg_pos_ratio=3.0,
                        score_thresh=0.2,
                        nms_thresh=0.3,
                        max_detections_per_img=200,
                        epochs=2,
                        early_stopping_rounds=None,
                        device=device,
                        save_model=True,
                        save_best_model=False,
                        epoch_save_interval=None,
                        SAVE_DIR=r"C:\Users\eblac\Documents\GitHub\self-driving-car\v2\saved_models",
                        timing=False,
                        past_train_dict=loss_dict,
                        compute_mAP=False,
                        )

    plot_losses(results)
