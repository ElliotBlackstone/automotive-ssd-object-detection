# train_model.py

# to use:
# & "c:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" train_ViT_model.py --epochs 5

import torch
from pathlib import Path
import sys
import argparse

from myViT import VisionTransformer
from ViTTrainer import ViT_train
from sched_optim import build_optimizer_and_scheduler
from build_dataloaders_ViT import build_train_dl
from plot_losses import plot_losses

sys.path.append(str(Path.cwd().parent / "self-driving-car"))
from v2.training_files.save_load_ckpt import load_checkpoint




device = "cuda" if torch.cuda.is_available() else "cpu"

# desktop, laptop, ubuntu
machine = 'desktop'

# Setup path to data folder
if machine == 'laptop':
    folder_path = Path(r"C:\self-driving-car\data")
elif machine == 'desktop':
    folder_path = Path(r"C:\Udacity_car_data\data")
elif machine == 'ubuntu':
    # folder_path = Path(r"/mnt/c/Udacity_car_data/data")
    folder_path = Path.home() / "datasets" / "Udacity_car_data" / "data"

train_path = folder_path / "train"



# ~11.3 million parameters
ViTmodel = VisionTransformer(class_to_idx_dict={'biker': 0, 'car': 1, 'pedestrian': 2, 'trafficLight': 3, 'truck': 4},
                             img_size=300,
                             patch_H=15,
                             patch_W=10,
                             in_channels=3,
                             embed_dim=256,
                             num_layers=6,
                             num_heads=4,
                             dim_feedforward=256*4,
                             dropout=0.1,
                             num_queries=50).to(device)


train_dataloader, val_dataloader = build_train_dl(train_path=train_path,
                                                  batch_size=64,
                                                  num_workers=4,
                                                  prefetch_factor=2)

optimizer, scheduler = build_optimizer_and_scheduler(model=ViTmodel,
                                                     train_dataloader=train_dataloader,
                                                     max_epochs=300,
                                                     warmup_epochs=10,
                                                     base_lr=1e-4,
                                                     min_lr=1e-7,
                                                     weight_decay=0.001)

scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

resume_path = Path("C:/Users/eblac/Documents/GitHub/myTransformer/saved_models/last.ckpt")
# Path.home() / "repos" / "automotive-ssd-object-detection" / "v2" / "saved_models" / "last.ckpt"
if resume_path.exists():
    start_epoch, best_map, loss_dict = load_checkpoint(
        resume_path, ViTmodel, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, map_location="cpu"  # safe load, then move to device
    )
    ViTmodel.to(device)
else:
    start_epoch, best_map, loss_dict = 0, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    args = parser.parse_args()

    results = ViT_train(model=ViTmodel,
                        train_dataloader=train_dataloader,
                        test_dataloader=val_dataloader,
                        optimizer=optimizer,
                        lambda_CE=1.0,
                        lambda_L1=5.0,
                        lambda_GIoU=2.0,
                        lambda_CE_HM=1.0,
                        lambda_L1_HM=5.0,
                        lambda_GIoU_HM=2.0,
                        scheduler=scheduler,
                        scaler=scaler,
                        sched_step_w_opt=True,
                        epochs=args.epochs,
                        early_stopping_rounds=None,
                        device=device,
                        save_model=True,
                        save_best_model=True,
                        epoch_save_interval=None,
                        SAVE_DIR=Path("C:/Users/eblac/Documents/GitHub/myTransformer/saved_models"),
                        timing=False,
                        past_train_dict=loss_dict,
                        compute_mAP_train=False,
                        compute_mAP_test=True,
                        include_test_step=True,
                        bg_weight=0.1,
                        aux_loss_weight=0.5,
                        grad_clip_val=1.0)


    plot_losses(losses=results, figsize=(10, 12))

if __name__ == "__main__":
    main()