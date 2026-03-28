# build_dataloaders.py
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from typing import Tuple
from pathlib import Path

from .CondIoUCrop import ConditionalIoUCrop
from ..CarImageClass import ImageClass, make_train_test_split


def collate_detection(batch):
    # batch: list of (img, target) tuples
    imgs  = [img for img, _ in batch]
    tgts  = [tgt for _, tgt in batch]
    return torch.stack(imgs, dim=0), tgts




def build_train_dl(train_path: Path) -> Tuple[DataLoader, DataLoader]:
    

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

    test_tfms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((300, 300), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageClass(targ_dir=train_path, transform=train_tfms, file_pct=1, rand_seed=724, include_area=False)

    train_data_init, val_data = make_train_test_split(full_set=train_set,
                                                      test_size=0.25,
                                                      rand_state=724,
                                                      transform_train=train_tfms,
                                                      transform_test=test_tfms,
                                                      include_area=False)
    
    df = train_data_init.annotate_df

    # 1) count non-empty objects per filename
    obj_counts = (
        df.loc[df["class"] != "empty"]
        .groupby("filename")
        .size()                      # number of rows (objects) per filename
    )

    # 2) map counts back to all rows, defaulting to 0 when there are no objects
    df["num_objects"] = (
        df["filename"]
        .map(obj_counts)            # NaN for filenames with only 'empty'
        .fillna(0)
        .astype(int)
    )

    filenames0 = df[df['num_objects'] == 0]['filename'].unique().tolist()
    filenames12 = df[(df['num_objects'] >= 1) & (df['num_objects'] <= 2)]['filename'].unique().tolist()
    filenames36 = df[(df['num_objects'] >= 3) & (df['num_objects'] <= 6)]['filename'].unique().tolist()
    filenames79 = df[(df['num_objects'] >= 7) & (df['num_objects'] <= 9)]['filename'].unique().tolist()
    filenames10p = df[df['num_objects'] >= 10]['filename'].unique().tolist()

    biglist = []
    
    # include background images once
    biglist += filenames0

    # include images with 1 to 2 objects twice
    biglist += filenames12 + filenames12

    # include images with 3 to 6 objects three times
    biglist += filenames36 + filenames36 + filenames36

    # include images with 7 to 9 objects 4 times
    biglist += filenames79 + filenames79 + filenames79 + filenames79

    # include images with 10+ objects 5 times
    biglist += filenames10p + filenames10p + filenames10p + filenames10p + filenames10p


    train_data = ImageClass(targ_dir=train_path, file_list=biglist, transform=train_tfms, file_pct=1, rand_seed=724, include_area=False)

    train_dataloader = DataLoader(train_data, 
                              batch_size=8, 
                              shuffle=True, 
                              num_workers=2,
                              persistent_workers=True,
                              prefetch_factor=2,
                              pin_memory=True,
                              collate_fn=collate_detection,
                              multiprocessing_context="spawn",
                              )

    val_dataloader = DataLoader(val_data, 
                                batch_size=8, 
                                shuffle=False,
                                num_workers=2,
                                persistent_workers=True,
                                prefetch_factor=2,
                                pin_memory=True,
                                collate_fn=collate_detection,
                                multiprocessing_context="spawn",
                                )


    return train_dataloader, val_dataloader



def build_test_dl(test_path: Path) -> DataLoader:

    test_tfms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((300, 300), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
    ])

    test_data = ImageClass(targ_dir=test_path, transform=test_tfms, file_pct=1)

    test_dataloader = DataLoader(test_data, 
                             batch_size=8, 
                             shuffle=False, 
                             num_workers=2,
                             persistent_workers=True,
                             prefetch_factor=2,
                             pin_memory=False,
                             collate_fn=collate_detection,
                             )

    return test_dataloader