from __future__ import annotations

from typing import Iterator, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset



def build_calibration_loader_from_dataset(dataset,
                                          *,
                                          num_samples: int | None = 1000,
                                          seed: int = 0,
                                          batch_size: int = 8,
                                          num_workers: int = 0,
                                          pin_memory: bool = False,
                                          ) -> DataLoader:
    """
    Builds a calibration DataLoader from an existing dataset (e.g., val_data).

    Assumptions:
      - dataset[i] returns (img, target)
      - img is a torch.Tensor or tv_tensors.Image shaped [3,300,300]
      - transforms are already applied (use your test_tfms / inference transforms)

    num_samples:
      - None => use the entire dataset
      - int  => deterministic subset of that size
    """

    ds = dataset

    if num_samples is not None and num_samples < len(ds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:num_samples].tolist()
        ds = Subset(ds, idx)

    def collate_images(batch):
        # batch elements are (img, target); ignore target
        imgs = []
        for img, _target in batch:
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            imgs.append(img.to(dtype=torch.float32))
        x = torch.stack(imgs, dim=0)  # [B,3,300,300]
        return x

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_images,
    )


class SSDCalibrationDataReader:
    """
    ONNX Runtime calibrator adapter.
    Yields {"images": np.ndarray float32 [B,3,300,300]}.
    """
    def __init__(self, loader: DataLoader, input_name: str = "images"):
        self.loader = loader
        self.input_name = input_name
        self._iter: Optional[Iterator[torch.Tensor]] = None

    def get_next(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            x = next(self._iter)
        except StopIteration:
            return None

        x = x.detach().cpu().contiguous()
        return {self.input_name: x.numpy().astype(np.float32)}

    def rewind(self):
        self._iter = None
