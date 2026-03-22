import torch
from typing import Dict
from pathlib import Path
import numpy as np
import random


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
