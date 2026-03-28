import matplotlib.pyplot as plt
from typing import Dict
import numpy as np

def plot_losses(losses: Dict, figsize=(10, 8)) -> None:
    """
    Plots train/test loss results

    Inputs
    losses: Dictionary with keys (all required): 
      "train_loss", "train_loss_loc", "train_loss_conf",
      "test_loss",  "test_loss_loc",  "test_loss_conf", "mAP"
    Values: lists of floats (except for mAP), all the same length.

    Output
    Produces a 2x2 matplotlib figure:
      (1) train_loss vs epoch and test_loss vs epoch
      (2) train_loss_conf vs epoch and test_loss_conf vs epoch
      (3) train_loss_loc  vs epoch and test_loss_loc  vs epoch
      (4) mAP vs epoch
    """
    required = [
        "train_loss", "train_loss_loc", "train_loss_conf",
        "test_loss",  "test_loss_loc",  "test_loss_conf", "mAP"
    ]
    # Key check
    missing = [k for k in required if k not in losses]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    # Type/length checks
    lens = []
    for k in ["train_loss", "train_loss_loc", "train_loss_conf",
              "test_loss",  "test_loss_loc",  "test_loss_conf"]:
        v = losses[k]
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"Value for '{k}' must be a list/tuple of floats.")
        lens.append(len(v))
        if any((not isinstance(x, (int, float)) or np.isnan(float(x)) or np.isinf(float(x))) for x in v):
            raise ValueError(f"Non-finite numeric in '{k}'.")
    if len(set(lens)) != 1:
        raise ValueError(f"All lists must have the same length. Got lengths: {dict(zip(required, lens))}")

    n = lens[0]
    x = list(range(n))

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # (1) total loss
    ax = axes[0,0]
    ax.plot(x, losses["train_loss"], label="train")
    ax.plot(x, losses["test_loss"],  label="validation")
    ax.set_title("Total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # (2) mAP
    mAP = [d.get('map_50', 0.0) for d in losses['mAP']]
    
    ax = axes[0,1]
    ax.plot(x, mAP, label="mAP")
    ax.set_title("mAP")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # (3) classification loss
    ax = axes[1,0]
    ax.plot(x, losses["train_loss_conf"], label="train")
    ax.plot(x, losses["test_loss_conf"],  label="validation")
    ax.set_title("Classification loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # (4) localization loss
    ax = axes[1,1]
    ax.plot(x, losses["train_loss_loc"], label="train")
    ax.plot(x, losses["test_loss_loc"],  label="validation")
    ax.set_title("Localization loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.show()