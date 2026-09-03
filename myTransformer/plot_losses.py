import matplotlib.pyplot as plt
from typing import Dict
import numpy as np


def plot_losses(losses: Dict, figsize=(10, 8)) -> None:
    """
    Plot training losses, with optional validation losses and validation mAP.

    Inputs
    ------
    losses : dict
        Dictionary with keys:
          "train_loss", "train_loss_loc", "train_loss_conf", "train_loss_GIoU",
          "mAP train",
          "test_loss", "test_loss_loc", "test_loss_conf", "test_loss_GIoU",
          "mAP test"

        All keys are required.

        Training entries must be nonempty lists/tuples of finite numeric values
        with equal length.

        Validation/test entries must either:
          1) all be empty, in which case only training data are plotted, or
          2) all be nonempty and have the same length as the training entries.

    figsize : tuple
        Matplotlib figure size.

    Output
    ------
    Produces:
      - a 4x1 figure when only training data are present:
          total loss, classification loss, localization loss, GIoU loss
      - a 5x1 figure when validation data are also present:
          total loss, mAP, classification loss, localization loss, GIoU loss
    """
    train_keys = [
        "train_loss",
        "train_loss_loc",
        "train_loss_conf",
        "train_loss_GIoU",
    ]

    test_keys = [
        "test_loss",
        "test_loss_loc",
        "test_loss_conf",
        "test_loss_GIoU",
    ]

    required = train_keys + test_keys

    # Key check
    missing = [k for k in required if k not in losses]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    # Type and numeric checks
    for k in required:
        v = losses[k]

        if not isinstance(v, (list, tuple)):
            raise TypeError(
                f"Value for '{k}' must be a list/tuple of numeric values."
            )

        if any(
            not isinstance(x, (int, float, np.number))
            or not np.isfinite(float(x))
            for x in v
        ):
            raise ValueError(f"Non-finite or non-numeric value in '{k}'.")

    # Training histories must all be nonempty and have equal length.
    train_lengths = {k: len(losses[k]) for k in train_keys}

    if any(n == 0 for n in train_lengths.values()):
        raise ValueError(
            f"Training histories must be nonempty. Got lengths: {train_lengths}"
        )

    if len(set(train_lengths.values())) != 1:
        raise ValueError(
            f"All training histories must have the same length. "
            f"Got lengths: {train_lengths}"
        )

    n = next(iter(train_lengths.values()))

    # Validation/test histories may all be empty, or all populated.
    test_lengths = {k: len(losses[k]) for k in test_keys}
    nonempty_test = [length > 0 for length in test_lengths.values()]

    if any(nonempty_test) and not all(nonempty_test):
        raise ValueError(
            "Validation/test histories must either all be empty or all be "
            f"populated. Got lengths: {test_lengths}"
        )

    has_test_data = all(nonempty_test)

    if has_test_data:
        if len(set(test_lengths.values())) != 1:
            raise ValueError(
                f"All validation/test histories must have the same length. "
                f"Got lengths: {test_lengths}"
            )

        test_n = next(iter(test_lengths.values()))
        if test_n != n:
            raise ValueError(
                "Training and validation/test histories must have the same "
                f"length. Training length: {n}; validation/test lengths: "
                f"{test_lengths}"
            )

    x = list(range(1, n + 1))

    nrows = 5 if has_test_data else 4
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        constrained_layout=True,
    )

    # Total loss
    ax = axes[0]
    ax.plot(x, losses["train_loss"], label="train")
    if has_test_data:
        ax.plot(x, losses["test_loss"], label="val")
    ax.set_title("Total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    if has_test_data:
        # Validation mAP
        ax = axes[1]
        ax.plot(x, losses["mAP test"] if len(losses["mAP test"]) != 0 else losses["mAP train"], label="mAP")
        ax.set_title("mAP")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()

        loss_axis_offset = 1
    else:
        loss_axis_offset = 0

    # Classification loss
    ax = axes[1 + loss_axis_offset]
    ax.plot(x, losses["train_loss_conf"], label="train")
    if has_test_data:
        ax.plot(x, losses["test_loss_conf"], label="val")
    ax.set_title("Classification loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # Localization loss
    ax = axes[2 + loss_axis_offset]
    ax.plot(x, losses["train_loss_loc"], label="train")
    if has_test_data:
        ax.plot(x, losses["test_loss_loc"], label="val")
    ax.set_title("Localization loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # GIoU loss
    ax = axes[3 + loss_axis_offset]
    ax.plot(x, losses["train_loss_GIoU"], label="train")
    if has_test_data:
        ax.plot(x, losses["test_loss_GIoU"], label="val")
    ax.set_title("GIoU loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.show()
