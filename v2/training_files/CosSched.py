import torch
import math


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    min_lr: float = 0.0,
                                    last_epoch: int = -1,
                                    ) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine decay with linear warmup.

    LR(t) = base_lr * f(t), where f(t) is:
      - warmup: linearly from 0 -> 1 over [0, num_warmup_steps)
      - cosine: from 1 -> (min_lr / base_lr) over [num_warmup_steps, num_training_steps]

    Arguments
    ---------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    num_warmup_steps : int
        Number of steps for linear warmup.
    num_training_steps : int
        Total number of training steps (epochs * steps_per_epoch).
    min_lr : float, default 0.0
        Absolute minimum learning rate. Implemented as a ratio of base_lr.
    last_epoch : int, default -1
        See PyTorch docs for LambdaLR (use -1 when creating scheduler).

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler to be stepped *once per optimizer step*.
    """
    # we implement min_lr by enforcing a minimum multiplicative factor
    # relative to base_lr; per param group the ratio may differ.
    # -> for each param group, factor(t) in [min_ratio, 1]
    # where min_ratio = min_lr / base_lr_group
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    # sanity check assumptions
    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be >= 0")
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")
    if num_warmup_steps > num_training_steps:
        raise ValueError("num_warmup_steps cannot exceed num_training_steps")

    def lr_lambda(current_step: int):
        # this returns one factor per param group
        factors = []
        for base_lr in base_lrs:
            if min_lr > base_lr:
                raise ValueError("min_lr cannot be larger than base_lr")

            min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

            if current_step < num_warmup_steps and num_warmup_steps > 0:
                # linear warmup: 0 -> 1
                warmup_frac = float(current_step) / float(max(1, num_warmup_steps))
                factor = warmup_frac  # in [0,1]
            else:
                # cosine phase
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)  # clamp numerically

                # pure cosine from 1 -> 0
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # in [0,1]

                # rescale to [min_ratio, 1]
                factor = min_ratio + (1.0 - min_ratio) * cosine

            factors.append(factor)
            
        return factors[0]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_optimizer_and_scheduler(model: torch.nn.Module,
                                  train_dataloader: torch.utils.data.DataLoader,
                                  max_epochs: int = 120,
                                  warmup_epochs: int = 5,
                                  base_lr: float = 3e-3,
                                  min_lr: float = 1e-5,
                                  momentum: float = 0.9,
                                  weight_decay: float = 5e-4):
    """
    Create SGD optimizer and cosine-with-warmup scheduler
    for an SSD-style detector.

    Arguments
    ---------
    model : nn.Module
        Model to be trained.
    train_dataloader : DataLoader
        Only used to infer steps_per_epoch.
    max_epochs : int
        Total number of epochs you plan to train.
    warmup_epochs : int
        Number of warmup epochs (linear LR increase).
    base_lr : float
        Peak learning rate after warmup.
    min_lr : float
        Minimum LR at the end of cosine decay.
    momentum : float
        Momentum for SGD.
    weight_decay : float
        L2 weight decay.

    Returns
    -------
    optimizer : torch.optim.SGD
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Must be stepped *once per optimizer step*.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    steps_per_epoch = len(train_dataloader)
    num_training_steps = max_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
    )

    return optimizer, scheduler