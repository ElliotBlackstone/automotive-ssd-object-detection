"""Deterministic overfitting checks for the repository's DETR-like ViT.

Run from a notebook:

    from sanity_test_overfit import run_sanity_suite
    sanity_runs = run_sanity_suite(train_path=train_path)

Or from PowerShell:

    python sanity_test_overfit.py --train-path C:\\Udacity_car_data\\data\\train
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import v2

from myTrainStep import move_targets_to_device, myTrainStep
from myViT import VisionTransformer


CLASS_TO_IDX = {
    "biker": 0,
    "car": 1,
    "pedestrian": 2,
    "trafficLight": 3,
    "truck": 4,
}


def collate_detection(batch):
    images = torch.stack([image for image, _ in batch], dim=0)
    targets = [target for _, target in batch]
    return images, targets


class FrozenDetectionDataset(Dataset):
    """Keep transformed samples in memory so every update sees identical data."""

    def __init__(self, samples: Sequence[tuple[torch.Tensor, dict]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, target = self.samples[index]
        # Return copies in case downstream code performs an in-place operation.
        image = image.clone()
        target = {
            key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in target.items()
        }
        return image, target


@dataclass
class SanityResult:
    model: VisionTransformer
    history: list[dict[str, float]]
    sample_indices: list[int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The model contains no convolution, but these settings prevent cuDNN from
    # introducing noise if the data pipeline changes later.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(device: torch.device) -> VisionTransformer:
    return VisionTransformer(
        class_to_idx_dict=CLASS_TO_IDX,
        img_size=300,
        patch_H=15,
        patch_W=10,
        in_channels=3,
        embed_dim=64,
        num_layers=4,
        num_heads=4,
        dim_feedforward=64 * 4,
        dropout=0.0,
        num_queries=15,
    ).to(device)


def load_fixed_samples(
    train_path: str | Path,
    *,
    num_images: int = 9,
    file_pct: float = 0.0005,
    dataset_seed: int = 724,
) -> tuple[list[tuple[torch.Tensor, dict]], list[int]]:
    """Load and freeze the same unaugmented subset used in the notebook."""
    sibling_repo = Path(__file__).resolve().parent.parent / "self-driving-car"
    if str(sibling_repo) not in sys.path:
        sys.path.append(str(sibling_repo))

    from v2.CarImageClass import ImageClass

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((300, 300), antialias=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    source = ImageClass(
        targ_dir=Path(train_path),
        transform=transforms,
        file_pct=file_pct,
        rand_seed=dataset_seed,
        include_area=False,
    )
    if len(source) < num_images:
        raise ValueError(
            f"The sampled dataset has only {len(source)} images; "
            f"requested {num_images}. Increase file_pct."
        )

    indices = list(range(num_images))
    samples = [source[index] for index in indices]

    object_counts = [int(target["labels"].numel()) for _, target in samples]
    max_objects = max(object_counts, default=0)
    if max_objects > 15:
        raise ValueError(
            f"A selected image has {max_objects} objects, but the model has only "
            "15 queries. Perfect recall is impossible; increase num_queries."
        )

    print(f"Frozen {len(samples)} images; object counts: {object_counts}")
    return samples, indices


def make_full_batch_loader(samples: Sequence[tuple[torch.Tensor, dict]]) -> DataLoader:
    dataset = FrozenDetectionDataset(samples)
    return DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_detection,
    )


def post_clip_gradient_norm(model: torch.nn.Module) -> float:
    """Return the gradient norm left by myTrainStep after its clipping call."""
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return float("nan")
    return torch.stack(norms).norm(2).item()


def train_step_with_pre_clip_norm(**train_step_kwargs):
    """Run myTrainStep and capture the norm returned by clip_grad_norm_.

    torch.nn.utils.clip_grad_norm_ returns the total gradient norm measured
    before it rescales any gradients. myTrainStep currently discards that
    return value, so this diagnostic wrapper records it without changing the
    production training function's API.

    This wrapper temporarily replaces the PyTorch function process-wide. The
    sanity runner is deliberately single-threaded, and the original function
    is restored in a finally block even if the training step raises.
    """
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_
    captured_pre_clip_norms: list[float] = []

    def recording_clip_grad_norm_(parameters, max_norm, *args, **kwargs):
        total_norm = original_clip_grad_norm(
            parameters,
            max_norm,
            *args,
            **kwargs,
        )
        captured_pre_clip_norms.append(float(total_norm.detach().float().item()))
        return total_norm

    torch.nn.utils.clip_grad_norm_ = recording_clip_grad_norm_
    try:
        train_metrics = myTrainStep(**train_step_kwargs)
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm

    if len(captured_pre_clip_norms) != 1:
        raise RuntimeError(
            "Expected exactly one clipping call for the one-batch sanity "
            f"loader, but observed {len(captured_pre_clip_norms)}."
        )

    return train_metrics, captured_pre_clip_norms[0]


@torch.inference_mode()
def evaluate_map50(
    model: VisionTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate every image using one fixed model state."""
    was_training = model.training
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.50],
        class_metrics=True,
    ).to(device)

    for images, targets in dataloader:
        images = images.to(device)
        targets = move_targets_to_device(targets, device, non_blocking=False)
        class_logits, boxes = model(images)
        predictions = model.predict(
            images,
            pre_class_logits=class_logits[-1],
            pre_bboxes=boxes[-1],
        )
        metric.update(predictions, targets)

    map50 = float(metric.compute()["map_50"].item())
    if was_training:
        model.train()
    return map50


def run_case(
    samples: Sequence[tuple[torch.Tensor, dict]],
    sample_indices: Iterable[int],
    *,
    name: str,
    steps: int,
    eval_every: int,
    learning_rate: float,
    grad_clip_val: float,
    seed: int,
    device: torch.device,
    target_map50: float = 0.995,
    target_evaluations: int = 2,
) -> SanityResult:
    """Run one deterministic, full-batch optimizer update per loop iteration."""
    seed_everything(seed)
    model = build_model(device)
    dataloader = make_full_batch_loader(samples)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    history: list[dict[str, float]] = []
    consecutive_target_evaluations = 0
    print(
        f"\n{name}: {len(samples)} image(s), one optimizer step per iteration, "
        f"lr={learning_rate:g}, grad_clip_val={grad_clip_val:g}"
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=1e-6,
    )

    for step in range(1, steps + 1):
        train_metrics, pre_clip_grad_norm = train_step_with_pre_clip_norm(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lambda_CE=1.0,
            lambda_L1=5.0,
            lambda_GIoU=2.0,
            lambda_CE_HM=1.0,
            lambda_L1_HM=5.0,
            lambda_GIoU_HM=2.0,
            scheduler=scheduler,
            scaler=None,
            device=device,
            timing=False,
            compute_mAP=False,
            bg_weight=0.1,
            grad_clip_val=grad_clip_val,
        )

        should_evaluate = step == 1 or step % eval_every == 0 or step == steps
        if not should_evaluate:
            continue

        map50 = evaluate_map50(model, dataloader, device)
        record = {
            "step": float(step),
            "loss": float(train_metrics["training loss"]),
            "ce": float(train_metrics["classification loss"]),
            "l1": float(train_metrics["localization loss"]),
            "giou": float(train_metrics["GIoU loss"]),
            "map50": map50,
            "pre_clip_grad_norm": pre_clip_grad_norm,
            "post_clip_grad_norm": post_clip_gradient_norm(model),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"step={step:5d}  loss={record['loss']:.4f}  "
            f"CE={record['ce']:.4f}  L1={record['l1']:.4f}  "
            f"GIoU={record['giou']:.4f}  mAP@.50={map50:.4f}  "
            f"pre-clip |g|={record['pre_clip_grad_norm']:.4f}  "
            f"post-clip |g|={record['post_clip_grad_norm']:.4f}"
        )

        if map50 >= target_map50:
            consecutive_target_evaluations += 1
        else:
            consecutive_target_evaluations = 0

        if consecutive_target_evaluations >= target_evaluations:
            print(
                f"Stopped: mAP@.50 remained >= {target_map50:.3f} for "
                f"{target_evaluations} evaluations."
            )
            break

    return SanityResult(
        model=model,
        history=history,
        sample_indices=list(sample_indices),
    )


def run_sanity_suite(
    train_path: str | Path,
    *,
    num_images: int = 9,
    file_pct: float = 0.0005,
    one_image_steps: int = 2_000,
    all_image_steps: int = 5_000,
    eval_every: int = 100,
    learning_rate: float = 3e-4,
    grad_clip_val: float = 1.0,
    seed: int = 724,
    device: str | torch.device | None = None,
) -> dict[str, SanityResult]:
    """Run the one-image test first, followed by the complete small subset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    samples, source_indices = load_fixed_samples(
        train_path,
        num_images=num_images,
        file_pct=file_pct,
        dataset_seed=seed,
    )

    nonempty_positions = [
        position
        for position, (_, target) in enumerate(samples)
        if 0 < int(target["labels"].numel()) <= 15
    ]
    if not nonempty_positions:
        raise ValueError("The selected subset contains no non-empty image.")

    # Use the least crowded non-empty image for the simplest possible test.
    one_position = min(
        nonempty_positions,
        key=lambda position: int(samples[position][1]["labels"].numel()),
    )

    one_result = run_case(
        [samples[one_position]],
        [source_indices[one_position]],
        name="ONE-IMAGE SANITY TEST",
        steps=one_image_steps,
        eval_every=eval_every,
        learning_rate=learning_rate,
        grad_clip_val=grad_clip_val,
        seed=seed,
        device=device,
    )

    all_result = run_case(
        samples,
        source_indices,
        name="ALL-IMAGE SANITY TEST",
        steps=all_image_steps,
        eval_every=eval_every,
        learning_rate=learning_rate,
        grad_clip_val=grad_clip_val,
        seed=seed,
        device=device,
    )

    return {"one_image": one_result, "all_images": all_result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=9)
    parser.add_argument("--file-pct", type=float, default=0.0005)
    parser.add_argument("--one-image-steps", type=int, default=2_000)
    parser.add_argument("--all-image-steps", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sanity_suite(
        train_path=args.train_path,
        num_images=args.num_images,
        file_pct=args.file_pct,
        one_image_steps=args.one_image_steps,
        all_image_steps=args.all_image_steps,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        grad_clip_val=args.grad_clip_val,
        seed=args.seed,
        device=args.device,
    )
