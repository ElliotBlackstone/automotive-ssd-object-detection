r"""Evaluate fine-grained transformer confidence thresholds on validation.

The validation split and matching protocol are shared with
``validation_threshold_sweep.py``. The script prints a Markdown table and
writes ``transformer_validation_fine_threshold_results.md`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from compare_models import (
    _metric_target,
    _prediction_at_native_size,
    collate_detection,
)
from validation_threshold_sweep import (
    MATCH_IOU_THRESHOLD,
    VALIDATION_FRACTION,
    VALIDATION_SEED,
    ThresholdSweep,
    _build_validation_dataset,
)
from video_predict import (
    CLASS_TO_IDX,
    DEFAULT_TRANSFORMER_CHECKPOINT,
    TRANSFORMER_DIR,
    _load_state_dict,
    _resolve_device,
)


THRESHOLDS = (0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.9975, 0.999)
DEFAULT_OUTPUT = Path("transformer_validation_fine_threshold_results.md")


def _load_transformer(
    checkpoint: Path, device: torch.device
) -> torch.nn.Module:
    transformer_dir = str(TRANSFORMER_DIR)
    if transformer_dir not in sys.path:
        sys.path.insert(0, transformer_dir)
    from myViT import VisionTransformer

    model = VisionTransformer(
        class_to_idx_dict=CLASS_TO_IDX,
        img_size=300,
        patch_H=15,
        patch_W=10,
        in_channels=3,
        embed_dim=256,
        num_layers=6,
        num_heads=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_queries=50,
    )
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def evaluate(
    args: argparse.Namespace,
) -> tuple[Dict[float, Dict[str, float]], int]:
    device = _resolve_device(args.device)
    base_dataset = _build_validation_dataset(args.train_dir)
    if base_dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            "Validation classes do not match the model. "
            f"Expected {CLASS_TO_IDX}, found {base_dataset.class_to_idx}."
        )

    dataset: Dataset = base_dataset
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_detection,
    )

    print(f"Validation split contains {len(base_dataset)} images.", flush=True)
    print(f"Loading transformer on {device}...", flush=True)
    transformer = _load_transformer(args.transformer_checkpoint, device)
    sweep = ThresholdSweep(THRESHOLDS, inclusive=True)
    image_count = 0
    amp_enabled = args.amp and device.type == "cuda"

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                class_logits, boxes = transformer(images)
                predictions = transformer.predict(
                    images=images,
                    pre_class_logits=class_logits[-1],
                    pre_bboxes=boxes[-1],
                    conf_thresh=THRESHOLDS[0],
                )

            metric_targets = [_metric_target(target) for target in targets]
            predictions_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(predictions, targets)
            ]
            sweep.update(predictions_cpu, metric_targets)

            image_count += len(targets)
            if image_count % args.progress_every < len(targets):
                print(f"Evaluated {image_count}/{len(dataset)} images", flush=True)

    return sweep.compute(), image_count


def render_markdown(
    results: Dict[float, Dict[str, float]],
    args: argparse.Namespace,
    validation_size: int,
) -> str:
    lines = [
        "# Transformer validation fine-threshold sweep",
        "",
        f"- Training directory: `{args.train_dir.resolve()}`",
        f"- Validation split: {VALIDATION_FRACTION:.0%}, seed {VALIDATION_SEED} "
        f"({validation_size} images evaluated)",
        f"- Matching: class-aware, one-to-one, IoU >= {MATCH_IOU_THRESHOLD:.2f}",
        "",
        "| Confidence threshold | Precision | Recall | F1 | False positives / image |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        result = results[threshold]
        lines.append(
            f"| {threshold:g} | {result['precision']:.4f} | "
            f"{result['recall']:.4f} | {result['f1']:.4f} | "
            f"{result['false_positives_per_image']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fine transformer thresholds on validation."
    )
    parser.add_argument(
        "train_dir",
        type=Path,
        help="Training-image directory used to reconstruct the validation split",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=DEFAULT_TRANSFORMER_CHECKPOINT,
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast")
    parser.add_argument("--max-images", type=int, help="Optional subset for a smoke test")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.train_dir.is_dir():
        raise NotADirectoryError(f"Training directory does not exist: {args.train_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be at least 1 when supplied.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1.")


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    results, validation_size = evaluate(args)
    markdown = render_markdown(results, args, validation_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print()
    print(markdown, end="")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
