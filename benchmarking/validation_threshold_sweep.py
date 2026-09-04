r"""Evaluate SSD-v2 and transformer confidence thresholds on validation data.

The validation set is reconstructed exactly as in both models' training code:
a 25% stratified group split of the training directory with random seed 724.

Example (PowerShell):
    & "C:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" `
        validation_threshold_sweep.py "C:\Udacity_car_data\data\train" `
        --device cuda

The script prints a Markdown table and writes it to
``validation_threshold_results.md`` by default.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

from compare_models import (
    EvaluationDataset,
    _metric_target,
    _prediction_at_native_size,
    collate_detection,
)
from v2.CarImageClass import ImageClass, make_train_test_split
from video_predict import (
    CLASS_TO_IDX,
    DEFAULT_SSD_CHECKPOINT,
    DEFAULT_TRANSFORMER_CHECKPOINT,
    _load_models,
    _resolve_device,
)


THRESHOLDS = tuple(index / 100.0 for index in range(5, 100, 5))
MATCH_IOU_THRESHOLD = 0.50
VALIDATION_FRACTION = 0.25
VALIDATION_SEED = 724
DEFAULT_OUTPUT = Path("validation_threshold_results.md")


@dataclass
class Counts:
    true_positives: int = 0
    false_positives: int = 0


class ThresholdSweep:
    """Accumulate score-threshold metrics from one greedy matching pass."""

    def __init__(self, thresholds: Sequence[float], *, inclusive: bool) -> None:
        self.thresholds = tuple(thresholds)
        self.inclusive = inclusive
        self.counts = {threshold: Counts() for threshold in self.thresholds}
        self.ground_truth_count = 0
        self.image_count = 0

    def update(
        self,
        predictions: Sequence[Dict[str, torch.Tensor]],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> None:
        for prediction, target in zip(predictions, targets):
            self.image_count += 1
            self.ground_truth_count += int(target["labels"].numel())
            scores, true_positive = self._match_image(prediction, target)

            for threshold in self.thresholds:
                selected = scores >= threshold if self.inclusive else scores > threshold
                selected_count = int(selected.sum())
                true_positive_count = int(true_positive[selected].sum())
                self.counts[threshold].true_positives += true_positive_count
                self.counts[threshold].false_positives += (
                    selected_count - true_positive_count
                )

    @staticmethod
    def _match_image(
        prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Label each score-ordered detection TP or FP at IoU 0.50."""
        image_scores: List[torch.Tensor] = []
        image_true_positive: List[torch.Tensor] = []

        for class_id in range(len(CLASS_TO_IDX)):
            pred_indices = torch.where(prediction["labels"] == class_id)[0]
            target_indices = torch.where(target["labels"] == class_id)[0]
            pred_indices = pred_indices[
                prediction["scores"][pred_indices].argsort(descending=True)
            ]
            class_scores = prediction["scores"][pred_indices]
            class_true_positive = torch.zeros(pred_indices.numel(), dtype=torch.bool)

            if pred_indices.numel() and target_indices.numel():
                overlaps = box_iou(
                    prediction["boxes"][pred_indices],
                    target["boxes"][target_indices],
                )
                matched_targets = torch.zeros(target_indices.numel(), dtype=torch.bool)
                for pred_row in range(pred_indices.numel()):
                    available = overlaps[pred_row].clone()
                    available[matched_targets] = -1.0
                    best_iou, best_target = available.max(dim=0)
                    if float(best_iou) >= MATCH_IOU_THRESHOLD:
                        class_true_positive[pred_row] = True
                        matched_targets[best_target] = True

            image_scores.append(class_scores)
            image_true_positive.append(class_true_positive)

        if not image_scores:
            return torch.empty(0), torch.empty(0, dtype=torch.bool)
        return torch.cat(image_scores), torch.cat(image_true_positive)

    def compute(self) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}
        for threshold, counts in self.counts.items():
            tp = counts.true_positives
            fp = counts.false_positives
            precision = tp / (tp + fp) if tp + fp else math.nan
            recall = tp / self.ground_truth_count if self.ground_truth_count else math.nan
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else math.nan
            )
            results[threshold] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives_per_image": fp / self.image_count,
            }
        return results


def _build_validation_dataset(train_dir: Path) -> EvaluationDataset:
    full_training_set = ImageClass(
        targ_dir=train_dir,
        transform=None,
        include_area=False,
    )
    _, validation_source = make_train_test_split(
        full_set=full_training_set,
        test_size=VALIDATION_FRACTION,
        rand_state=VALIDATION_SEED,
        transform_train=None,
        transform_test=None,
        include_area=False,
    )
    return EvaluationDataset(source=validation_source)


def evaluate(
    args: argparse.Namespace,
) -> tuple[Dict[float, Dict[str, float]], Dict[float, Dict[str, float]], int]:
    device = _resolve_device(args.device)
    base_dataset = _build_validation_dataset(args.train_dir)
    if base_dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            "Validation classes do not match the models. "
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
    print(f"Loading models on {device}...", flush=True)
    ssd, transformer, used_nms_fallback = _load_models(
        args.ssd_checkpoint,
        args.transformer_checkpoint,
        device,
    )
    if used_nms_fallback:
        print("gen_nms is unavailable; using the PyTorch DIoU-NMS fallback.")

    # SSD predict uses score > threshold; transformer predict uses score >= threshold.
    ssd_sweep = ThresholdSweep(THRESHOLDS, inclusive=False)
    transformer_sweep = ThresholdSweep(THRESHOLDS, inclusive=True)
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
                ssd_locations, ssd_logits = ssd(images)
                transformer_logits, transformer_boxes = transformer(images)
                ssd_predictions = ssd.predict(
                    images=images,
                    score_thresh=THRESHOLDS[0],
                    nms_thresh=args.ssd_nms_threshold,
                    iou_variant="DIoU",
                    max_per_img=args.ssd_max_detections,
                    class_agnostic=False,
                    pre_loc_all=ssd_locations,
                    pre_conf_all=ssd_logits,
                )
                transformer_predictions = transformer.predict(
                    images=images,
                    pre_class_logits=transformer_logits[-1],
                    pre_bboxes=transformer_boxes[-1],
                    conf_thresh=THRESHOLDS[0],
                )

            metric_targets = [_metric_target(target) for target in targets]
            ssd_predictions_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(ssd_predictions, targets)
            ]
            transformer_predictions_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(transformer_predictions, targets)
            ]
            ssd_sweep.update(ssd_predictions_cpu, metric_targets)
            transformer_sweep.update(transformer_predictions_cpu, metric_targets)

            image_count += len(targets)
            if image_count % args.progress_every < len(targets):
                print(f"Evaluated {image_count}/{len(dataset)} images", flush=True)

    return ssd_sweep.compute(), transformer_sweep.compute(), image_count


def _format(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def render_markdown(
    ssd_results: Dict[float, Dict[str, float]],
    transformer_results: Dict[float, Dict[str, float]],
    args: argparse.Namespace,
    validation_size: int,
) -> str:
    lines = [
        "# Validation-set confidence-threshold sweep",
        "",
        f"- Training directory: `{args.train_dir.resolve()}`",
        f"- Validation split: {VALIDATION_FRACTION:.0%}, seed {VALIDATION_SEED} "
        f"({validation_size} images evaluated)",
        f"- Matching: class-aware, one-to-one, IoU >= {MATCH_IOU_THRESHOLD:.2f}",
        f"- SSD NMS: DIoU, threshold {args.ssd_nms_threshold:.2f}, "
        f"maximum {args.ssd_max_detections} detections/image",
        "",
        "| Confidence threshold | SSD precision | SSD recall | SSD F1 | "
        "SSD FP/image | Transformer precision | Transformer recall | "
        "Transformer F1 | Transformer FP/image |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        ssd = ssd_results[threshold]
        transformer = transformer_results[threshold]
        lines.append(
            f"| {threshold:.2f} | {_format(ssd['precision'])} | "
            f"{_format(ssd['recall'])} | {_format(ssd['f1'])} | "
            f"{_format(ssd['false_positives_per_image'])} | "
            f"{_format(transformer['precision'])} | "
            f"{_format(transformer['recall'])} | {_format(transformer['f1'])} | "
            f"{_format(transformer['false_positives_per_image'])} |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep SSD-v2 and transformer confidence thresholds on validation."
    )
    parser.add_argument(
        "train_dir",
        type=Path,
        help="Training-image directory used to reconstruct the validation split",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ssd-checkpoint", type=Path, default=DEFAULT_SSD_CHECKPOINT)
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=DEFAULT_TRANSFORMER_CHECKPOINT,
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ssd-nms-threshold", type=float, default=0.45)
    parser.add_argument("--ssd-max-detections", type=int, default=200)
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
    if not 0.0 < args.ssd_nms_threshold < 1.0:
        raise ValueError("--ssd-nms-threshold must be in (0, 1).")
    if args.ssd_max_detections < 1:
        raise ValueError("--ssd-max-detections must be at least 1.")


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    ssd_results, transformer_results, validation_size = evaluate(args)
    markdown = render_markdown(
        ssd_results,
        transformer_results,
        args,
        validation_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print()
    print(markdown, end="")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
