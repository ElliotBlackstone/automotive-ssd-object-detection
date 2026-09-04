r"""Compare the SSD-v2 and transformer detectors on a labeled test set.

The dataset directory must contain the test ``.jpg`` files and exactly one
annotation CSV in the format consumed by :class:`v2.CarImageClass.ImageClass`.

Example (PowerShell):
    & "C:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" compare_models.py `
        "C:\Udacity_car_data\data\test" --device cuda

The output is a Markdown report (``model_comparison_results.md`` by default).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors
from torchvision.ops import box_iou
from torchvision.transforms import v2

from v2.CarImageClass import ImageClass
from video_predict import (
    CLASS_TO_IDX,
    DEFAULT_SSD_CHECKPOINT,
    DEFAULT_TRANSFORMER_CHECKPOINT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    _load_models,
    _resolve_device,
)


DEFAULT_OUTPUT = Path("model_comparison_results.md")
AP_IOU_THRESHOLDS = [0.50 + 0.05 * index for index in range(10)]
MATCH_IOU_THRESHOLD = 0.50


class EvaluationDataset(Dataset):
    """Return model-ready images while retaining ground truth at native size."""

    def __init__(
        self,
        test_dir: Path | None = None,
        *,
        source: ImageClass | None = None,
    ) -> None:
        if (test_dir is None) == (source is None):
            raise ValueError("Provide exactly one of test_dir or source.")
        self.source = (
            source
            if source is not None
            else ImageClass(targ_dir=test_dir, transform=None, include_area=False)
        )
        self.class_to_idx = self.source.class_to_idx
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((MODEL_HEIGHT, MODEL_WIDTH), antialias=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image, target = self.source[index]
        height, width = image.shape[-2:]
        boxes = target["boxes"].as_subclass(torch.Tensor).to(torch.float32)
        boxes = boxes.clone()
        labels = target["labels"].to(torch.int64).clone()
        area = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        )
        clean_target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(labels.shape, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }
        return self.transform(image), clean_target


def collate_detection(
    batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images)), list(targets)


def _metric_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in target.items() if key != "orig_size"}


def _prediction_at_native_size(
    prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Move a prediction to CPU, rescale it, and discard invalid boxes."""
    boxes = prediction["boxes"].detach().to(device="cpu", dtype=torch.float32)
    scores = prediction["scores"].detach().to(device="cpu", dtype=torch.float32)
    labels = prediction["labels"].detach().to(device="cpu", dtype=torch.int64)

    height, width = target["orig_size"].tolist()
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= float(width) / MODEL_WIDTH
    boxes[:, [1, 3]] *= float(height) / MODEL_HEIGHT
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)

    valid = (
        torch.isfinite(boxes).all(dim=1)
        & torch.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (labels >= 0)
        & (labels < len(CLASS_TO_IDX))
    )
    return {"boxes": boxes[valid], "scores": scores[valid], "labels": labels[valid]}


def _filter_prediction(
    prediction: Dict[str, torch.Tensor], score_threshold: float
) -> Dict[str, torch.Tensor]:
    """Apply an operating threshold to an already score-ordered prediction."""
    keep = prediction["scores"] > score_threshold
    return {key: value[keep] for key, value in prediction.items()}


@dataclass
class OperatingPoint:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    image_count: int = 0
    true_positive_iou_sum: float = 0.0

    def update(
        self,
        predictions: Sequence[Dict[str, torch.Tensor]],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> None:
        for prediction, target in zip(predictions, targets):
            self.image_count += 1
            self._update_image(prediction, target)

    def _update_image(
        self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> None:
        pred_labels = prediction["labels"]
        target_labels = target["labels"]
        image_tp = 0
        image_fp = 0

        for class_id in range(len(CLASS_TO_IDX)):
            pred_indices = torch.where(pred_labels == class_id)[0]
            target_indices = torch.where(target_labels == class_id)[0]
            pred_indices = pred_indices[
                prediction["scores"][pred_indices].argsort(descending=True)
            ]

            if pred_indices.numel() == 0:
                continue
            if target_indices.numel() == 0:
                image_fp += int(pred_indices.numel())
                continue

            overlaps = box_iou(
                prediction["boxes"][pred_indices], target["boxes"][target_indices]
            )
            matched_targets = torch.zeros(target_indices.numel(), dtype=torch.bool)
            for pred_row in range(pred_indices.numel()):
                available = overlaps[pred_row].clone()
                available[matched_targets] = -1.0
                best_iou, best_target = available.max(dim=0)
                if float(best_iou) >= MATCH_IOU_THRESHOLD:
                    matched_targets[best_target] = True
                    image_tp += 1
                    self.true_positive_iou_sum += float(best_iou)
                else:
                    image_fp += 1

        self.true_positives += image_tp
        self.false_positives += image_fp
        self.false_negatives += int(target_labels.numel()) - image_tp

    def compute(self) -> Dict[str, float]:
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        precision = tp / (tp + fp) if tp + fp else math.nan
        recall = tp / (tp + fn) if tp + fn else math.nan
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else math.nan
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives_per_image": fp / self.image_count,
            "mean_true_positive_iou": (
                self.true_positive_iou_sum / tp if tp else math.nan
            ),
        }


def _new_map_metric() -> MeanAveragePrecision:
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=AP_IOU_THRESHOLDS,
        class_metrics=True,
        max_detection_thresholds=[1, 10, 100],
    )


def _float(value: torch.Tensor | float) -> float:
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def _collect_results(
    ap: Dict[str, torch.Tensor], operating_point: OperatingPoint
) -> Dict[str, float]:
    results = {
        "mAP@0.50": _float(ap["map_50"]),
        "mAP@0.75": _float(ap["map_75"]),
        "mAP@[0.50:0.95]": _float(ap["map"]),
        "AP small": _float(ap["map_small"]),
        "AP medium": _float(ap["map_medium"]),
        "AP large": _float(ap["map_large"]),
    }
    per_class = {
        int(class_id): _float(value)
        for class_id, value in zip(ap["classes"], ap["map_per_class"])
    }
    for class_name, class_id in CLASS_TO_IDX.items():
        display_name = "traffic light" if class_name == "trafficLight" else class_name
        results[f"AP {display_name}"] = per_class.get(class_id, math.nan)
    results.update(operating_point.compute())
    return results


def evaluate(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float], int]:
    device = _resolve_device(args.device)
    dataset: Dataset = EvaluationDataset(args.test_dir)
    class_mapping = dataset.class_to_idx  # type: ignore[attr-defined]
    if class_mapping != CLASS_TO_IDX:
        raise ValueError(
            "Dataset classes do not match the models. "
            f"Expected {CLASS_TO_IDX}, found {class_mapping}."
        )
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

    print(f"Loading models on {device}...", flush=True)
    ssd, transformer, used_nms_fallback = _load_models(
        args.ssd_checkpoint, args.transformer_checkpoint, device
    )
    if used_nms_fallback:
        print("gen_nms is unavailable; using the PyTorch DIoU-NMS fallback.")

    ssd_ap = _new_map_metric()
    transformer_ap = _new_map_metric()
    ssd_point = OperatingPoint()
    transformer_point = OperatingPoint()
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
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                ssd_locations, ssd_logits = ssd(images)
                transformer_logits, transformer_boxes = transformer(images)

                ssd_ap_predictions = ssd.predict(
                    images=images,
                    score_thresh=args.ssd_ap_score_floor,
                    nms_thresh=args.ssd_nms_threshold,
                    iou_variant="DIoU",
                    max_per_img=args.ssd_max_detections,
                    class_agnostic=False,
                    pre_loc_all=ssd_locations,
                    pre_conf_all=ssd_logits,
                )
                # NMS is score ordered, so thresholding its low-floor output is
                # equivalent to a second predict call at the operating cutoff.
                ssd_threshold_predictions = [
                    _filter_prediction(prediction, args.ssd_score_threshold)
                    for prediction in ssd_ap_predictions
                ]
                transformer_ap_predictions = transformer.predict(
                    images=images,
                    pre_class_logits=transformer_logits[-1],
                    pre_bboxes=transformer_boxes[-1],
                    conf_thresh=None,
                )
                transformer_threshold_predictions = transformer.predict(
                    images=images,
                    pre_class_logits=transformer_logits[-1],
                    pre_bboxes=transformer_boxes[-1],
                    conf_thresh=args.transformer_confidence_threshold,
                )

            metric_targets = [_metric_target(target) for target in targets]
            ssd_ap_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(ssd_ap_predictions, targets)
            ]
            transformer_ap_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(transformer_ap_predictions, targets)
            ]
            ssd_threshold_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(ssd_threshold_predictions, targets)
            ]
            transformer_threshold_cpu = [
                _prediction_at_native_size(prediction, target)
                for prediction, target in zip(transformer_threshold_predictions, targets)
            ]

            ssd_ap.update(ssd_ap_cpu, metric_targets)
            transformer_ap.update(transformer_ap_cpu, metric_targets)
            ssd_point.update(ssd_threshold_cpu, metric_targets)
            transformer_point.update(transformer_threshold_cpu, metric_targets)

            image_count += len(targets)
            if image_count % args.progress_every < len(targets):
                print(f"Evaluated {image_count}/{len(dataset)} images", flush=True)

    print("Computing COCO metrics...", flush=True)
    return (
        _collect_results(ssd_ap.compute(), ssd_point),
        _collect_results(transformer_ap.compute(), transformer_point),
        image_count,
    )


def _format_metric(value: float) -> str:
    return "N/A" if not math.isfinite(value) or value < 0 else f"{value:.4f}"


def render_markdown(
    ssd_results: Dict[str, float],
    transformer_results: Dict[str, float],
    args: argparse.Namespace,
    image_count: int,
) -> str:
    rows = [
        "mAP@0.50",
        "mAP@0.75",
        "mAP@[0.50:0.95]",
        "AP small",
        "AP medium",
        "AP large",
        "AP biker",
        "AP car",
        "AP pedestrian",
        "AP traffic light",
        "AP truck",
        "precision",
        "recall",
        "f1",
        "false_positives_per_image",
        "mean_true_positive_iou",
    ]
    display_names = {
        "precision": "Precision @ chosen threshold",
        "recall": "Recall @ chosen threshold",
        "f1": "F1 @ chosen threshold",
        "false_positives_per_image": "False positives / image",
        "mean_true_positive_iou": "Mean IoU of true positives",
    }
    lines = [
        "# Test-set detector comparison",
        "",
        f"- Test set: `{args.test_dir.resolve()}` ({image_count} images)",
        f"- SSD threshold: {args.ssd_score_threshold:.3f}",
        f"- Transformer threshold: {args.transformer_confidence_threshold:.3f}",
        f"- Threshold matching: class-aware, one-to-one, IoU >= {MATCH_IOU_THRESHOLD:.2f}",
        f"- SSD AP score floor: {args.ssd_ap_score_floor:.3f}",
        "- AP protocol: COCO interpolation at IoU 0.50:0.05:0.95, maxDets=100; "
        "size categories use native-image object areas",
        "",
        "| Metric | SSD v2 | Transformer |",
        "|---|---:|---:|",
    ]
    for key in rows:
        lines.append(
            f"| {display_names.get(key, key)} | {_format_metric(ssd_results[key])} "
            f"| {_format_metric(transformer_results[key])} |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the SSD-v2 and transformer models on a test set."
    )
    parser.add_argument(
        "test_dir", type=Path, help="Directory containing test JPGs and annotation CSV"
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
    parser.add_argument("--ssd-score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--ssd-ap-score-floor",
        type=float,
        default=0.05,
        help="Low pre-NMS cutoff used only for SSD AP curves (default: 0.05)",
    )
    parser.add_argument("--ssd-nms-threshold", type=float, default=0.45)
    parser.add_argument("--ssd-max-detections", type=int, default=200)
    parser.add_argument("--transformer-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast")
    parser.add_argument("--max-images", type=int, help="Optional subset for a smoke test")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.test_dir.is_dir():
        raise NotADirectoryError(f"Test directory does not exist: {args.test_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be at least 1 when supplied.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1.")
    for name in (
        "ssd_score_threshold",
        "ssd_ap_score_floor",
        "transformer_confidence_threshold",
    ):
        if not 0.0 <= getattr(args, name) < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1).")
    if args.ssd_ap_score_floor > args.ssd_score_threshold:
        raise ValueError("--ssd-ap-score-floor cannot exceed --ssd-score-threshold.")
    if not 0.0 < args.ssd_nms_threshold < 1.0:
        raise ValueError("--ssd-nms-threshold must be in (0, 1).")
    if args.ssd_max_detections < 1:
        raise ValueError("--ssd-max-detections must be at least 1.")


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    ssd_results, transformer_results, image_count = evaluate(args)
    markdown = render_markdown(ssd_results, transformer_results, args, image_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print()
    print(markdown, end="")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
