r"""Benchmark end-to-end FP32 image prediction latency for SSD and transformer.

The timed path is intentionally the complete single-image prediction path:
read/decode the image from disk, convert BGR to RGB, resize and normalize it,
transfer it to the selected device, run model inference and postprocessing, and
copy the predictions to CPU memory. Model construction/checkpoint loading and
warm-up are excluded.

Example (PowerShell):
    & "C:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" `
        benchmarking\benchmark_image_prediction.py `
        "C:\Udacity_car_data\data\test\example.jpg"

The Markdown table is printed and the complete report is written to
``benchmarking/image_prediction_benchmark_results.md`` by default.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_predict import (  # noqa: E402
    DEFAULT_SSD_CHECKPOINT,
    DEFAULT_TRANSFORMER_CHECKPOINT,
    _load_models,
    _preprocess_frames,
)


DEFAULT_OUTPUT = Path(__file__).with_name("image_prediction_benchmark_results.md")


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    device: str
    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float

    @property
    def median_fps(self) -> float:
        return 1000.0 / self.median_ms


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without an extra dependency."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _materialize_predictions(predictions: Sequence[Dict[str, torch.Tensor]]) -> None:
    """Copy every output to host memory so asynchronous GPU work is measured."""
    for prediction in predictions:
        for name in ("labels", "scores", "boxes"):
            prediction[name].detach().to("cpu").numpy()


def _make_prediction_call(
    model_name: str,
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Callable[[], None]:
    if model_name == "SSD (FP32)":
        predict = lambda images: model.predict(
            images=images,
            score_thresh=args.ssd_score_threshold,
            nms_thresh=args.ssd_nms_threshold,
            iou_variant="DIoU",
            max_per_img=args.ssd_max_detections,
            class_agnostic=False,
        )
    else:
        predict = lambda images: model.predict(
            images=images,
            conf_thresh=args.transformer_confidence_threshold,
        )

    def predict_one_image() -> None:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"OpenCV could not decode image: {image_path}")
        images = _preprocess_frames([frame], device)
        with torch.inference_mode():
            predictions = predict(images)
        _materialize_predictions(predictions)

    return predict_one_image


def _benchmark(
    model_name: str,
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> BenchmarkResult:
    predict_one_image = _make_prediction_call(
        model_name, model, image_path, device, args
    )

    for _ in range(args.warmup):
        predict_one_image()
    _synchronize(device)

    times_ms: List[float] = []
    for _ in range(args.iterations):
        _synchronize(device)
        started = time.perf_counter_ns()
        predict_one_image()
        _synchronize(device)
        times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    return BenchmarkResult(
        model=model_name,
        device=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
        samples=len(times_ms),
        mean_ms=statistics.fmean(times_ms),
        median_ms=statistics.median(times_ms),
        p95_ms=_percentile(times_ms, 0.95),
        minimum_ms=min(times_ms),
        maximum_ms=max(times_ms),
    )


def _markdown_table(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        "| Model | Device | N | Mean (ms/image) | p50 (ms/image) | p95 (ms/image) | Min (ms) | Max (ms) | FPS at p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.model} | {result.device} | {result.samples} "
            f"| {result.mean_ms:.3f} | {result.median_ms:.3f} "
            f"| {result.p95_ms:.3f} | {result.minimum_ms:.3f} "
            f"| {result.maximum_ms:.3f} | {result.median_fps:.2f} |"
        )
    return "\n".join(lines)


def _cpu_name() -> str:
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "Unknown"


def _gen_nms_source() -> str:
    spec = importlib.util.find_spec("gen_nms")
    return str(spec.origin) if spec is not None else "Unavailable (built-in fallback)"


def _build_report(
    results: Sequence[BenchmarkResult], args: argparse.Namespace
) -> str:
    table = _markdown_table(results)
    image = args.image.resolve()
    ssd_checkpoint = args.ssd_checkpoint.resolve()
    transformer_checkpoint = args.transformer_checkpoint.resolve()
    device_names = ", ".join(args.devices)
    return f"""# End-to-end image prediction benchmark

{table}

## Configuration

- Timestamp: {datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
- Image: `{image}`
- Input/batch size: 300 x 300, batch size 1
- Precision: FP32 (automatic mixed precision disabled)
- Requested devices: {device_names}
- Warm-up iterations: {args.warmup} per model/device
- Measured iterations: {args.iterations} per model/device
- Python executable: `{sys.executable}`
- PyTorch: {torch.__version__}
- CUDA build used by PyTorch: {getattr(torch.version, "cuda", None)}
- gen_nms: `{_gen_nms_source()}`
- CPU: {_cpu_name()}
- CPU threads used by PyTorch: {torch.get_num_threads()}
- SSD checkpoint: `{ssd_checkpoint}`
- Transformer checkpoint: `{transformer_checkpoint}`
- SSD thresholds: score={args.ssd_score_threshold}, DIoU-NMS={args.ssd_nms_threshold}, max detections={args.ssd_max_detections}
- Transformer confidence threshold: {args.transformer_confidence_threshold}

## Measurement boundary

Each sample starts before `cv2.imread` and ends after the prediction tensors have
been copied to CPU memory. It therefore includes disk image read/decode,
preprocessing, host-to-device transfer where applicable, FP32 model inference,
and model postprocessing. Checkpoint loading and warm-up are excluded. Repeated
reads normally benefit from the operating-system file cache, so these numbers do
not represent cold-storage latency.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SSD and transformer end-to-end image prediction."
    )
    parser.add_argument("image", type=Path, help="Image used for every timed sample")
    parser.add_argument("--ssd-checkpoint", type=Path, default=DEFAULT_SSD_CHECKPOINT)
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=DEFAULT_TRANSFORMER_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "cuda"),
        default=("cpu", "cuda"),
        help="Devices to test (default: cpu cuda)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--ssd-score-threshold", type=float, default=0.2)
    parser.add_argument("--ssd-nms-threshold", type=float, default=0.45)
    parser.add_argument("--ssd-max-detections", type=int, default=200)
    parser.add_argument("--transformer-confidence-threshold", type=float, default=0.4)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not args.ssd_checkpoint.is_file():
        parser.error(f"SSD checkpoint does not exist: {args.ssd_checkpoint}")
    if not args.transformer_checkpoint.is_file():
        parser.error(
            f"transformer checkpoint does not exist: {args.transformer_checkpoint}"
        )
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not 0.0 <= args.ssd_score_threshold < 1.0:
        parser.error("--ssd-score-threshold must be in [0, 1)")
    if not 0.0 < args.ssd_nms_threshold < 1.0:
        parser.error("--ssd-nms-threshold must be in (0, 1)")
    if args.ssd_max_detections < 1:
        parser.error("--ssd-max-detections must be at least 1")
    if not 0.0 <= args.transformer_confidence_threshold < 1.0:
        parser.error("--transformer-confidence-threshold must be in [0, 1)")
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA was requested, but torch.cuda.is_available() is False")
    return args


def main() -> None:
    args = _parse_args()
    results: List[BenchmarkResult] = []

    for device_name in args.devices:
        device = torch.device(device_name)
        print(f"Loading both FP32 models on {device}...", flush=True)
        ssd, transformer, used_nms_fallback = _load_models(
            args.ssd_checkpoint, args.transformer_checkpoint, device
        )
        if used_nms_fallback:
            print(
                "gen_nms is unavailable; SSD uses the built-in PyTorch DIoU-NMS fallback.",
                flush=True,
            )

        for model_name, model in (
            ("SSD (FP32)", ssd),
            ("Transformer (FP32)", transformer),
        ):
            print(f"Benchmarking {model_name} on {device}...", flush=True)
            results.append(_benchmark(model_name, model, args.image, device, args))

        del ssd, transformer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = _markdown_table(results)
    report = _build_report(results, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n{table}\n")
    print(f"Wrote report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
