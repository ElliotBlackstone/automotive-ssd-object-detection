"""Benchmark the end-to-end custom VisionTransformer training step.

The script calls the project's existing myTrainStep() on an in-memory synthetic
object-detection dataset. It benchmarks CPU FP32 and, when available, CUDA FP32
and CUDA AMP/FP16.

Expected project files in the same directory:
    myViT.py
    myTrainStep.py
    HungarianMatch.py
    HungarianMatchBatched.py
    plus modules imported by myViT.py

Examples:
    python benchmark_vit_training.py
    python benchmark_vit_training.py --batch-size 8 --num-batches 10 --repeats 5
    python benchmark_vit_training.py --no-amp
"""

# to use:
# & "c:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" benchmark_vit_training.py --batch-size 4 --num-batches 10 --warmup 2 --repeats 5 --embed-dim 256 --num-layers 16 --num-heads 4 --dim-feedforward 1024 --num-queries 50 --num-workers 4 --patch-h 10 --patch-w 10

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from myTrainStep import myTrainStep
from myViT import VisionTransformer


CLASS_TO_IDX = {
    "biker": 0,
    "car": 1,
    "pedestrian": 2,
    "trafficLight": 3,
    "truck": 4,
}


class SyntheticDetectionDataset(Dataset):
    """Pre-generated synthetic detection data; generation is not timed."""

    def __init__(
        self,
        num_samples: int,
        img_size: int,
        num_classes: int,
        min_objects: int,
        max_objects: int,
        seed: int,
    ) -> None:
        if min_objects < 0:
            raise ValueError("min_objects must be >= 0")
        if max_objects < min_objects:
            raise ValueError("max_objects must be >= min_objects")

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        self.images: List[torch.Tensor] = []
        self.targets: List[Dict[str, torch.Tensor]] = []

        for _ in range(num_samples):
            image = torch.randn(3, img_size, img_size, generator=g)

            if min_objects == max_objects:
                m = min_objects
            else:
                m = int(torch.randint(min_objects, max_objects + 1, (1,), generator=g).item())

            if m == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.long)
            else:
                xy1 = torch.rand((m, 2), generator=g) * (0.75 * img_size)
                min_side = max(2.0, 0.02 * img_size)
                max_side = max(min_side + 1.0, 0.25 * img_size)
                wh = torch.rand((m, 2), generator=g) * (max_side - min_side) + min_side
                xy2 = torch.minimum(xy1 + wh, torch.full_like(xy1, float(img_size)))
                boxes = torch.cat((xy1, xy2), dim=1).float()
                labels = torch.randint(0, num_classes, (m,), generator=g, dtype=torch.long)

            self.images.append(image)
            self.targets.append({"boxes": boxes, "labels": labels})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        target = self.targets[index]
        return self.images[index], {
            "boxes": target["boxes"].clone(),
            "labels": target["labels"].clone(),
        }


def collate_detection(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


@dataclass
class ModelConfig:
    img_size: int
    patch_h: int
    patch_w: int
    embed_dim: int
    num_layers: int
    num_heads: int
    dim_feedforward: int
    dropout: float
    num_queries: int


@dataclass
class BenchmarkResult:
    device: str
    precision: str
    transfer_ms: float
    forward_ms: float
    matching_ms: float
    loss_ms: float
    backward_optimizer_ms: float
    total_batch_ms: float
    other_ms: float
    images_per_second: float
    peak_memory_mb: float | None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(cfg: ModelConfig) -> VisionTransformer:
    return VisionTransformer(
        class_to_idx_dict=CLASS_TO_IDX,
        img_size=cfg.img_size,
        patch_H=cfg.patch_h,
        patch_W=cfg.patch_w,
        in_channels=3,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        num_queries=cfg.num_queries,
    )


def make_loader(dataset, batch_size: int, device: torch.device, num_workers: int):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_detection,
        drop_last=True,
        prefetch_factor=4,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def make_optimizer(model, lr: float, fused: bool, device: torch.device):
    kwargs = {"params": model.parameters(), "lr": lr, "weight_decay": 0.0}
    if fused:
        if device.type != "cuda":
            raise ValueError("fused AdamW is only used on CUDA")
        kwargs["fused"] = True
    return torch.optim.AdamW(**kwargs)


def make_scaler(device: torch.device, use_amp: bool):
    if device.type != "cuda" or not use_amp:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def run_one_training_step(model, loader, optimizer, device, scaler, args):
    """Run myTrainStep() once and externally time the entire call."""
    synchronize(device)
    t0 = time.perf_counter()

    result = myTrainStep(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lambda_CE=args.lambda_ce,
        lambda_L1=args.lambda_l1,
        lambda_GIoU=args.lambda_giou,
        lambda_CE_HM=args.lambda_ce_hm,
        lambda_L1_HM=args.lambda_l1_hm,
        lambda_GIoU_HM=args.lambda_giou_hm,
        device=str(device),
        timing=True,
        scheduler=None,
        scaler=scaler,
        compute_mAP=False,
        bg_weight=args.bg_weight,
    )

    synchronize(device)
    elapsed = time.perf_counter() - t0
    return result["timing"], elapsed


def med(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def benchmark_configuration(device, use_amp, dataset, model_cfg, args):
    precision = "AMP/FP16" if use_amp else "FP32"
    print(f"\nBenchmarking {device} / {precision}")

    # Same initialization seed for each configuration.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = make_model(model_cfg).to(device)
    optimizer = make_optimizer(
        model,
        args.lr,
        fused=(args.fused_adamw and device.type == "cuda"),
        device=device,
    )
    scaler = make_scaler(device, use_amp)
    loader = make_loader(dataset, args.batch_size, device, args.num_workers)

    if len(loader) == 0:
        raise RuntimeError("DataLoader has zero batches.")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup is not included in results.
    for _ in range(args.warmup):
        run_one_training_step(model, loader, optimizer, device, scaler, args)

    keys = ["to device", "model forward", "matching", "compute loss", "backward pass"]
    samples = {k: [] for k in keys}
    total_batch_samples = []

    for r in range(args.repeats):
        timing, total_s = run_one_training_step(model, loader, optimizer, device, scaler, args)
        for k in keys:
            samples[k].append(float(timing[k]))

        batch_s = total_s / len(loader)
        total_batch_samples.append(batch_s)
        print(f"  repeat {r + 1:>2}/{args.repeats}: {1000.0 * batch_s:8.3f} ms/batch")

    transfer = med(samples["to device"])
    forward = med(samples["model forward"])
    matching = med(samples["matching"])
    loss = med(samples["compute loss"])
    backward_opt = med(samples["backward pass"])
    total_batch = med(total_batch_samples)

    measured = transfer + forward + matching + loss + backward_opt
    other = max(0.0, total_batch - measured)

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    result = BenchmarkResult(
        device=str(device),
        precision=precision,
        transfer_ms=1000 * transfer,
        forward_ms=1000 * forward,
        matching_ms=1000 * matching,
        loss_ms=1000 * loss,
        backward_optimizer_ms=1000 * backward_opt,
        total_batch_ms=1000 * total_batch,
        other_ms=1000 * other,
        images_per_second=args.batch_size / total_batch,
        peak_memory_mb=peak_mb,
    )

    del model, optimizer, loader, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def print_results(results):
    headers = [
        "Device", "Precision", "Transfer", "Forward", "Match", "Loss",
        "Backward+Opt", "Total/batch", "Other", "Images/s", "Peak MB",
    ]
    rows = []
    for x in results:
        rows.append([
            x.device,
            x.precision,
            f"{x.transfer_ms:.3f}",
            f"{x.forward_ms:.3f}",
            f"{x.matching_ms:.3f}",
            f"{x.loss_ms:.3f}",
            f"{x.backward_optimizer_ms:.3f}",
            f"{x.total_batch_ms:.3f}",
            f"{x.other_ms:.3f}",
            f"{x.images_per_second:.2f}",
            "-" if x.peak_memory_mb is None else f"{x.peak_memory_mb:.1f}",
        ])

    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]

    def fmt(row):
        return " | ".join(row[j].ljust(widths[j]) for j in range(len(row)))

    print("\n" + fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    print("\nAll component columns except Images/s and Peak MB are milliseconds per batch.")
    print("'Backward+Opt' is the current myTrainStep.py timer: backward + optimizer.step().")
    print("compute_mAP=False so mAP inference/evaluation is excluded.")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark VisionTransformer training.")

    # Workload.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=724)
    p.add_argument("--min-objects", type=int, default=1)
    p.add_argument("--max-objects", type=int, default=20)

    # Modes.
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--fused-adamw", action="store_true")
    p.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="highest",
    )

    # Current model defaults.
    p.add_argument("--img-size", type=int, default=300)
    p.add_argument("--patch-h", type=int, default=20)
    p.add_argument("--patch-w", type=int, default=15)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num-queries", type=int, default=100)

    # Training settings.
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--lambda-l1", type=float, default=5.0)
    p.add_argument("--lambda-giou", type=float, default=1.0)
    p.add_argument("--lambda-ce-hm", type=float, default=1.0)
    p.add_argument("--lambda-l1-hm", type=float, default=5.0)
    p.add_argument("--lambda-giou-hm", type=float, default=2.0)
    p.add_argument("--bg-weight", type=float, default=0.05)

    return p.parse_args()


def validate_args(args):
    for name in [
        "batch_size", "num_batches", "repeats", "img_size", "patch_h",
        "patch_w", "embed_dim", "num_layers", "num_heads",
        "dim_feedforward", "num_queries", "max_objects",
    ]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.min_objects < 0 or args.max_objects < args.min_objects:
        raise ValueError("Require 0 <= min_objects <= max_objects")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")


def main():
    args = parse_args()
    validate_args(args)
    torch.set_float32_matmul_precision(args.matmul_precision)

    cfg = ModelConfig(
        img_size=args.img_size,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_queries=args.num_queries,
    )

    dataset = SyntheticDetectionDataset(
        num_samples=args.batch_size * args.num_batches,
        img_size=args.img_size,
        num_classes=len(CLASS_TO_IDX),
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        seed=args.seed,
    )

    print("PyTorch:", torch.__version__)
    try:
        import torchvision
        print("TorchVision:", torchvision.__version__)
    except Exception:
        pass
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print(
        f"Model: image={args.img_size}, patch=({args.patch_h},{args.patch_w}), "
        f"D={args.embed_dim}, layers={args.num_layers}, heads={args.num_heads}, "
        f"FF={args.dim_feedforward}, Q={args.num_queries}"
    )
    print(
        f"Benchmark: B={args.batch_size}, batches/run={args.num_batches}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )

    configs: List[Tuple[torch.device, bool]] = [(torch.device("cpu"), False)]
    if torch.cuda.is_available() and not args.cpu_only:
        configs.append((torch.device("cuda"), False))
        if not args.no_amp:
            configs.append((torch.device("cuda"), True))

    results = [
        benchmark_configuration(device, amp, dataset, cfg, args)
        for device, amp in configs
    ]
    print_results(results)


if __name__ == "__main__":
    main()
