import argparse
import statistics
import time
from pathlib import Path
import importlib.util

import torch

from HungarianMatchBatched import hungarian_match_batched
from HungarianMatch import hungarian_match

# to use:
# & "c:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" benchmark_hungarian_match.py --batch-size 16 --num-queries 100 --num-classes 5 --max-targets 30 --warmup 10 --repeats 100 



def make_random_batch(
    device: torch.device,
    batch_size: int,
    num_queries: int,
    num_classes: int,
    min_targets: int,
    max_targets: int,
    seed: int,
):
    """Create valid synthetic detector outputs and variable-length targets."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    pred_class_logits = torch.randn(
        batch_size, num_queries, num_classes + 1, generator=g
    )

    # Valid normalized predicted boxes in cxcywh format.
    centers = 0.1 + 0.8 * torch.rand(batch_size, num_queries, 2, generator=g)
    sizes = 0.02 + 0.30 * torch.rand(batch_size, num_queries, 2, generator=g)
    pred_bbox = torch.cat((centers, sizes), dim=-1)

    gt_classes = []
    gt_bbox = []
    target_counts = torch.randint(
        low=min_targets,
        high=max_targets + 1,
        size=(batch_size,),
        generator=g,
    ).tolist()

    for M_i in target_counts:
        labels_i = torch.randint(0, num_classes, (M_i,), generator=g)

        # Generate valid normalized xyxy boxes.
        xy1 = 0.75 * torch.rand(M_i, 2, generator=g)
        box_wh = 0.02 + 0.23 * torch.rand(M_i, 2, generator=g)
        xy2 = torch.minimum(xy1 + box_wh, torch.ones_like(xy1))
        boxes_i = torch.cat((xy1, xy2), dim=-1)

        gt_classes.append(labels_i.to(device))
        gt_bbox.append(boxes_i.to(device))

    return (
        pred_class_logits.to(device),
        pred_bbox.to(device),
        gt_classes,
        gt_bbox,
    )


def old_match_batch(
    pred_class_logits,
    pred_bbox,
    gt_classes,
    gt_bbox,
    lambda_CE,
    lambda_L1,
    lambda_GIoU,
):
    """Current training-loop behavior: cost construction + solve per image."""
    matches = []
    for i in range(pred_bbox.shape[0]):
        matches.append(
            hungarian_match(
                pred_class_logits=pred_class_logits[i],
                pred_bbox=pred_bbox[i],
                gt_classes=gt_classes[i],
                gt_bbox=gt_bbox[i],
                lambda_CE=lambda_CE,
                lambda_L1=lambda_L1,
                lambda_GIoU=lambda_GIoU,
            )
        )
    return matches


def new_match_batch(
    pred_class_logits,
    pred_bbox,
    gt_classes,
    gt_bbox,
    lambda_CE,
    lambda_L1,
    lambda_GIoU,
):
    return hungarian_match_batched(
        pred_class_logits=pred_class_logits,
        pred_bbox=pred_bbox,
        gt_classes=gt_classes,
        gt_bbox=gt_bbox,
        lambda_CE=lambda_CE,
        lambda_L1=lambda_L1,
        lambda_GIoU=lambda_GIoU,
    )


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def matches_are_identical(old_matches, new_matches):
    if len(old_matches) != len(new_matches):
        return False

    for (old_pred, old_gt), (new_pred, new_gt) in zip(old_matches, new_matches):
        if not torch.equal(old_pred.cpu(), new_pred.cpu()):
            return False
        if not torch.equal(old_gt.cpu(), new_gt.cpu()):
            return False
    return True


def benchmark_function(fn, args, device, warmup, repeats):
    # Warm-up is particularly important on CUDA.
    for _ in range(warmup):
        fn(*args)
    synchronize(device)

    times_ms = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn(*args)
        synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
    }


def run_device(device, cfg):
    data = make_random_batch(
        device=device,
        batch_size=cfg.batch_size,
        num_queries=cfg.num_queries,
        num_classes=cfg.num_classes,
        min_targets=cfg.min_targets,
        max_targets=cfg.max_targets,
        seed=cfg.seed,
    )

    common_args = (
        *data,
        cfg.lambda_CE,
        cfg.lambda_L1,
        cfg.lambda_GIoU,
    )

    # Correctness check before timing.
    old_matches = old_match_batch(*common_args)
    new_matches = new_match_batch(*common_args)
    identical = matches_are_identical(old_matches, new_matches)
    if not identical:
        raise AssertionError(
            f"Old and batched matchers produced different assignments on {device}."
        )

    old_timing = benchmark_function(
        old_match_batch, common_args, device, cfg.warmup, cfg.repeats
    )
    new_timing = benchmark_function(
        new_match_batch, common_args, device, cfg.warmup, cfg.repeats
    )

    speedup = old_timing["median_ms"] / new_timing["median_ms"]
    return {
        "device": str(device),
        "identical": identical,
        "old_median_ms": old_timing["median_ms"],
        "batched_median_ms": new_timing["median_ms"],
        "speedup": speedup,
        "old_mean_ms": old_timing["mean_ms"],
        "batched_mean_ms": new_timing["mean_ms"],
    }


def print_results(rows):
    headers = [
        "Device",
        "Same result",
        "Old median (ms)",
        "Batched median (ms)",
        "Speedup",
    ]
    formatted = []
    for row in rows:
        formatted.append(
            [
                row["device"],
                str(row["identical"]),
                f'{row["old_median_ms"]:.3f}',
                f'{row["batched_median_ms"]:.3f}',
                f'{row["speedup"]:.2f}x',
            ]
        )

    widths = [
        max(len(headers[j]), max(len(r[j]) for r in formatted))
        for j in range(len(headers))
    ]

    def row_string(values):
        return " | ".join(v.ljust(widths[j]) for j, v in enumerate(values))

    print(row_string(headers))
    print("-+-".join("-" * w for w in widths))
    for row in formatted:
        print(row_string(row))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark original per-image vs batched Hungarian cost construction."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--lambda-CE", dest="lambda_CE", type=float, default=1.0)
    parser.add_argument("--lambda-L1", dest="lambda_L1", type=float, default=5.0)
    parser.add_argument("--lambda-GIoU", dest="lambda_GIoU", type=float, default=2.0)
    return parser.parse_args()


def main():
    cfg = parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = [run_device(device, cfg) for device in devices]
    print_results(rows)

    if not torch.cuda.is_available():
        print("\nCUDA is not available; GPU benchmark was skipped.")


if __name__ == "__main__":
    main()
