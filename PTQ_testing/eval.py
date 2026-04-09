from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import v2.CarImageClass as CarImageClass
from v2.model_files.SSD_from_scratch import mySSD
from v1.SSD_trainer import collate_detection


def build_test_tfms() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((300, 300), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def build_model() -> torch.nn.Module:
    return mySSD(
        class_to_idx_dict={
            "biker": 0,
            "car": 1,
            "pedestrian": 2,
            "trafficLight": 3,
            "truck": 4,
        },
        in_channels=3,
        variances=(0.1, 0.2),
    )


def load_weights(model: torch.nn.Module, weights_path: str) -> None:
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    res = model.load_state_dict(state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        print("WARNING load_state_dict(strict=False):")
        print("  missing_keys   =", res.missing_keys)
        print("  unexpected_keys=", res.unexpected_keys)


def build_val_data_from_train_dir(
    train_path: str,
    *,
    rand_state: int = 724,
    test_size: float = 0.25,
):
    train_path = Path(train_path)
    test_tfms = build_test_tfms()

    # Train transform is irrelevant for evaluation; required by splitter signature.
    train_tfms_dummy = test_tfms

    full_set = CarImageClass.ImageClass(
        targ_dir=train_path,
        transform=train_tfms_dummy,
        file_pct=1,
        rand_seed=rand_state,
        include_area=False,
    )

    _train_data, val_data = CarImageClass.make_train_test_split(
        full_set=full_set,
        test_size=test_size,
        rand_state=rand_state,
        transform_train=train_tfms_dummy,
        transform_test=test_tfms,
        include_area=False,
    )
    return val_data



def make_ort_session(onnx_path: str, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def to_torch_det_format(preds: List[Dict[str, object]]) -> List[Dict[str, torch.Tensor]]:
    """
    torchmetrics expects list[dict] with keys: boxes, scores, labels.
    Your predict() already returns that, but with object typing.
    """
    out: List[Dict[str, torch.Tensor]] = []
    for p in preds:
        out.append({
            "boxes": p["boxes"],
            "scores": p["scores"],
            "labels": p["labels"],
        })
    return out


def targets_to_torch_det_format(targets: List[Dict]) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        boxes = t["boxes"]
        # tv_tensors.BoundingBoxes -> Tensor view for torchmetrics
        if hasattr(boxes, "as_subclass"):
            boxes = boxes.as_subclass(torch.Tensor)
        out.append({
            "boxes": boxes.to(torch.float32),
            "labels": t["labels"].to(torch.int64),
        })
    return out


def eval_backend(
    *,
    backend: str,
    model: torch.nn.Module,
    loader: DataLoader,
    onnx_path: Optional[str],
    score_thresh: float,
    nms_thresh: float,
    max_per_img: int,
    class_agnostic: bool,
    threads: int,
) -> Dict[str, object]:
    """
    Returns:
      {
        "backend": str,
        "map_50": float,
        "ap_50_per_class": list[float] length=num_classes-1 (foreground classes),
      }
    """
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as e:
        raise RuntimeError(
            "torchmetrics is required. Install with: pip install torchmetrics"
        ) from e

    # IoU=0.5 only
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)

    ort_sess = None
    input_name = None
    if backend != "torch":
        if not onnx_path:
            raise ValueError(f"backend={backend} requires --onnx_fp32/--onnx_int8")
        ort_sess = make_ort_session(onnx_path, threads=threads)
        input_name = ort_sess.get_inputs()[0].name

    model.eval()

    with torch.inference_mode():
        for images, targets in loader:
            # targets/preds for torchmetrics must be lists of dicts
            gt = targets_to_torch_det_format(targets)

            if backend == "torch":
                preds = model.predict(
                    images=images,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                    iou_variant="DIoU",
                    max_per_img=max_per_img,
                    class_agnostic=class_agnostic,
                )
            else:
                # ORT forward -> numpy loc/conf
                assert ort_sess is not None and input_name is not None
                x_np = images.numpy().astype(np.float32)
                loc_np, conf_np = ort_sess.run(["loc", "conf"], {input_name: x_np})
                # convert to torch for your existing postprocess
                loc_t = torch.from_numpy(loc_np)
                conf_t = torch.from_numpy(conf_np)
                preds = model.predict(
                    images=images,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                    iou_variant="DIoU",
                    max_per_img=max_per_img,
                    class_agnostic=class_agnostic,
                    pre_loc_all=loc_t,
                    pre_conf_all=conf_t,
                )

            pr = to_torch_det_format(preds)
            metric.update(pr, gt)

    res = metric.compute()

    # torchmetrics returns tensors; convert to python
    map_50 = float(res["map_50"].cpu().item()) if torch.is_tensor(res["map_50"]) else float(res["map_50"])
    ap_pc = res.get("map_per_class", None)
    if ap_pc is None:
        ap_50_per_class = []
    else:
        ap_50_per_class = [float(x) for x in ap_pc.cpu().tolist()]

    return {
        "backend": backend,
        "map_50": map_50,
        "ap_50_per_class": ap_50_per_class,
        "raw": {k: (v.cpu().tolist() if torch.is_tensor(v) else v) for k, v in res.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--rand_state", type=int, default=724)

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit_batches", type=int, default=0, help="0 = no limit")

    ap.add_argument("--onnx_fp32", type=str, default=None)
    ap.add_argument("--onnx_int8", type=str, default=None)

    ap.add_argument("--score_thresh", type=float, default=0.2)
    ap.add_argument("--nms_thresh", type=float, default=0.5)
    ap.add_argument("--max_per_img", type=int, default=100)
    ap.add_argument("--class_agnostic", action="store_true")

    ap.add_argument("--backends", nargs="+", default=["torch", "ort_fp32", "ort_int8"],
                    choices=["torch", "ort_fp32", "ort_int8"])
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    # Build deterministic val set with test transforms
    val_data = build_val_data_from_train_dir(
        args.train_path,
        rand_state=args.rand_state,
        test_size=args.test_size,
    )

    loader = DataLoader(
        val_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_detection,
    )

    model = build_model().to("cpu").float().eval()
    load_weights(model, args.weights)

    # Optional: limit batches for quick iteration
    if args.limit_batches > 0:
        # wrap loader iterator
        def limited(iterable, n):
            for i, x in enumerate(iterable):
                if i >= n:
                    break
                yield x
        loader_iter = limited(loader, args.limit_batches)
    else:
        loader_iter = loader

    # Run evals
    results: Dict[str, Dict[str, object]] = {}
    for b in args.backends:
        onnx_path = None
        if b == "ort_fp32":
            onnx_path = args.onnx_fp32
        elif b == "ort_int8":
            onnx_path = args.onnx_int8

        r = eval_backend(
            backend=b,
            model=model,
            loader=loader_iter if args.limit_batches > 0 else loader,
            onnx_path=onnx_path,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            max_per_img=args.max_per_img,
            class_agnostic=args.class_agnostic,
            threads=args.threads,
        )
        results[b] = r
        print(f"{b}: mAP@0.5 = {r['map_50']:.4f}")

    # Deltas (relative to torch if present)
    if "torch" in results:
        base = results["torch"]
        base_map = base["map_50"]
        base_ap = base["ap_50_per_class"]

        for b, r in results.items():
            if b == "torch":
                continue
            d_map = r["map_50"] - base_map
            print(f"\nDelta vs torch for {b}:")
            print(f"  mAP@0.5 delta = {d_map:+.4f}")

            ap = r["ap_50_per_class"]
            if base_ap and ap and len(base_ap) == len(ap):
                deltas = [a - b0 for a, b0 in zip(ap, base_ap)]
                print("  per-class AP@0.5 deltas:", [f"{x:+.4f}" for x in deltas])


if __name__ == "__main__":
    main()
