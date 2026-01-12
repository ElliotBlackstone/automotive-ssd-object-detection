from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import time

import onnxruntime as ort

# --- make repo root importable ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from SSD_from_scratch import mySSD


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


def make_ort_session(onnx_path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess


def percentile_report(ms: np.ndarray, label: str) -> None:
    p50 = float(np.percentile(ms, 50))
    p95 = float(np.percentile(ms, 95))
    mean = float(ms.mean())
    print(f"{label}: p50={p50:.3f} ms, p95={p95:.3f} ms, mean={mean:.3f} ms")


def prepare_inputs(batch: int, runs: int, seed: int = 0) -> List[Tuple[torch.Tensor, np.ndarray]]:
    """
    Pre-generate inputs to avoid measuring random generation time.
    Returns list of (x_torch, x_numpy_view) pairs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs: List[Tuple[torch.Tensor, np.ndarray]] = []
    for _ in range(runs):
        x = torch.randn(batch, 3, 300, 300, dtype=torch.float32)  # CPU tensor
        x_np = x.numpy()  # zero-copy view (shares memory) on CPU
        xs.append((x, x_np))
    return xs


def run_bench(
    *,
    backend: str,
    mode: str,
    model: torch.nn.Module,
    ort_sess: ort.InferenceSession | None,
    xs: List[Tuple[torch.Tensor, np.ndarray]],
    warmup: int,
    score_thresh: float,
    nms_thresh: float,
    max_per_img: int,
    class_agnostic: bool,
) -> np.ndarray:
    """
    Returns array of per-iteration latencies in milliseconds.
    mode:
      - forward: measures only stage B (network forward)
      - e2e: measures forward + postprocess (predict with pre_loc/pre_conf)
            Note: does NOT include image file I/O or resize/normalize.
    """
    times: List[float] = []

    if backend == "torch":
        assert ort_sess is None
        model.eval()

        # warmup
        for i in range(warmup):
            x, _ = xs[i]
            with torch.inference_mode():
                loc, conf = model(x)
                if mode == "e2e":
                    _ = model.predict(
                        x,
                        score_thresh=score_thresh,
                        nms_thresh=nms_thresh,
                        max_per_img=max_per_img,
                        class_agnostic=class_agnostic,
                        pre_loc_all=loc,
                        pre_conf_all=conf,
                    )

        # timed
        for i in range(warmup, len(xs)):
            x, _ = xs[i]
            t0 = time.perf_counter()
            with torch.inference_mode():
                loc, conf = model(x)
                if mode == "e2e":
                    _ = model.predict(
                        x,
                        score_thresh=score_thresh,
                        nms_thresh=nms_thresh,
                        max_per_img=max_per_img,
                        class_agnostic=class_agnostic,
                        pre_loc_all=loc,
                        pre_conf_all=conf,
                    )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return np.asarray(times, dtype=np.float64)

    elif backend in ("ort_fp32", "ort_int8"):
        assert ort_sess is not None
        model.eval()  # used only for predict() postprocess in e2e

        input_name = ort_sess.get_inputs()[0].name

        # warmup
        for i in range(warmup):
            x_t, x_np = xs[i]
            loc_o, conf_o = ort_sess.run(["loc", "conf"], {input_name: x_np})
            if mode == "e2e":
                loc_t = torch.from_numpy(loc_o)
                conf_t = torch.from_numpy(conf_o)
                _ = model.predict(
                    x_t,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                    max_per_img=max_per_img,
                    class_agnostic=class_agnostic,
                    pre_loc_all=loc_t,
                    pre_conf_all=conf_t,
                )

        # timed
        for i in range(warmup, len(xs)):
            x_t, x_np = xs[i]
            t0 = time.perf_counter()
            loc_o, conf_o = ort_sess.run(["loc", "conf"], {input_name: x_np})
            if mode == "e2e":
                # convert to torch for your existing postprocess
                loc_t = torch.from_numpy(loc_o)
                conf_t = torch.from_numpy(conf_o)
                _ = model.predict(
                    x_t,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                    max_per_img=max_per_img,
                    class_agnostic=class_agnostic,
                    pre_loc_all=loc_t,
                    pre_conf_all=conf_t,
                )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return np.asarray(times, dtype=np.float64)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, required=True, choices=["torch", "ort_fp32", "ort_int8"])
    ap.add_argument("--mode", type=str, required=True, choices=["forward", "e2e"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)

    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--onnx_fp32", type=str, default=None, help="Path to FP32 ONNX (needed for ort_fp32)")
    ap.add_argument("--onnx_int8", type=str, default=None, help="Path to INT8 ONNX (needed for ort_int8)")

    # postprocess controls (for e2e mode)
    ap.add_argument("--score_thresh", type=float, default=0.2)
    ap.add_argument("--nms_thresh", type=float, default=0.5)
    ap.add_argument("--max_per_img", type=int, default=100)
    ap.add_argument("--class_agnostic", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # thread control matters for comparisons
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    model = build_model().to("cpu").float().eval()
    load_weights(model, args.weights)

    ort_sess = None
    if args.backend == "ort_fp32":
        if not args.onnx_fp32:
            raise ValueError("--onnx_fp32 is required for backend=ort_fp32")
        ort_sess = make_ort_session(args.onnx_fp32, threads=args.threads)
    elif args.backend == "ort_int8":
        if not args.onnx_int8:
            raise ValueError("--onnx_int8 is required for backend=ort_int8")
        ort_sess = make_ort_session(args.onnx_int8, threads=args.threads)

    # pre-generate inputs (includes warmup + timed)
    total = args.warmup + args.runs
    xs = prepare_inputs(batch=args.batch, runs=total, seed=args.seed)

    ms = run_bench(
        backend=args.backend,
        mode=args.mode,
        model=model,
        ort_sess=ort_sess,
        xs=xs,
        warmup=args.warmup,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        max_per_img=args.max_per_img,
        class_agnostic=args.class_agnostic,
    )

    label = f"{args.backend} | {args.mode} | batch={args.batch} | threads={args.threads}"
    percentile_report(ms, label)


if __name__ == "__main__":
    main()
