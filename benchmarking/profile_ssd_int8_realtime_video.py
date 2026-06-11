"""
Profile the SSD INT8 ONNX real-time video pipeline.

The script breaks the loop into measurable stages:
  capture/synthetic frame generation
  preprocessing
  ONNX Runtime inference
  NumPy -> torch tensor transfer
  PyTorch SSD postprocessing / NMS
  CPU result conversion + box scaling
  drawing
  optional display

Example camera run:
  python profile_ssd_int8_realtime_video.py \
      --model PTQ_testing/ssd_int8_v2.onnx \
      --device cuda \
      --camera 0 \
      --batch-size 1 \
      --warmup-batches 10 \
      --profile-batches 200 \
      --csv profile_cuda_b1.csv
    
  (windows)
  python benchmarking\\profile_ssd_int8_realtime_video.py `
      --model "C:\\Users\\eblac\\Documents\\GitHub\\self-driving-car\\PTQ_testing\\ssd_int8_v2.onnx" `
      --device cpu `
      --camera 0 `
      --batch-size 1 `
      --warmup-batches 10 `
      --profile-batches 200 `
      --csv profile_cpu_b1.csv
  

Example synthetic run to remove camera/display effects:
  python profile_ssd_int8_realtime_video.py \
      --model PTQ_testing/ssd_int8_v2.onnx \
      --device cuda \
      --synthetic \
      --batch-size 4 \
      --profile-batches 300 \
      --no-draw \
      --csv profile_cuda_b4_synthetic.csv

Optional cProfile run:
  python profile_ssd_int8_realtime_video.py \
      --model PTQ_testing/ssd_int8_v2.onnx \
      --device cuda \
      --synthetic \
      --profile-batches 100 \
      --cprofile-out profile_stats.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics as stats
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from SSDInt8_ONNX_Pred_v2_gpu import SSDInt8ONNXPredictorRaw, PreprocessConfig


# Keep this mapping consistent with your real-time script.
CLASS_TO_IDX = {
    "biker": 0,
    "car": 1,
    "pedestrian": 2,
    "trafficLight": 3,
    "truck": 4,
}

STAGE_NAMES = [
    "capture_ms",
    "preprocess_ms",
    "ort_ms",
    "np_to_torch_ms",
    "torch_post_ms",
    "cpu_result_ms",
    "draw_ms",
    "display_ms",
    "loop_ms",
]


def mb(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x) / (1024.0 * 1024.0)


def fmt_ms(x: float) -> str:
    return f"{x:9.3f}"


def pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def mean(values: Sequence[float]) -> float:
    return float(stats.fmean(values)) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    return float(stats.stdev(values)) if len(values) >= 2 else 0.0


def get_process_rss_mb() -> Optional[float]:
    """Current process RSS in MB. Uses psutil when available; otherwise returns None."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        return mb(proc.memory_info().rss)
    except Exception:
        return None


def get_nvidia_smi_memory_mb() -> Optional[float]:
    """
    Return GPU memory used according to nvidia-smi, if available.

    This is useful because ONNX Runtime CUDAExecutionProvider does not allocate through
    PyTorch's CUDA caching allocator, so torch.cuda.memory_allocated() does not tell
    the whole GPU-memory story.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        first = out.strip().splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


def cuda_is_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def sync_cuda(device: torch.device) -> None:
    if cuda_is_active(device):
        torch.cuda.synchronize(device)


@contextmanager
def timed_stage(row: Dict[str, Any], key: str, sync_device: Optional[torch.device] = None):
    if sync_device is not None:
        sync_cuda(sync_device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_device is not None:
            sync_cuda(sync_device)
        row[key] = (time.perf_counter() - t0) * 1000.0


def draw_predictions_bgr(
    frame_bgr: np.ndarray,
    pred: Dict[str, Any],
    show_labels: bool = True,
    score_fmt: str = "{:.2f}",
) -> np.ndarray:
    """
    Lightweight copy of the draw function from the real-time script.

    pred = {"labels": [str], "scores": [float], "boxes": np.ndarray (K,4) xyxy in frame pixels}
    """
    out = frame_bgr
    H, W = out.shape[:2]

    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]

    if boxes is None or len(labels) == 0:
        return out

    boxes = np.asarray(boxes, dtype=np.float32)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if show_labels:
            txt = f"{labels[i]}:{score_fmt.format(scores[i])}"
            cv2.putText(
                out,
                txt,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    return out


def open_camera(source: Any, backend: str):
    backend_map = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
    }

    sysname = platform.system().lower()

    if backend != "auto":
        trial = [backend]
    else:
        if sysname == "windows":
            trial = ["dshow", "msmf", "any"]
        elif sysname == "linux":
            trial = ["v4l2", "gstreamer", "any"]
        else:
            trial = ["any"]

    last_err = None
    for b in trial:
        cap = cv2.VideoCapture(source, backend_map[b])
        if cap.isOpened():
            return cap, b
        last_err = b
        cap.release()

    raise RuntimeError(
        f"Could not open camera/video source={source!r} with backends={trial} "
        f"(last tried: {last_err})"
    )


def configure_camera(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def collect_frame_batch(
    cap: Optional[cv2.VideoCapture],
    batch_size: int,
    *,
    synthetic: bool,
    width: int,
    height: int,
    synthetic_random: bool,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], bool]:
    frames: List[np.ndarray] = []

    if synthetic:
        if synthetic_random:
            for _ in range(batch_size):
                frames.append(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))
        else:
            # Use copy() to avoid accidentally measuring mutation reuse artifacts.
            base = np.zeros((height, width, 3), dtype=np.uint8)
            frames = [base.copy() for _ in range(batch_size)]
        return frames, False

    if cap is None:
        raise RuntimeError("cap is None but synthetic=False")

    stream_ended = False
    for _ in range(batch_size):
        ok, frame_bgr = cap.read()
        if not ok:
            stream_ended = True
            break
        frames.append(frame_bgr)

    return frames, stream_ended


def predictor_pipeline_profiled(
    predictor: SSDInt8ONNXPredictorRaw,
    frames_batch: Sequence[np.ndarray],
    row: Dict[str, Any],
    sync_device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Same logical work as predictor.predict_batch(...), split into timed stages.

    This deliberately uses predictor internals because predictor.predict_batch(...) is too
    coarse to tell whether the bottleneck is preprocessing, ONNX Runtime, torch tensor
    movement, PyTorch predict/NMS, or final CPU conversion.
    """
    with timed_stage(row, "preprocess_ms"):
        x_np, orig_sizes = predictor.preprocess_batch(frames_batch)
        batch_size = int(x_np.shape[0])

    with timed_stage(row, "ort_ms", sync_device):
        loc_all_np, conf_all_np = predictor.sess.run(
            [predictor.out_loc, predictor.out_conf],
            {predictor.input_name: x_np},
        )

    if loc_all_np is None or conf_all_np is None:
        return [predictor._empty_result() for _ in range(batch_size)]

    with timed_stage(row, "np_cast_ms"):
        loc_all_np = np.asarray(loc_all_np, dtype=np.float32)
        conf_all_np = np.asarray(conf_all_np, dtype=np.float32)

    if loc_all_np.size == 0 or conf_all_np.size == 0:
        return [predictor._empty_result() for _ in range(batch_size)]

    with timed_stage(row, "np_to_torch_ms", sync_device):
        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(
            device=predictor.postprocess_device, dtype=torch.float32
        )
        loc_all_t = torch.from_numpy(np.ascontiguousarray(loc_all_np)).to(
            device=predictor.postprocess_device, dtype=torch.float32
        )
        conf_all_t = torch.from_numpy(np.ascontiguousarray(conf_all_np)).to(
            device=predictor.postprocess_device, dtype=torch.float32
        )

    with timed_stage(row, "torch_post_ms", sync_device):
        with torch.inference_mode():
            pred = predictor.pytorch_post_model.predict(
                x_t,
                score_thresh=predictor.score_thresh,
                nms_thresh=predictor.nms_thresh,
                iou_variant=predictor.iou_variant,
                max_per_img=predictor.max_per_img,
                pre_loc_all=loc_all_t,
                pre_conf_all=conf_all_t,
            )

    with timed_stage(row, "cpu_result_ms", sync_device):
        pred_list = predictor._normalize_predict_output_batch(
            pred,
            expected_batch_size=batch_size,
        )
        results: List[Dict[str, Any]] = []
        for pred_i, (orig_w, orig_h) in zip(pred_list, orig_sizes):
            results.append(
                predictor._postprocess_single_result(
                    pred_i,
                    orig_w=orig_w,
                    orig_h=orig_h,
                )
            )

    return results


def memory_snapshot(
    *,
    cuda_device: torch.device,
    enable_tracemalloc: bool,
    sample_nvidia_smi: bool,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "cpu_rss_mb": get_process_rss_mb(),
        "py_current_mb": None,
        "py_peak_mb": None,
        "torch_cuda_allocated_mb": None,
        "torch_cuda_reserved_mb": None,
        "torch_cuda_peak_allocated_mb": None,
        "torch_cuda_peak_reserved_mb": None,
        "nvidia_smi_used_mb": None,
    }

    if enable_tracemalloc and tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        out["py_current_mb"] = mb(cur)
        out["py_peak_mb"] = mb(peak)

    if cuda_is_active(cuda_device):
        out["torch_cuda_allocated_mb"] = mb(torch.cuda.memory_allocated(cuda_device))
        out["torch_cuda_reserved_mb"] = mb(torch.cuda.memory_reserved(cuda_device))
        out["torch_cuda_peak_allocated_mb"] = mb(torch.cuda.max_memory_allocated(cuda_device))
        out["torch_cuda_peak_reserved_mb"] = mb(torch.cuda.max_memory_reserved(cuda_device))

    if sample_nvidia_smi:
        out["nvidia_smi_used_mb"] = get_nvidia_smi_memory_mb()

    return out


def print_summary(rows: Sequence[Dict[str, Any]], batch_size: int) -> None:
    if not rows:
        print("[warn] no rows collected")
        return

    print("\n=== Timing summary, milliseconds per batch ===")
    print(f"{'stage':<22} {'mean':>10} {'p50':>10} {'p90':>10} {'p95':>10} {'stdev':>10}")
    print("-" * 76)

    for key in STAGE_NAMES:
        vals = [float(r.get(key, 0.0)) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        print(
            f"{key:<22} "
            f"{fmt_ms(mean(vals))} "
            f"{fmt_ms(pct(vals, 0.50))} "
            f"{fmt_ms(pct(vals, 0.90))} "
            f"{fmt_ms(pct(vals, 0.95))} "
            f"{fmt_ms(stdev(vals))}"
        )

    loop_vals = [float(r["loop_ms"]) for r in rows]
    frame_count = sum(int(r["frames"]) for r in rows)
    elapsed_s = sum(loop_vals) / 1000.0
    fps = frame_count / elapsed_s if elapsed_s > 0 else float("nan")

    print("\n=== End-to-end ===")
    print(f"profiled batches: {len(rows)}")
    print(f"profiled frames:  {frame_count}")
    print(f"batch size arg:   {batch_size}")
    print(f"mean loop FPS:    {fps:.2f} frames/s")
    print(f"mean loop latency:{mean(loop_vals):.3f} ms/batch")

    # Sort non-loop stages by mean contribution.
    stage_means = []
    for key in STAGE_NAMES:
        if key == "loop_ms":
            continue
        vals = [float(r.get(key, 0.0)) for r in rows if r.get(key) is not None]
        if vals:
            stage_means.append((key, mean(vals)))
    stage_means.sort(key=lambda kv: kv[1], reverse=True)

    print("\n=== Largest average stage costs ===")
    for key, val in stage_means[:8]:
        print(f"{key:<22} {val:9.3f} ms/batch")

    mem_keys = [
        "cpu_rss_mb",
        "py_peak_mb",
        "torch_cuda_peak_allocated_mb",
        "torch_cuda_peak_reserved_mb",
        "nvidia_smi_used_mb",
    ]
    print("\n=== Peak observed memory ===")
    for key in mem_keys:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if vals:
            print(f"{key:<32} {max(float(v) for v in vals):9.1f} MB")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return

    all_keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[info] wrote CSV: {path}")


def run_profile(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.profile_batches < 1:
        raise ValueError("--profile-batches must be >= 1")

    if args.tracemalloc:
        tracemalloc.start()

    ort_providers, postprocess_device = SSDInt8ONNXPredictorRaw.resolve_runtime_device(args.device)
    print(f"[info] requested compute device={args.device!r}")
    print(f"[info] ORT providers={ort_providers} | torch postprocess device={postprocess_device}")

    predictor = SSDInt8ONNXPredictorRaw(
        onnx_model_path=args.model,
        class_to_idx=CLASS_TO_IDX,
        providers=ort_providers,
        preprocess_cfg=PreprocessConfig(input_color="bgr"),
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        max_per_img=args.max_per_img,
        postprocess_device=postprocess_device,
    )
    print(f"[info] active ORT providers={predictor.active_providers}")
    print(f"[info] ONNX input={predictor.input_name!r} outputs=({predictor.out_loc!r}, {predictor.out_conf!r})")

    if cuda_is_active(postprocess_device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(postprocess_device)
        sync_cuda(postprocess_device)

    cap: Optional[cv2.VideoCapture] = None
    backend_used = "synthetic"

    if not args.synthetic:
        source: Any
        if args.input_video:
            source = args.input_video
        else:
            source = args.camera_device if args.camera_device is not None else args.camera

        cap, backend_used = open_camera(source, args.backend)
        if not args.input_video:
            configure_camera(cap, args.width, args.height, args.fps)
        print(f"[info] opened source={source!r} using backend={backend_used}")
    else:
        print(f"[info] synthetic source: {args.width}x{args.height}, random={args.synthetic_random}")

    rng = np.random.default_rng(args.seed)

    try:
        # Warmup avoids attributing CUDA/ORT initialization to steady-state runtime.
        print(f"[info] warmup batches={args.warmup_batches}")
        for _ in range(args.warmup_batches):
            frames, stream_ended = collect_frame_batch(
                cap,
                args.batch_size,
                synthetic=args.synthetic,
                width=args.width,
                height=args.height,
                synthetic_random=args.synthetic_random,
                rng=rng,
            )
            if not frames:
                raise RuntimeError("No frames available during warmup.")
            _ = predictor.predict_batch(frames)
            if stream_ended:
                raise RuntimeError("Input stream ended during warmup; use fewer warmup batches or a longer video.")

        if cuda_is_active(postprocess_device):
            torch.cuda.reset_peak_memory_stats(postprocess_device)
            sync_cuda(postprocess_device)

        rows: List[Dict[str, Any]] = []
        print(f"[info] profiling batches={args.profile_batches}")

        for batch_idx in range(args.profile_batches):
            row: Dict[str, Any] = {"batch_idx": batch_idx}
            loop_t0 = time.perf_counter()

            with timed_stage(row, "capture_ms"):
                frames_batch, stream_ended = collect_frame_batch(
                    cap,
                    args.batch_size,
                    synthetic=args.synthetic,
                    width=args.width,
                    height=args.height,
                    synthetic_random=args.synthetic_random,
                    rng=rng,
                )

            if not frames_batch:
                print("[info] input stream ended")
                break

            row["frames"] = len(frames_batch)
            preds_batch = predictor_pipeline_profiled(
                predictor,
                frames_batch,
                row,
                sync_device=postprocess_device,
            )

            row["detections"] = sum(len(p["labels"]) for p in preds_batch)

            with timed_stage(row, "draw_ms"):
                if args.no_draw:
                    vis_batch = list(frames_batch)
                else:
                    vis_batch = [
                        draw_predictions_bgr(fr, pred, show_labels=not args.no_labels)
                        for fr, pred in zip(frames_batch, preds_batch)
                    ]

            with timed_stage(row, "display_ms"):
                if args.display:
                    for vis in vis_batch:
                        cv2.imshow("SSD INT8 profile", vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            stream_ended = True
                            break

            row["loop_ms"] = (time.perf_counter() - loop_t0) * 1000.0
            row["fps_e2e"] = (len(frames_batch) / (row["loop_ms"] / 1000.0)) if row["loop_ms"] > 0 else 0.0

            row.update(
                memory_snapshot(
                    cuda_device=postprocess_device,
                    enable_tracemalloc=args.tracemalloc,
                    sample_nvidia_smi=args.sample_nvidia_smi,
                )
            )

            rows.append(row)

            if args.print_every > 0 and (batch_idx + 1) % args.print_every == 0:
                print(
                    f"[batch {batch_idx + 1:>5}/{args.profile_batches}] "
                    f"fps={row['fps_e2e']:.1f} "
                    f"loop={row['loop_ms']:.2f}ms "
                    f"cap={row.get('capture_ms', 0.0):.2f} "
                    f"prep={row.get('preprocess_ms', 0.0):.2f} "
                    f"ort={row.get('ort_ms', 0.0):.2f} "
                    f"to_torch={row.get('np_to_torch_ms', 0.0):.2f} "
                    f"post={row.get('torch_post_ms', 0.0):.2f} "
                    f"cpu={row.get('cpu_result_ms', 0.0):.2f} "
                    f"draw={row.get('draw_ms', 0.0):.2f} "
                    f"dets={row.get('detections', 0)}"
                )

            if stream_ended:
                print("[info] stopping because stream ended or user requested exit")
                break

        return rows

    finally:
        if cap is not None:
            cap.release()
        if args.display:
            cv2.destroyAllWindows()
        if args.tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Profile SSD INT8 ONNX real-time video pipeline.")
    ap.add_argument("--model", required=True, type=str, help="Path to raw INT8 ONNX model.")
    ap.add_argument("--device", default="cpu", type=str, help="cpu, cuda, or cuda:N")

    source = ap.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="Use generated frames instead of camera/video.")
    source.add_argument("--input-video", default=None, type=str, help="Path to a video file to profile.")

    ap.add_argument("--camera", default=0, type=int, help="Webcam index if --input-video and --synthetic are not set.")
    ap.add_argument("--camera-device", default=None, help="Linux camera path such as /dev/video2. Overrides --camera.")
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "any", "dshow", "msmf", "v4l2", "gstreamer"],
        help="OpenCV VideoCapture backend.",
    )

    ap.add_argument("--width", default=1280, type=int, help="Requested camera width or synthetic width.")
    ap.add_argument("--height", default=720, type=int, help="Requested camera height or synthetic height.")
    ap.add_argument("--fps", default=30, type=int, help="Requested camera FPS.")
    ap.add_argument("--batch-size", default=1, type=int)
    ap.add_argument("--warmup-batches", default=10, type=int)
    ap.add_argument("--profile-batches", default=200, type=int)

    ap.add_argument("--score-thresh", default=0.20, type=float)
    ap.add_argument("--nms-thresh", default=0.30, type=float)
    ap.add_argument("--max-per-img", default=100, type=int)

    ap.add_argument("--no-draw", action="store_true", help="Skip drawing boxes/labels.")
    ap.add_argument("--no-labels", action="store_true", help="Draw boxes but not text labels.")
    ap.add_argument("--display", action="store_true", help="Call cv2.imshow/waitKey. Usually disable for clean profiling.")

    ap.add_argument("--synthetic-random", action="store_true", help="Use random synthetic frames instead of zero frames.")
    ap.add_argument("--seed", default=0, type=int)

    ap.add_argument("--tracemalloc", action="store_true", help="Track Python allocations. Adds overhead.")
    ap.add_argument(
        "--sample-nvidia-smi",
        action="store_true",
        help="Sample nvidia-smi memory.used each batch. Adds overhead but captures non-PyTorch GPU memory.",
    )
    ap.add_argument("--csv", default="ssd_realtime_profile.csv", type=str, help="CSV output path.")
    ap.add_argument("--print-every", default=10, type=int)
    ap.add_argument("--cprofile-out", default=None, type=str, help="Optional text file for cProfile output.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.cprofile_out:
        import cProfile
        import io
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        rows = run_profile(args)
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
        ps.print_stats(80)
        Path(args.cprofile_out).write_text(s.getvalue(), encoding="utf-8")
        print(f"[info] wrote cProfile report: {args.cprofile_out}")
    else:
        rows = run_profile(args)

    print_summary(rows, batch_size=args.batch_size)

    if args.csv:
        write_csv(Path(args.csv), rows)

    print("\nNotes:")
    print("  - For GPU timings, this script synchronizes PyTorch CUDA work around timed stages.")
    print("  - torch_cuda_* memory only tracks PyTorch's CUDA allocator.")
    print("  - ONNX Runtime CUDAExecutionProvider GPU allocations are not included in torch_cuda_* memory.")
    print("  - Use --sample-nvidia-smi for a coarse total GPU-memory view, but it adds measurement overhead.")
    print("  - Use --synthetic --no-draw to isolate model/pre/postprocess from camera and rendering costs.")


if __name__ == "__main__":
    main()
