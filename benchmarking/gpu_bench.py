#gpu_bench.py
import torch
from typing import Iterable, Callable, Dict, Any, List
import time
from dataclasses import dataclass



def _ms(ns: int) -> float:
    return ns / 1e6

def _pct(sorted_vals: List[float], p: float) -> float:
    # p in [0,100]
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

@dataclass
class StageStats:
    n: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

def summarize(times_ms: List[float]) -> StageStats:
    s = sorted(times_ms)
    return StageStats(
        n=len(s),
        mean_ms=sum(s) / len(s),
        median_ms=_pct(s, 50),
        p90_ms=_pct(s, 90),
        p95_ms=_pct(s, 95),
        p99_ms=_pct(s, 99),
        min_ms=s[0],
        max_ms=s[-1],
    )




def bench_inference_gpu(
    inputs: Iterable[Any],
    preprocess: Callable[[Any], Any],
    model: torch.nn.Module,
    warmup_iters: int = 20,
    measure_iters: int = 200,
    device: torch.device | str = "cuda",
    pin_memory: bool = False,
    score_thresh: float = 0.3,
    nms_thresh: float = 0.5,
    new_model: bool = False,
    max_per_img: int = 50,
    enable_cudnn_benchmark: bool = False,
    restore_training_mode: bool = True,
) -> Dict[str, StageStats]:
    """
    Benchmark single-model GPU inference as *latency*.

    Reported stages are all measured with CPU wall time:
        - preprocess: host-side preprocessing, including conversion to torch.Tensor
        - h2d: host-to-device transfer latency
        - forward: model forward latency
        - postprocess: latency of model.predict(...)
        - end_to_end: full latency from preprocess start to postprocess end

    Notes
    -----
    - This is a latency benchmark, not a throughput benchmark.
    - Because all stages are wall-clock times, they are directly comparable.
    - H2D is reported separately instead of being mixed into preprocess or forward.
    """

    inputs = list(inputs)
    if not inputs:
        raise ValueError("inputs must be non-empty (and already in memory).")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be >= 0.")
    if measure_iters <= 0:
        raise ValueError("measure_iters must be > 0.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device for GPU benchmarking.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested device {device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
        )

    def _to_cpu_tensor(x: Any) -> torch.Tensor:
        """
        Convert preprocess output to a tensor.
        This conversion is counted as part of CPU-side preprocessing.
        """
        t = x if torch.is_tensor(x) else torch.as_tensor(x)

        # Preprocess is expected to produce CPU data. If it already produced
        # a CUDA tensor, leave it alone; H2D time will then be ~0.
        if t.device.type == "cpu" and pin_memory and not t.is_pinned():
            t = t.pin_memory()

        return t

    def _move_to_device(t: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to the target CUDA device.
        """
        if t.device == device:
            return t

        non_blocking = (t.device.type == "cpu") and pin_memory
        return t.to(device, non_blocking=non_blocking)

    model_was_training = model.training
    cudnn_benchmark_prev = torch.backends.cudnn.benchmark

    model = model.to(device)
    model.eval()

    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    pre_t: list[float] = []
    h2d_t: list[float] = []
    fwd_t: list[float] = []
    post_t: list[float] = []
    e2e_t: list[float] = []

    try:
        with torch.cuda.device(device), torch.inference_mode():
            # --- Warmup ---
            for wi in range(warmup_iters):
                x = inputs[wi % len(inputs)]

                mi_cpu = _to_cpu_tensor(preprocess(x))
                mi_gpu = _move_to_device(mi_cpu)

                loc_all, conf_all = model(mi_gpu)
                if new_model:
                    _ = _ = model.predict(mi_gpu,
                                    score_thresh=score_thresh,
                                    nms_thresh=nms_thresh,
                                    iou_variant="DIoU",
                                    max_per_img=max_per_img,
                                    pre_loc_all=loc_all,
                                    pre_conf_all=conf_all,
                                    )
                else:
                    _ = model.predict(mi_gpu,
                                    score_thresh=score_thresh,
                                    nms_thresh=nms_thresh,
                                    max_per_img=max_per_img,
                                    pre_loc_all=loc_all,
                                    pre_conf_all=conf_all,
                                    )

            torch.cuda.synchronize(device)

            # --- Measurement ---
            for i in range(measure_iters):
                x = inputs[i % len(inputs)]

                t_e2e0 = time.perf_counter_ns()

                # 1) CPU preprocess
                t0 = time.perf_counter_ns()
                mi_cpu = _to_cpu_tensor(preprocess(x))
                t1 = time.perf_counter_ns()

                # 2) Host -> device
                mi_gpu = _move_to_device(mi_cpu)
                torch.cuda.synchronize(device)
                t2 = time.perf_counter_ns()

                # 3) Forward
                loc_all, conf_all = model(mi_gpu)
                torch.cuda.synchronize(device)
                t3 = time.perf_counter_ns()

                # 4) Postprocess
                if new_model:
                    _ = _ = model.predict(mi_gpu,
                                          score_thresh=score_thresh,
                                          nms_thresh=nms_thresh,
                                          iou_variant="DIoU",
                                          max_per_img=max_per_img,
                                          pre_loc_all=loc_all,
                                          pre_conf_all=conf_all,
                                          )
                else:
                    _ = model.predict(mi_gpu,
                                      score_thresh=score_thresh,
                                      nms_thresh=nms_thresh,
                                      max_per_img=max_per_img,
                                      pre_loc_all=loc_all,
                                      pre_conf_all=conf_all,
                                      )
                torch.cuda.synchronize(device)
                t4 = time.perf_counter_ns()

                pre_t.append(_ms(t1 - t0))
                h2d_t.append(_ms(t2 - t1))
                fwd_t.append(_ms(t3 - t2))
                post_t.append(_ms(t4 - t3))
                e2e_t.append(_ms(t4 - t_e2e0))

    finally:
        torch.backends.cudnn.benchmark = cudnn_benchmark_prev
        if restore_training_mode and model_was_training:
            model.train()

    out = {
        "preprocess": summarize(pre_t),
        "h2d": summarize(h2d_t),
        "forward": summarize(fwd_t),
        "postprocess": summarize(post_t),
        "end_to_end": summarize(e2e_t),
    }

    # These are now all wall-clock latencies, so this sanity check is meaningful.
    sum_means = (
        out["preprocess"].mean_ms
        + out["h2d"].mean_ms
        + out["forward"].mean_ms
        + out["postprocess"].mean_ms
    )
    rel_err = abs(out["end_to_end"].mean_ms - sum_means) / max(out["end_to_end"].mean_ms, 1e-9)
    if rel_err > 0.05:
        print(
            f"[warn] end_to_end mean ({out['end_to_end'].mean_ms:.3f} ms) "
            f"!= sum of stage means ({sum_means:.3f} ms)."
        )
        print("Small differences can still occur from Python overhead between stage boundaries.")

    return out


def print_report(report: Dict[str, StageStats]) -> None:
    for name, s in report.items():
        print(
            f"{name:>11}: n={s.n:4d}  mean={s.mean_ms:8.3f}  "
            f"p50={s.median_ms:8.3f}  p95={s.p95_ms:8.3f}  p99={s.p99_ms:8.3f}  "
            f"min={s.min_ms:8.3f}  max={s.max_ms:8.3f}"
        )