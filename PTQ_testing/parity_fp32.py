from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import onnxruntime as ort

# --- make repo root importable (adjust if needed) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from SSD_from_scratch import mySSD  # adjust if your import path differs


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


def diff_stats(a: np.ndarray, b: np.ndarray, name: str, eps: float = 1e-12) -> None:
    assert a.shape == b.shape, f"{name} shape mismatch: {a.shape} vs {b.shape}"
    diff = a - b
    absd = np.abs(diff)

    max_abs = float(absd.max())
    mean_abs = float(absd.mean())
    rmse = float(np.sqrt((diff * diff).mean()))

    rel = absd / (np.abs(a) + eps)
    max_rel = float(rel.max())
    mean_rel = float(rel.mean())

    # a few useful percentiles
    p50 = float(np.percentile(absd, 50))
    p90 = float(np.percentile(absd, 90))
    p99 = float(np.percentile(absd, 99))

    print(f"\n[{name}]")
    print(f"  max_abs  : {max_abs:.3e}")
    print(f"  mean_abs : {mean_abs:.3e}")
    print(f"  rmse     : {rmse:.3e}")
    print(f"  max_rel  : {max_rel:.3e}")
    print(f"  mean_rel : {mean_rel:.3e}")
    print(f"  abs diff percentiles: p50={p50:.3e}, p90={p90:.3e}, p99={p99:.3e}")
    print(f"  finite(a)={np.isfinite(a).all()}  finite(b)={np.isfinite(b).all()}")

    # show worst offender index/value
    flat_idx = int(absd.reshape(-1).argmax())
    idx = np.unravel_index(flat_idx, absd.shape)
    print(f"  worst idx={idx}, torch={a[idx]:.6g}, ort={b[idx]:.6g}, abs_diff={absd[idx]:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for torch + ORT")
    ap.add_argument("--runs", type=int, default=1, help="How many random inputs to test")
    args = ap.parse_args()

    # determinism / consistency
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    # --- load torch model ---
    model = build_model().to("cpu").float().eval()
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    res = model.load_state_dict(state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        print("WARNING load_state_dict(strict=False):")
        print("  missing_keys   =", res.missing_keys)
        print("  unexpected_keys=", res.unexpected_keys)

    # --- load ORT session ---
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])

    # sanity: check io names match your export
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    if set(out_names) != {"loc", "conf"}:
        print("WARNING: ONNX outputs are:", out_names, "(expected ['loc','conf'])")

    for i in range(args.runs):
        # IMPORTANT: one input tensor used for BOTH torch and ORT
        x = torch.randn(1, 3, 300, 300, dtype=torch.float32)
        x_np = x.numpy()  # shared values

        with torch.inference_mode():
            loc_t, conf_t = model(x)

        loc_o, conf_o = sess.run(["loc", "conf"], {input_name: x_np.astype(np.float32)})

        loc_t_np = loc_t.numpy()
        conf_t_np = conf_t.numpy()

        print(f"\n=== Run {i+1}/{args.runs} ===")
        diff_stats(loc_t_np, loc_o, "loc")
        diff_stats(conf_t_np, conf_o, "conf")

        # assert_close (torch vs ORT) with configurable tolerances
        torch.testing.assert_close(
            torch.from_numpy(loc_o), loc_t, rtol=args.rtol, atol=args.atol
        )
        torch.testing.assert_close(
            torch.from_numpy(conf_o), conf_t, rtol=args.rtol, atol=args.atol
        )
        print(f"assert_close: PASS (rtol={args.rtol}, atol={args.atol})")

    print("\nDone.")


if __name__ == "__main__":
    main()
