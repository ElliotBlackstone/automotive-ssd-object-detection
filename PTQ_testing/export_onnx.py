# export_onnx.py
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from v2.model_files.SSD_from_scratch import mySSD



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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to .pth state_dict")
    ap.add_argument("--out", type=str, default="ssd.onnx", help="Output .onnx path")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--static-batch", action="store_true", help="Export with fixed batch=1")
    args = ap.parse_args()

    out_path = str(pathlib.Path(args.out).resolve())

    model = build_model().to("cpu").float().eval()

    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    res = model.load_state_dict(state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        print("WARNING: load_state_dict(strict=False) reported:")
        print("  Missing keys:", res.missing_keys)
        print("  Unexpected keys:", res.unexpected_keys)

    # Input contract: float32, NCHW, 300x300
    dummy = torch.randn(1, 3, 300, 300, dtype=torch.float32)

    dynamic_axes = None
    if not args.static_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "loc": {0: "batch"},
            "conf": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["images"],
        output_names=["loc", "conf"],
        dynamic_axes=dynamic_axes,
        external_data=False,
    )

    print(f"Wrote: {out_path}")

    # Optional: structural validity check (recommended)
    try:
        import onnx  # type: ignore

        m = onnx.load(out_path)
        onnx.checker.check_model(m)
        print("ONNX checker: PASS")
    except Exception as e:
        print("ONNX checker: FAILED")
        print(e)


if __name__ == "__main__":
    main()