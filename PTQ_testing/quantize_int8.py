from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.transforms import v2

# ORT quantization
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

# --- repo root import fix ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import v2.CarImageClass as CarImageClass
from calibration_data import build_calibration_loader_from_dataset, SSDCalibrationDataReader


def build_test_tfms() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((300, 300), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def build_val_data_from_train_dir(
    train_path: str,
    *,
    rand_state: int = 724,
    test_size: float = 0.25,
):
    """
    Reconstruct val_data deterministically from your full training directory.
    Uses test transforms for val split (as you described).
    """
    test_tfms = build_test_tfms()

    # For the split function signature, we must pass a train transform too.
    # It doesn't matter for calibration because we'll only use val_data.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_fp32", type=str, required=True, help="FP32 ONNX input model")
    ap.add_argument("--onnx_int8", type=str, default="ssd_int8.onnx", help="INT8 ONNX output model")
    ap.add_argument("--train_path", type=str, required=True, help="Training directory used to build val split")

    ap.add_argument("--calib_samples", type=int, default=1000, help="Number of val images for calibration (500-2000 typical)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=724)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--calib_method", type=str, default="MinMax", choices=["MinMax", "Entropy", "Percentile"])

    args = ap.parse_args()

    # Build val_data with deterministic inference transforms
    val_data = build_val_data_from_train_dir(
        args.train_path,
        rand_state=args.seed,
        test_size=args.test_size,
    )

    calib_loader = build_calibration_loader_from_dataset(
        val_data,
        num_samples=args.calib_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # The input name must match your ONNX export (you used input_names=["images"])
    reader = SSDCalibrationDataReader(calib_loader, input_name="images")

    # Pick calibration method
    calib_method = getattr(CalibrationMethod, args.calib_method)

    # Quantize: QDQ format with QUInt8 activations and QInt8 weights is the common CPU recipe.
    quantize_static(
        model_input=args.onnx_fp32,
        model_output=args.onnx_int8,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=calib_method,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={
            # Symmetric weights are common; activations often left asymmetric
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
    )

    print(f"Wrote INT8 model: {args.onnx_int8}")


if __name__ == "__main__":
    main()
