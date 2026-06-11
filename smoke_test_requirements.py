import importlib
import sys

packages = [
    "numpy",
    "cv2",
    "PIL",
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "fastapi",
    "uvicorn",
    "matplotlib",
    "pandas",
    "sklearn",
    "tqdm",
    "gen_nms",
]

failed = []

for name in packages:
    try:
        importlib.import_module(name)
    except Exception as e:
        failed.append((name, repr(e)))

repo_imports = [
    ("v1.SSD_from_scratch", "mySSD"),
    ("v2.model_files.SSD_from_scratch", "mySSD"),
]

for module_name, attr_name in repo_imports:
    try:
        module = importlib.import_module(module_name)
        getattr(module, attr_name)
    except Exception as e:
        failed.append((module_name, repr(e)))

# Verify the custom NMS package exposes the expected API.
try:
    gen_nms = importlib.import_module("gen_nms")
    expected_gen_nms_fns = [
        "iou_nms",
        "batched_iou_nms",
        "giou_nms",
        "batched_giou_nms",
        "diou_nms",
        "batched_diou_nms",
        "ciou_nms",
        "batched_ciou_nms",
    ]

    for fn_name in expected_gen_nms_fns:
        if not hasattr(gen_nms, fn_name):
            failed.append(("gen_nms", f"missing expected function: {fn_name}"))
except Exception as e:
    failed.append(("gen_nms API check", repr(e)))

if failed:
    print("Smoke test failed:")
    for name, err in failed:
        print(f"  {name}: {err}")
    sys.exit(1)

import torch
import onnxruntime
import gen_nms

print("Smoke test passed.")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"ONNX Runtime providers: {onnxruntime.get_available_providers()}")
print(f"gen_nms module: {gen_nms.__file__}")