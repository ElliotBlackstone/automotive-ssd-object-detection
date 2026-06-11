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

if failed:
    print("Smoke test failed:")
    for name, err in failed:
        print(f"  {name}: {err}")
    sys.exit(1)

import torch
import onnxruntime

print("Smoke test passed.")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"ONNX Runtime providers: {onnxruntime.get_available_providers()}")