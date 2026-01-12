# repro.py
from __future__ import annotations

import os
import platform
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> None:
    """
    Seed python/numpy/torch. Optionally request deterministic algorithms.
    Notes:
      - Determinism can reduce performance.
      - Some ops remain nondeterministic depending on backend/device.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    # If you ever use CUDA later:
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Enforce deterministic algorithms where supported
        torch.use_deterministic_algorithms(True)
        # cuDNN knobs (harmless on CPU, useful if you later benchmark GPU)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch_threads(*, intra_op: int, inter_op: int = 1) -> None:
    """
    Control CPU thread pools. Without this, benchmarks are not comparable across runs/machines.
    """
    torch.set_num_threads(int(intra_op))
    torch.set_num_interop_threads(int(inter_op))


def configure_omp_env(*, num_threads: int) -> None:
    """
    Control common BLAS/OMP thread env vars.
    Set these BEFORE importing heavy numeric libs in a fresh process when possible.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    # sometimes helpful for latency in service settings
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def file_sha256(path: str, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class RunMeta:
    seed: int
    deterministic: bool
    device: str
    torch_version: str
    numpy_version: str
    python_version: str
    platform: str
    cpu_threads_intra: Optional[int] = None
    cpu_threads_inter: Optional[int] = None
    weights_path: Optional[str] = None
    weights_sha256: Optional[str] = None
    onnx_path: Optional[str] = None
    onnx_sha256: Optional[str] = None
    opset: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_meta(
    *,
    seed: int,
    deterministic: bool,
    device: str,
    intra: Optional[int] = None,
    inter: Optional[int] = None,
    weights_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    opset: Optional[int] = None,
) -> RunMeta:
    return RunMeta(
        seed=seed,
        deterministic=deterministic,
        device=device,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_threads_intra=intra,
        cpu_threads_inter=inter,
        weights_path=weights_path,
        weights_sha256=file_sha256(weights_path) if weights_path else None,
        onnx_path=onnx_path,
        onnx_sha256=file_sha256(onnx_path) if onnx_path else None,
        opset=opset,
    )
