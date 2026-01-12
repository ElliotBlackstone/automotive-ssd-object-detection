# parity.py
import numpy as np
import torch
import onnxruntime as ort

from repro import seed_everything, configure_torch_threads, collect_meta

def compare_np(a: np.ndarray, b: np.ndarray, name: str, eps: float = 1e-12):
    assert a.shape == b.shape, (name, a.shape, b.shape)
    diff = a - b
    absd = np.abs(diff)
    max_abs = absd.max()
    mean_abs = absd.mean()
    rmse = np.sqrt((diff * diff).mean())
    rel = absd / (np.abs(a) + eps)
    print(f"{name}: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rmse={rmse:.3e}, max_rel={rel.max():.3e}")

def parity(model, onnx_path: str):
    seed_everything(0, deterministic=True)
    configure_torch_threads(intra_op=1, inter_op=1)
    model.eval().to("cpu").float()

    x = torch.randn(1, 3, 300, 300, dtype=torch.float32)
    x_np = x.numpy()

    with torch.inference_mode():
        loc_t, conf_t = model(x)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    loc_o, conf_o = sess.run(["loc", "conf"], {"images": x_np.astype(np.float32)})

    compare_np(loc_t.numpy(), loc_o, "loc")
    compare_np(conf_t.numpy(), conf_o, "conf")

    meta = collect_meta(seed=0, deterministic=True, device="cpu", intra=1, inter=1, onnx_path=onnx_path)
    print(meta.to_dict())
