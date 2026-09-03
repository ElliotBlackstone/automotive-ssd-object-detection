import time
import statistics
import torch
import torch.nn as nn

from myMultiHeadAttn import myMultiHeadAttn, myMultiHeadSelfAttn

# to use:
# & "c:\Users\eblac\anaconda3\envs\torchGPUenv\python.exe" benchmark_multihead_attention.py

def copy_weights(old_model: myMultiHeadAttn, new_model: myMultiHeadSelfAttn):
    """
    Copy the old model's Q, K, V and output-projection parameters into the
    packed-QKV model so the two models perform the same computation.

    PyTorch Linear weights have shape (out_features, in_features), so the
    three E x E weight matrices are concatenated along dimension 0 to form
    the packed (3E) x E matrix.
    """
    with torch.no_grad():
        new_model.qkv.weight.copy_(
            torch.cat(
                [
                    old_model.query.weight,
                    old_model.key.weight,
                    old_model.value.weight,
                ],
                dim=0,
            )
        )

        new_model.qkv.bias.copy_(
            torch.cat(
                [
                    old_model.query.bias,
                    old_model.key.bias,
                    old_model.value.bias,
                ],
                dim=0,
            )
        )

        new_model.proj.weight.copy_(old_model.proj.weight)
        new_model.proj.bias.copy_(old_model.proj.bias)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_forward(fn, device, warmup=50, iterations=200):
    """
    Benchmark a forward-pass callable and return a list of elapsed times in ms.

    CUDA is synchronized immediately before and after each timed iteration so
    perf_counter measures actual GPU execution rather than only kernel launch.
    """
    with torch.inference_mode():
        for _ in range(warmup):
            fn()

        synchronize(device)

        elapsed_ms = []

        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter()

            fn()

            synchronize(device)
            end = time.perf_counter()

            elapsed_ms.append((end - start) * 1000.0)

    return elapsed_ms


def summarize(times):
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
    }


def print_table(rows):
    headers = [
        "Device",
        "Model",
        "Mean (ms)",
        "Median (ms)",
        "Min (ms)",
        "Median speedup",
    ]

    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            [
                row["device"],
                row["model"],
                f'{row["mean"]:.4f}',
                f'{row["median"]:.4f}',
                f'{row["min"]:.4f}',
                row["speedup"],
            ]
        )

    widths = []
    for j, header in enumerate(headers):
        widths.append(
            max(
                len(header),
                max(len(row[j]) for row in formatted_rows),
            )
        )

    def format_row(row):
        return " | ".join(
            value.ljust(widths[j]) for j, value in enumerate(row)
        )

    print()
    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))

    for row in formatted_rows:
        print(format_row(row))


def benchmark_device(
    device,
    batch_size,
    seq_len,
    embed_dim,
    num_heads,
    warmup,
    iterations,
):
    torch.manual_seed(0)

    old_model = myMultiHeadAttn(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
    ).to(device)

    new_model = myMultiHeadSelfAttn(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
    ).to(device)

    copy_weights(old_model, new_model)

    old_model.eval()
    new_model.eval()

    x = torch.randn(
        batch_size,
        seq_len,
        embed_dim,
        device=device,
    )

    # Verify that packing Q/K/V into one Linear layer did not change the result.
    with torch.inference_mode():
        old_output = old_model(x, x, x)
        new_output = new_model(x)

    max_abs_error = (old_output - new_output).abs().max().item()

    # Floating-point operation ordering can differ slightly between separate
    # and packed GEMMs, especially on GPU, so use a numerical tolerance.
    if not torch.allclose(old_output, new_output, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"Outputs do not match on {device}. "
            f"Maximum absolute error = {max_abs_error:.6e}"
        )

    old_times = benchmark_forward(
        lambda: old_model(x, x, x),
        device=device,
        warmup=warmup,
        iterations=iterations,
    )

    new_times = benchmark_forward(
        lambda: new_model(x),
        device=device,
        warmup=warmup,
        iterations=iterations,
    )

    old_stats = summarize(old_times)
    new_stats = summarize(new_times)

    speedup = old_stats["median"] / new_stats["median"]

    rows = [
        {
            "device": str(device),
            "model": "myMultiHeadAttn",
            **old_stats,
            "speedup": "1.000x",
        },
        {
            "device": str(device),
            "model": "myMultiHeadSelfAttn",
            **new_stats,
            "speedup": f"{speedup:.3f}x",
        },
    ]

    return rows, max_abs_error


def main():
    # ViT-like dimensions. Adjust these to match the configuration you want
    # to benchmark.
    batch_size = 4
    seq_len = 300
    embed_dim = 256
    num_heads = 8

    warmup = 50
    iterations = 200

    print("Self-attention benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(
        f"Input shape: ({batch_size}, {seq_len}, {embed_dim}), "
        f"heads={num_heads}"
    )
    print(f"Warmup iterations: {warmup}")
    print(f"Timed iterations: {iterations}")

    all_rows = []

    # CPU
    cpu_rows, cpu_error = benchmark_device(
        device=torch.device("cpu"),
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        warmup=warmup,
        iterations=iterations,
    )
    all_rows.extend(cpu_rows)

    print(f"CPU max |old - new|: {cpu_error:.6e}")

    # GPU, if CUDA is available.
    if torch.cuda.is_available():
        gpu = torch.device("cuda")

        print(f"CUDA device: {torch.cuda.get_device_name(gpu)}")

        gpu_rows, gpu_error = benchmark_device(
            device=gpu,
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            warmup=warmup,
            iterations=iterations,
        )
        all_rows.extend(gpu_rows)

        print(f"GPU max |old - new|: {gpu_error:.6e}")
    else:
        print("CUDA GPU not available; skipping GPU benchmark.")

    print_table(all_rows)


if __name__ == "__main__":
    main()
