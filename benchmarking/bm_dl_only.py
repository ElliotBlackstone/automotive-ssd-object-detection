# bm_dl_only.py
import time
import torch
from torch.utils.data import DataLoader



def benchmark_dataloader_only(
    dataset,
    collate_fn,
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    multiprocessing_context=None,
    shuffle=False,
    warmup_batches=20,
    measure_batches=100,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context

    loader = DataLoader(**kwargs)

    it = iter(loader)

    # Warmup
    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    times = []
    n_samples = 0

    for _ in range(measure_batches):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t1 = time.perf_counter()

        times.append(t1 - t0)

        # assumes batch = (images, targets) and images is a list/tensor
        images = batch[0]
        if torch.is_tensor(images):
            n_samples += images.shape[0]
        else:
            n_samples += len(images)

    total_time = sum(times)
    return {
        "avg_batch_time_s": total_time / len(times),
        "batches_per_sec": len(times) / total_time,
        "samples_per_sec": n_samples / total_time,
        "median_batch_time_s": float(torch.tensor(times).median().item()),
    }