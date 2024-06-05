import os
import jsonlines


def estimate_skip_length(output_path: str):
    if os.path.exists(output_path):
        with jsonlines.open(output_path, "r") as f:
            skip_length = len(list(f))
    else:
        skip_length = 0

    return skip_length


def batched_iteration(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch