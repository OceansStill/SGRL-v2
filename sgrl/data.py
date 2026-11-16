"""Upstream puzzle format and a resumable, single-process training cursor."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def puzzle_dataset(path, batch_size, seed, *, test=False, epochs_per_iter=16, split=None):
    return PuzzleDataset(PuzzleDatasetConfig(seed=seed, dataset_paths=[str(path)],
        global_batch_size=batch_size, test_set_mode=test, epochs_per_iter=epochs_per_iter,
        rank=0, num_replicas=1), split=split or ("test" if test else "train"))


def dataset_identity(path, split="train"):
    """Cheap change detection; full reproducibility assumes immutable arrays."""
    root = Path(path).resolve()
    metadata = root / split / "dataset.json"
    files = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
             for p in sorted((root / split).glob("*.npy"))]
    return {"root": str(root), "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "arrays": files}


class StatefulPuzzleStream:
    """Replay the current finite sampler block on resume, without model calls.

    Upstream samples augmentations via NumPy's global RNG. Both the block-entry
    RNG and the current global RNG are preserved; num_workers must stay zero.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.iterator = None
        self.block_start_iters = 0
        self.block_numpy_state = None
        self.consumed = 0

    def _new_block(self):
        self.block_start_iters = self.dataset._iters
        self.block_numpy_state = copy.deepcopy(np.random.get_state())
        self.consumed = 0
        self.iterator = iter(self.dataset)

    def __next__(self):
        if self.iterator is None:
            self._new_block()
        try:
            item = next(self.iterator)
        except StopIteration:
            self._new_block()
            try:
                item = next(self.iterator)
            except StopIteration as exc:
                raise ValueError("dataset yields no complete training batch; lower batch_size or increase epochs_per_iter") from exc
        self.consumed += 1
        return item

    def state_dict(self):
        return {"block_start_iters": self.block_start_iters,
                "block_numpy_state": copy.deepcopy(self.block_numpy_state), "consumed": self.consumed}

    def load_state_dict(self, state):
        self.block_start_iters = state["block_start_iters"]
        self.block_numpy_state = copy.deepcopy(state["block_numpy_state"])
        self.consumed = state["consumed"]
        self.dataset._iters = self.block_start_iters
        if self.block_numpy_state is None:
            self.iterator = None
            return
        current_numpy = np.random.get_state()
        try:
            np.random.set_state(self.block_numpy_state)
            self.iterator = iter(self.dataset)
            for _ in range(self.consumed):
                next(self.iterator)
        finally:
            np.random.set_state(current_numpy)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def append_metrics(path, record):
    # Nonfinite diagnostics are rendered as null, keeping the file valid JSONL.
    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        return value
    line = json.dumps(finite(record), ensure_ascii=False, allow_nan=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)
