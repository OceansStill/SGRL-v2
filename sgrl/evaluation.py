"""Fixed-loop evaluation: paired snapshots of the same uninterrupted trajectory."""
import torch

from sgrl.data import to_device
from sgrl.objectives import exact_match


@torch.no_grad()
def evaluate(model, dataset, device, loop_caps, max_batches=None, progress=None, arc_evaluators=None):
    caps = sorted(set(loop_caps))
    if not caps or caps[0] <= 0:
        raise ValueError("positive loop caps required")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")
    was_training = model.training
    model.eval()
    counts = {cap: {"correct": 0, "tokens_correct": 0, "tokens": 0} for cap in caps}
    # Pair the first and last cap; values are counts, not percentages.
    paired = {"correct_correct": 0, "correct_wrong": 0, "wrong_correct": 0, "wrong_wrong": 0}
    total, batches = 0, 0
    try:
        for _, batch, _ in dataset:
            batch = to_device(batch, device)
            inputs, labels = batch["inputs"], batch["labels"]
            valid = labels.ne(-100)
            valid_rows = valid.any(-1)
            x = model.encode(inputs, batch["puzzle_identifiers"])
            z = model.initial_state(x)
            snapshots = {}
            for k in range(1, caps[-1] + 1):
                z = model.step(z, x)
                if k not in counts:
                    continue
                logits = model.decode(z)
                correct = exact_match(logits, labels)
                snapshots[k] = correct
                counts[k]["correct"] += int(correct.sum())
                counts[k]["tokens_correct"] += int(((logits.argmax(-1) == labels) & valid).sum())
                counts[k]["tokens"] += int(valid.sum())
                if arc_evaluators:
                    # The method has no Q head; use an explicit constant vote.
                    arc_evaluators[k].update_batch(batch, {"preds": logits.argmax(-1),
                        "q_halt_logits": logits.new_zeros((len(labels), 1))})
            first, last = snapshots[caps[0]], snapshots[caps[-1]]
            paired["correct_correct"] += int((first & last & valid_rows).sum())
            paired["correct_wrong"] += int((first & ~last & valid_rows).sum())
            paired["wrong_correct"] += int((~first & last & valid_rows).sum())
            paired["wrong_wrong"] += int((~first & ~last & valid_rows).sum())
            total += int(valid_rows.sum())
            batches += 1
            if progress:
                progress({"batches": batches, "count": total,
                          "correct": {cap: item["correct"] for cap, item in counts.items()}})
            if max_batches is not None and batches >= max_batches:
                break
    finally:
        model.train(was_training)
    return {"count": total, "batches": batches, "max_batches": max_batches,
            "protocol": "fixed_loops_no_early_stop", "loop_caps": caps,
            "results": {cap: {**item, "wrong": total - item["correct"],
                "sequence_accuracy": item["correct"] / max(total, 1),
                "token_accuracy": item["tokens_correct"] / max(item["tokens"], 1)} for cap, item in counts.items()},
            "paired_first_last": paired}
