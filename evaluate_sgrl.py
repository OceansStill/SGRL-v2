"""Evaluate raw or EMA weights at multiple fixed loop caps in one trajectory."""
import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from sgrl.checkpoint import load_weights, read_checkpoint
from sgrl.config import ModelConfig
from sgrl.data import append_metrics, puzzle_dataset
from sgrl.evaluation import evaluate
from sgrl.model import SGRLModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True, help="a new JSON result path; progress uses .progress.jsonl")
    parser.add_argument("--weights", choices=["raw", "ema"], default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loops", type=int, nargs="+", default=[1000])
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arc-voting", action="store_true",
                        help="also run upstream inverse-augmentation/crop/pass@K with constant confidence")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing results: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state = read_checkpoint(args.checkpoint)
    values = dict(state["model_config"], batch_size=args.batch_size)
    model = SGRLModel(ModelConfig(**values)).to(args.device).eval()
    load_weights(model, state, args.weights)
    dataset = puzzle_dataset(args.data_path, args.batch_size, args.seed, test=True)
    for key in ("seq_len", "vocab_size", "num_puzzle_identifiers"):
        if getattr(dataset.metadata, key) != getattr(model.config, key):
            raise ValueError(f"evaluation data {key} differs from checkpoint")
    arc_evaluators = None
    group_initialized = False
    with tempfile.TemporaryDirectory(prefix="sgrl-arc-") as temporary:
        try:
            if args.arc_voting:
                from evaluators.arc import ARC
                import torch.distributed as dist
                if dist.is_initialized():
                    raise RuntimeError("standalone ARC evaluation requires its own single-rank process group")
                dist.init_process_group("gloo", init_method=Path(temporary, "rendezvous").as_uri(), rank=0, world_size=1)
                group_initialized = True
                arc_evaluators = {cap: ARC(args.data_path, dataset.metadata) for cap in sorted(set(args.loops))}
                for evaluator in arc_evaluators.values():
                    evaluator.begin_eval()
            result = evaluate(model, dataset, args.device, args.loops, args.max_batches,
                progress=lambda values: append_metrics(destination.with_suffix(".progress.jsonl"), values),
                arc_evaluators=arc_evaluators)
            result.update(checkpoint=str(Path(args.checkpoint).resolve()), step=state["step"],
                          stage=state["train_config"]["stage"], weights=args.weights, seed=args.seed)
            if arc_evaluators:
                result["arc_voting"] = {}
                for cap, evaluator in arc_evaluators.items():
                    save_path = destination.parent / f"{destination.stem}_arc_loop{cap}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    result["arc_voting"][cap] = evaluator.result(str(save_path), rank=0, world_size=1)
                result["arc_confidence"] = "constant; no learned Q head"
            with open(destination, "x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            print(json.dumps(result, indent=2))
        finally:
            if group_initialized:
                import torch.distributed as dist
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
