"""Single-device SGRL-v2 training. No GPU allocation or external logging."""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from sgrl.checkpoint import (EMA, load_weights, read_checkpoint, restore_rng,
                             rng_state, save_checkpoint)
from sgrl.config import ModelConfig, TrainConfig
from sgrl.data import (StatefulPuzzleStream, append_metrics, dataset_identity,
                       puzzle_dataset, to_device)
from sgrl.evaluation import evaluate
from sgrl.model import SGRLModel
from sgrl.recovery import real_objective, recovery_step


def resolve_model_config(config, metadata):
    values = dict(config.model)
    for key in ("seq_len", "vocab_size", "num_puzzle_identifiers"):
        expected = getattr(metadata, key)
        if key in values and values[key] != expected:
            raise ValueError(f"model.{key} disagrees with dataset metadata")
        values[key] = expected
    values["batch_size"] = config.batch_size
    return ModelConfig(**values)


def make_optimizers(model, config):
    if config.stage == "recovery":
        return []  # SGRL's backtracked update is stateless SGD, not AdamW.
    optimizers = [torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        betas=(0.9, 0.95), weight_decay=config.weight_decay)]
    if hasattr(model, "puzzle_emb"):
        optimizers.append(CastedSparseEmbeddingSignSGD_Distributed(
            [model.puzzle_emb.local_weights, model.puzzle_emb.local_ids, model.puzzle_emb.weights],
            world_size=1, lr=config.learning_rate, weight_decay=config.weight_decay))
    return optimizers


def base_step(model, batch, optimizers, config):
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    x = model.encode(batch["inputs"], batch["puzzle_identifiers"])
    z0 = model.initial_state(x)
    loss, _ = real_objective(model, x, z0, batch["labels"], 0, config.recovery)
    if not torch.isfinite(loss):
        raise FloatingPointError("base training produced a nonfinite loss")
    loss.backward()
    parameters = list(model.parameters())
    if hasattr(model, "puzzle_emb"):
        parameters.append(model.puzzle_emb.local_weights)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"accepted": True, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}


def compatible_model(saved, current):
    # Batch size/dtype/checkpointing affect execution, not weight tensor shapes.
    ignored = {"batch_size", "forward_dtype", "activation_checkpointing"}
    return {k: v for k, v in saved.items() if k not in ignored} == {
        k: v for k, v in current.items() if k not in ignored}


def update_latest(output, checkpoint):
    temporary = output / "latest.pt.tmp"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(checkpoint.name)
    os.replace(temporary, output / "latest.pt")


def train(config: TrainConfig, dry_run=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this trainer supports one device; do not launch with multi-rank torchrun")
    dataset = puzzle_dataset(config.data_path, config.batch_size, config.seed,
                             epochs_per_iter=config.epochs_per_iter)
    model_config = resolve_model_config(config, dataset.metadata)
    if dry_run:
        print(json.dumps({"config": config.model_dump(), "model": model_config.model_dump(),
                          "metadata": dataset.metadata.model_dump()}, indent=2))
        return
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device=cpu for CPU checks")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and not config.resume:
        raise ValueError("output_dir already contains a run; use a new directory or explicit resume")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = SGRLModel(model_config).to(device)
    model.set_recovery_mode(config.stage == "recovery")
    stream = StatefulPuzzleStream(dataset)
    identity = dataset_identity(config.data_path)
    state = read_checkpoint(config.resume or config.init_from) if (config.resume or config.init_from) else None
    if state:
        if not compatible_model(state["model_config"], model_config.model_dump()):
            raise ValueError("checkpoint architecture/initialization differs from this configuration")
        load_weights(model, state, "raw" if config.resume else config.init_weights)
    optimizers = make_optimizers(model, config)
    ema = EMA(model, config.ema_decay)
    start_step, accepted_steps = 0, 0
    if config.resume:
        # Exact sampler replay requires unchanged recipe, data and batch size.
        allowed_changes = {"steps", "output_dir", "resume", "init_from", "init_weights",
                           "save_every", "eval_every", "eval_batches", "eval_loops", "device"}
        old = {k: v for k, v in state["train_config"].items() if k not in allowed_changes}
        new = {k: v for k, v in config.model_dump().items() if k not in allowed_changes}
        if old != new or state["dataset_identity"] != identity:
            raise ValueError("resume recipe/data mismatch; use init_from for a new experiment")
        if len(optimizers) != len(state["optimizers"]):
            raise ValueError("optimizer count does not match resume stage")
        for optimizer, saved in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        ema.load_state_dict(state["ema"])
        stream.load_state_dict(state["sampler"])
        restore_rng(state["rng"])
        start_step, accepted_steps = state["step"], state["accepted_steps"]
    if config.steps <= start_step:
        raise ValueError("steps must exceed the resumed checkpoint step")
    with open(output / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"train": config.model_dump(), "model": model_config.model_dump(),
                   "dataset": identity}, handle, indent=2)
    append_metrics(output / "metrics.jsonl", {"event": "start", "stage": config.stage,
        "step": start_step, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "time_utc": datetime.now(timezone.utc).isoformat()})
    for step in range(start_step + 1, config.steps + 1):
        start = time.monotonic()
        _, batch, _ = next(stream)
        batch = to_device(batch, device)
        model.train()
        metrics = (recovery_step(model, batch, config.recovery) if config.stage == "recovery"
                   else base_step(model, batch, optimizers, config))
        if metrics["accepted"]:
            ema.update(model)
            accepted_steps += 1
        append_metrics(output / "metrics.jsonl", {"event": "train", "stage": config.stage,
            "step": step, "accepted_steps": accepted_steps, "seconds": time.monotonic() - start,
            **metrics})
        should_evaluate = config.eval_every > 0 and step % config.eval_every == 0
        if step % config.save_every == 0 or step == config.steps or should_evaluate:
            # Save before expensive evaluation, so timeout cannot lose this EMA.
            path = output / f"step_{step:08d}.pt"
            save_checkpoint(path, model, ema, optimizers, config, stream, step, accepted_steps, identity)
            update_latest(output, path)
        if should_evaluate:
            preserved_rng = rng_state()
            try:
                evaluation_data = puzzle_dataset(config.data_path, config.batch_size, config.seed, test=True)
                with ema.apply(model):
                    result = evaluate(model, evaluation_data, device, [config.eval_loops], config.eval_batches)
                append_metrics(output / "metrics.jsonl", {"event": "eval_ema", "step": step, **result})
            finally:
                restore_rng(preserved_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sgrl/sudoku.yaml")
    parser.add_argument("--dry-run", action="store_true", help="validate metadata/config only; no model calls")
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args()
    raw = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    config = TrainConfig(**OmegaConf.to_container(raw, resolve=True))
    train(config, args.dry_run)


if __name__ == "__main__":
    main()
