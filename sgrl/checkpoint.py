"""Atomic raw/EMA checkpoints with RNG, sampler, stage and optimizer state."""
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


FORMAT = "sgrl-v2-1"


def cpu_copy(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        tracked = {name for name, param in model.named_parameters() if param.requires_grad}
        if not model.recovery_mode and hasattr(model, "puzzle_emb"):
            tracked.add("puzzle_emb.weights")
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items() if name in tracked}

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for name, value in self.shadow.items():
            value.lerp_(state[name], 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": cpu_copy(self.shadow)}

    def load_state_dict(self, state):
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA fields do not match the current training stage")
        self.decay, self.updates = state["decay"], state["updates"]
        for name in self.shadow:
            self.shadow[name].copy_(state["shadow"][name])

    @contextmanager
    def apply(self, model):
        state = model.state_dict()
        original = {name: state[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    state[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in original.items():
                    state[name].copy_(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, ema, optimizers, config, stream, step, accepted_steps, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "model_config": model.config.model_dump(),
        "train_config": config.model_dump(), "model": cpu_copy(model.state_dict()),
        "ema": ema.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
        "step": step, "accepted_steps": accepted_steps, "rng": rng_state(),
        "sampler": stream.state_dict(), "dataset_identity": identity}
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path):
    # Only load locally trusted checkpoints: optimizer/RNG payloads use pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("expected a SGRL-v2 checkpoint; upstream FPRM weights need explicit conversion")
    return state


def load_weights(model, state, kind="raw"):
    if kind not in {"raw", "ema"}:
        raise ValueError("weight kind must be raw or ema")
    weights = dict(state["model"])
    if kind == "ema":
        weights.update(state["ema"]["shadow"])
    model.load_state_dict(weights, strict=True)
