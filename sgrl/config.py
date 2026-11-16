"""Strict configuration: budgets count shared-core calls, not optimizer steps."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfig):
    seq_len: int = Field(gt=0)
    vocab_size: int = Field(gt=1)
    num_puzzle_identifiers: int = Field(default=1, gt=0)
    batch_size: int = Field(default=8, gt=0)
    hidden_size: int = Field(default=512, gt=0)
    num_heads: int = Field(default=8, gt=0)
    expansion: float = Field(default=4.0, gt=0)
    puzzle_emb_ndim: int = Field(default=0, ge=0)
    puzzle_emb_len: int = Field(default=0, ge=0)
    pos_encodings: Literal["rope", "learned", "none"] = "rope"
    forward_dtype: Literal["float32", "bfloat16"] = "float32"
    alpha_1_init: float = Field(default=0.75, gt=0, lt=1)
    alpha_2_init: float = Field(default=0.25, gt=0, lt=1)
    init_std: float = Field(default=6.0, ge=0)
    activation_checkpointing: bool = True

    @model_validator(mode="after")
    def check_dimensions(self):
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.pos_encodings == "rope" and (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.puzzle_emb_ndim == 0 and self.puzzle_emb_len:
            raise ValueError("puzzle_emb_len must be zero without puzzle embeddings")
        if self.puzzle_emb_len and self.puzzle_emb_ndim > self.puzzle_emb_len * self.hidden_size:
            raise ValueError("puzzle embedding does not fit in its register tokens")
        return self


class RecoveryConfig(StrictConfig):
    loops: int = Field(default=72, gt=0)  # K, total calls from initialization
    terminal_window: int = Field(default=2, ge=0)  # S: S+1 observations, K-S..K
    anchor_min: int = Field(default=12, ge=0)
    anchor_max: int = Field(default=36, ge=0)
    bridge_steps: int = Field(default=4, gt=0)  # m
    max_anchors: int = Field(default=2, ge=0)  # bounded teacher work per batch
    teacher_restarts: int = Field(default=2, gt=0)
    teacher_steps: int = Field(default=16, ge=0)
    teacher_lr: float = Field(default=0.03, gt=0)
    teacher_grad_clip: float = Field(default=1.0, gt=0)
    displacement_limit: float = Field(default=12.0, gt=0)
    bridge_radius: float = Field(default=0.02, gt=0)
    suffix_radius: float = Field(default=0.04, gt=0)
    gap: float = Field(default=0.01, gt=0)
    kappa: float = Field(default=0.5, ge=0, lt=1)
    teacher_dynamics_weight: float = Field(default=0.1, ge=0)
    teacher_clearance_weight: float = Field(default=10.0, ge=0)
    bridge_weight: float = Field(default=1.0, ge=0)
    min_margin: float = Field(default=0.001, gt=0)
    min_teacher_improvement: float = Field(default=1e-6, ge=0)
    probes: int = Field(default=2, gt=0)  # random starts, plus center
    probe_ascent_steps: int = Field(default=1, ge=0)
    probe_step_size: float = Field(default=0.01, gt=0)
    suffix_gain_limit: float = Field(default=4.0, gt=0)
    preservation_tolerance: float = Field(default=0.01, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    backtrack_factor: float = Field(default=0.5, gt=0, lt=1)
    backtrack_attempts: int = Field(default=8, gt=0)
    min_objective_decrease: float = Field(default=1e-8, ge=0)
    real_loss_slack: float = Field(default=0.0, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    loss: Literal["stablemax", "softmax"] = "stablemax"

    @model_validator(mode="after")
    def check_budget(self):
        if self.anchor_min > self.anchor_max:
            raise ValueError("anchor_min must not exceed anchor_max")
        if self.anchor_max + self.terminal_window + self.bridge_steps > self.loops:
            raise ValueError("need anchor_max + terminal_window + bridge_steps <= loops")
        if self.suffix_radius < self.bridge_radius:
            raise ValueError("suffix_radius must be >= bridge_radius")
        if self.displacement_limit <= self.bridge_steps * (self.bridge_radius + self.suffix_radius + self.gap):
            raise ValueError("displacement_limit is too small for the required corridor clearance")
        return self


class TrainConfig(StrictConfig):
    data_path: str
    output_dir: str = "runs/sgrl-v2"
    stage: Literal["base", "recovery"] = "base"
    device: str = "cuda"
    seed: int = 0
    batch_size: int = Field(default=8, gt=0)
    steps: int = Field(default=10000, gt=0)
    epochs_per_iter: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)  # base AdamW only
    weight_decay: float = Field(default=0.001, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    ema_decay: float = Field(default=0.999, ge=0, lt=1)
    save_every: int = Field(default=100, gt=0)
    eval_every: int = Field(default=0, ge=0)
    eval_batches: int | None = Field(default=None, gt=0)
    eval_loops: int = Field(default=1000, gt=0)
    init_from: str | None = None
    init_weights: Literal["raw", "ema"] = "ema"
    resume: str | None = None
    model: dict = Field(default_factory=dict)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)

    @model_validator(mode="after")
    def check_initialization(self):
        if self.resume and self.init_from:
            raise ValueError("resume and init_from are mutually exclusive")
        if self.stage == "recovery" and not (self.init_from or self.resume):
            raise ValueError("recovery requires a trained SGRL checkpoint via init_from or resume")
        return self
