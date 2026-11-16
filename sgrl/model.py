"""FPRM's two-block core, with the SGRL-v2 fixed-loop state map."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.common import trunc_normal_init_
from models.config import ReasoningModelConfig
from models.layers import CastedEmbedding, CastedLinear, RotaryEmbedding
from models.sparse_embedding import CastedSparseEmbedding
from models.transformer import FixedPointTransformer
from sgrl.config import ModelConfig


class SGRLModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.puzzle_emb_len = (config.puzzle_emb_len or math.ceil(config.puzzle_emb_ndim / config.hidden_size))
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size,
                                            1 / self.embed_scale, self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        if config.puzzle_emb_ndim:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers,
                config.puzzle_emb_ndim, config.batch_size, 0, self.forward_dtype)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads,
                config.seq_len + self.puzzle_emb_len, base=10000.0)
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(config.seq_len + self.puzzle_emb_len,
                config.hidden_size, 1 / self.embed_scale, self.forward_dtype)
        # Enforce the chosen method; FPOPT, convolutions, Q/ACT heads and dropout
        # are deliberately not instantiated in this model.
        trunk_config = ReasoningModelConfig(
            **config.model_dump(exclude={"init_std", "activation_checkpointing"}),
            H_cycles=1, L_cycles=1, H_layers=0, L_layers=2,
            halt_max_steps=1, halt_exploration_prob=0.0,
            conv_type="none", norm_type="pre-norm", norm_placement="none",
            residual_scale="input-independent", normalize_input_injection=False,
            outer_only=False, weight_dropout=0.0,
        )
        self.core = FixedPointTransformer(trunk_config, n_layers=2)
        self.recovery_mode = False

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if hasattr(self, "puzzle_emb"):
            # Upstream's trainable sparse cache is a buffer, not a Parameter.
            # Device/dtype transfers can make it non-leaf; the sparse optimizer
            # must receive a fresh leaf after such transfers.
            self.puzzle_emb.local_weights = self.puzzle_emb.local_weights.detach().requires_grad_(
                not self.recovery_mode)
        return self

    def set_recovery_mode(self, enabled=True):
        self.recovery_mode = enabled
        self.requires_grad_(not enabled)
        self.core.requires_grad_(True)
        if hasattr(self, "puzzle_emb"):
            self.puzzle_emb.local_weights.requires_grad_(not enabled)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.recovery_mode and hasattr(self, "puzzle_emb"):
            # Frozen direct lookup; no mutable batch-local embedding cache.
            self.puzzle_emb.eval()
        return self

    def encode(self, inputs, puzzle_identifiers):
        x = self.embed_tokens(inputs.long())
        if hasattr(self, "puzzle_emb"):
            p = self.puzzle_emb(puzzle_identifiers.long())
            p = F.pad(p, (0, self.puzzle_emb_len * self.config.hidden_size - p.shape[-1]))
            x = torch.cat((p.reshape(-1, self.puzzle_emb_len, self.config.hidden_size), x), dim=1)
        if hasattr(self, "embed_pos"):
            x = (x + self.embed_pos.embedding_weight[:x.shape[1]].to(x.dtype)) * (2 ** -0.5)
        return self.embed_scale * x

    def initial_state(self, x):
        return trunc_normal_init_(torch.empty_like(x), std=self.config.init_std)

    def state_mask(self, labels):
        """Training-only metric mask, including registers of nonempty examples."""
        valid = labels.ne(-100)
        prefix = valid.any(-1, keepdim=True).expand(-1, self.puzzle_emb_len)
        return torch.cat((prefix, valid), dim=1)

    def step(self, z, x, core=None):
        # Teacher search variables stay fp32, but every native call uses the
        # same configured arithmetic as the student and deployment.
        z, x = z.to(self.forward_dtype), x.to(self.forward_dtype)
        selected_core = self.core if core is None else core
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            cos_sin = (cos[:z.shape[1]], sin[:z.shape[1]])

        def call(state, injection):
            return selected_core(state, injection, cos_sin=cos_sin,
                                 puzzle_emb_len=self.puzzle_emb_len)

        # Non-reentrant checkpointing supports state gradients through a frozen
        # teacher, and parameter gradients through an un-detached student rollout.
        if self.config.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(call, z, x, use_reentrant=False)
        return call(z, x)

    def decode(self, z):
        return self.lm_head(z[:, self.puzzle_emb_len:].to(self.forward_dtype)).float()

    def frozen_core(self):
        # Do not duplicate the potentially large ARC identifier embedding table.
        return copy.deepcopy(self.core).eval().requires_grad_(False)

    def forward(self, inputs, puzzle_identifiers, loops: int):
        """Inference accepts no labels, teacher paths or loop/stage identifiers."""
        if loops < 1:
            raise ValueError("loops must be positive")
        x = self.encode(inputs, puzzle_identifiers)
        z = self.initial_state(x)
        for _ in range(loops):
            z = self.step(z, x)
        return self.decode(z)
