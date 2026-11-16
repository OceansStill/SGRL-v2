"""CPU regressions for mathematical boundaries, transactions and resumability."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sgrl.checkpoint import EMA, load_weights, read_checkpoint
from sgrl.config import ModelConfig, RecoveryConfig, TrainConfig
from sgrl.data import StatefulPuzzleStream, puzzle_dataset
from sgrl.evaluation import evaluate
from sgrl.model import SGRLModel
from sgrl.objectives import exact_match, metric_norm, metric_sq, min_margin, task_loss
from sgrl.recovery import (backtracked_step, clearance, collect_failures, make_bridge,
                          proposal_objective, recovery_step, search_teacher)
from train_sgrl import make_optimizers, train


def small_recovery(**changes):
    values = dict(loops=5, terminal_window=1, anchor_min=1, anchor_max=1,
                  bridge_steps=2, max_anchors=1, teacher_restarts=1,
                  teacher_steps=40, teacher_lr=0.2, probes=1, probe_ascent_steps=0,
                  teacher_dynamics_weight=0, teacher_clearance_weight=0,
                  loss="softmax", learning_rate=0.1, preservation_tolerance=0.2)
    values.update(changes)
    return RecoveryConfig(**values)


class BiasCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, 2))

    def forward(self, z, x):
        return z + self.bias


class ToyModel(nn.Module):
    """Identity old dynamics with a known reachable, persistent correct basin."""
    def __init__(self):
        super().__init__()
        self.core = BiasCore()
        self.recovery_mode = True

    def encode(self, inputs, puzzle_identifiers):
        return torch.zeros(len(inputs), 1, 2)

    def initial_state(self, x):
        return -torch.ones_like(x)

    def step(self, z, x, core=None):
        return (self.core if core is None else core)(z, x)

    def decode(self, z):
        return torch.stack((-z[..., 0], z[..., 0]), dim=-1)

    def state_mask(self, labels):
        return labels.ne(-100)

    def frozen_core(self):
        return copy.deepcopy(self.core).requires_grad_(False)


class ScriptedModel(ToyModel):
    """Depth/id are hidden state, enabling exact failure-window counterexamples."""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode(self, inputs, puzzle_identifiers):
        return torch.stack((torch.zeros_like(inputs).float(), inputs.float()), -1)

    def initial_state(self, x):
        return x.clone()

    def step(self, z, x, core=None):
        self.calls += 1
        return z + z.new_tensor([1.0, 0.0])

    def decode(self, z):
        depth, identifier = z[..., 0], z[..., 1]
        correct = ((identifier == 0) & (depth == 3) |
                   (identifier == 1) & (depth >= 6) |
                   (identifier == 2) & (depth == 2))
        score = correct.float() * 2 - 1
        return torch.stack((-score, score), -1)


def batch(size=1):
    return {"inputs": torch.zeros(size, 1, dtype=torch.long),
            "puzzle_identifiers": torch.zeros(size, dtype=torch.long),
            "labels": torch.ones(size, 1, dtype=torch.long)}


def write_fixture(path):
    path = Path(path)
    metadata = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                    vocab_size=3, seq_len=4, num_puzzle_identifiers=5,
                    total_groups=4, mean_puzzle_examples=3.0, total_puzzles=4, sets=["all"])
    for split in ("train", "test"):
        folder = path / split
        folder.mkdir(parents=True)
        (folder / "dataset.json").write_text(json.dumps(metadata))
        inputs = (np.arange(48).reshape(12, 4) % 2 + 1).astype(np.int32)
        labels = np.ones((12, 4), dtype=np.int32)
        labels[:, -1] = 0
        arrays = dict(inputs=inputs, labels=labels,
                      puzzle_identifiers=np.arange(1, 5, dtype=np.int32),
                      puzzle_indices=np.arange(0, 13, 3, dtype=np.int32),
                      group_indices=np.arange(5, dtype=np.int32))
        for key, value in arrays.items():
            np.save(folder / f"all__{key}.npy", value)


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        torch.set_num_threads(1)

    def test_failure_selection_uses_anchor_and_terminal_window_only(self):
        model = ScriptedModel()
        data = batch(4)
        data["inputs"][:, 0] = torch.arange(4)
        data["labels"][-1] = -100
        cfg = small_recovery(loops=7, terminal_window=2, anchor_min=2, anchor_max=2)
        collected = collect_failures(model, model.frozen_core(), data, cfg, t=2)
        self.assertEqual(collected.failed.tolist(), [True, False, False, False])
        self.assertEqual(collected.terminal_correct.shape, (3, 4))
        self.assertFalse(collected.anchor.requires_grad)
        # Sample 0 was correct at depth 3, but still qualifies: the definition
        # does not say every intervening loop must be wrong.
        altered = {**data, "labels": torch.zeros_like(data["labels"])}
        other = collect_failures(model, model.frozen_core(), altered, cfg, t=2)
        torch.testing.assert_close(collected.anchor, other.anchor)

    def test_masked_task_metrics_exclude_padding(self):
        logits = torch.tensor([[[0., 3.], [5., 0.]], [[0., 1.], [0., 1.]]])
        labels = torch.tensor([[1, -100], [-100, -100]])
        self.assertEqual(exact_match(logits, labels).tolist(), [True, False])
        self.assertEqual(task_loss(logits, labels)[1].item(), 0)
        self.assertEqual(min_margin(logits, labels).tolist(), [3., -float("inf")])
        value = torch.tensor([[[3., 4.], [100., 100.]]])
        self.assertAlmostEqual(metric_sq(value, torch.tensor([[True, False]])).item(), 12.5)

    def test_bridge_geometry_and_endpoint_membership(self):
        a = torch.zeros(1, 2, 2)
        v = torch.ones_like(a) * 2
        mask = torch.tensor([[True, False]])
        bends = torch.randn(1, 1, 2, 2)
        q = make_bridge(a, v, bends, mask, 2)
        torch.testing.assert_close(q[0], a)
        torch.testing.assert_close(q[-1], a + v * mask.unsqueeze(-1))
        transverse = q[1] - a - 0.5 * v * mask.unsqueeze(-1)
        self.assertAlmostEqual(float((transverse * v * mask.unsqueeze(-1)).sum()), 0, places=5)
        suffix = q[-1:].repeat(3, 1, 1, 1)
        _, passes = clearance(q, suffix, mask, small_recovery())
        self.assertTrue(passes)  # q_m == w_0 must not be compared to itself.

    def test_teacher_frozen_core_keeps_input_gradients(self):
        model = SGRLModel(ModelConfig(seq_len=2, vocab_size=3, hidden_size=8,
                                     num_heads=2, activation_checkpointing=True))
        model.set_recovery_mode()
        self.assertFalse(any(p.requires_grad for p in model.embed_tokens.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.lm_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.core.parameters()))
        frozen = model.frozen_core()
        x = model.encode(torch.ones(1, 2, dtype=torch.long), torch.zeros(1, dtype=torch.long)).detach()
        z = model.initial_state(x).requires_grad_(True)
        loss = model.decode(model.step(z, x, core=frozen)).square().sum()
        grad, = torch.autograd.grad(loss, z)
        self.assertGreater(float(grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))

    def test_sparse_embedding_cache_stays_leaf_after_transfer(self):
        config = ModelConfig(seq_len=2, vocab_size=3, hidden_size=8,
                             num_heads=2, puzzle_emb_ndim=8, puzzle_emb_len=1)
        model = SGRLModel(config).to(dtype=torch.float64)
        self.assertTrue(model.puzzle_emb.local_weights.is_leaf)
        make_optimizers(model, TrainConfig(data_path="unused"))
        model.set_recovery_mode()
        model.to(dtype=torch.float32)
        self.assertFalse(model.puzzle_emb.local_weights.requires_grad)

    def test_teacher_search_is_detached_and_does_not_mutate_model(self):
        model = ToyModel()
        cfg = small_recovery()
        before = copy.deepcopy(model.state_dict())
        old = model.frozen_core()
        collected = collect_failures(model, old, batch(), cfg)
        teacher = search_teacher(model, old, collected, batch()["labels"], 0, cfg)
        self.assertIsNotNone(teacher)
        self.assertFalse(teacher.q.requires_grad)
        self.assertFalse(teacher.suffix.requires_grad)
        self.assertTrue(all(not point.requires_grad for point in teacher.suffix_inputs))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in old.parameters()))
        self.assertGreater(min_margin(model.decode(teacher.suffix[-1]), batch()["labels"]).item(), 0)

    def test_real_rollout_keeps_wrong_samples_without_teachers(self):
        model = ToyModel()
        cfg = small_recovery(max_anchors=0)
        collected = collect_failures(model, model.frozen_core(), batch(2), cfg)
        objective, real, bridge, final = proposal_objective(model, collected, batch(2)["labels"], [], cfg)
        self.assertEqual(bridge.item(), 0)
        objective.backward()
        self.assertGreater(model.core.bias.grad.abs().sum().item(), 0)
        self.assertEqual(final.shape, collected.anchor.shape)
        self.assertEqual(objective.item(), real.item())

    def test_successful_teacher_enters_actual_recovery_update(self):
        model = ToyModel()
        metrics = recovery_step(model, batch(), small_recovery())
        self.assertEqual(metrics["teacher_qualified"], 1)
        self.assertTrue(metrics["accepted"])
        self.assertLess(metrics["real_loss_after"], metrics["real_loss_before"])
        self.assertLessEqual(metrics["preservation_error"], 0.2)

    def test_rejected_batch_restores_weights_and_reports_actual_result(self):
        model = ToyModel()
        before = model.core.bias.detach().clone()
        metrics = recovery_step(model, batch(), small_recovery(max_anchors=0, min_objective_decrease=100))
        self.assertFalse(metrics["accepted"])
        torch.testing.assert_close(model.core.bias, before, rtol=0, atol=0)
        self.assertEqual(metrics["wrong_to_correct"], 0)
        self.assertEqual(metrics["objective_after"], metrics["objective_before"])

    def test_backtracking_accepts_later_step(self):
        p = nn.Parameter(torch.ones(1))
        def check():
            return bool(p.square() < 1), {"value": p.item()}
        accepted, result = backtracked_step([p], [torch.ones_like(p)], check,
                                           small_recovery(learning_rate=2))
        self.assertTrue(accepted)
        self.assertEqual(result["backtracks"], 1)
        self.assertEqual(p.item(), 0)

    def test_backtracking_rolls_back_on_exception_and_nonfinite_gradients(self):
        p = nn.Parameter(torch.ones(1))
        def explode():
            raise RuntimeError("candidate check failed")
        with self.assertRaisesRegex(RuntimeError, "candidate check failed"):
            backtracked_step([p], [torch.ones_like(p)], explode, small_recovery())
        self.assertEqual(p.item(), 1)
        accepted, _ = backtracked_step([p], [torch.full_like(p, float("nan"))], explode, small_recovery())
        self.assertFalse(accepted)
        self.assertEqual(p.item(), 1)

    def test_evaluation_uses_one_trajectory_and_counts_regressions(self):
        model = ScriptedModel()
        data = batch(4)
        data["inputs"][:, 0] = torch.arange(4)
        data["labels"][-1] = -100
        result = evaluate(model, [("all", data, 3)], "cpu", [2, 7])
        self.assertEqual(model.calls, 7)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["paired_first_last"], dict(correct_correct=0, correct_wrong=1,
                                                         wrong_correct=1, wrong_wrong=1))

    def test_budget_validation_rejects_invalid_or_unknown_fields(self):
        with self.assertRaises(ValueError):
            small_recovery(loops=3)
        with self.assertRaises(ValueError):
            small_recovery(unknown_flag=True)
        with self.assertRaises(ValueError):
            TrainConfig(data_path="unused", stage="recovery")

    def test_sampler_resume_across_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_fixture(temporary)
            stream = StatefulPuzzleStream(puzzle_dataset(temporary, 2, 5, epochs_per_iter=1))
            next(stream)
            saved, rng = stream.state_dict(), np.random.get_state()
            expected = [next(stream)[1] for _ in range(5)]
            restored = StatefulPuzzleStream(puzzle_dataset(temporary, 2, 5, epochs_per_iter=1))
            restored.load_state_dict(saved)
            np.random.set_state(rng)
            actual = [next(restored)[1] for _ in range(5)]
            for first, second in zip(expected, actual):
                for key in first:
                    torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_checkpoint_resume_and_rejected_ema_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            write_fixture(root / "data")
            recipe = dict(data_path=str(root / "data"), device="cpu", seed=21,
                batch_size=2, steps=2, epochs_per_iter=2, save_every=1,
                model=dict(hidden_size=8, num_heads=2, puzzle_emb_ndim=8,
                           puzzle_emb_len=1, activation_checkpointing=True),
                recovery=small_recovery(loops=4, bridge_steps=1, teacher_steps=0, max_anchors=0))
            train(TrainConfig(**recipe, output_dir=str(root / "base")))
            checkpoint_path = root / "base/latest.pt"
            saved = read_checkpoint(checkpoint_path)
            self.assertEqual(saved["step"], 2)
            self.assertEqual(saved["ema"]["updates"], 2)
            recipe["steps"] = 3
            train(TrainConfig(**recipe, output_dir=str(root / "base"), resume=str(checkpoint_path)))
            resumed = read_checkpoint(checkpoint_path)
            train(TrainConfig(**recipe, output_dir=str(root / "uninterrupted")))
            uninterrupted = read_checkpoint(root / "uninterrupted/latest.pt")
            for name in resumed["model"]:
                torch.testing.assert_close(resumed["model"][name], uninterrupted["model"][name], rtol=0, atol=0)
            for name in resumed["ema"]["shadow"]:
                torch.testing.assert_close(resumed["ema"]["shadow"][name], uninterrupted["ema"]["shadow"][name], rtol=0, atol=0)
            # Force rejection and check that neither raw parameters nor EMA advance.
            recipe["steps"] = 1
            recipe["recovery"] = recipe["recovery"].model_copy(update={"min_objective_decrease": 100.0})
            train(TrainConfig(**recipe, stage="recovery", output_dir=str(root / "recovery"),
                              init_from=str(checkpoint_path), init_weights="ema"))
            rejected = read_checkpoint(root / "recovery/latest.pt")
            self.assertEqual(rejected["accepted_steps"], 0)
            self.assertEqual(rejected["ema"]["updates"], 0)
            weights = dict(resumed["model"], **resumed["ema"]["shadow"])
            for name in weights:
                torch.testing.assert_close(rejected["model"][name], weights[name], rtol=0, atol=0)
            self.assertFalse((root / "recovery/latest.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
