"""Bounded teacher search, neighborhood supervision and transactional SGD.

This implements finite probes, not a whole-neighborhood certificate. Nothing
from a teacher path is ever spliced into the student's real trajectory.
"""
from dataclasses import dataclass
from typing import Callable

import torch

from sgrl.config import RecoveryConfig
from sgrl.objectives import (exact_match, mean_valid, metric_norm, metric_sq,
    min_margin, project_ball, random_delta, task_loss)


@dataclass
class Collection:
    x: torch.Tensor
    anchor: torch.Tensor
    t: int
    failed: torch.Tensor
    anchor_correct: torch.Tensor
    terminal_correct: torch.Tensor  # [S+1, B], not all intermediate decodes
    baseline_loss: torch.Tensor  # [B]


@dataclass
class Corridor:
    sample: int
    q: torch.Tensor  # [m+1, 1, N, D], q_m belongs to suffix only
    suffix: torch.Tensor  # [K-t-m+1, 1, N, D]
    bridge_deltas: list[torch.Tensor]  # each [P+1, 1, N, D]
    suffix_inputs: list[torch.Tensor]  # frozen finite probe inputs
    suffix_targets: list[torch.Tensor]  # old-core outputs, detached
    teacher_loss: float


@torch.no_grad()
def collect_failures(model, core, batch, cfg: RecoveryConfig, t: int | None = None):
    if t is None:
        t = int(torch.randint(cfg.anchor_min, cfg.anchor_max + 1, ()).item())
    if not cfg.anchor_min <= t <= cfg.anchor_max:
        raise ValueError("anchor depth is outside the configured range")
    labels = batch["labels"]
    x = model.encode(batch["inputs"], batch["puzzle_identifiers"]).detach()
    z = model.initial_state(x)
    anchor = z.detach().clone() if t == 0 else None
    correctness, losses = [], []
    for depth in range(1, cfg.loops + 1):
        z = model.step(z, x, core=core)
        if depth == t:
            anchor = z.detach().clone()
        if depth >= cfg.loops - cfg.terminal_window:
            logits = model.decode(z)
            correctness.append(exact_match(logits, labels))
            losses.append(task_loss(logits, labels, cfg.loss))
    assert anchor is not None
    anchor_correct = exact_match(model.decode(anchor), labels)
    terminal_correct = torch.stack(correctness)
    failed = (~anchor_correct & ~terminal_correct.any(0) & labels.ne(-100).any(-1))
    return Collection(x, anchor, t, failed, anchor_correct,
                      terminal_correct, torch.stack(losses).mean(0))


def make_bridge(anchor, displacement, bends, mask, m):
    """Orthogonal transverse bends: U_0=U_m=0, q_0=a, q_m=a+v."""
    v = displacement * mask.unsqueeze(-1)
    bends = bends * mask.unsqueeze(-1)
    denominator = (v.square() * mask.unsqueeze(-1)).sum((-2, -1)).clamp_min(1e-12)
    dot = (bends * v * mask.unsqueeze(-1)).sum((-2, -1))
    transverse = bends - v * (dot / denominator)[..., None, None]
    middle = [anchor + (j / m) * v + transverse[j - 1] for j in range(1, m)]
    return torch.stack([anchor, *middle, anchor + v])


def native_suffix(model, core, start, x, length):
    states = [start]
    for _ in range(length):
        states.append(model.step(states[-1], x, core=core))
    return torch.stack(states)


def suffix_objective(model, suffix, labels, cfg):
    logits = [model.decode(z) for z in suffix[-(cfg.terminal_window + 1):]]
    losses = torch.stack([task_loss(logit, labels, cfg.loss) for logit in logits])
    margins = torch.stack([min_margin(logit, labels) for logit in logits])
    return losses.mean(), margins.amin()


def clearance(q, suffix, mask, cfg):
    """Explicit distances; exclude q_m=w_0 from the bridge domains."""
    threshold = cfg.bridge_radius + cfg.suffix_radius + cfg.gap
    v_size = metric_norm(q[-1] - q[0], mask).amin()
    distances = [v_size / cfg.bridge_steps]
    for j in range(cfg.bridge_steps):
        distances.extend(metric_norm(q[j] - suffix[k], mask).amin()
                         for k in range(len(suffix)))
        distances.extend(metric_norm(q[j] - q[i], mask).amin()
                         for i in range(j + 1, cfg.bridge_steps))
    distances = torch.stack(distances)
    penalty = (threshold - distances).clamp_min(0).square().mean()
    return penalty, bool((distances.detach() >= threshold).all())


def probe_deltas(center, mask, radius, cfg, objective: Callable):
    """Center + projected first-order ascent from a finite random probe set."""
    deltas = [torch.zeros_like(center, dtype=torch.float32)]
    for _ in range(cfg.probes):
        delta = random_delta(center, radius, mask).detach()
        for _ in range(cfg.probe_ascent_steps):
            delta.requires_grad_(True)
            score = objective(delta)
            if not score.requires_grad:
                delta = delta.detach()
                break
            (grad,) = torch.autograd.grad(score, delta, create_graph=False)
            if not torch.isfinite(grad).all():
                delta = delta.detach()
                break
            direction = grad / metric_norm(grad, mask)[..., None, None]
            delta = project_ball(delta + cfg.probe_step_size * direction,
                                 radius, mask).detach()
        deltas.append(delta.detach())
    return torch.stack(deltas)


def qualify_corridor(model, core, q, suffix, x, labels, mask, cfg, baseline_loss, sample):
    """Hard checks after search; small penalties alone never qualify a bridge."""
    q, suffix = q.detach(), suffix.detach()
    if not torch.isfinite(q).all() or not torch.isfinite(suffix).all():
        return None
    with torch.no_grad():
        loss, margin = suffix_objective(model, suffix, labels, cfg)
        _, clear = clearance(q, suffix, mask, cfg)
    if not (clear and torch.isfinite(loss) and margin >= cfg.min_margin
            and loss < baseline_loss - cfg.min_teacher_improvement):
        return None
    # Perturb the suffix entrance and check real native continuation, not just
    # a convenient decoded endpoint. This remains a finite empirical check.
    def terminal_loss(delta):
        perturbed = native_suffix(model, core, suffix[0] + delta, x, len(suffix) - 1)
        return suffix_objective(model, perturbed, labels, cfg)[0]

    entrance_deltas = probe_deltas(suffix[0], mask, cfg.suffix_radius, cfg, terminal_loss)
    with torch.no_grad():
        for delta in entrance_deltas:
            continued = native_suffix(model, core, suffix[0] + delta, x, len(suffix) - 1)
            perturbed_loss, perturbed_margin = suffix_objective(model, continued, labels, cfg)
            if not (torch.isfinite(continued).all() and torch.isfinite(perturbed_loss)
                    and perturbed_margin >= cfg.min_margin):
                return None

    bridge_deltas = []
    for j in range(cfg.bridge_steps):
        def bridge_error(delta, j=j):
            target = q[j + 1] + cfg.kappa * delta
            return metric_sq(model.step(q[j] + delta, x, core=core) - target, mask).mean()
        bridge_deltas.append(probe_deltas(q[j], mask, cfg.bridge_radius, cfg, bridge_error))

    suffix_inputs, suffix_targets = [], []
    for k in range(len(suffix) - 1):
        with torch.no_grad():
            center_target = model.step(suffix[k], x, core=core).detach()

        def local_change(delta, k=k, center_target=center_target):
            return metric_sq(model.step(suffix[k] + delta, x, core=core) - center_target, mask).mean()

        deltas = probe_deltas(suffix[k], mask, cfg.suffix_radius, cfg, local_change)
        with torch.no_grad():
            for delta in deltas:
                point = (suffix[k] + delta).detach()
                target = model.step(point, x, core=core).detach()
                # Finite one-step gain, not a Lipschitz certificate.
                deviation = metric_norm(target - center_target, mask)
                limit = cfg.suffix_gain_limit * metric_norm(delta, mask) + 1e-6
                if not torch.isfinite(target).all() or not (deviation <= limit).all():
                    return None
                suffix_inputs.append(point)
                suffix_targets.append(target)
    return Corridor(sample, q, suffix, bridge_deltas, suffix_inputs, suffix_targets, float(loss))


def search_teacher(model, core, collection, labels, sample, cfg):
    """Optimize v,U only. The old core/head remain parameter-frozen."""
    a = collection.anchor[sample:sample + 1].detach().float()
    x = collection.x[sample:sample + 1].detach()
    y = labels[sample:sample + 1]
    mask = model.state_mask(y)
    m = cfg.bridge_steps
    suffix_length = cfg.loops - collection.t - m
    minimum_length = m * (cfg.bridge_radius + cfg.suffix_radius + cfg.gap)
    for _ in range(cfg.teacher_restarts):
        v = (random_delta(a, minimum_length * 1.1, mask)).requires_grad_(True)
        u = torch.zeros((m - 1, *a.shape), device=a.device, dtype=torch.float32, requires_grad=True)
        parameters = [v, u] if m > 1 else [v]
        optimizer = torch.optim.Adam(parameters, lr=cfg.teacher_lr)
        for search_step in range(cfg.teacher_steps + 1):
            q = make_bridge(a, v, u, mask, m)
            suffix = native_suffix(model, core, q[-1], x, suffix_length)
            loss, margin = suffix_objective(model, suffix, y, cfg)
            geometry_loss, clear = clearance(q, suffix, mask, cfg)
            if (clear and bool((margin.detach() >= cfg.min_margin).item())
                    and bool((loss.detach() < collection.baseline_loss[sample] - cfg.min_teacher_improvement).item())):
                corridor = qualify_corridor(model, core, q, suffix, x, y, mask, cfg,
                                             collection.baseline_loss[sample], sample)
                if corridor is not None:
                    return corridor
            if search_step == cfg.teacher_steps:
                break
            dynamics = torch.stack([metric_sq(q[j + 1] - model.step(q[j], x, core=core), mask).mean()
                                    for j in range(m)]).mean()
            objective = loss + cfg.teacher_dynamics_weight * dynamics + cfg.teacher_clearance_weight * geometry_loss
            if not torch.isfinite(objective):
                break
            optimizer.zero_grad(set_to_none=True)
            # autograd.grad prevents accidental accumulation on student params.
            grads = torch.autograd.grad(objective, parameters, create_graph=False)
            if not all(torch.isfinite(grad).all() for grad in grads):
                break
            for parameter, grad in zip(parameters, grads):
                parameter.grad = grad
            torch.nn.utils.clip_grad_norm_(parameters, cfg.teacher_grad_clip)
            optimizer.step()
            with torch.no_grad():
                v.copy_(project_ball(v, cfg.displacement_limit, mask))
                u.copy_(project_ball(u, cfg.displacement_limit, mask))
    return None


def real_objective(model, x, anchor, labels, t, cfg):
    """All examples, from old natural anchors; no teacher-state insertion."""
    z = anchor.detach()
    horizon = cfg.loops - t
    losses = []
    for k in range(1, horizon + 1):
        z = model.step(z, x)
        if k >= horizon - cfg.terminal_window:
            losses.append(task_loss(model.decode(z), labels, cfg.loss))
    return mean_valid(torch.stack(losses).mean(0), labels), z


def proposal_objective(model, collection, labels, corridors, cfg):
    real, final_state = real_objective(model, collection.x, collection.anchor, labels, collection.t, cfg)
    bridge = real.new_zeros(())
    for corridor in corridors:
        i = corridor.sample
        x = collection.x[i:i + 1]
        mask = model.state_mask(labels[i:i + 1])
        for j, deltas in enumerate(corridor.bridge_deltas):
            errors = []
            for delta in deltas:
                prediction = model.step(corridor.q[j] + delta, x)
                target = corridor.q[j + 1] + cfg.kappa * delta
                errors.append(metric_sq(prediction - target, mask).mean())
            bridge = bridge + torch.stack(errors).max()
    bridge = bridge / labels.ne(-100).any(-1).sum().clamp_min(1)
    return real + cfg.bridge_weight * bridge, real, bridge, final_state


@torch.no_grad()
def preservation_error(model, collection, labels, corridors):
    maximum = 0.0
    for corridor in corridors:
        i = corridor.sample
        x = collection.x[i:i + 1]
        mask = model.state_mask(labels[i:i + 1])
        for point, target in zip(corridor.suffix_inputs, corridor.suffix_targets):
            difference = metric_sq(model.step(point, x) - target, mask).sqrt().amax().item()
            if not torch.isfinite(torch.tensor(difference)):
                return float("inf")
            maximum = max(maximum, difference)
    return maximum


def backtracked_step(parameters, gradients, check, cfg):
    """Stateless SGD. Restore all parameters on rejection or any exception.

    check() returns (passes, metrics). No optimizer/EMA state is mutated here.
    """
    parameters, gradients = list(parameters), list(gradients)
    original = [parameter.detach().clone() for parameter in parameters]
    accepted = False
    last = {}
    attempts = 0
    try:
        if not all(torch.isfinite(g).all() for g in gradients):
            return False, {"reason": "nonfinite_gradient", "backtracks": 0}
        for b in range(cfg.backtrack_attempts):
            attempts = b + 1
            rate = cfg.learning_rate * cfg.backtrack_factor ** b
            with torch.no_grad():
                for parameter, initial, grad in zip(parameters, original, gradients):
                    parameter.copy_(initial - rate * grad)
                passes, last = check()
            if passes:
                accepted = True
                return True, {**last, "learning_rate": rate, "backtracks": b}
        return False, {**last, "reason": "all_candidates_rejected", "backtracks": attempts}
    finally:
        if not accepted:
            with torch.no_grad():
                for parameter, initial in zip(parameters, original):
                    parameter.copy_(initial)


def recovery_step(model, batch, cfg: RecoveryConfig):
    if not model.recovery_mode:
        raise ValueError("freeze the encoder and readout with set_recovery_mode() first")
    model.zero_grad(set_to_none=True)
    snapshot = model.frozen_core()
    collection = collect_failures(model, snapshot, batch, cfg)
    failed_indices = collection.failed.nonzero(as_tuple=False).flatten()
    order = torch.randperm(len(failed_indices), device=failed_indices.device)
    selected = failed_indices[order[:cfg.max_anchors]].tolist()
    corridors = []
    for sample in selected:
        corridor = search_teacher(model, snapshot, collection, batch["labels"], sample, cfg)
        if corridor is not None:
            corridors.append(corridor)
    labels = batch["labels"]
    objective, real, bridge, before_state = proposal_objective(model, collection, labels, corridors, cfg)
    with torch.no_grad():
        before_correct = exact_match(model.decode(before_state), labels)
    initial_objective, initial_real = float(objective.detach()), float(real.detach())
    metrics = {"anchor_depth": collection.t, "failed": int(collection.failed.sum()),
               "teacher_attempted": len(selected), "teacher_qualified": len(corridors),
               "objective_before": initial_objective, "real_loss_before": initial_real,
               "bridge_loss": float(bridge.detach()), "count": int(labels.ne(-100).any(-1).sum())}
    if not torch.isfinite(objective):
        return {**metrics, "accepted": False, "reason": "nonfinite_objective"}
    objective.backward()
    parameters = [parameter for parameter in model.core.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
    gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]

    def check():
        drift = preservation_error(model, collection, labels, corridors)
        if drift > cfg.preservation_tolerance:
            return False, {"preservation_error": drift}
        candidate, candidate_real, _, final_state = proposal_objective(model, collection, labels, corridors, cfg)
        after_correct = exact_match(model.decode(final_state), labels)
        valid = labels.ne(-100).any(-1)
        improves = bool(torch.isfinite(candidate)
                        and candidate < initial_objective - cfg.min_objective_decrease
                        and candidate_real <= initial_real + cfg.real_loss_slack)
        return improves, {"objective_after": float(candidate), "real_loss_after": float(candidate_real),
            "preservation_error": drift,
            "wrong_to_correct": int((~before_correct & after_correct & valid).sum()),
            "correct_to_wrong": int((before_correct & ~after_correct & valid).sum()),
            "correct_after": int(after_correct.sum())}

    accepted, update_metrics = backtracked_step(parameters, gradients, check, cfg)
    if not accepted:
        # Rejected candidate results must not masquerade as deployed-model results.
        update_metrics = {"candidate_" + key: value for key, value in update_metrics.items()}
        update_metrics.update(objective_after=initial_objective, real_loss_after=initial_real,
                              wrong_to_correct=0, correct_to_wrong=0,
                              correct_after=int(before_correct.sum()))
    model.zero_grad(set_to_none=True)
    return {**metrics, **update_metrics, "accepted": accepted, "grad_norm": float(grad_norm)}
