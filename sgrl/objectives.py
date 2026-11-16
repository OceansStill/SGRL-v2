"""Masked per-example task objectives and the coordinate-mean state metric."""
import torch

from models.losses import stablemax_cross_entropy, softmax_cross_entropy


def task_loss(logits, labels, kind="stablemax"):
    fn = stablemax_cross_entropy if kind == "stablemax" else softmax_cross_entropy
    per_token = fn(logits, labels)
    return per_token.sum(-1) / labels.ne(-100).sum(-1).clamp_min(1)


def exact_match(logits, labels):
    valid = labels.ne(-100)
    return ((logits.argmax(-1) == labels) | ~valid).all(-1) & valid.any(-1)


def min_margin(logits, labels):
    valid = labels.ne(-100)
    indices = labels.masked_fill(~valid, 0).long().unsqueeze(-1)
    correct = logits.gather(-1, indices).squeeze(-1)
    alternatives = logits.scatter(-1, indices, -torch.inf).amax(-1)
    margins = (correct - alternatives).masked_fill(~valid, torch.inf)
    return margins.amin(-1).masked_fill(~valid.any(-1), -torch.inf)


def mean_valid(values, labels):
    valid = labels.ne(-100).any(-1)
    return (values * valid).sum() / valid.sum().clamp_min(1)


def metric_sq(value, mask):
    """RMS squared; last axes are [tokens, channels], mask is [batch,tokens]."""
    value = value.float()
    numerator = (value.square() * mask.unsqueeze(-1)).sum((-2, -1))
    denominator = mask.sum(-1).clamp_min(1) * value.shape[-1]
    return numerator / denominator


def metric_norm(value, mask):
    # Clamp before sqrt to keep zero-distance clearance derivatives finite.
    return metric_sq(value, mask).clamp_min(1e-12).sqrt()


def project_ball(delta, radius, mask):
    delta = delta * mask.unsqueeze(-1)
    scale = (radius / metric_norm(delta, mask)).clamp(max=1)
    return delta * scale[..., None, None]


def random_delta(reference, radius, mask):
    delta = torch.randn_like(reference, dtype=torch.float32) * mask.unsqueeze(-1)
    return delta * (radius / metric_norm(delta, mask))[..., None, None]
