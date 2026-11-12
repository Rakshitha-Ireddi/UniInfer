"""Confidence signals computed from logits the model already produces (zero extra forward passes).

Used by the confidence cascade (ADR-0008): answer with a cheap configuration, and escalate when the
model is unsure *under that compression*.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def token_confidence(logits: torch.Tensor) -> dict[str, float]:
    """Confidence of the greedy choice from one next-token logit vector.

    Returns:
      prob:    probability of the argmax token
      margin:  prob(top1) - prob(top2)
      entropy: Shannon entropy of the distribution (nats)
      logprob: log-probability of the argmax token
    """
    logp = torch.log_softmax(logits.float().flatten(), dim=-1)
    top = logp.topk(2).values
    p1, p2 = top[0].exp().item(), top[1].exp().item()
    entropy = -(logp.exp() * logp).sum().item()
    return {"prob": p1, "margin": p1 - p2, "entropy": entropy, "logprob": top[0].item()}


def answer_confidence(token_logprobs: Sequence[float]) -> dict[str, float]:
    """Sequence-level confidence from the greedy tokens' log-probs (EOS included when generated)."""
    if not token_logprobs:
        return {"mean_logprob": float("nan"), "min_logprob": float("nan"), "sum_logprob": float("nan")}
    s = float(sum(token_logprobs))
    return {"mean_logprob": s / len(token_logprobs), "min_logprob": float(min(token_logprobs)),
            "sum_logprob": s}


def geometric_mean_prob(token_logprobs: Sequence[float]) -> float:
    return math.exp(answer_confidence(token_logprobs)["mean_logprob"]) if token_logprobs else float("nan")
