import math

import pytest
import torch

from uniinfer.signals import answer_confidence, geometric_mean_prob, token_confidence


def test_token_confidence_peaked_vs_flat():
    peaked = token_confidence(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    flat = token_confidence(torch.zeros(4))
    assert peaked["prob"] > 0.99 and peaked["margin"] > 0.98
    assert flat["prob"] == pytest.approx(0.25) and flat["margin"] == pytest.approx(0.0)
    assert flat["entropy"] == pytest.approx(math.log(4))
    assert peaked["entropy"] < flat["entropy"]
    assert peaked["logprob"] == pytest.approx(math.log(peaked["prob"]), rel=1e-3)  # float32 log-softmax


def test_token_confidence_handles_bf16_and_2d():
    c = token_confidence(torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.bfloat16))
    assert 0 < c["prob"] <= 1 and c["margin"] >= 0


def test_answer_confidence():
    a = answer_confidence([-0.1, -2.0, -0.3])
    assert a["min_logprob"] == -2.0
    assert a["mean_logprob"] == pytest.approx(-0.8)
    assert a["sum_logprob"] == pytest.approx(-2.4)
    assert geometric_mean_prob([-0.1, -2.0, -0.3]) == pytest.approx(math.exp(-0.8))


def test_answer_confidence_empty_is_nan():
    assert math.isnan(answer_confidence([])["mean_logprob"])
