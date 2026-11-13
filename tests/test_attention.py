import math

import pytest
import torch
from transformers.cache_utils import DynamicCache

from uniinfer.compression import (
    compress_kv_snapkv,
    select_visual_tokens,
    snapkv_scores,
    window_attention,
)


def _naive(q, k):
    """Reference: full causal attention for the last W rows, computed row by row."""
    B, H, W, D = q.shape
    L = k.shape[2]
    kk = k.repeat_interleave(H // k.shape[1], 1)
    out = torch.zeros(B, H, W, L)
    for w in range(W):
        pos = L - W + w
        s = (q[:, :, w : w + 1] @ kk[:, :, : pos + 1].transpose(-1, -2))[:, :, 0] / math.sqrt(D)
        out[:, :, w, : pos + 1] = s.softmax(-1)
    return out


def test_window_attention_matches_naive_causal_gqa():
    torch.manual_seed(0)
    q, k = torch.randn(1, 8, 5, 16), torch.randn(1, 2, 30, 16)   # 4 query heads per KV head
    p = window_attention(q, k)
    assert torch.allclose(p, _naive(q, k), atol=1e-6)
    assert torch.allclose(p.sum(-1), torch.ones(1, 8, 5))
    assert (p[0, :, 0, 26:] == 0).all()   # first window row (position 25) cannot see positions > 25


def test_snapkv_scores_group_mean_and_pool():
    probs = torch.zeros(1, 4, 2, 10)
    probs[0, 0, :, 3] = 1.0                       # one query head of KV-group 0 looks at token 3
    s = snapkv_scores(probs, n_kv_heads=2, pool=1)
    assert s.shape == (1, 2, 10)
    assert s[0, 0, 3] == pytest.approx(1.0) and s[0, 1].abs().sum() == 0   # (2 rows * 1.0) / 2 heads in group
    pooled = snapkv_scores(probs, n_kv_heads=2, pool=3)
    assert pooled[0, 0, 2] == pooled[0, 0, 4] == pytest.approx(1.0)        # neighbours inherit the max


def _cache(layers, H, L, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = DynamicCache()
    for li in range(layers):
        c.update(torch.randn(1, H, L, D, generator=g), torch.randn(1, H, L, D, generator=g), li)
    return c


def test_snapkv_keeps_window_and_attended_tokens():
    torch.manual_seed(1)
    layers, Hkv, L, D, W = 2, 2, 64, 16, 16
    cache = _cache(layers, Hkv, L, D)
    base = torch.randn(D)
    q = base + 0.1 * torch.randn(1, 4, W, D)      # every window query points roughly along `base`
    planted = [5, 20, 33]
    for li in range(layers):                       # planted prefix keys align with the queries
        for t in planted:
            cache.key_cache[li][0, :, t] = 3 * base
    queries = {li: q for li in range(layers)}
    out = compress_kv_snapkv(cache, queries, ratio=0.5, window=W, pool=1)
    n_keep = 32
    for li in range(layers):
        assert out.key_cache[li].shape == (1, Hkv, n_keep, D)
        for h in range(Hkv):
            kept = {tuple(r.tolist()) for r in out.key_cache[li][0, h]}
            for t in list(range(L - W, L)) + planted:
                assert tuple(cache.key_cache[li][0, h, t].tolist()) in kept
    assert cache.key_cache[0].shape[-2] == L   # input cache untouched


def test_snapkv_ratio_one_is_identity_copy():
    cache = _cache(2, 2, 20, 8)
    out = compress_kv_snapkv(cache, {}, ratio=1.0)
    assert all(torch.equal(a, b) for a, b in zip(out.key_cache, cache.key_cache))
    assert out.key_cache[0] is not cache.key_cache[0]


def test_attn_visual_selection_uses_scores():
    mask = torch.tensor([False, True, True, True, True, False])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    keep = select_visual_tokens(mask, torch.zeros(4, 3), 0.5, "attn", scores)
    assert keep.tolist() == [True, False, True, False, True, True]   # top-2 scores: image tokens 2 and 4
    with pytest.raises(ValueError):
        select_visual_tokens(mask, torch.zeros(4, 3), 0.5, "attn")
