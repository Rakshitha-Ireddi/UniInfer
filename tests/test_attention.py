import math

import pytest
import torch
from transformers.cache_utils import DynamicCache

from uniinfer.compression import (
    compress_kv_pyramid,
    compress_kv_snapkv,
    kv_tokens,
    pyramid_budgets,
    select_visual_tokens,
    snapkv_scores,
    window_attention,
)
from uniinfer.compression.attention import PYRAMID_BETA, WINDOW


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


def test_pyramid_budgets_keep_the_average_and_decrease():
    b = pyramid_budgets(36, 1000, 0.1)
    assert sum(b) == 36 * 100                       # same average as SnapKV at 10%
    assert all(x >= y for x, y in zip(b, b[1:]))    # bottom layers keep more
    assert b[-1] == 16 and b[0] > 2 * 100 - 20      # top = max(window, avg / 20), bottom ~ 2 * avg - top
    capped = pyramid_budgets(36, 300, 0.75)         # bottom would exceed L: the cut is redistributed
    assert max(capped) == 300 and sum(capped) == round(0.75 * 300 * 36)
    assert pyramid_budgets(4, 50, 1.0) == [50] * 4


def test_pyramid_cache_has_per_layer_lengths_and_average_cost():
    layers, Hkv, L, D = 6, 2, 200, 8
    cache = _cache(layers, Hkv, L, D)
    queries = {li: torch.randn(1, 4, 16, D) for li in range(layers)}
    out = compress_kv_pyramid(cache, queries, ratio=0.25)
    lens = [k.shape[-2] for k in out.key_cache]
    assert lens == pyramid_budgets(layers, L, 0.25) and len(set(lens)) > 1
    assert kv_tokens(out) == pytest.approx(0.25 * L)
    assert kv_tokens(compress_kv_snapkv(cache, queries, 0.25)) == 50


def test_flat_pyramid_is_snapkv():
    """beta = 1 makes every layer's budget equal, so PyramidKV must reduce to SnapKV exactly."""
    cache = _cache(4, 2, 120, 8, seed=3)
    queries = {li: torch.randn(1, 4, 16, 8) for li in range(4)}
    a = compress_kv_pyramid(cache, queries, ratio=0.3, beta=1.0)
    b = compress_kv_snapkv(cache, queries, ratio=0.3)
    assert all(torch.equal(x, y) for x, y in zip(a.key_cache, b.key_cache))
    assert all(torch.equal(x, y) for x, y in zip(a.value_cache, b.value_cache))


@pytest.mark.parametrize("compress", [compress_kv_snapkv, compress_kv_pyramid])
@pytest.mark.parametrize("ratio", [0.01, 0.05, 0.25, 0.75])
def test_kv_compression_always_keeps_observation_window_in_order(compress, ratio):
    """Every layer and KV head keeps the last WINDOW tokens and preserves original sequence order."""
    layers, Hkv, H, L, D = 6, 2, 4, 400, 16
    cache = _cache(layers, Hkv, L, D, seed=7)
    g = torch.Generator().manual_seed(11)
    queries = {li: torch.randn(1, H, WINDOW, D, generator=g) for li in range(layers)}

    out = compress(cache, queries, ratio=ratio)

    assert [k.shape[-2] for k in cache.key_cache] == [L] * layers
    for li in range(layers):
        for h in range(Hkv):
            orig_k = cache.key_cache[li][0, h]
            kept_k = out.key_cache[li][0, h]
            if ratio * L < WINDOW:
                assert kept_k.shape[0] == WINDOW
            else:
                assert kept_k.shape[0] >= WINDOW
            matches = (kept_k[:, None, :] == orig_k[None, :, :]).all(dim=-1)
            assert (matches.sum(dim=-1) == 1).all()
            orig_idx = matches.int().argmax(dim=-1)
            assert (orig_idx[1:] > orig_idx[:-1]).all()
            assert orig_idx[-WINDOW:].tolist() == list(range(L - WINDOW, L))
            assert torch.equal(kept_k[-WINDOW:], orig_k[-WINDOW:])
            assert torch.equal(out.value_cache[li][0, h], cache.value_cache[li][0, h, orig_idx])


def test_pyramid_top_layer_floor_and_average_preservation():
    """Top layer keeps exactly WINDOW tokens when avg / beta < WINDOW (uncapped), and average stays within 1."""
    for n_layers in (4, 6, 36):
        for L, ratio in ((200, 0.01), (200, 0.05), (400, 0.05), (400, 0.25), (400, 0.5), (1000, 0.25)):
            avg = max(WINDOW, ratio * L)
            assert avg / PYRAMID_BETA < WINDOW
            assert pyramid_budgets(n_layers, L, ratio)[-1] == WINDOW

    for n_layers in (3, 6, 36):
        for L in (101, 200, 333, 400, 1000):
            for ratio in (0.05, 0.1, 0.25, 0.5, 0.75):
                if ratio * L >= WINDOW:
                    budgets = pyramid_budgets(n_layers, L, ratio)
                    assert abs(sum(budgets) / len(budgets) - ratio * L) <= 1.0


def test_pyramid_cache_top_layer_floor_and_average():
    """compress_kv_pyramid hits the WINDOW floor on the top layer and preserves average cost within 1 token."""
    layers, Hkv, H, L, D = 6, 2, 4, 400, 16
    cache = _cache(layers, Hkv, L, D, seed=13)
    g = torch.Generator().manual_seed(17)
    queries = {li: torch.randn(1, H, WINDOW, D, generator=g) for li in range(layers)}

    floor_out = compress_kv_pyramid(cache, queries, ratio=0.01)
    assert [k.shape[-2] for k in floor_out.key_cache] == [WINDOW] * layers

    for ratio in (0.05, 0.25):
        out = compress_kv_pyramid(cache, queries, ratio=ratio)
        assert out.key_cache[-1].shape[-2] == WINDOW
        assert abs(kv_tokens(out) - ratio * L) <= 1.0

    capped_out = compress_kv_pyramid(cache, queries, ratio=0.75)
    assert abs(kv_tokens(capped_out) - 0.75 * L) <= 1.0
