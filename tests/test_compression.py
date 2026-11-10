import pytest
import torch

from uniinfer.compression import compress_kv_l2norm, select_visual_tokens

transformers = pytest.importorskip("transformers")
from transformers.cache_utils import DynamicCache  # noqa: E402


def _mask(n_text_before=3, n_vis=20, n_text_after=5):
    return torch.tensor([False] * n_text_before + [True] * n_vis + [False] * n_text_after)


@pytest.mark.parametrize("method", ["uniform", "dissim"])
@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_visual_selection_keeps_ratio_and_all_text(method, ratio):
    mask = _mask()
    vis = torch.randn(int(mask.sum()), 8)
    keep = select_visual_tokens(mask, vis, ratio, method)
    assert keep[~mask].all(), "text tokens must never be dropped"
    assert int(keep[mask].sum()) == round(ratio * mask.sum().item())


def test_visual_selection_full_ratio_is_identity():
    mask = _mask()
    assert select_visual_tokens(mask, torch.randn(20, 8), 1.0).all()


def test_dissim_drops_redundant_tokens():
    mask = _mask(0, 10, 0)
    vis = torch.ones(10, 4)
    vis[3] = torch.tensor([1.0, -1.0, 1.0, -1.0])  # the one distinctive token
    keep = select_visual_tokens(mask, vis, 0.1, "dissim")
    assert keep.nonzero().squeeze().item() == 3


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        select_visual_tokens(_mask(), torch.randn(20, 8), 0.5, "nope")


def _cache(layers=3, heads=2, L=100, d=8):
    c = DynamicCache()
    torch.manual_seed(0)
    for li in range(layers):
        c.update(torch.randn(1, heads, L, d), torch.randn(1, heads, L, d), li)
    return c


def test_kv_compression_budget_and_protected_tokens():
    c = _cache()
    out = compress_kv_l2norm(c, 0.5)
    for li in range(3):
        assert out.key_cache[li].shape[-2] == 50
        k_full, k_new = c.key_cache[li], out.key_cache[li]
        assert torch.equal(k_new[:, :, :4], k_full[:, :, :4]), "attention sinks kept"
        assert torch.equal(k_new[:, :, -16:], k_full[:, :, -16:]), "question tail kept"


def test_kv_compression_keeps_low_norm_keys():
    c = DynamicCache()
    k = torch.ones(1, 1, 40, 4)
    k[0, 0, 10] *= 0.01  # tiny norm, so it must be kept
    k[0, 0, 11] *= 100   # huge norm, so it must be evicted
    c.update(k, k.clone(), 0)
    out = compress_kv_l2norm(c, 0.6)
    kept_norms = out.key_cache[0].norm(dim=-1).flatten()
    assert kept_norms.min() < 0.1 and kept_norms.max() < 50


def test_kv_compression_does_not_mutate_input():
    c = _cache()
    before = [k.clone() for k in c.key_cache]
    compress_kv_l2norm(c, 0.25)
    assert all(torch.equal(a, b) for a, b in zip(before, c.key_cache))


def test_kv_full_ratio_is_copy():
    c = _cache()
    out = compress_kv_l2norm(c, 1.0)
    assert torch.equal(out.key_cache[0], c.key_cache[0])
    assert out.key_cache[0].data_ptr() != c.key_cache[0].data_ptr()
