import pytest
import torch
from transformers.cache_utils import DynamicCache

from uniinfer.runtime.escalation import CascadeRun, cache_to, kv_bytes


def _cache(layers=3, L=12):
    c = DynamicCache()
    for li in range(layers):
        c.update(torch.randn(1, 2, L, 8, dtype=torch.bfloat16), torch.randn(1, 2, L, 8, dtype=torch.bfloat16), li)
    return c


def test_cache_to_cpu_is_an_independent_exact_copy():
    c = _cache()
    out = cache_to(c, "cpu")
    for a, b in zip(c.key_cache + c.value_cache, out.key_cache + out.value_cache):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    out.key_cache[0].add_(1)
    assert not torch.equal(c.key_cache[0], out.key_cache[0])


def test_kv_bytes_counts_keys_and_values():
    assert kv_bytes(_cache(layers=3, L=12)) == 2 * 3 * (1 * 2 * 12 * 8) * 2   # K and V, 3 layers, bf16


def test_cascade_costs():
    run = CascadeRun("a", True, -1.0, [5, 15], 100, 0)
    assert run.cost_max == pytest.approx(0.15) and run.cost_sum == pytest.approx(0.20)
