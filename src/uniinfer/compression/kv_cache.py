"""Post-prefill KV-cache compression.

L2-norm eviction (Devoto et al., 2024, "A Simple and Effective L2 Norm-Based Strategy for KV Cache
Compression"): keys with a LOW L2 norm tend to receive high attention, so they are kept. The method is
attention-free, so it works with SDPA/flash kernels. RoPE is a rotation, so key norms are unchanged by it.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

KEEP_FIRST = 4  # attention-sink tokens
KEEP_LAST = 16  # question tail and generation prompt


def _select(keys: torch.Tensor, n_keep: int, keep_first: int, keep_last: int) -> torch.Tensor:
    L = keys.shape[-2]
    score = keys.float().norm(dim=-1).mean(dim=(0, 1))  # (L,), averaged over batch and kv heads
    score[:keep_first] = -float("inf")
    score[max(0, L - keep_last) :] = -float("inf")
    return score.topk(n_keep, largest=False).indices.sort().values


def compress_kv_l2norm(
    cache: DynamicCache, ratio: float, keep_first: int = KEEP_FIRST, keep_last: int = KEEP_LAST
) -> DynamicCache:
    """Return a NEW cache keeping `ratio` of the tokens in every layer (same count per layer).

    The input cache is not modified, so several ratios can be applied to one prefill.
    """
    L = cache.key_cache[0].shape[-2]
    n_keep = L if ratio >= 1.0 else min(L, max(keep_first + keep_last, int(round(ratio * L))))
    out = DynamicCache()
    for li, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        if n_keep >= L:
            out.update(k.clone(), v.clone(), li)
        else:
            idx = _select(k, n_keep, keep_first, keep_last)
            out.update(k[:, :, idx].contiguous(), v[:, :, idx].contiguous(), li)
    return out


def compress_kv_recent(cache: DynamicCache, ratio: float, keep_first: int = 4) -> DynamicCache:
    """Copy attention sinks and the most recent tokens within an exact token budget.

    Keep round(ratio * L) tokens per layer, prioritizing the first keep_first tokens
    and filling the remaining budget from the tail. Even a full budget returns
    independent tensors so the original prefill cache can be reused safely.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    if not isinstance(keep_first, int) or keep_first < 0:
        raise ValueError("keep_first must be a nonnegative integer")

    out = DynamicCache()
    for li, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        L = k.shape[-2]
        n_keep = round(ratio * L)
        if n_keep == L:
            out.update(k.clone(), v.clone(), li)
        else:
            n_first = min(keep_first, n_keep)
            tail_start = L - (n_keep - n_first)
            # tail_start >= n_first: prefix and tail never overlap, including at zero budget.
            keys = torch.cat((k[..., :n_first, :], k[..., tail_start:, :]), dim=-2)
            values = torch.cat((v[..., :n_first, :], v[..., tail_start:, :]), dim=-2)
            out.update(keys, values, li)
    return out
