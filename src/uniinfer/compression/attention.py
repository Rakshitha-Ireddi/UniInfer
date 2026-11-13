"""Attention-guided compression (SnapKV, FastV-style) on top of SDPA prefill.

SDPA does not return attention weights. Instead, a read-only hook stores the RoPE'd queries of the last
`window` prompt positions in every layer during prefill. The rotated keys are already in the KV cache, so
the attention those positions paid to every earlier token can be rebuilt exactly, at a cost of
O(window · L) per layer. Model outputs are unchanged (the hook only reads). `scripts/check_attention.py`
verifies the reconstruction against the model's own attention output.

* SnapKV (Li et al., 2024, "SnapKV: LLM Knows What You are Looking for Before Generation"): per layer and per
  KV head, keep the prefix tokens with the highest pooled attention from the observation window, plus the
  window itself.
* FastV-style (Chen et al., 2024, "An Image is Worth 1/2 Tokens After Layer 2"): rank image tokens by the
  attention they receive at layer 2. FastV drops them inside the forward pass after layer 2. Here they are
  dropped *before* prefill, so every layer's KV shrinks. That matches the KV-memory cost model (ADR-0005).
  The scores come from layers 0-2 only, so a deployment needs a 3-layer shallow pass (about 8% of a prefill
  for a 36-layer model).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

WINDOW = 16       # observation window: question tail and generation prompt (matches kv_cache.KEEP_LAST)
POOL = 7          # SnapKV max-pool kernel over the token axis
FASTV_LAYER = 2   # FastV's default K


def window_attention(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Attention probabilities of the last W positions over all L keys, causal.

    q: (B, H, W, D) RoPE'd queries of the last W positions.  k: (B, Hkv, L, D) RoPE'd keys.
    Returns (B, H, W, L) float32 probabilities (rows sum to 1).
    """
    B, H, W, D = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    kk = k.float().repeat_interleave(H // Hkv, dim=1)
    s = q.float() @ kk.transpose(-1, -2) / math.sqrt(D)                         # (B, H, W, L)
    qpos = torch.arange(L - W, L, device=q.device)[:, None]
    s = s.masked_fill(torch.arange(L, device=q.device)[None, :] > qpos, float("-inf"))
    return s.softmax(-1)


def snapkv_scores(probs: torch.Tensor, n_kv_heads: int, pool: int = POOL) -> torch.Tensor:
    """(B, H, W, L) probabilities -> (B, Hkv, L) importance: sum over the window, mean within each GQA group,
    then max-pooling along tokens (keeps neighbours of important tokens, as in SnapKV)."""
    B, H, W, L = probs.shape
    s = probs.sum(2).view(B, n_kv_heads, H // n_kv_heads, L).mean(2)
    if pool > 1:
        s = F.max_pool1d(s.reshape(B * n_kv_heads, 1, L), pool, stride=1, padding=pool // 2).view(B, n_kv_heads, L)
    return s


def compress_kv_snapkv(cache: DynamicCache, queries: dict, ratio: float, window: int = WINDOW,
                       pool: int = POOL) -> DynamicCache:
    """Return a NEW cache keeping `ratio` of the tokens in every layer (same count per layer and head).

    Each KV head keeps the whole observation window plus its own top prefix tokens, so different heads may
    keep different tokens. Positions are already baked into the rotated keys, so no re-indexing is needed.
    """
    L = cache.key_cache[0].shape[-2]
    n_keep = L if ratio >= 1.0 else min(L, max(window, int(round(ratio * L))))
    out = DynamicCache()
    for li, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        if n_keep >= L:
            out.update(k.clone(), v.clone(), li)
            continue
        Hkv = k.shape[1]
        s = snapkv_scores(window_attention(queries[li], k), Hkv, pool)          # (B, Hkv, L)
        s[..., L - window:] = float("inf")                                      # always keep the window
        idx = s.topk(n_keep, dim=-1).indices.sort(-1).values                    # (B, Hkv, n_keep)
        g = idx[..., None].expand(-1, -1, -1, k.shape[-1])
        out.update(k.gather(2, g).contiguous(), v.gather(2, g).contiguous(), li)
    return out


def visual_attention_scores(cache: DynamicCache, queries: dict, image_mask: torch.Tensor,
                            layer: int = FASTV_LAYER) -> torch.Tensor:
    """Attention each image token receives at `layer` from the observation window, averaged over heads and
    window positions. Needs a prefill over the FULL sequence. Returns (n_vis,) in sequence order."""
    p = window_attention(queries[layer], cache.key_cache[layer])               # (1, H, W, L)
    return p.mean(dim=(0, 1, 2))[image_mask]


class WindowQueryCapture:
    """Context manager: during a prefill, store the RoPE'd queries of the last `window` positions per layer.

    Read-only forward pre-hooks on every self-attention module. Single-token decode steps are ignored.
        with WindowQueryCapture(model) as cap:
            model(...)                      # prefill
        cap.queries[layer_idx]              # (1, H, window, D)
    """

    def __init__(self, model, window: int = WINDOW):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

        self._rope = apply_multimodal_rotary_pos_emb
        self.layers = model.model.layers
        self.window = window
        self.queries: dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook(self, mod, args, kwargs):
        h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        if h.shape[1] < self.window:  # decode step or tiny input
            return
        W = self.window
        q = mod.q_proj(h[:, -W:]).view(h.shape[0], W, -1, mod.head_dim).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, _ = self._rope(q, q, cos[..., -W:, :], sin[..., -W:, :], mod.rope_scaling["mrope_section"])
        self.queries[mod.layer_idx] = q.detach()

    def __enter__(self):
        self.queries = {}
        self._handles = [layer.self_attn.register_forward_pre_hook(self._hook, with_kwargs=True)
                         for layer in self.layers]
        return self

    def __exit__(self, *exc):
        for hd in self._handles:
            hd.remove()
        self._handles = []
        return False
