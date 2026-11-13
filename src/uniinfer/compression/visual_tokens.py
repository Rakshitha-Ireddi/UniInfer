"""Visual-token selection applied before the language model's prefill.

Pruned tokens are removed from the input sequence; surviving tokens keep their original
(M-RoPE) position ids, so spatial positions stay correct.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHODS = ("uniform", "dissim", "attn")


def select_visual_tokens(image_mask: torch.Tensor, visual_embeds: torch.Tensor, ratio: float,
                         method: str = "dissim", scores: torch.Tensor | None = None) -> torch.Tensor:
    """Return a boolean keep-mask over the full sequence.

    image_mask:    (L,) bool, True at image-token positions.
    visual_embeds: (n_vis, d) vision-encoder outputs, in sequence order.
    ratio:         fraction of visual tokens to keep (1.0 keeps all).
    method:
      * "uniform": evenly spaced tokens in raster order.
      * "dissim":  drop the tokens most similar to the image's mean embedding (background or redundant).
      * "attn":    FastV-style; keep the tokens with the highest `scores` (attention received at layer 2, from
                   `attention.visual_attention_scores`).
    scores:        (n_vis,) importance per visual token, required for "attn".
    Text tokens are always kept.
    """
    keep = torch.ones_like(image_mask, dtype=torch.bool)
    if ratio >= 1.0:
        return keep
    vis_pos = image_mask.nonzero().squeeze(-1)
    n = vis_pos.numel()
    k = max(1, int(round(ratio * n)))
    if method == "uniform":
        sel = torch.linspace(0, n - 1, k, device=image_mask.device).round().long().unique()
    elif method == "dissim":
        v = F.normalize(visual_embeds.float(), dim=-1)
        mean = F.normalize(v.mean(0, keepdim=True), dim=-1)
        sel = (v @ mean.T).squeeze(-1).topk(k, largest=False).indices.sort().values
    elif method == "attn":
        if scores is None or scores.numel() != n:
            raise ValueError("method 'attn' needs scores with one value per visual token")
        sel = scores.float().topk(k).indices.sort().values
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    keep[vis_pos] = False
    keep[vis_pos[sel]] = True
    return keep
