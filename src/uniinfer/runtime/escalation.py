"""Cheap escalation for KV-cache cascades (v0.3 groundwork).

A two-stage cascade answers with a small KV budget and retries with a larger one when the answer's
confidence is low. Done naively, the retry re-runs the vision encoder and the prefill. Because KV compression
is applied *after* prefill, both caches can come from ONE prefill:

* ``offload``: after the prefill, build the escalation cache (larger ratio) and park it in pinned CPU memory,
  build the cheap cache on the GPU, and free the full prefill cache. On escalation, free the cheap cache and
  copy the parked cache back over PCIe. GPU-resident decode KV never exceeds the stage being decoded.
* ``reprefill``: the naive baseline. Escalation re-runs encode + prefill + compression.
* ``static``: one config, no cascade.

All three produce, per stage, exactly the answer of the corresponding single config (same prefill, same
compression, same greedy decode), so only cost and latency differ. `check_escalation` verifies this.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from ..signals import answer_confidence


def cache_to(cache: DynamicCache, device: torch.device | str) -> DynamicCache:
    """Copy of `cache` on `device`. CPU copies are pinned so the copy back to the GPU is fast."""
    out = DynamicCache()
    pin = torch.device(device).type == "cpu" and torch.cuda.is_available()
    for li, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        if pin:
            kk = torch.empty(k.shape, dtype=k.dtype, pin_memory=True).copy_(k)
            vv = torch.empty(v.shape, dtype=v.dtype, pin_memory=True).copy_(v)
        else:
            kk, vv = k.to(device), v.to(device)
        out.update(kk, vv, li)
    return out


def kv_bytes(cache: DynamicCache) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in zip(cache.key_cache, cache.value_cache))


@dataclass
class CascadeRun:
    text: str
    escalated: bool
    confidence: float                       # min token log-prob of the first-stage answer
    kv_tokens: list                         # per-layer-average KV tokens decoded at each stage
    full_kv_tokens: int
    peak_decode_kv_bytes: int               # largest GPU-resident KV during any decode
    offloaded_bytes: int = 0
    timings: dict = field(default_factory=dict)

    @property
    def cost_max(self) -> float:            # memory freed before the retry (ADR-0008 "max" model)
        return max(self.kv_tokens) / self.full_kv_tokens

    @property
    def cost_sum(self) -> float:            # every attempted stage is paid for
        return sum(self.kv_tokens) / self.full_kv_tokens


class _Clock:
    def __init__(self, runner):
        self.r, self.t = runner, {}

    def __call__(self, name):
        clock = self

        class _C:
            def __enter__(self):
                clock.r._sync()
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                clock.r._sync()
                clock.t[name] = clock.t.get(name, 0.0) + time.perf_counter() - self.t0
        return _C()


@torch.inference_mode()
def run_kv_cascade(runner, image, prompt: str, max_new_tokens: int, r1: float, r2: float | None,
                   threshold: float = float("-inf"), mode: str = "offload") -> CascadeRun:
    """Answer with SnapKV at ratio r1; if min log-prob < threshold, escalate to ratio r2 (full visual tokens).

    r2=None (or mode='static') runs the single config r1. Visual tokens are never pruned here, so the
    escalation needs no new prefill.
    """
    from ..compression import kv_tokens

    clk = _Clock(runner)
    with clk("encode"):
        enc = runner.encode(image, prompt)
    keep = torch.ones_like(enc.image_mask)
    with clk("prefill"):
        pre = runner.prefill(enc, keep, capture=True)
    full_tokens = pre.cache.key_cache[0].shape[-2]
    parked = None
    with clk("compress"):                    # includes the offload copy, also reported on its own
        if mode == "offload" and r2 is not None:
            c2 = runner.compress(pre, "snapkv", r2)
            with clk("offload"):
                parked = cache_to(c2, "cpu")
            del c2
        c1 = runner.compress(pre, "snapkv", r1)
    pre.cache, pre.queries = None, None     # free the full prefill KV before decoding
    tokens, peak = [kv_tokens(c1)], kv_bytes(c1)
    with clk("decode"):
        dec = runner.decode(pre, c1, enc.next_position, max_new_tokens)
    conf = answer_confidence(dec.token_logprobs)["min_logprob"]
    run = CascadeRun(dec.text, False, conf, tokens, full_tokens, peak,
                     kv_bytes(parked) if parked is not None else 0, clk.t)
    if r2 is None or mode == "static" or conf >= threshold:
        return run
    del c1
    run.escalated = True
    if mode == "offload":
        with clk("transfer"):
            c2 = cache_to(parked, runner.dev)
        del parked
    elif mode == "reprefill":
        with clk("encode"):
            enc = runner.encode(image, prompt)
        with clk("prefill"):
            pre = runner.prefill(enc, keep, capture=True)
        with clk("compress"):
            c2 = runner.compress(pre, "snapkv", r2)
        pre.cache, pre.queries = None, None
    else:
        raise ValueError(f"unknown mode {mode!r}")
    run.kv_tokens.append(kv_tokens(c2))
    run.peak_decode_kv_bytes = max(run.peak_decode_kv_bytes, kv_bytes(c2))
    with clk("decode"):
        dec = runner.decode(pre, c2, enc.next_position, max_new_tokens)
    run.text = dec.text
    return run


@torch.inference_mode()
def check_escalation(runner, image, prompt: str, max_new_tokens: int, r1: float, r2: float) -> dict:
    """Both cascade modes, forced to escalate, must return exactly the single-config answers."""
    single = {r: run_kv_cascade(runner, image, prompt, max_new_tokens, r, None, mode="static").text
              for r in (r1, r2)}
    out = {"single_r1": single[r1], "single_r2": single[r2]}
    for mode in ("offload", "reprefill"):
        stay = run_kv_cascade(runner, image, prompt, max_new_tokens, r1, r2, float("-inf"), mode)
        esc = run_kv_cascade(runner, image, prompt, max_new_tokens, r1, r2, float("inf"), mode)
        out[mode] = dict(stay=stay.text, escalated=esc.text,
                         ok=stay.text == single[r1] and esc.text == single[r2] and esc.escalated)
    return out
