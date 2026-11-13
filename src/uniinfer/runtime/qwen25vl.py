"""Qwen2.5-VL runtime with explicit control over visual tokens and the KV cache.

We bypass `generate()` and run prefill/decode ourselves so that we can
  1. drop visual tokens while keeping each survivor's original 3-D M-RoPE position id, and
  2. swap in a compressed KV cache after prefill.
With no compression, the output is token-identical to HF `generate()` (greedy); `check_equivalence`
verifies this. ADR-0003 explains the design.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from ..compression import (
    WindowQueryCapture,
    compress_kv_l2norm,
    compress_kv_snapkv,
    select_visual_tokens,
    visual_attention_scores,
)
from ..data import MAX_NEW_TOKENS
from ..metrics import SCORERS
from ..signals import answer_confidence, token_confidence

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass
class Prefill:
    first_token: int
    first_conf: dict          # token_confidence() of the first generated token
    cache: DynamicCache
    seq_len: int
    seconds: float
    queries: dict | None = None  # observation-window queries per layer (attention-guided methods only)


@dataclass
class Decoded:
    text: str
    n_tokens: int
    seconds: float
    token_logprobs: list      # log-prob of each greedy token, first token included


def confidence_record(pre: Prefill, dec: Decoded) -> dict:
    """Flat confidence fields stored per (request, config) in sweep results."""
    rec = {f"first_{k}": round(v, 6) for k, v in pre.first_conf.items()}
    rec.update({f"ans_{k}": round(v, 6) for k, v in answer_confidence(dec.token_logprobs).items()})
    return rec


@dataclass
class Encoded:
    embeds: torch.Tensor      # (1, L, d) text and visual embeddings
    position_ids: torch.Tensor  # (3, 1, L) M-RoPE ids
    image_mask: torch.Tensor  # (L,) bool
    visual: torch.Tensor      # (n_vis, d)
    timings: dict = field(default_factory=dict)

    @property
    def next_position(self) -> int:
        return int(self.position_ids.max()) + 1


class QwenVLRunner:
    def __init__(self, model_id: str = DEFAULT_MODEL, max_pixels: int = 1024 * 28 * 28,
                 min_pixels: int = 256 * 28 * 28, device: str = "cuda:0"):
        self.dev = torch.device(device)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.image_token_id = self.model.config.image_token_id

    # ------------------------------------------------------------------ helpers
    def _sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def processor_inputs(self, image, prompt: str):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.proc(text=[text], images=[image], return_tensors="pt").to(self.dev)

    # ------------------------------------------------------------------ stages
    @torch.inference_mode()
    def encode(self, image, prompt: str) -> Encoded:
        """Run the vision encoder once and build full-length embeddings and M-RoPE position ids."""
        self._sync(); t0 = time.perf_counter()
        inp = self.processor_inputs(image, prompt)
        m = self.model
        emb = m.model.embed_tokens(inp["input_ids"])
        vis = m.visual(inp["pixel_values"].type(m.visual.dtype), grid_thw=inp["image_grid_thw"])
        mask = inp["input_ids"][0] == self.image_token_id
        emb[0, mask] = vis.to(emb.dtype)
        pos, _ = m.get_rope_index(inp["input_ids"], inp["image_grid_thw"], None, None, inp["attention_mask"])
        self._sync()
        return Encoded(emb, pos, mask, vis, {"encode_s": time.perf_counter() - t0})

    @torch.inference_mode()
    def prefill(self, enc: Encoded, keep: torch.Tensor, capture: bool = False) -> Prefill:
        """Prefill on the kept tokens only; also reports first-token confidence (free: logits exist).

        capture=True also stores the observation-window queries per layer (read-only hooks; outputs are
        unchanged), which SnapKV and FastV-style scoring need.
        """
        e, p = enc.embeds[:, keep], enc.position_ids[:, :, keep]
        L = e.shape[1]
        self._sync(); t0 = time.perf_counter()
        cap = WindowQueryCapture(self.model) if capture else None
        if cap is not None:
            cap.__enter__()
        try:
            out = self.model(inputs_embeds=e, position_ids=p,
                             attention_mask=torch.ones(1, L, device=self.dev, dtype=torch.long),
                             past_key_values=DynamicCache(), use_cache=True)
        finally:
            if cap is not None:
                cap.__exit__(None, None, None)
        last = out.logits[0, -1]
        first = int(last.argmax())
        conf = token_confidence(last)
        self._sync()
        return Prefill(first, conf, out.past_key_values, L, time.perf_counter() - t0,
                       cap.queries if cap is not None else None)

    @staticmethod
    def compress(pre: Prefill, kv_method: str, kv_ratio: float) -> DynamicCache:
        """New compressed cache; `pre.cache` is left untouched so several settings can share one prefill."""
        if kv_method == "l2norm" or kv_ratio >= 1.0:
            return compress_kv_l2norm(pre.cache, kv_ratio)
        if kv_method == "snapkv":
            if pre.queries is None:
                raise ValueError("snapkv needs prefill(..., capture=True)")
            return compress_kv_snapkv(pre.cache, pre.queries, kv_ratio)
        raise ValueError(f"unknown kv_method {kv_method!r}")

    def keep_mask(self, enc: Encoded, ratio: float, method: str, full: Prefill | None = None) -> torch.Tensor:
        """Visual keep-mask. Method 'attn' needs a captured prefill over the full sequence (`full`)."""
        scores = None
        if method == "attn" and ratio < 1.0:
            if full is None or full.queries is None:
                raise ValueError("v_method 'attn' needs a full-sequence prefill with capture=True")
            scores = visual_attention_scores(full.cache, full.queries, enc.image_mask)
        return select_visual_tokens(enc.image_mask, enc.visual, ratio, method, scores)

    @torch.inference_mode()
    def decode(self, pre: Prefill, cache: DynamicCache, next_position: int, max_new_tokens: int) -> Decoded:
        """Greedy decode. Decode-step positions continue from max(prompt position) + 1, as in HF.

        Records the log-prob of every greedy token (the first one comes from the prefill logits).
        """
        tok = pre.first_token
        toks, logprobs = [tok], [pre.first_conf["logprob"]]
        self._sync(); t0 = time.perf_counter()
        for t in range(max_new_tokens - 1):
            if tok in self.eos:
                break
            cur = cache.key_cache[0].shape[-2]
            pid = torch.full((3, 1, 1), next_position + t, device=self.dev, dtype=torch.long)
            o = self.model(input_ids=torch.tensor([[tok]], device=self.dev), position_ids=pid,
                           attention_mask=torch.ones(1, cur + 1, device=self.dev, dtype=torch.long),
                           past_key_values=cache, use_cache=True)
            step = o.logits[0, -1]
            tok = int(step.argmax())
            toks.append(tok)
            logprobs.append(torch.log_softmax(step.float(), dim=-1)[tok].item())
        self._sync()
        text = self.proc.tokenizer.decode([x for x in toks if x not in self.eos], skip_special_tokens=True)
        return Decoded(text.strip(), len(toks), time.perf_counter() - t0, logprobs)

    # ------------------------------------------------------------------ end-to-end
    @torch.inference_mode()
    def run(self, image, prompt: str, v_ratio=1.0, v_method="dissim", kv_ratio=1.0, max_new_tokens=32,
            kv_method="l2norm") -> str:
        """Answer one request under a single configuration."""
        return self.run_with_confidence(image, prompt, v_ratio, v_method, kv_ratio, max_new_tokens, kv_method)[0]

    @torch.inference_mode()
    def run_with_confidence(self, image, prompt: str, v_ratio=1.0, v_method="dissim", kv_ratio=1.0,
                            max_new_tokens=32, kv_method="l2norm") -> tuple[str, dict]:
        """Answer plus confidence signals: first_* (first token) and answer-level log-prob stats.

        v_method 'attn' first runs a full prefill to score image tokens (in deployment layers 0-2 suffice).
        """
        enc = self.encode(image, prompt)
        full = None
        if v_method == "attn" and v_ratio < 1.0:
            full = self.prefill(enc, torch.ones_like(enc.image_mask), capture=True)
        keep = self.keep_mask(enc, v_ratio, v_method, full)
        del full
        pre = self.prefill(enc, keep, capture=kv_method == "snapkv" and kv_ratio < 1.0)
        dec = self.decode(pre, self.compress(pre, kv_method, kv_ratio), enc.next_position, max_new_tokens)
        return dec.text, confidence_record(pre, dec)

    @torch.inference_mode()
    def sweep(self, sample: dict, v_ratios, v_methods, kv_ratios, kv_methods=("l2norm",)) -> dict:
        """Evaluate every configuration on one sample, reusing the encoder pass and the prefills.

        Config order: visual configs ("full" first); for each, KV ratio 1.0 once, then every
        (kv_method, kv_ratio < 1). With kv_methods=("l2norm",) this is exactly the v0.1/v0.2 order.
        """
        enc = self.encode(sample["image"], sample["prompt"])
        scorer, max_new = SCORERS[sample["dataset"]], MAX_NEW_TOKENS[sample["dataset"]]
        vconfigs = [("full", 1.0)] + [(m, r) for m in v_methods for r in v_ratios if r < 1.0]
        kconfigs = [(kv_methods[0], 1.0)] + [(km, r) for km in kv_methods for r in kv_ratios if r < 1.0]
        capture = "snapkv" in kv_methods or "attn" in v_methods
        results, full = [], None
        for method, vr in vconfigs:
            keep = self.keep_mask(enc, vr, "uniform" if method == "full" else method, full)
            pre = self.prefill(enc, keep, capture=capture)
            if method == "full":
                full = pre  # FastV-style scores come from the full-sequence prefill
            for km, kr in kconfigs:
                cache = self.compress(pre, km, kr)
                kv_len = cache.key_cache[0].shape[-2]
                dec = self.decode(pre, cache, enc.next_position, max_new)
                results.append(dict(v_method=method, v_ratio=vr, kv_method=km, kv_ratio=kr, pred=dec.text,
                                    score=scorer(dec.text, sample["answers"]),
                                    n_vis_kept=int(keep[enc.image_mask].sum()), seq_len=pre.seq_len, kv_len=kv_len,
                                    prefill_s=round(pre.seconds, 4), decode_s=round(dec.seconds, 4),
                                    n_new=dec.n_tokens, **confidence_record(pre, dec)))
                del cache
            if pre is not full:
                del pre
        del full
        return dict(id=sample["id"], dataset=sample["dataset"], answers=list(sample["answers"]),
                    category=sample.get("category"), n_vis=int(enc.image_mask.sum()),
                    seq_len=int(enc.image_mask.numel()), encode_s=round(enc.timings["encode_s"], 4),
                    configs=results)

    @torch.inference_mode()
    def check_attention(self, image, prompt: str) -> list[float]:
        """Per-layer relative error between the model's own attention output for the observation window and
        the one rebuilt from captured queries + cached keys/values (what SnapKV/FastV-style scoring uses).

        An error in the captured queries, RoPE, scaling, causal mask or GQA grouping shows up here. Expect
        about 1e-3 (bf16 noise).
        """
        from ..compression import window_attention

        enc = self.encode(image, prompt)
        outs: dict[int, torch.Tensor] = {}
        W = WindowQueryCapture(self.model).window

        def grab(mod, args, kwargs, output):
            outs[mod.layer_idx] = output[0][:, -W:].float()

        hooks = [layer.self_attn.register_forward_hook(grab, with_kwargs=True) for layer in self.model.model.layers]
        try:
            pre = self.prefill(enc, torch.ones_like(enc.image_mask), capture=True)
        finally:
            for h in hooks:
                h.remove()
        errs = []
        for li, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            p = window_attention(pre.queries[li], pre.cache.key_cache[li])                  # (1, H, W, L)
            v = pre.cache.value_cache[li].float().repeat_interleave(p.shape[1] // pre.cache.value_cache[li].shape[1], 1)
            o = (p @ v).transpose(1, 2).reshape(1, W, -1)                                     # (1, W, H*D)
            o = attn.o_proj(o.to(attn.o_proj.weight.dtype)).float()
            errs.append(float((o - outs[li]).norm() / outs[li].norm()))
        return errs

    @torch.inference_mode()
    def check_equivalence(self, sample: dict) -> tuple[str, str]:
        """(HF generate output, our output) for the uncompressed configuration; they must match."""
        inp = self.processor_inputs(sample["image"], sample["prompt"])
        self.model.rope_deltas = None
        max_new = MAX_NEW_TOKENS[sample["dataset"]]
        g = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        ref = self.proc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return ref, self.run(sample["image"], sample["prompt"], max_new_tokens=max_new)
