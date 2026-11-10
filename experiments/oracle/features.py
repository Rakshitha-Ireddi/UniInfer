"""Step 2: extract per-request features for the learned selector (same samples as sweep.py).

Tiers, by cost:
  T1  question-text stats + image / visual-token statistics (from the vision encoder, which runs anyway)
  T2  + mean question-token embedding (an embedding lookup only)
  T3  + mid-layer last-token hidden state of a FULL prefill (expensive; reported as a reference point)
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from common import RESULTS

from uniinfer.data import iter_samples
from uniinfer.runtime import QwenVLRunner

KEYWORDS = ["what is written", "what does", "say", "name", "number", "how many", "how much", "total", "date",
            "year", "brand", "title", "color", "colour", "which", "where", "who", "when", "time", "price", "$",
            "percentage", "%", "amount", "page", "table", "figure"]


def text_features(prompt: str) -> list[float]:
    q = prompt.lower()
    return [len(q), len(q.split()), sum(ch.isdigit() for ch in q), float(bool(re.search(r"\nA\. ", prompt))),
            float("hint:" in q)] + [float(k in q) for k in KEYWORDS]


def visual_features(vis: torch.Tensor, image) -> list[float]:
    v = F.normalize(vis.float(), dim=-1)
    cos = (v @ F.normalize(v.mean(0, keepdim=True), dim=-1).T).squeeze(-1)
    norms = vis.float().norm(dim=-1)
    n = v.shape[0]
    sub = v[torch.randperm(n, generator=torch.Generator().manual_seed(0))[:256].to(v.device)]
    pair = (sub @ sub.T).fill_diagonal_(0)
    w, h = image.size
    q = torch.quantile(cos, torch.tensor([0.1, 0.5, 0.9], device=cos.device)).tolist()
    return [n, float(np.log(n)), w / h, float(np.log(w * h)), cos.mean().item(), cos.std().item(), *q,
            (cos > 0.9).float().mean().item(), (cos > 0.95).float().mean().item(),
            norms.mean().item(), norms.std().item(), pair.sum().item() / max(1, sub.shape[0] ** 2 - sub.shape[0])]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="textvqa,docvqa,mmbench")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--out", default=str(RESULTS / "features.npz"))
    a = ap.parse_args()
    r = QwenVLRunner(max_pixels=a.max_pixels)
    m = r.model
    mid = m.config.num_hidden_layers // 2
    ids, t1, t2, t3, times = [], [], [], [], []
    for name in a.datasets.split(","):
        for s in iter_samples(name, a.n, a.seed):
            enc = r.encode(s["image"], s["prompt"])
            t0 = time.perf_counter()
            f1 = visual_features(enc.visual, s["image"]) + text_features(s["prompt"])
            qtok = r.proc.tokenizer(s["prompt"], return_tensors="pt").input_ids.to(r.dev)
            qemb = m.model.embed_tokens(qtok)[0].float().mean(0).cpu().numpy()
            r._sync(); t_cheap = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = m.model(inputs_embeds=enc.embeds, position_ids=enc.position_ids,
                          output_hidden_states=True, use_cache=False)
            hid = out.hidden_states[mid][0, -1].float().cpu().numpy()
            r._sync(); t_prefill = time.perf_counter() - t0
            del out
            ids.append(s["id"]); t1.append(f1); t2.append(qemb); t3.append(hid)
            times.append([enc.timings["encode_s"], t_cheap, t_prefill])
            if len(ids) % 50 == 0:
                print(f"{len(ids)} done", flush=True)
    np.savez(a.out, ids=np.array(ids), t1=np.array(t1, np.float32), t2=np.array(t2, np.float16),
             t3=np.array(t3, np.float16), times=np.array(times, np.float32))
    med = np.median(np.array(times), 0)
    print(json.dumps(dict(n=len(ids), median_s=dict(vision_encode=float(med[0]), cheap_features=float(med[1]),
                                                   full_prefill_T3=float(med[2])))))
    print("FEATURES_DONE")


if __name__ == "__main__":
    main()
