"""Step 1: evaluate every (visual-token ratio x pruning method x KV ratio) config on every sample.

    python experiments/oracle/sweep.py --check    # uncompressed path must equal HF generate()
    python experiments/oracle/sweep.py --n 150    # resumable; appends to results/.../oracle_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import torch
from common import KV_METHODS, KV_RATIOS, RESULTS, V_METHODS, V_RATIOS

from uniinfer.data import iter_samples
from uniinfer.runtime import QwenVLRunner
from uniinfer.runtime.qwen25vl import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="textvqa,docvqa,mmbench")
    ap.add_argument("--n", type=int, default=150, help="samples per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--out", default=str(RESULTS / "oracle_sweep.jsonl"))
    ap.add_argument("--v-methods", default=",".join(V_METHODS), help="e.g. uniform,dissim,attn")
    ap.add_argument("--kv-methods", default=",".join(KV_METHODS), help="e.g. l2norm,snapkv")
    ap.add_argument("--v-ratios", default=",".join(map(str, V_RATIOS)), help="visual keep ratios (1.0 = full)")
    ap.add_argument("--kv-ratios", default=",".join(map(str, KV_RATIOS)), help="KV keep ratios (1.0 = full)")
    ap.add_argument("--check", action="store_true", help="verify equivalence with HF generate(), then exit")
    ap.add_argument("--check-attn", action="store_true",
                    help="verify the attention reconstruction used by SnapKV/FastV-style scoring, then exit")
    a = ap.parse_args()
    v_methods, kv_methods = a.v_methods.split(","), a.kv_methods.split(",")
    v_ratios, kv_ratios = [float(x) for x in a.v_ratios.split(",")], [float(x) for x in a.kv_ratios.split(",")]

    runner = QwenVLRunner(a.model, max_pixels=a.max_pixels)
    if a.check:
        ok = True
        for name in a.datasets.split(","):
            for s in iter_samples(name, 2, a.seed):
                ref, ours = runner.check_equivalence(s)
                print(f"[{s['id']}] generate={ref!r} ours={ours!r} {'OK' if ref == ours else 'MISMATCH'}")
                ok &= ref == ours
        print("EQUIVALENCE", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)
    if a.check_attn:
        worst = 0.0
        for name in a.datasets.split(","):
            for s in iter_samples(name, 2, a.seed):
                err = runner.check_attention(s["image"], s["prompt"])
                worst = max(worst, max(err))
                print(f"[{s['id']}] relative error of rebuilt attention output: max {max(err):.4f} "
                      f"median {sorted(err)[len(err) // 2]:.4f} over {len(err)} layers", flush=True)
        ok = worst < 0.02
        print("ATTENTION", "PASS" if ok else "FAIL", f"(worst {worst:.4f}, threshold 0.02)")
        raise SystemExit(0 if ok else 1)

    from pathlib import Path
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    meta = dict(model=a.model, gpu=torch.cuda.get_device_name(0), max_pixels=a.max_pixels, seed=a.seed,
                v_ratios=v_ratios, v_methods=v_methods, kv_ratios=kv_ratios, kv_methods=kv_methods,
                torch=torch.__version__)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
    with out.open("a") as f:
        for name in a.datasets.split(","):
            t0, k = time.perf_counter(), 0
            for s in iter_samples(name, a.n, a.seed):
                if s["id"] in done:
                    continue
                try:
                    rec = runner.sweep(s, v_ratios, v_methods, kv_ratios, kv_methods)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM on {s['id']}, skipping", flush=True)
                    torch.cuda.empty_cache()
                    continue
                except Exception:
                    print(f"ERROR on {s['id']}\n{traceback.format_exc()}", flush=True)
                    continue
                f.write(json.dumps(rec) + "\n")
                f.flush()
                k += 1
                full = rec["configs"][0]
                print(f"{name} {k}/{a.n} {s['id']} nvis={rec['n_vis']} full={full['score']:.2f} "
                      f"pred={full['pred'][:30]!r} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
