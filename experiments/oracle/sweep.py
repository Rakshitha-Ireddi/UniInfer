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
from common import KV_RATIOS, RESULTS, V_METHODS, V_RATIOS

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
    ap.add_argument("--check", action="store_true", help="verify equivalence with HF generate(), then exit")
    a = ap.parse_args()

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

    from pathlib import Path
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    meta = dict(model=a.model, gpu=torch.cuda.get_device_name(0), max_pixels=a.max_pixels, seed=a.seed,
                v_ratios=V_RATIOS, v_methods=V_METHODS, kv_ratios=KV_RATIOS, torch=torch.__version__)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
    with out.open("a") as f:
        for name in a.datasets.split(","):
            t0, k = time.perf_counter(), 0
            for s in iter_samples(name, a.n, a.seed):
                if s["id"] in done:
                    continue
                try:
                    rec = runner.sweep(s, V_RATIOS, V_METHODS, KV_RATIOS)
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
