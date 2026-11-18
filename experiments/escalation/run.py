"""Step 2 (GPU): run the planned cascades end to end on NEW requests and measure cost and latency.

Requests are drawn with a different shuffle seed and every request used in v0.1/v0.2 is skipped, so the
plan (fit on the v0.2 requests) is evaluated on held-out data. For each request and budget we run:
  * uncompressed (reference), and the best static SnapKV ratio for the budget,
  * the per-task cascade with cheap escalation (offload: one prefill, parked escalation cache),
  * the same cascade with naive escalation (reprefill: encode + prefill again on a retry).

    python experiments/escalation/run.py --check          # both modes must reproduce single-config answers
    python experiments/escalation/run.py --n 100          # resumable; results/v0.3-escalation/runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from uniinfer.data import MAX_NEW_TOKENS, iter_samples
from uniinfer.metrics import SCORERS
from uniinfer.runtime import QwenVLRunner
from uniinfer.runtime.escalation import check_escalation, run_kv_cascade
from uniinfer.runtime.qwen25vl import DEFAULT_MODEL

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "v0.3-escalation"


def record(run, scorer, answers):
    return dict(pred=run.text, score=scorer(run.text, answers), escalated=run.escalated,
                confidence=round(run.confidence, 6), kv_tokens=run.kv_tokens, full_kv_tokens=run.full_kv_tokens,
                cost_max=round(run.cost_max, 6), cost_sum=round(run.cost_sum, 6),
                peak_decode_kv_bytes=run.peak_decode_kv_bytes, offloaded_bytes=run.offloaded_bytes,
                timings={k: round(v, 5) for k, v in run.timings.items()},
                total_s=round(sum(v for k, v in run.timings.items() if k != "offload"), 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="textvqa,docvqa,mmbench")
    ap.add_argument("--n", type=int, default=100, help="new requests per dataset")
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed for NEW requests (v0.2 used 0)")
    ap.add_argument("--exclude", default=str(REPO / "results" / "v0.2-baselines" / "oracle_sweep.jsonl"))
    ap.add_argument("--plan", default=str(OUT / "plan.json"))
    ap.add_argument("--out", default=str(OUT / "runs.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    # fail fast on a bad --exclude path, before loading the model (the check needs no exclusions)
    seen = set() if a.check else {json.loads(line)["id"] for line in open(a.exclude)}
    runner = QwenVLRunner(a.model)

    if a.check:
        ok = True
        for name in a.datasets.split(","):
            for s in iter_samples(name, 2, a.seed):
                res = check_escalation(runner, s["image"], s["prompt"], MAX_NEW_TOKENS[name], 0.05, 0.15)
                good = res["offload"]["ok"] and res["reprefill"]["ok"]
                ok &= good
                print(f"[{s['id']}] {'OK' if good else 'MISMATCH'} {res}", flush=True)
        print("ESCALATION", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    out = Path(a.out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    meta = dict(model=a.model, gpu=torch.cuda.get_device_name(0), seed=a.seed, n=a.n, plan=plan,
                torch=torch.__version__, excluded=len(seen))
    out.with_name("runs_meta.json").write_text(json.dumps(meta, indent=2))
    with out.open("a") as f:
        for name in a.datasets.split(","):
            t0, k = time.perf_counter(), 0
            for s in iter_samples(name, 10 * a.n, a.seed):
                if k >= a.n:
                    break
                if s["id"] in seen:
                    continue
                k += 1
                if s["id"] in done:
                    continue
                scorer, mx = SCORERS[name], MAX_NEW_TOKENS[name]
                try:
                    rec = dict(id=s["id"], dataset=name, answers=list(s["answers"]))
                    rec["full"] = record(run_kv_cascade(runner, s["image"], s["prompt"], mx, 1.0, None,
                                                        mode="static"), scorer, s["answers"])
                    for B, entry in plan["budgets"].items():
                        t = entry["tasks"][name]
                        rec[f"static@{B}"] = record(run_kv_cascade(runner, s["image"], s["prompt"], mx,
                                                                   entry["static"]["ratio"], None, mode="static"),
                                                    scorer, s["answers"])
                        for mode in ("offload", "reprefill"):
                            thr = t["threshold"] if t["threshold"] is not None else float("-inf")
                            rec[f"{mode}@{B}"] = record(run_kv_cascade(runner, s["image"], s["prompt"], mx,
                                                                       t["r1"], t["r2"], thr, mode),
                                                        scorer, s["answers"])
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM on {s['id']}, skipping", flush=True)
                    torch.cuda.empty_cache()
                    continue
                except Exception:
                    print(f"ERROR on {s['id']}\n{traceback.format_exc()}", flush=True)
                    continue
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{name} {k}/{a.n} {s['id']} full={rec['full']['score']:.2f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
