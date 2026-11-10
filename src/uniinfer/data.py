"""Benchmark loaders (streamed from the Hugging Face Hub) with the prompts used in all experiments."""
from __future__ import annotations

import math
from collections.abc import Iterator

SHORT_ANSWER = "\nAnswer the question using a single word or phrase."
MULTIPLE_CHOICE = "\nAnswer with the option's letter from the given choices directly."

DATASETS = {
    "textvqa": ("lmms-lab/textvqa", None, "validation"),
    "docvqa": ("lmms-lab/DocVQA", "DocVQA", "validation"),
    "mmbench": ("lmms-lab/MMBench", "en", "dev"),
}
MAX_NEW_TOKENS = {"textvqa": 16, "docvqa": 24, "mmbench": 4}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip().lower() == "nan"


def build_mc_prompt(row: dict) -> str:
    opts = [(c, row[c]) for c in "ABCD" if not _missing(row.get(c))]
    hint = "" if _missing(row.get("hint")) else f"Hint: {row['hint']}\n"
    return hint + row["question"] + "\n" + "\n".join(f"{c}. {t}" for c, t in opts) + MULTIPLE_CHOICE


def iter_samples(name: str, n: int, seed: int = 0, buffer_size: int = 1000) -> Iterator[dict]:
    """Yield `n` shuffled samples as dicts: id, dataset, image (PIL RGB), prompt, answers."""
    from datasets import load_dataset

    repo, cfg, split = DATASETS[name]
    ds = load_dataset(repo, cfg, split=split, streaming=True).shuffle(seed=seed, buffer_size=buffer_size)
    for k, r in enumerate(ds):
        if k >= n:
            break
        img = r["image"].convert("RGB")
        if name == "textvqa":
            yield dict(id=f"textvqa-{r['question_id']}", dataset=name, image=img,
                       prompt=r["question"] + SHORT_ANSWER, answers=r["answers"])
        elif name == "docvqa":
            yield dict(id=f"docvqa-{r['questionId']}", dataset=name, image=img,
                       prompt=r["question"] + SHORT_ANSWER, answers=r["answers"])
        else:
            yield dict(id=f"mmbench-{r['index']}", dataset=name, image=img, prompt=build_mc_prompt(r),
                       answers=[r["answer"]], category=r.get("category"))
