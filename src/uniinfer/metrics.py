"""Task metrics: TextVQA accuracy, DocVQA ANLS, MMBench multiple-choice accuracy."""
from __future__ import annotations

import re
import string
from collections.abc import Callable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace."""
    s = s.lower().strip().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def vqa_accuracy(pred: str, answers: Sequence[str]) -> float:
    """VQA accuracy: min(1, #annotators that gave this answer / 3), after normalization."""
    p = normalize_answer(pred)
    return min(1.0, sum(normalize_answer(a) == p for a in answers) / 3.0)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, answers: Sequence[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA), best over reference answers."""
    p = " ".join(pred.lower().strip().split())
    best = 0.0
    for a in answers:
        a = " ".join(a.lower().strip().split())
        nl = levenshtein(p, a) / max(len(p), len(a), 1)
        best = max(best, 1.0 - nl if nl < tau else 0.0)
    return best


def mc_accuracy(pred: str, answers: Sequence[str]) -> float:
    """Multiple-choice accuracy: first standalone option letter A-D must equal the key."""
    m = re.search(r"\b([A-D])\b", pred.strip())
    return float(bool(m) and m.group(1) == answers[0])


_NUMBER = re.compile(r"^[-+]?\d*\.?\d+(e[-+]?\d+)?$")


def _to_number(s: str) -> float | None:
    """Parse '1,234.5', '$12', '45%', '12.' as numbers; None if not numeric."""
    t = s.strip().lower().rstrip(".").replace(",", "").replace("$", "").replace("%", "").strip()
    return float(t) if _NUMBER.match(t) else None


def relaxed_accuracy(pred: str, answers: Sequence[str], tol: float = 0.05) -> float:
    """ChartQA relaxed accuracy (Masry et al., 2022): numeric answers within `tol` relative error count
    as correct; non-numeric answers need a case-insensitive exact match.

    Unlike the reference implementation, a trailing '%' is stripped rather than divided by 100, so '45%'
    matches '45'. Both forms appear in ChartQA answers and model outputs, so this only removes
    formatting false-negatives.
    """
    p = pred.strip()
    pn = _to_number(p)
    for a in answers:
        an = _to_number(a)
        if pn is not None and an is not None:
            if (an == 0 and pn == 0) or (an != 0 and abs(pn - an) / abs(an) <= tol):
                return 1.0
        elif p.lower().rstrip(".") == a.strip().lower().rstrip("."):
            return 1.0
    return 0.0


SCORERS: dict[str, Callable[[str, Sequence[str]], float]] = {
    "textvqa": vqa_accuracy,
    "docvqa": anls,
    "mmbench": mc_accuracy,
    "chartqa": relaxed_accuracy,
}
