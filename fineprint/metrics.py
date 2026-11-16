"""Evaluation metrics. All pure functions over plain dicts, so they are unit-testable
and can be used both locally and as LangSmith code evaluators."""
from __future__ import annotations

from .validators import normalize

_MIN = 20  # ignore trivially short quotes when matching


def _overlap(q: str, e: str) -> bool:
    q, e = normalize(q), normalize(e)
    return len(q) >= _MIN and (q in e or e in q)


def item_matches_trap(item: dict, trap: dict) -> bool:
    quotes = [ev["quote"] for ev in item.get("evidence", [])]
    evidence = trap["evidence"]
    if len(evidence) > 1:  # conflict traps: the item must cite BOTH conflicting sentences
        return all(any(_overlap(q, e) for q in quotes) for e in evidence)
    return any(_overlap(q, e) for q in quotes for e in evidence)


def trap_recall(plan: dict, truth: dict) -> float | None:
    traps = truth["traps"]
    if not traps:
        return None
    found = sum(1 for t in traps if any(item_matches_trap(it, t) for it in plan["items"]
                                        if it["kind"] in ("risk", "conflict", "question")))
    return found / len(traps)


def high_severity_recall(plan: dict, truth: dict) -> float | None:
    traps = [t for t in truth["traps"] if t["severity"] == "high"]
    if not traps:
        return None
    found = sum(1 for t in traps if any(item_matches_trap(it, t) for it in plan["items"]))
    return found / len(traps)


def deadline_recall(plan: dict, truth: dict) -> float | None:
    ds = truth["deadlines"]
    if not ds:
        return None
    found = 0
    for d in ds:
        for it in plan["items"]:
            if (it["kind"] == "deadline" or it.get("deadline")) and any(_overlap(e["quote"], d["evidence"]) for e in it["evidence"]):
                found += 1
                break
    return found / len(ds)


def unlabeled_flag_count(plan: dict, truth: dict) -> int:
    """High/medium risk or conflict items that match no planted trap.

    This is an UPPER BOUND on false positives: a flag may be legitimate but simply
    not labelled in the synthetic ground truth. On the clean lease it is the
    cleanest false-alarm signal we have.
    """
    n = 0
    for it in plan["items"]:
        if it["kind"] not in ("risk", "conflict") or it["severity"] == "low":
            continue
        if not any(item_matches_trap(it, t) for t in truth["traps"]):
            n += 1
    return n
