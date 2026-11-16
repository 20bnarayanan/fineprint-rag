"""Evaluators in LangSmith's (outputs, reference_outputs) form.

They wrap the pure functions in fineprint.metrics, so the exact same numbers come out
whether you run locally or through langsmith.evaluate().

Returning score=None means "not applicable for this example" (e.g. trap recall on the
clean lease) and is excluded from averages.
"""
from __future__ import annotations

from fineprint import metrics


def _r(key: str, score, comment: str | None = None) -> dict:
    out = {"key": key, "score": score}
    if comment:
        out["comment"] = comment
    return out


def trap_recall(outputs: dict, reference_outputs: dict) -> dict:
    return _r("trap_recall", metrics.trap_recall(outputs["plan"], reference_outputs))


def high_severity_recall(outputs: dict, reference_outputs: dict) -> dict:
    return _r("high_severity_recall", metrics.high_severity_recall(outputs["plan"], reference_outputs))


def deadline_recall(outputs: dict, reference_outputs: dict) -> dict:
    return _r("deadline_recall", metrics.deadline_recall(outputs["plan"], reference_outputs))


def extra_flags(outputs: dict, reference_outputs: dict) -> dict:
    """High/medium flags that match no planted trap, on leases that HAVE traps (upper bound on false positives)."""
    if not reference_outputs["traps"]:
        return _r("extra_flags_per_lease", None)
    return _r("extra_flags_per_lease", metrics.unlabeled_flag_count(outputs["plan"], reference_outputs))


def clean_lease_flags(outputs: dict, reference_outputs: dict) -> dict:
    """High/medium flags on the lease with NO planted traps: the cleanest false-alarm signal."""
    if reference_outputs["traps"]:
        return _r("clean_lease_flags", None)
    return _r("clean_lease_flags", metrics.unlabeled_flag_count(outputs["plan"], reference_outputs))


def fabricated_first_draft(outputs: dict, reference_outputs: dict) -> dict:
    return _r("fabricated_quote_rate_first_draft", outputs.get("first_draft_fab_rate"))


def fabricated_final(outputs: dict, reference_outputs: dict) -> dict:
    return _r("fabricated_quote_rate_final", outputs.get("final_fab_rate"))


def gate_failures_first_draft(outputs: dict, reference_outputs: dict) -> dict:
    return _r("hard_gate_failures_first_draft", outputs.get("first_draft_gate_failures"))


def escalated(outputs: dict, reference_outputs: dict) -> dict:
    return _r("escalated", 1.0 if outputs["status"] == "escalated" else 0.0)


def drafts(outputs: dict, reference_outputs: dict) -> dict:
    return _r("drafts", float(outputs["attempts"]))


ALL_EVALUATORS = [trap_recall, high_severity_recall, deadline_recall, extra_flags, clean_lease_flags,
                  fabricated_first_draft, fabricated_final, gate_failures_first_draft, escalated, drafts]
