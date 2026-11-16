"""Judge calibration without human labelling.

Because plans are built from ground truth, we know exactly how each degraded plan
differs from a correct one. That gives the LLM judge a labelled test set:

  complete plan                      -> judge should PASS      (else: false alarm)
  plan missing a high-severity trap  -> judge should FAIL and name the right clause
  plan missing a conflict            -> judge should FAIL      (rubric: unreported contradiction)
  plan with an unsupported claim     -> judge should FAIL
  plan with a legal conclusion       -> judge should FAIL      (scope_ok = false)

Limits (be upfront about them): synthetic degradations are easier than real model
mistakes, so this measures a *floor* on judge quality, not agreement with humans.
Hand-label a sample of real planner outputs for that.
"""
from __future__ import annotations

import json
from pathlib import Path

from .ingest import split_clauses
from .synth import plan_from_truth

ROOT = Path(__file__).resolve().parents[1]


def _clause_ids_for(clauses, sentences):
    return [c.id for c in clauses for s in sentences if s in c.text]


def build_calibration_cases(lease_ids: list[str] | None = None) -> list[dict]:
    truth_all = json.loads((ROOT / "data" / "traps.json").read_text())
    cases: list[dict] = []
    for lease_id, truth in truth_all.items():
        if lease_ids and lease_id not in lease_ids:
            continue
        text = (ROOT / "data" / "leases" / f"{lease_id}.txt").read_text()
        clauses = split_clauses(text)
        base = dict(lease_id=lease_id, clauses=clauses)
        cases.append({**base, "name": "complete", "expect": "pass", "dropped_clause_ids": [],
                      "plan": plan_from_truth(clauses, truth)})
        for trap in truth["traps"]:
            if trap["severity"] == "high" or trap["id"].endswith("conflict"):
                cases.append({**base, "name": f"drop:{trap['id']}", "expect": "fail",
                              "dropped_clause_ids": _clause_ids_for(clauses, trap["evidence"]),
                              "plan": plan_from_truth(clauses, truth, drop=(trap["id"],))})
        if truth["traps"]:
            cases.append({**base, "name": "unsupported_claim", "expect": "fail", "dropped_clause_ids": [],
                          "plan": plan_from_truth(clauses, truth, unsupported_claim=True)})
            cases.append({**base, "name": "legal_conclusion", "expect": "fail", "dropped_clause_ids": [],
                          "plan": plan_from_truth(clauses, truth, legal_conclusion=True)})
    return cases


def score_calibration(results: list[dict]) -> dict:
    """results: [{expect, name, verdict, flagged_clause_ids, dropped_clause_ids}]"""
    complete = [r for r in results if r["expect"] == "pass"]
    degraded = [r for r in results if r["expect"] == "fail"]
    drops = [r for r in degraded if r["dropped_clause_ids"]]
    out = {
        "n_complete": len(complete), "n_degraded": len(degraded),
        "false_alarm_rate": (sum(r["verdict"] == "fail" for r in complete) / len(complete)) if complete else None,
        "detection_rate": (sum(r["verdict"] == "fail" for r in degraded) / len(degraded)) if degraded else None,
        "localization_rate": (sum(any(c in r["flagged_clause_ids"] for c in r["dropped_clause_ids"]) for r in drops) / len(drops)) if drops else None,
    }
    by_kind: dict[str, list[bool]] = {}
    for r in degraded:
        by_kind.setdefault(r["name"].split(":")[0], []).append(r["verdict"] == "fail")
    out["detection_by_kind"] = {k: sum(v) / len(v) for k, v in by_kind.items()}
    return out
