"""Build reference plans from ground truth, optionally degraded in known ways.

Used for (1) unit tests of the graph with scripted models and (2) judge
calibration: because we know exactly what was removed or corrupted, the judge's
decisions can be scored without any human labelling.
"""
from __future__ import annotations

from .schemas import Clause, DeadlineSpec, Evidence, Plan, PlanItem


def _clause_for(sentence: str, clauses: list[Clause]) -> str:
    for c in clauses:
        if sentence in c.text:
            return c.id
    raise KeyError(f"evidence sentence not found in any clause: {sentence[:60]!r}")


def plan_from_truth(clauses: list[Clause], truth: dict, *, drop: tuple[str, ...] = (),
                    fabricate: bool = False, legal_conclusion: bool = False,
                    unsupported_claim: bool = False, include_deadlines: bool = True) -> Plan:
    items: list[PlanItem] = []
    for trap in truth["traps"]:
        if trap["id"] in drop:
            continue
        ev = [Evidence(clause_id=_clause_for(s, clauses), quote=s) for s in trap["evidence"]]
        items.append(PlanItem(
            kind="conflict" if trap["id"].endswith("conflict") else "risk",
            category=trap["category"], severity=trap["severity"],
            title=trap["summary"][:80], explanation=trap["summary"], evidence=ev,
            question_for_landlord=f"Can you clarify this term: {trap['summary']}"))
    if include_deadlines:
        for d in truth["deadlines"]:
            items.append(PlanItem(
                kind="deadline", category="deadlines", severity="low", title=d["label"],
                explanation=f"{d['label']} ({d['offset_days']} days from {d['anchor']}).",
                evidence=[Evidence(clause_id=_clause_for(d["evidence"], clauses), quote=d["evidence"])],
                deadline=DeadlineSpec(anchor=d["anchor"], offset_days=d["offset_days"])))
    if fabricate and items:
        first = items[0]
        items.append(first.model_copy(update={
            "title": "Invented clause",
            "evidence": [Evidence(clause_id=first.evidence[0].clause_id,
                                  quote="Tenant forfeits all belongings left in the unit after five days.")]}))
    if legal_conclusion and items:
        items[0] = items[0].model_copy(update={"explanation": "This clause is illegal and unenforceable."})
    if unsupported_claim and items:
        items[0] = items[0].model_copy(update={
            "explanation": "The clause caps the tenant's total exposure at $10, so this is not a concern."})
    return Plan(summary="Reference plan built from ground truth.", items=items, not_found=[])
