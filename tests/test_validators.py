from fineprint.schemas import DeadlineSpec, Evidence, Plan, PlanItem
from fineprint.synth import plan_from_truth
from fineprint.validators import fabricated_quote_rate, normalize, run_hard_gates


def _item(**kw):
    base = dict(kind="risk", category="maintenance", severity="high", title="t", explanation="e",
                evidence=[Evidence(clause_id="s06", quote="Tenant is responsible for the first $300 of the cost of any repair")])
    base.update(kw)
    return PlanItem(**base)


def _plan(*items, not_found=()):
    return Plan(summary="s", items=list(items), not_found=list(not_found))


def test_reference_plan_passes_all_gates(lease_a_clauses, truth):
    plan = plan_from_truth(lease_a_clauses, truth["lease_a"])
    assert run_hard_gates(plan, lease_a_clauses) == []


def test_verbatim_quote_ok_with_typography_and_whitespace(lease_a_clauses):
    q = "Tenant is responsible for the first $300   of the cost of any repair"
    assert run_hard_gates(_plan(_item(evidence=[Evidence(clause_id="s06", quote=q)])), lease_a_clauses) == []
    assert normalize("it’s “fine”") == "it's \"fine\""


def test_paraphrase_is_rejected(lease_a_clauses):
    q = "Tenants must pay the first three hundred dollars of any repair"
    fails = run_hard_gates(_plan(_item(evidence=[Evidence(clause_id="s06", quote=q)])), lease_a_clauses)
    assert [f.gate for f in fails] == ["verbatim_quote"]


def test_quote_must_come_from_the_cited_clause(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(evidence=[Evidence(clause_id="s07", quote="Tenant is responsible for the first $300 of the cost of any repair")])), lease_a_clauses)
    assert fails and fails[0].gate == "verbatim_quote"


def test_unknown_clause_id_and_short_quote(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(evidence=[Evidence(clause_id="s99", quote="x" * 30)]),
                                 _item(evidence=[Evidence(clause_id="s06", quote="repair")])), lease_a_clauses)
    assert {f.gate for f in fails} == {"clause_exists", "quote_too_short"}


def test_missing_evidence_and_bad_category(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(evidence=[], category="vibes")), lease_a_clauses)
    assert {f.gate for f in fails} == {"evidence_required", "category"}


def test_conflict_needs_two_clauses(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(kind="conflict")), lease_a_clauses)
    assert "conflict_needs_two_clauses" in {f.gate for f in fails}


def test_deadline_needs_spec(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(kind="deadline", category="deadlines")), lease_a_clauses)
    assert "deadline_spec" in {f.gate for f in fails}
    ok = _item(kind="deadline", category="deadlines", deadline=DeadlineSpec(anchor="move_in", offset_days=7))
    assert "deadline_spec" not in {f.gate for f in run_hard_gates(_plan(ok), lease_a_clauses)}


def test_scope_gate_blocks_legal_conclusions(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(explanation="This clause is illegal.")), lease_a_clauses)
    assert [f.gate for f in fails] == ["scope"]
    assert run_hard_gates(_plan(_item(explanation="This is not legal advice, but it is costly.")), lease_a_clauses) == []


def test_not_found_contradicting_the_document(lease_a_clauses):
    # lease_a mentions renewals, so claiming "renewal" is not addressed must fail
    fails = run_hard_gates(_plan(not_found=["renewal"]), lease_a_clauses)
    assert [f.gate for f in fails] == ["not_found_keyword"]
    assert "s02" in fails[0].message


def test_not_found_ok_when_truly_absent(lease_a_clauses):
    assert run_hard_gates(_plan(not_found=["arbitration_x"]), lease_a_clauses)[0].gate == "not_found_category"
    assert run_hard_gates(_plan(not_found=["dispute_resolution"]), lease_a_clauses) == []


def test_not_found_and_item_same_category(lease_a_clauses):
    fails = run_hard_gates(_plan(_item(), not_found=["maintenance"]), lease_a_clauses)
    assert "not_found_contradiction" in {f.gate for f in fails}


def test_fabricated_quote_rate(lease_a_clauses, truth):
    good = plan_from_truth(lease_a_clauses, truth["lease_a"])
    assert fabricated_quote_rate(good, lease_a_clauses) == 0.0
    bad = plan_from_truth(lease_a_clauses, truth["lease_a"], fabricate=True)
    assert 0 < fabricated_quote_rate(bad, lease_a_clauses) < 0.2


def test_stringified_json_lists_are_coerced():
    """Regression (seen live): the model returned `items` as a JSON string instead of a list."""
    import json
    from fineprint.schemas import JudgeVerdict
    item = {"kind": "risk", "category": "maintenance", "severity": "high", "title": "t", "explanation": "e",
            "evidence": json.dumps([{"clause_id": "s06", "quote": "Tenant is responsible for the first $300"}]),
            "deadline": json.dumps({"anchor": "move_in", "offset_days": 7})}
    p = Plan.model_validate({"summary": "s", "items": json.dumps([item]), "not_found": "[]"})
    assert p.items[0].evidence[0].clause_id == "s06" and p.items[0].deadline.offset_days == 7 and p.not_found == []
    v = JudgeVerdict.model_validate({"groundedness": 5, "scope_ok": True, "verdict": "pass", "rationale": "r",
                                     "missed_risks": '[{"clause_id": "s01", "why": "w"}]', "unsupported_claims": "[]"})
    assert v.missed_risks[0].clause_id == "s01"
    import pytest
    with pytest.raises(Exception):  # genuinely malformed strings must still fail, not be swallowed
        Plan.model_validate({"summary": "s", "items": "[not json"})
