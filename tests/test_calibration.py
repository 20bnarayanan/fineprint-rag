from fineprint.calibration import build_calibration_cases, score_calibration
from fineprint.validators import run_hard_gates


def test_case_builder_labels_and_dropped_clauses():
    cases = build_calibration_cases()
    names = {c["name"] for c in cases}
    assert "complete" in names and "unsupported_claim" in names and "legal_conclusion" in names
    drops = [c for c in cases if c["name"].startswith("drop:")]
    assert drops and all(c["expect"] == "fail" and c["dropped_clause_ids"] for c in drops)
    # lease_a: auto_renewal(high), repair_cost_shift(high), entry_no_notice(high), late_fee_conflict -> 4 drops
    assert len([c for c in drops if c["lease_id"] == "lease_a"]) == 4
    # clean lease contributes only a complete plan
    assert [c["name"] for c in cases if c["lease_id"] == "lease_e_clean"] == ["complete"]


def test_calibration_plans_pass_hard_gates_except_scope_case():
    # Calibration probes the JUDGE, so plans must reach it: only the legal-conclusion case trips a gate.
    for c in build_calibration_cases():
        fails = run_hard_gates(c["plan"], c["clauses"])
        assert (not fails) or c["name"] == "legal_conclusion", (c["lease_id"], c["name"], fails)


def test_score_calibration():
    rs = [
        dict(expect="pass", name="complete", verdict="pass", flagged_clause_ids=[], dropped_clause_ids=[]),
        dict(expect="pass", name="complete", verdict="fail", flagged_clause_ids=[], dropped_clause_ids=[]),
        dict(expect="fail", name="drop:x", verdict="fail", flagged_clause_ids=["s02"], dropped_clause_ids=["s02"]),
        dict(expect="fail", name="drop:y", verdict="fail", flagged_clause_ids=["s09"], dropped_clause_ids=["s07"]),
        dict(expect="fail", name="unsupported_claim", verdict="pass", flagged_clause_ids=[], dropped_clause_ids=[]),
    ]
    s = score_calibration(rs)
    assert s["false_alarm_rate"] == 0.5
    assert abs(s["detection_rate"] - 2 / 3) < 1e-9
    assert s["localization_rate"] == 0.5
    assert s["detection_by_kind"] == {"drop": 1.0, "unsupported_claim": 0.0}
