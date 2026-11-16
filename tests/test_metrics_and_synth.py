from fineprint.metrics import deadline_recall, high_severity_recall, trap_recall, unlabeled_flag_count
from fineprint.synth import plan_from_truth


def _d(plan):
    return plan.model_dump()


def test_perfect_plan_scores_one(lease_a_clauses, truth):
    p = _d(plan_from_truth(lease_a_clauses, truth["lease_a"]))
    t = truth["lease_a"]
    assert trap_recall(p, t) == 1.0 and high_severity_recall(p, t) == 1.0
    assert deadline_recall(p, t) == 1.0 and unlabeled_flag_count(p, t) == 0


def test_dropping_a_trap_lowers_recall(lease_a_clauses, truth):
    t = truth["lease_a"]
    p = _d(plan_from_truth(lease_a_clauses, t, drop=("auto_renewal",)))
    assert trap_recall(p, t) == 0.75
    assert high_severity_recall(p, t) == 2 / 3


def test_conflict_requires_both_sentences(lease_a_clauses, truth):
    t = truth["lease_a"]
    p = _d(plan_from_truth(lease_a_clauses, t))
    for it in p["items"]:
        if it["kind"] == "conflict":
            it["evidence"] = it["evidence"][:1]  # cite only one side
    assert trap_recall(p, t) == 0.75


def test_clean_lease_recall_is_none_and_false_flags_counted(truth):
    t = truth["lease_e_clean"]
    plan = {"items": [{"kind": "risk", "severity": "high", "evidence": [{"clause_id": "s01", "quote": "A quote that matches no trap at all."}]}]}
    assert trap_recall(plan, t) is None
    assert unlabeled_flag_count(plan, t) == 1


def test_degradations_are_applied(lease_a_clauses, truth):
    t = truth["lease_a"]
    assert "illegal" in plan_from_truth(lease_a_clauses, t, legal_conclusion=True).items[0].explanation
    assert "$10" in plan_from_truth(lease_a_clauses, t, unsupported_claim=True).items[0].explanation


def test_deadline_recall_counts_dated_items_labelled_with_another_kind(lease_a_clauses, truth):
    t = truth["lease_a"]
    p = _d(plan_from_truth(lease_a_clauses, t))
    for it in p["items"]:
        if it["kind"] == "deadline":
            it["kind"] = "risk"   # planner mislabels the kind but still fills in the deadline spec
    assert deadline_recall(p, t) == 1.0
    for it in p["items"]:
        it["deadline"] = None
    assert deadline_recall(p, t) == 0.0
