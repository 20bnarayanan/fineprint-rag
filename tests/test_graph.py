"""End-to-end graph tests with scripted models (no network)."""
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from fakes import ScriptedChatModel
from fineprint.config import SETTINGS
from fineprint.graph import run_pipeline
from fineprint.schemas import JudgeVerdict, MissedRisk
from fineprint.synth import plan_from_truth

PASS = JudgeVerdict(groundedness=5, scope_ok=True, verdict="pass", rationale="ok")


def _executor_script(plan, *, all_questions=True):
    """A well-behaved executor: one event per deadline item, and (optionally) every question in the email."""
    def call(name, args, i):
        return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
    dates = {("move_in", 7): "2027-01-08", ("lease_start", 10): "2027-01-11"}
    calls = [call("add_calendar_event", {"title": it.title, "iso_date": dates[(it.deadline.anchor, it.deadline.offset_days)]}, n)
             for n, it in enumerate([i for i in plan.items if i.deadline], start=1)]
    qs = list(dict.fromkeys(i.question_for_landlord for i in plan.items if i.question_for_landlord))
    calls.append(call("draft_questions_email", {"questions": qs if all_questions else qs[:1]}, 99))
    return [AIMessage(content="", tool_calls=calls), AIMessage(content="Done.")]


@pytest.fixture
def good(lease_a_clauses, truth):
    return plan_from_truth(lease_a_clauses, truth["lease_a"])


@pytest.fixture
def fabricated(lease_a_clauses, truth):
    return plan_from_truth(lease_a_clauses, truth["lease_a"], fabricate=True)


def test_repair_loop_then_execute(lease_a_text, good, fabricated, tmp_path):
    planner = ScriptedChatModel(responses=[fabricated, good])
    judge = ScriptedChatModel(responses=[PASS])
    executor = ScriptedChatModel(responses=_executor_script(good))
    out = run_pipeline(lease_a_text, retrieval="full", out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": executor})
    assert out["status"] == "accepted"
    assert out["attempts"] == 2
    assert out["first_draft_fab_rate"] > 0           # the unvalidated first draft had a fabricated quote
    assert [h["hard_failures"] for h in out["history"]] == [["verbatim_quote"], []]
    assert len(judge.calls) == 1                      # judge only ran after hard gates passed
    # the repair feedback reached the planner's second call
    assert "does not appear verbatim" in str(planner.calls[1][-1].content)
    # tools ran, and the executor was verified against the plan
    assert (tmp_path / "questions_for_landlord.md").exists()
    assert (tmp_path / "lease_deadlines.ics").read_text().count("BEGIN:VEVENT") == 2
    assert out["actions_verified"] is True
    assert out["actions_detail"] == {"events": "2/2", "email_questions": "4/4"}


def test_escalates_when_repairs_exhausted_and_never_executes(lease_a_text, fabricated, tmp_path):
    planner = ScriptedChatModel(responses=[fabricated] * (1 + SETTINGS.max_repairs))
    judge = ScriptedChatModel()
    executor = ScriptedChatModel()
    out = run_pipeline(lease_a_text, retrieval="full", out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": executor})
    assert out["status"] == "escalated"
    assert out["attempts"] == 1 + SETTINGS.max_repairs
    assert judge.calls == [] and executor.calls == []
    assert not list(tmp_path.iterdir())               # no side effects on an unverified plan


def test_judge_failure_widens_retrieval_context(lease_a_text, good, tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "top_k_per_category", 1)
    miss = JudgeVerdict(groundedness=5, scope_ok=True, verdict="fail", rationale="missed one",
                        missed_risks=[MissedRisk(clause_id="s01", why="The premises clause names the wrong unit.")])
    planner = ScriptedChatModel(responses=[good, good])
    judge = ScriptedChatModel(responses=[miss, PASS])
    out = run_pipeline(lease_a_text, retrieval="rag", run_executor=False, out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": ScriptedChatModel()})
    first_prompt = str(planner.calls[0][-1].content)
    second_prompt = str(planner.calls[1][-1].content)
    assert "[s01]" not in first_prompt                # retrieval never showed the planner this clause
    assert "[s01]" in second_prompt                   # judge saw the full lease; repair widened context
    assert "A reviewer found a risk you missed in clause s01" in second_prompt
    assert out["status"] == "accepted" and out["attempts"] == 2


def test_validator_off_accepts_first_draft(lease_a_text, fabricated, tmp_path):
    planner = ScriptedChatModel(responses=[fabricated])
    judge = ScriptedChatModel()
    out = run_pipeline(lease_a_text, retrieval="full", use_validator=False, run_executor=False,
                       out_dir=str(tmp_path), llms={"planner": planner, "judge": judge, "executor": ScriptedChatModel()})
    assert out["status"] == "accepted" and out["attempts"] == 1
    assert out["first_draft_fab_rate"] > 0 and judge.calls == []


def test_parse_flake_is_retried_without_using_the_repair_budget(lease_a_text, good, tmp_path):
    """Regression (seen live): garbled JSON from the model must not burn a repair attempt."""
    planner = ScriptedChatModel(responses=[OutputParserException("garbled json"), good])
    judge = ScriptedChatModel(responses=[PASS])
    out = run_pipeline(lease_a_text, retrieval="full", run_executor=False, out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": ScriptedChatModel()})
    assert out["status"] == "accepted" and out["attempts"] == 1
    assert out["history"][0]["hard_failures"] == []


def test_persistent_schema_failure_is_a_failed_draft_not_a_crash(lease_a_text, good, tmp_path):
    n_fail = 1 + SETTINGS.parse_retries  # every retry fails -> counts as one failed draft
    planner = ScriptedChatModel(responses=[OutputParserException("could not parse")] * n_fail + [good])
    judge = ScriptedChatModel(responses=[PASS])
    out = run_pipeline(lease_a_text, retrieval="full", run_executor=False, out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": ScriptedChatModel()})
    assert out["status"] == "accepted" and out["attempts"] == 2
    assert out["history"][0]["hard_failures"] == ["schema"]
    assert out["first_draft_fab_rate"] is None


def test_scope_violation_is_caught_by_hard_gate(lease_a_text, good, lease_a_clauses, truth, tmp_path):
    legal = plan_from_truth(lease_a_clauses, truth["lease_a"], legal_conclusion=True)
    planner = ScriptedChatModel(responses=[legal, good])
    judge = ScriptedChatModel(responses=[PASS])
    out = run_pipeline(lease_a_text, retrieval="full", run_executor=False, out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": ScriptedChatModel()})
    assert out["history"][0]["hard_failures"] == ["scope"]
    assert out["status"] == "accepted"


def test_verification_catches_an_executor_that_drops_questions(lease_a_text, good, tmp_path):
    """Regression (seen live): the executor emailed 1 of ~12 questions and verification still said True."""
    planner = ScriptedChatModel(responses=[good])
    judge = ScriptedChatModel(responses=[PASS])
    executor = ScriptedChatModel(responses=_executor_script(good, all_questions=False))
    out = run_pipeline(lease_a_text, retrieval="full", out_dir=str(tmp_path),
                       llms={"planner": planner, "judge": judge, "executor": executor})
    assert out["status"] == "accepted"
    assert out["actions_verified"] is False and out["actions_detail"]["email_questions"] == "1/4"


def test_executor_schedules_dated_items_even_when_kind_is_not_deadline(lease_a_text, lease_a_clauses, truth, tmp_path):
    """Regression (seen live): an insurance deadline labelled kind='risk' was skipped."""
    plan = plan_from_truth(lease_a_clauses, truth["lease_a"])
    plan.items = [it.model_copy(update={"kind": "risk"}) if it.deadline else it for it in plan.items]
    out = run_pipeline(lease_a_text, retrieval="full", out_dir=str(tmp_path), llms={
        "planner": ScriptedChatModel(responses=[plan]), "judge": ScriptedChatModel(responses=[PASS]),
        "executor": ScriptedChatModel(responses=_executor_script(plan))})
    assert out["actions_detail"]["events"] == "2/2" and out["actions_verified"] is True


def test_api_errors_propagate_instead_of_looking_like_content_failures(lease_a_text, tmp_path):
    """Regression (seen live): a 'credit balance too low' error was swallowed, retried, and reported as an
    escalation. Non-parse errors must fail loudly, immediately, with no retries and no side effects."""
    planner = ScriptedChatModel(responses=[RuntimeError("credit balance is too low")])
    with pytest.raises(RuntimeError, match="credit balance"):
        run_pipeline(lease_a_text, retrieval="full", out_dir=str(tmp_path),
                     llms={"planner": planner, "judge": ScriptedChatModel(), "executor": ScriptedChatModel()})
    assert len(planner.calls) == 1                    # not retried
    assert not list(tmp_path.iterdir())
