"""Offline test of the eval harness: scripted models stand in for the planner and judge."""
import json

from fakes import ScriptedChatModel
from evals.evaluators import ALL_EVALUATORS
from evals.run_evals import Config, aggregate, load_cases, run_local, run_target, to_markdown
from fineprint.ingest import split_clauses
from fineprint.schemas import JudgeVerdict
from fineprint.synth import plan_from_truth

PASS = JudgeVerdict(groundedness=5, scope_ok=True, verdict="pass", rationale="ok")


def _scripted_target(dropped: tuple[str, ...] = ()):
    """A target whose planner returns a plan built from ground truth, optionally missing traps."""
    def target(inputs, cfg):
        case = next(c for c in load_cases() if c["lease_id"] == inputs["lease_id"])
        plan = plan_from_truth(split_clauses(case["text"]), case["truth"],
                               drop=tuple(d for d in dropped if any(t["id"] == d for t in case["truth"]["traps"])))
        llms = {"planner": ScriptedChatModel(responses=[plan]), "judge": ScriptedChatModel(responses=[PASS]),
                "executor": ScriptedChatModel()}
        return run_target(inputs, cfg, llms=llms)
    return target


def test_evaluators_return_langsmith_shaped_results():
    case = next(c for c in load_cases() if c["lease_id"] == "lease_a")
    out = _scripted_target()({"lease_id": "lease_a", "text": case["text"]}, Config(retrieval="full"))
    for ev in ALL_EVALUATORS:
        r = ev(out, case["truth"])
        assert set(r) >= {"key", "score"} and isinstance(r["key"], str)


def test_perfect_planner_scores_perfectly_and_clean_lease_is_clean():
    rows = run_local([Config(retrieval="full")], load_cases(), target=_scripted_target())
    agg = aggregate(rows)["v4-full-val"]
    assert agg["trap_recall"] == 1.0 and agg["high_severity_recall"] == 1.0 and agg["deadline_recall"] == 1.0
    assert agg["clean_lease_flags"] == 0.0 and agg["extra_flags_per_lease"] == 0.0
    assert agg["fabricated_quote_rate_final"] == 0.0 and agg["escalated"] == 0.0


def test_missing_traps_show_up_in_recall():
    rows = run_local([Config(retrieval="full")], load_cases(), target=_scripted_target(dropped=("auto_renewal",)))
    agg = aggregate(rows)["v4-full-val"]
    # auto_renewal exists in lease_a and lease_d (2 of 16 traps) -> recall < 1; others unaffected
    assert 0.8 < agg["trap_recall"] < 1.0


def test_markdown_table_has_a_row_per_config(tmp_path):
    rows = run_local([Config(retrieval="full"), Config("v3", "full", False)], load_cases(["lease_a", "lease_e_clean"]),
                     target=_scripted_target())
    md = to_markdown(aggregate(rows), 2)
    assert md.count("\n| v") == 2 and "trap recall" in md and "n/a" not in md.split("\n")[2]


def test_partial_results_are_written_as_runs_finish(tmp_path):
    part = tmp_path / "runs.partial.jsonl"
    run_local([Config(retrieval="full")], load_cases(["lease_a", "lease_e_clean"]),
              target=_scripted_target(), partial_path=part)
    rows = [json.loads(line) for line in part.read_text().splitlines()]
    assert {r["lease_id"] for r in rows} == {"lease_a", "lease_e_clean"} and all("scores" in r for r in rows)
