"""The pipeline as a LangGraph state machine: plan -> validate -> (repair | escalate | execute).

            +--------+   +----------+   +------+   +----------+
 START ---> | ingest |-->| retrieve |-->| plan |-->| validate |
            +--------+   +----------+   +------+   +----------+
                                           ^            | pass            | fail, budget left     | fail, no budget
                                           |            v                 v                       v
                                           |        +---------+      +--------+             +----------+
                                           +--------| execute |      | repair |------------>| escalate |
                                           (repair loops to plan)     +--------+             +----------+

Key properties:
  * Tools run only on the `execute` edge, i.e. only after validation passed.
  * The judge sees the FULL lease while the planner may see retrieved clauses only,
    so the judge can catch retrieval misses; `repair` then widens the planner context.
  * After `max_repairs` failed repairs the run is escalated to a human instead of
    shipping an unverified plan.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import SETTINGS
from .ingest import extract_term_dates, format_clauses, split_clauses
from .llm import PARSE_ERRORS, default_llms, run_judge, run_planner
from .prompts import EXECUTOR_SYSTEM
from .retrieval import build_index, get_embeddings, retrieve_for_checklist
from .schemas import Clause, Failure, JudgeVerdict, Plan, ValidationReport
from .tools import compute_deadline_date, make_tools
from .validators import fabricated_quote_rate, run_hard_gates


class State(TypedDict, total=False):
    # inputs / settings
    lease_text: str
    prompt_version: str
    retrieval_mode: Literal["rag", "full"]
    use_validator: bool
    run_executor: bool
    out_dir: str
    move_in: Optional[str]
    # derived
    clauses: list[Clause]
    key_dates: dict[str, Optional[date]]
    context_ids: list[str]
    retrieval_hits: dict[str, list[str]]
    # loop state
    plan: Optional[Plan]
    feedback: list[str]
    attempts: int
    report: Optional[ValidationReport]
    history: list[dict]
    first_draft_fab_rate: Optional[float]
    first_draft_gate_failures: Optional[int]
    # outcome
    status: Literal["pending", "accepted", "escalated"]
    actions_summary: str
    actions_verified: Optional[bool]
    actions_detail: Optional[dict]


def build_graph(llms: dict[str, Any] | None = None, embeddings=None):
    """Compile the graph. `llms` and `embeddings` can be injected (tests use fakes)."""
    llms = llms or default_llms()

    # ---------------------------------------------------------------- nodes
    def ingest(state: State) -> dict:
        clauses = split_clauses(state["lease_text"])
        start, end = extract_term_dates(state["lease_text"])
        move_in = date.fromisoformat(state["move_in"]) if state.get("move_in") else start
        return {
            "clauses": clauses,
            "key_dates": {"lease_start": start, "lease_end": end, "move_in": move_in},
            "attempts": 0, "feedback": [], "history": [], "status": "pending",
        }

    def retrieve(state: State) -> dict:
        clauses = state["clauses"]
        if state.get("retrieval_mode", "rag") == "full":
            return {"context_ids": [c.id for c in clauses], "retrieval_hits": {}}
        index = build_index(clauses, embeddings or get_embeddings())
        chosen, hits = retrieve_for_checklist(index, clauses)
        return {"context_ids": [c.id for c in chosen], "retrieval_hits": hits}

    def _context_text(state: State) -> str:
        wanted = set(state["context_ids"])
        return format_clauses([c for c in state["clauses"] if c.id in wanted])

    def plan_node(state: State) -> dict:
        attempts = state["attempts"] + 1
        try:
            plan = run_planner(llms["planner"], state["prompt_version"], _context_text(state),
                               state["feedback"] or None)
            err = None
        except PARSE_ERRORS as exc:  # unparseable output counts as a failed draft; API errors propagate
            plan, err = None, f"The previous output could not be parsed into the required schema: {exc}"
        update: dict[str, Any] = {"attempts": attempts, "plan": plan}
        if err:
            update["feedback"] = [err]
        if attempts == 1:  # remember how the *unvalidated* first draft did
            if plan is not None:
                update["first_draft_fab_rate"] = fabricated_quote_rate(plan, state["clauses"])
                update["first_draft_gate_failures"] = len(run_hard_gates(plan, state["clauses"]))
            else:
                update["first_draft_fab_rate"] = None
                update["first_draft_gate_failures"] = None
        return update

    def validate(state: State) -> dict:
        plan = state.get("plan")
        history = list(state["history"])
        if plan is None:
            report = ValidationReport(hard_failures=[Failure(gate="schema", message=state["feedback"][0])])
            history.append({"attempt": state["attempts"], "hard_failures": ["schema"], "judge": None})
            return {"report": report, "history": history, "status": "pending"}

        if not state.get("use_validator", True):
            history.append({"attempt": state["attempts"], "hard_failures": [], "judge": "skipped"})
            return {"report": ValidationReport(), "history": history, "status": "accepted"}

        failures = run_hard_gates(plan, state["clauses"])
        report = ValidationReport(hard_failures=failures)
        judge_note: Any = "not_run"
        if not failures:  # fail fast: only spend a judge call on plans that pass the cheap gates
            verdict = run_judge(llms["judge"], format_clauses(state["clauses"]), plan)
            report.judge = verdict
            judge_note = verdict.verdict
        history.append({"attempt": state["attempts"], "hard_failures": [f.gate for f in failures], "judge": judge_note})
        ok = report.passed_hard_gates and report.judge is not None and report.judge.verdict == "pass"
        return {"report": report, "history": history, "status": "accepted" if ok else "pending"}

    def repair(state: State) -> dict:
        report = state["report"]
        feedback: list[str] = []
        context_ids = set(state["context_ids"])
        if report is not None:
            feedback += [f.message for f in report.hard_failures]
            if report.judge is not None:
                j: JudgeVerdict = report.judge
                valid_ids = {c.id for c in state["clauses"]}
                for m in j.missed_risks:
                    feedback.append(f"A reviewer found a risk you missed in clause {m.clause_id}: {m.why} "
                                    "Add an item for it with a verbatim quote.")
                    if m.clause_id in valid_ids:
                        context_ids.add(m.clause_id)  # widen retrieval: the planner may never have seen it
                feedback += [f"Unsupported claim: {u}" for u in j.unsupported_claims]
                if not j.scope_ok:
                    feedback.append("Remove any statement about legality or enforceability.")
                if j.groundedness < 4 and not (j.missed_risks or j.unsupported_claims):
                    feedback.append(f"Reviewer rationale: {j.rationale}")
        return {"feedback": feedback or ["The plan did not pass review; improve grounding and coverage."],
                "context_ids": sorted(context_ids)}

    def escalate(state: State) -> dict:
        return {"status": "escalated"}

    def execute(state: State) -> dict:
        plan = state["plan"]
        if not state.get("run_executor", True) or plan is None:
            return {"status": "accepted", "actions_summary": "executor skipped", "actions_verified": None}
        from langchain.agents import create_agent  # lazy import

        kd = state["key_dates"]
        out_dir = state.get("out_dir") or "outputs"
        tools = make_tools(out_dir, kd)
        agent = create_agent(model=llms["executor"], tools=tools, system_prompt=EXECUTOR_SYSTEM)
        # Select by the presence of a deadline spec, not by `kind`: live runs showed the planner sometimes
        # labels a dated obligation kind="risk" while still filling in the deadline.
        deadline_items = [it.model_dump() for it in plan.items if it.deadline]
        questions = list(dict.fromkeys(q.strip() for it in plan.items
                                       if (q := it.question_for_landlord) and q.strip()))
        msg = ("Key dates (ISO; UNKNOWN means do not guess): "
               + json.dumps({k: (v.isoformat() if v else "UNKNOWN") for k, v in kd.items()})
               + "\n\nDeadline items:\n" + json.dumps(deadline_items, indent=2)
               + "\n\nQuestions for the landlord:\n" + json.dumps(questions, indent=2))
        result = agent.invoke({"messages": [{"role": "user", "content": msg}]})
        summary = result["messages"][-1].content
        if isinstance(summary, list):
            summary = " ".join(b.get("text", "") for b in summary if isinstance(b, dict))

        # Verify the executor did what the plan says (the executor is an LLM too): every resolvable deadline
        # became a calendar event, and the email contains EVERY question, not a subset.
        expected_events = sum(1 for it in deadline_items
                              if compute_deadline_date(it["deadline"]["anchor"], it["deadline"]["offset_days"], kd))
        ics = Path(out_dir) / "lease_deadlines.ics"
        actual_events = ics.read_text().count("BEGIN:VEVENT") if ics.exists() else 0
        email = Path(out_dir) / "questions_for_landlord.md"
        actual_questions = sum(1 for ln in email.read_text().splitlines()
                               if ln[:1].isdigit() and ". " in ln[:5]) if email.exists() else 0
        detail = {"events": f"{actual_events}/{expected_events}", "email_questions": f"{actual_questions}/{len(questions)}"}
        ok = actual_events == expected_events and actual_questions == len(questions)
        return {"actions_summary": str(summary), "actions_verified": ok, "actions_detail": detail, "status": "accepted"}

    # -------------------------------------------------------------- routing
    def after_validate(state: State) -> str:
        if state["status"] == "accepted":
            return "execute"
        # attempts counts drafts; allow 1 initial draft + max_repairs repairs
        return "repair" if state["attempts"] <= SETTINGS.max_repairs else "escalate"

    g = StateGraph(State)
    g.add_node("ingest", ingest)
    g.add_node("retrieve", retrieve)
    g.add_node("plan", plan_node)
    g.add_node("validate", validate)
    g.add_node("repair", repair)
    g.add_node("escalate", escalate)
    g.add_node("execute", execute)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "validate")
    g.add_conditional_edges("validate", after_validate,
                            {"execute": "execute", "repair": "repair", "escalate": "escalate"})
    g.add_edge("repair", "plan")
    g.add_edge("execute", END)
    g.add_edge("escalate", END)
    return g.compile()


def run_pipeline(lease_text: str, *, prompt_version: str = "v4", retrieval: str = "rag",
                 use_validator: bool = True, run_executor: bool = True, out_dir: str = "outputs",
                 move_in: str | None = None, llms=None, embeddings=None, tags: list[str] | None = None,
                 metadata: dict | None = None) -> State:
    """Run the pipeline once. Tags/metadata land on the LangSmith trace when tracing is enabled."""
    graph = build_graph(llms=llms, embeddings=embeddings)
    config = {
        "run_name": "fineprint",
        "tags": [f"prompt:{prompt_version}", f"retrieval:{retrieval}",
                 f"validator:{'on' if use_validator else 'off'}"] + (tags or []),
        "metadata": {"prompt_version": prompt_version, "retrieval": retrieval,
                     "use_validator": use_validator, **(metadata or {})},
    }
    return graph.invoke({
        "lease_text": lease_text, "prompt_version": prompt_version, "retrieval_mode": retrieval,
        "use_validator": use_validator, "run_executor": run_executor, "out_dir": out_dir,
        "move_in": move_in,
    }, config=config)
