"""Render a finished pipeline state as Markdown (used by the CLI)."""
from __future__ import annotations

_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
_SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def render_markdown(state: dict) -> str:
    plan = state.get("plan")
    lines: list[str] = []
    status = state.get("status", "pending")
    badge = {"accepted": "✅ accepted", "escalated": "⚠️ ESCALATED to a human reviewer"}.get(status, status)
    lines.append(f"**Status:** {badge}  |  **drafts:** {state.get('attempts', 0)}")
    if status == "escalated":
        lines.append("\n> The plan below did not pass validation after the allowed repairs. "
                     "Nothing was executed. Treat it as an unverified draft.")
    if plan is None:
        return "\n".join(lines + ["\nNo plan could be produced."])
    lines.append(f"\n{plan.summary}\n")
    for it in sorted(plan.items, key=lambda i: _SEV_ORDER[i.severity]):
        lines.append(f"### {_SEV_ICON[it.severity]} {it.title}  \n`{it.kind}` · `{it.category}` · {it.severity}")
        lines.append(it.explanation)
        for e in it.evidence:
            lines.append(f"> **{e.clause_id}:** “{e.quote}”")
        if it.deadline:
            lines.append(f"*Deadline:* {it.deadline.offset_days:+d} days from `{it.deadline.anchor}`")
        if it.question_for_landlord:
            lines.append(f"*Ask:* {it.question_for_landlord}")
        lines.append("")
    if plan.not_found:
        lines.append("**Not addressed in this document:** " + ", ".join(plan.not_found))
    hist = state.get("history", [])
    if hist:
        lines.append("\n**Validation history**")
        for h in hist:
            gates = ", ".join(h["hard_failures"]) or "all hard gates passed"
            lines.append(f"- draft {h['attempt']}: {gates}; judge: {h['judge']}")
    if state.get("actions_summary"):
        lines.append(f"\n**Actions:** {state['actions_summary']} "
                     f"(verified against plan: {state.get('actions_verified')}; {state.get('actions_detail')})")
    return "\n".join(lines)
