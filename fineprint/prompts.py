"""Versioned prompts.

The versions are deliberately cumulative so the eval can attribute gains to a
specific change. Each version adds exactly one idea:

  v1  baseline: "list the risks"
  v2  + grounding: every item must carry a verbatim quote and clause id
  v3  + structure: checklist, severity rubric, conflict detection, deadlines,
        abstention (`not_found`), and a no-legal-advice scope rule
  v4  + procedure & calibration: step-by-step method, minimal-quote rule,
        "do not flag standard terms", self-check, and one worked example

The worked example in v4 uses clause topics (bounced-payment fees, quiet hours) that are not
planted traps in data/leases, so the eval is not contaminated.
"""
from __future__ import annotations

from .config import RISK_CATEGORIES

_V1 = """You are an assistant that reads apartment leases for tenants.
Read the lease clauses and list the risks the tenant should know about."""

_V2 = _V1 + """

Rules:
- Every item you produce must include evidence: the clause_id and a quote.
- The quote must be copied EXACTLY, character for character, from that clause. Never paraphrase inside a quote."""

_V3_RUBRIC = """
Severity rubric:
- high: could cost the tenant more than one month's rent, remove a core protection, or limit the tenant's legal remedies.
- medium: a notable cost, fee, or restriction that is worth negotiating or at least understanding.
- low: a minor restriction or inconvenience.

Item kinds:
- risk: one clause that disadvantages the tenant.
- conflict: two clauses that disagree with each other. Include a quote from EACH clause as evidence.
- deadline: a date-driven obligation. Fill `deadline` with an anchor (move_in, lease_start, lease_end) and offset_days (negative = before the anchor).
- question: something the tenant should ask the landlord to clarify.
"""

_V3 = _V2 + """

Checklist categories (use these exact strings in `category`):
{categories}

Method:
- Review each checklist category against the clauses.
- If a category is genuinely not addressed anywhere in the clauses you were given, put it in `not_found`. Do NOT invent a risk for it.
- Do not give legal advice and do not say whether any clause is legal, illegal, or enforceable. Describe what the text says and what it could mean for the tenant.
""" + _V3_RUBRIC

_V4_EXAMPLES = """
Worked examples (different lease, for format only):

Clauses:
[s09] SECTION 9 - Payment Fees: Tenant shall pay a fee of $40 for any returned or bounced payment, and a fee of $25 for each written reminder notice that Landlord sends about overdue charges.
[s14] SECTION 14 - Quiet Hours: Quiet hours are 10 p.m. to 7 a.m. Tenant is responsible for the conduct of guests.

Good item:
  kind=risk, category=late_fees, severity=low,
  title="Fees for bounced payments and reminder notices",
  explanation="Each bounced payment costs $40 and each reminder notice costs $25, which can add up quickly if you are ever late.",
  evidence=[{clause_id: "s09", quote: "a fee of $40 for any returned or bounced payment, and a fee of $25 for each written reminder notice"}]
Why it is good: the quote is short, contiguous, exactly as written, and the category comes from the checklist.

Not an item: the quiet-hours sentence. It is a standard term. Flagging standard terms makes the real risks harder to see.
"""

_V4 = _V3 + """
Procedure (follow in order):
1. List every clause that touches a checklist category.
2. For each, decide: is this a standard term, or does it shift cost/risk/rights onto the tenant beyond the norm? Only produce items for the second kind.
3. Compare clauses across sections for contradictions (for example two different numbers of days or fees for the same thing). Report each contradiction as ONE conflict item quoting both clauses.
4. Extract every date-driven obligation as a deadline item.
5. Quote the shortest span that proves the point (one sentence or less). Copy it exactly.
6. Before answering, re-read each quote against its clause. If you cannot find it character for character, shorten or fix it.
7. Order items by severity, high first.
""" + _V4_EXAMPLES

PROMPTS = {"v1": _V1, "v2": _V2, "v3": _V3, "v4": _V4}


def planner_system_prompt(version: str) -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; choose from {sorted(PROMPTS)}")
    text = PROMPTS[version]
    if "{categories}" in text:
        text = text.replace("{categories}", ", ".join(RISK_CATEGORIES))
    return text


def planner_user_prompt(clauses_text: str, feedback: list[str] | None = None) -> str:
    parts = ["Lease clauses:\n\n" + clauses_text]
    if feedback:
        parts.append(
            "Your previous draft failed automated validation. Problems found:\n"
            + "\n".join(f"- {f}" for f in feedback)
            + "\n\nReturn the full corrected plan. Fix these problems and keep everything that was correct."
        )
    return "\n\n".join(parts)


JUDGE_SYSTEM = """You are a careful reviewer checking an AI-generated lease review plan for a tenant.
You are given the FULL lease (every clause) and the plan.

Your job:
1. groundedness (1-5): are the plan's claims supported by the quoted text? 5 = fully supported, 1 = mostly unsupported.
2. scope_ok: false if the plan gives legal advice or asserts that a clause is legal, illegal, or enforceable.
3. missed_risks: scan the FULL lease for clauses that are unusual or tenant-unfavourable beyond a standard residential lease, or pairs of clauses that contradict each other, that the plan does NOT cover. List each with its clause_id and a one-sentence reason. Do NOT list standard, benign terms. Do not list something the plan already covers.
4. unsupported_claims: statements in the plan's explanations that the quoted text does not support.
5. verdict: "fail" if there is any missed HIGH-impact risk (cost over one month's rent, loss of a core protection, or limited legal remedies), any contradiction left unreported, scope_ok is false, or groundedness is below 4. Otherwise "pass".

Be strict about missed risks but do not invent risks. Judge only from the text provided."""


def judge_user_prompt(clauses_text: str, plan_json: str) -> str:
    return f"FULL LEASE:\n\n{clauses_text}\n\nPLAN TO REVIEW (JSON):\n\n{plan_json}"


EXECUTOR_SYSTEM = """You turn an already-validated lease review plan into actions using the tools provided.
Rules:
- Schedule EVERY deadline item listed: for each one, call compute_deadline, then add_calendar_event once with the date it returned.
- At the end, call draft_questions_email exactly once, passing EVERY question from the "Questions for the landlord" list, unchanged and in the same order. Never select a subset or merge them.
- Use only values that appear in the plan or in the provided key dates. Never invent dates, quotes, or questions.
- If an anchor date is listed as unknown, do not guess it: skip that item and say so in your final message.
- When finished, reply with one short sentence summarising what you did."""
