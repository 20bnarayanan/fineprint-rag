"""Deterministic hard gates.

Design rule: anything that can be checked with code is checked with code. An LLM
judge is only used for what code cannot do (judging coverage and faithfulness of
explanations). Gates return `Failure`s phrased as repair instructions, so the same
object is both the validation report and the feedback prompt.
"""
from __future__ import annotations

import re

from .config import RISK_CATEGORIES, SETTINGS
from .schemas import Clause, Failure, Plan

_QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def normalize(text: str) -> str:
    """Whitespace- and typography-insensitive form used ONLY for quote matching.

    Case and wording are preserved: a paraphrase must never pass.
    """
    return re.sub(r"\s+", " ", text.translate(_QUOTE_TRANSLATION)).strip()


def quote_in_clause(quote: str, clause: Clause) -> bool:
    return normalize(quote) in normalize(clause.text)


_SCOPE_PATTERNS = [
    r"\billegal\b", r"\bunlawful\b", r"\bunenforceable\b", r"\bnot enforceable\b",
    r"\bagainst the law\b", r"\bviolates? (?:state|federal|local)? ?(?:law|statute)",
    r"\byou should (?:sue|hire a lawyer|get a lawyer)\b", r"\bwould not hold up in court\b",
]
_SCOPE_RE = re.compile("|".join(_SCOPE_PATTERNS), re.IGNORECASE)


def _plan_text_fields(plan: Plan) -> list[tuple[int | None, str]]:
    fields: list[tuple[int | None, str]] = [(None, plan.summary)]
    for i, it in enumerate(plan.items):
        fields.append((i, it.title))
        fields.append((i, it.explanation))
        if it.question_for_landlord:
            fields.append((i, it.question_for_landlord))
    return fields


def run_hard_gates(plan: Plan, clauses: list[Clause]) -> list[Failure]:
    by_id = {c.id: c for c in clauses}
    failures: list[Failure] = []

    for i, item in enumerate(plan.items):
        # G1: category must come from the checklist
        if item.category not in RISK_CATEGORIES:
            failures.append(Failure(
                gate="category", item_index=i,
                message=f"Item {i} uses category {item.category!r}, which is not in the checklist. "
                        f"Use one of: {', '.join(RISK_CATEGORIES)}."))

        # G2: evidence required (conflicts need two distinct clauses)
        if not item.evidence:
            failures.append(Failure(
                gate="evidence_required", item_index=i,
                message=f"Item {i} ({item.title!r}) has no evidence. Add a verbatim quote or remove the item."))
        elif item.kind == "conflict" and len({e.clause_id for e in item.evidence}) < 2:
            failures.append(Failure(
                gate="conflict_needs_two_clauses", item_index=i,
                message=f"Item {i} is a conflict but quotes fewer than two different clauses. "
                        "Quote each conflicting clause, or change the kind."))

        # G3: every quote must exist verbatim in the clause it cites
        for e in item.evidence:
            clause = by_id.get(e.clause_id)
            if clause is None:
                failures.append(Failure(
                    gate="clause_exists", item_index=i,
                    message=f"Item {i} cites clause_id {e.clause_id!r}, which does not exist. "
                            f"Valid ids: {', '.join(sorted(by_id))}."))
                continue
            if len(normalize(e.quote)) < SETTINGS.min_quote_chars:
                failures.append(Failure(
                    gate="quote_too_short", item_index=i,
                    message=f"Item {i}: quote {e.quote!r} is too short to be meaningful "
                            f"(minimum {SETTINGS.min_quote_chars} characters)."))
            elif not quote_in_clause(e.quote, clause):
                failures.append(Failure(
                    gate="verbatim_quote", item_index=i,
                    message=f"Item {i}: the quote {e.quote!r} does not appear verbatim in clause "
                            f"{e.clause_id}. Copy the text exactly or remove the item."))

        # G4: deadline items need a deadline spec
        if item.kind == "deadline" and item.deadline is None:
            failures.append(Failure(
                gate="deadline_spec", item_index=i,
                message=f"Item {i} is a deadline but has no `deadline` anchor/offset_days."))

    # G5: scope - no legal conclusions
    for idx, text in _plan_text_fields(plan):
        m = _SCOPE_RE.search(text)
        if m:
            where = "the summary" if idx is None else f"item {idx}"
            failures.append(Failure(
                gate="scope", item_index=idx,
                message=f"In {where}: the phrase {m.group(0)!r} makes a legal conclusion. "
                        "Describe what the text says without judging legality or enforceability."))

    # G6: abstention honesty - "not found" must not contradict the document
    flagged_categories = {it.category for it in plan.items}
    for cat in plan.not_found:
        if cat not in RISK_CATEGORIES:
            failures.append(Failure(
                gate="not_found_category",
                message=f"not_found contains {cat!r}, which is not a checklist category."))
            continue
        if cat in flagged_categories:
            failures.append(Failure(
                gate="not_found_contradiction",
                message=f"Category {cat!r} is in not_found but also has plan items. Pick one."))
            continue
        for kw in RISK_CATEGORIES[cat]["keywords"]:
            pat = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            hit = next((c for c in clauses if pat.search(c.text)), None)
            if hit:
                failures.append(Failure(
                    gate="not_found_keyword",
                    message=f"You listed {cat!r} as not addressed, but clause {hit.id} mentions {kw!r}. "
                            f"Re-read {hit.id}: either flag it or explain why it is a standard term."))
                break
    return failures


def fabricated_quote_rate(plan: Plan, clauses: list[Clause]) -> float:
    """Fraction of evidence quotes that are NOT verbatim in their cited clause."""
    by_id = {c.id: c for c in clauses}
    total = bad = 0
    for item in plan.items:
        for e in item.evidence:
            total += 1
            clause = by_id.get(e.clause_id)
            if clause is None or not quote_in_clause(e.quote, clause):
                bad += 1
    return bad / total if total else 0.0
