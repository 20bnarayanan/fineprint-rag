"""Typed contracts between pipeline stages.

The plan is a *typed artifact*, not prose. That is what makes it checkable.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _coerce_json(v: Any) -> Any:
    """Models occasionally emit a list/object as a JSON *string* in tool-call arguments
    (seen live: `items` returned as '[{"kind": ...}]'). Parse it instead of failing the draft."""
    if isinstance(v, str) and v.strip()[:1] in ("[", "{"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


class Clause(BaseModel):
    id: str  # e.g. "s06"
    number: int
    title: str
    text: str


class Evidence(BaseModel):
    clause_id: str = Field(description="Id of the clause the quote comes from, e.g. 's06'.")
    quote: str = Field(description="A verbatim, contiguous quote copied exactly from that clause.")


class DeadlineSpec(BaseModel):
    anchor: Literal["move_in", "lease_start", "lease_end"] = Field(
        description="What the deadline is measured from.")
    offset_days: int = Field(
        description="Days relative to the anchor. Negative means before (e.g. -90 for 'at least 90 days before the end of the term').")


class PlanItem(BaseModel):
    kind: Literal["risk", "conflict", "deadline", "question"]
    category: str = Field(description="One of the risk categories in the checklist.")
    severity: Literal["high", "medium", "low"]
    title: str = Field(description="Short headline, at most 12 words.")
    explanation: str = Field(description="Plain-English explanation of why this matters to the tenant, 1-3 sentences.")
    evidence: list[Evidence] = Field(
        description="Verbatim quotes supporting this item. A conflict item needs a quote from each conflicting clause.")
    question_for_landlord: Optional[str] = Field(
        default=None, description="A question the tenant could ask the landlord about this item.")
    deadline: Optional[DeadlineSpec] = Field(
        default=None, description="Only for kind='deadline'.")

    @field_validator("evidence", "deadline", mode="before")
    @classmethod
    def _parse_json_strings(cls, v: Any) -> Any:
        return _coerce_json(v)


class Plan(BaseModel):
    summary: str = Field(description="Two sentences: the overall picture for the tenant.")
    items: list[PlanItem]
    not_found: list[str] = Field(
        default_factory=list,
        description="Checklist categories that this document genuinely does not address.")

    @field_validator("items", "not_found", mode="before")
    @classmethod
    def _parse_json_strings(cls, v: Any) -> Any:
        return _coerce_json(v)


class MissedRisk(BaseModel):
    clause_id: str
    why: str


class JudgeVerdict(BaseModel):
    groundedness: int = Field(ge=1, le=5, description="5 = every claim is supported by the cited text.")
    scope_ok: bool = Field(description="False if the plan gives legal advice or asserts legality/enforceability.")
    missed_risks: list[MissedRisk] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    verdict: Literal["pass", "fail"]
    rationale: str

    @field_validator("missed_risks", "unsupported_claims", mode="before")
    @classmethod
    def _parse_json_strings(cls, v: Any) -> Any:
        return _coerce_json(v)


class Failure(BaseModel):
    """One validation failure, phrased so it can be fed back to the planner."""
    gate: str  # e.g. "verbatim_quote"
    item_index: Optional[int] = None
    message: str


class ValidationReport(BaseModel):
    hard_failures: list[Failure] = Field(default_factory=list)
    judge: Optional[JudgeVerdict] = None

    @property
    def passed_hard_gates(self) -> bool:
        return not self.hard_failures
