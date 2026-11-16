"""Model construction and the two LLM-backed stages (planner, judge).

The judge runs on a different, stronger model than the planner. Reasons:
  - different models make different mistakes, so the judge is less likely to share
    the planner's blind spots,
  - self-preference bias: a model tends to rate its own style of output favourably.
"""
from __future__ import annotations

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .config import SETTINGS
from .prompts import JUDGE_SYSTEM, judge_user_prompt, planner_system_prompt, planner_user_prompt
from .schemas import JudgeVerdict, Plan


class EmptyStructuredOutput(ValueError):
    """The model returned nothing parseable at all."""


# ONLY these count as "the model's output was unparseable". Everything else (billing, auth, rate limits,
# network) must propagate: swallowing it makes an outage look like a content failure. Seen live: a
# "credit balance too low" error was retried, "repaired", and reported as an escalation.
PARSE_ERRORS = (OutputParserException, ValidationError, json.JSONDecodeError, EmptyStructuredOutput)


def ensure_api_key() -> None:
    """Fail with a readable message (not a stack trace) when the Anthropic key is missing."""
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                         "or export it in your shell.")


def get_chat_model(name: str, *, max_tokens: int = 16000) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic  # lazy: keeps tests free of API config
    # No `temperature`: the current Claude models reject non-default values, so sampling settings are left at defaults.
    return ChatAnthropic(model=name, max_tokens=max_tokens)


def default_llms() -> dict[str, BaseChatModel]:
    return {
        "planner": get_chat_model(SETTINGS.planner_model),
        "judge": get_chat_model(SETTINGS.judge_model, max_tokens=4096),
        "executor": get_chat_model(SETTINGS.executor_model, max_tokens=4096),
    }


def _invoke_with_parse_retries(structured, messages, retries: int):
    """Retry when the model's output cannot be parsed into the schema.

    A parse failure says nothing about the quality of the plan, so it must not consume the
    repair budget. The last exception is raised if every attempt fails.
    """
    last: Exception | None = None
    for _ in range(1 + retries):
        try:
            out = structured.invoke(messages)
            if out is None:
                raise EmptyStructuredOutput("model returned no structured output")
            return out
        except PARSE_ERRORS as exc:
            last = exc
    raise last  # type: ignore[misc]


def run_planner(llm: BaseChatModel, version: str, clauses_text: str, feedback: list[str] | None = None) -> Plan:
    # Tool-calling mode: in live tests JSON-schema mode on Haiku failed to parse far more often (6/8 vs 1/8).
    # max_tokens is generous on purpose: a full plan is 3-5k tokens, and a truncated plan fails to parse
    # (seen live with the v1 prompt at a 4096 cap, which caused spurious escalations).
    structured = llm.with_structured_output(Plan)
    out = _invoke_with_parse_retries(structured, [
        SystemMessage(content=planner_system_prompt(version)),
        HumanMessage(content=planner_user_prompt(clauses_text, feedback)),
    ], SETTINGS.parse_retries)
    return out if isinstance(out, Plan) else Plan.model_validate(out)


def run_judge(llm: BaseChatModel, all_clauses_text: str, plan: Plan) -> JudgeVerdict:
    # JSON-schema mode: forced tool calling is not supported on claude-sonnet-5-5 (LangChain warns about it).
    structured = llm.with_structured_output(JudgeVerdict, method="json_schema")
    out = _invoke_with_parse_retries(structured, [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=judge_user_prompt(all_clauses_text, plan.model_dump_json(indent=2))),
    ], SETTINGS.parse_retries)
    return out if isinstance(out, JudgeVerdict) else JudgeVerdict.model_validate(out)
