"""Scripted chat model for offline tests. No network, no API keys."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns queued responses in order.

    * with_structured_output(schema): pops an object/dict/Exception and validates it.
    * plain chat (used by the executor agent): pops an AIMessage.
    """
    responses: list[Any] = Field(default_factory=list)
    calls: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _pop(self):
        if not self.responses:
            raise AssertionError("ScriptedChatModel script exhausted")
        return self.responses.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        r = self._pop()
        msg = r if isinstance(r, AIMessage) else AIMessage(content=str(r))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.calls.append(messages)
            r = self._pop()
            if isinstance(r, Exception):
                raise r
            return r if isinstance(r, schema) else schema.model_validate(r)
        return RunnableLambda(_call)
