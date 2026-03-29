from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class ToolNotFoundError(RuntimeError):
    pass


@dataclass
class AgentSettings:
    max_steps: int = 12
    max_repeat_calls: int = 2  # same tool+args repeating => stop


def _content_to_str(content: Any) -> str:
    """AIMessage.content can be str or list[blocks]; normalize to str."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


class ToolRouter:
    def __init__(self, tools: Sequence[BaseTool]):
        self._tools = {t.name: t for t in tools}

    def invoke(self, tool_name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool not found: {tool_name}")

        out = tool.invoke(args)  # BaseTool.invoke accepts dict inputs
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


class ToolCallingAgent:
    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        settings: AgentSettings | None = None,
    ):
        self._settings = settings or AgentSettings()
        self._router = ToolRouter(tools)
        self._model: Runnable[Sequence[BaseMessage] | str, BaseMessage] = model.bind_tools(list(tools))

    def run(self, user_text: str, system_prompt: str | None = None) -> List[dict[str, Any]]:
        messages: list[BaseMessage] = []
        events: List[dict[str, Any]] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_text))

        repeat_counter: dict[str, int] = {}

        for step in range(self._settings.max_steps):
            ai_msg = self._model.invoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []

            if not tool_calls:
                events.append({"output": _content_to_str(getattr(ai_msg, "content", ""))})
                return events

            for i, tc in enumerate(tool_calls):
                name = str(tc["name"])
                args = dict(tc.get("args", {}) or {})
                call_id = str(tc.get("id") or f"call_{step}_{i}")

                events.append({"tool_name": name, "tool_args": args})

                signature = f"{name}:{json.dumps(args, sort_keys=True)}"
                repeat_counter[signature] = repeat_counter.get(signature, 0) + 1
                if repeat_counter[signature] > self._settings.max_repeat_calls:
                    events.append(
                        {
                            "output": "Stopping because the model is repeating the same tool call.\n"
                            f"Repeated: {name}({args})"
                        }
                    )
                    return events

                tool_output = self._router.invoke(name, args)
                messages.append(ToolMessage(content=tool_output, tool_call_id=call_id))

        events.append({"output": "Stopped: max_steps reached (to prevent infinite loops)."})
        return events
