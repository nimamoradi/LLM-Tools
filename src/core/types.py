from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    # tool response messages
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    # assistant messages that *request* tools
    tool_calls: Optional[list[ToolCall]] = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall]
    raw: Any = None  # keep provider-native response for debugging
