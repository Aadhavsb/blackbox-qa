"""OpenAI-compatible chat client — the only module that talks to the LLM provider.

Provider-agnostic by design: point OPENAI_BASE_URL / LLM_MODEL at Gemini's
OpenAI-compatible endpoint, a local Ollama, or OpenAI itself via .env — no code
change. The agent loop calls `chat()` and never touches the SDK directly, which
also means tests can monkeypatch `chat` to exercise the loop with no network,
no API key, and zero tokens spent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model emitted it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


@cache
def _client():
    # Lazy import so the package loads without the openai dep present.
    from openai import OpenAI

    from blackbox_qa.config import settings

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty. Set it in .env (a free Google AI Studio key "
            "works with OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/)."
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    model: str | None = None,
    temperature: float = 0.0,
) -> LLMResponse:
    """Send one chat completion. Returns either text content or tool calls.

    `tool_choice` is only forwarded when tools are provided; pass "none" on the
    final turn to force a text answer (the loop's graceful floor).
    """
    from blackbox_qa.config import settings

    kwargs: dict[str, Any] = {
        "model": model or settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    resp = _client().chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (msg.tool_calls or [])
    ]
    return LLMResponse(content=msg.content, tool_calls=calls)
