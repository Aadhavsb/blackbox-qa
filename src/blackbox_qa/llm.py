"""OpenAI-compatible chat client — the only module that talks to the LLM provider.

Provider-agnostic by design: point OPENAI_BASE_URL / LLM_MODEL at Gemini's
OpenAI-compatible endpoint, a local Ollama, or OpenAI itself via .env — no code
change. The agent loop calls `chat()` and never touches the SDK directly, which
also means tests can monkeypatch `chat` to exercise the loop with no network,
no API key, and zero tokens spent.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import cache
from typing import Any

MAX_RETRIES = 5
_RETRY_HINT_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model emitted it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: dict[str, int] | None = None  # prompt/completion/total tokens, if reported

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


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Prefer the server's suggested delay; else exponential backoff. Capped at 65s."""
    m = _RETRY_HINT_RE.search(str(exc))
    if m:
        return min(float(m.group(1)) + 1.0, 65.0)
    return min(2.0 ** attempt, 60.0)


def _is_rate_limited(exc: Exception) -> bool:
    """True for 429s and Groq's 413 tokens-per-minute rate-limit (both clear by waiting)."""
    if type(exc).__name__ == "RateLimitError":
        return True
    code = getattr(exc, "status_code", None)
    return code == 413 and "rate_limit" in str(exc).lower()


def _create_with_backoff(kwargs: dict[str, Any]):
    """Call the API, retrying on rate-limit errors with backoff."""
    client = _client()
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
            if not _is_rate_limited(exc) or attempt == MAX_RETRIES:
                raise
            time.sleep(_retry_delay(exc, attempt))


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

    resp = _create_with_backoff(kwargs)
    msg = resp.choices[0].message
    calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (msg.tool_calls or [])
    ]
    usage = None
    if resp.usage is not None:
        usage = {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
            "total": resp.usage.total_tokens,
        }
    return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)
