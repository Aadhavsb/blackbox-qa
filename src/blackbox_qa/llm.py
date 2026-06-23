"""OpenAI-compatible chat client — the only module that talks to the LLM provider.

Provider-agnostic by design: point OPENAI_BASE_URL / LLM_MODEL at Gemini's
OpenAI-compatible endpoint, a local Ollama, or OpenAI itself via .env — no code
change. The agent loop calls `chat()` and never touches the SDK directly, which
also means tests can monkeypatch `chat` to exercise the loop with no network,
no API key, and zero tokens spent.
"""

from __future__ import annotations

import ast
import json
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
            "OPENAI_API_KEY is empty. Set it in .env (a free Groq key works with "
            "OPENAI_BASE_URL=https://api.groq.com/openai/v1)."
        )
    if not settings.openai_base_url:
        # Cost guard: an empty base_url lets the OpenAI SDK default to the paid
        # api.openai.com. Require an explicit (free) endpoint so the project can
        # never silently bill. Point at Groq/Gemini/Ollama via .env.
        raise RuntimeError(
            "OPENAI_BASE_URL is empty. Set it to a free OpenAI-compatible endpoint "
            "(e.g. https://api.groq.com/openai/v1) to avoid accidentally billing "
            "api.openai.com. To intentionally use paid OpenAI, set it to "
            "https://api.openai.com/v1 explicitly."
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Prefer the server's suggested delay; else exponential backoff. Capped at 65s."""
    m = _RETRY_HINT_RE.search(str(exc))
    if m:
        return min(float(m.group(1)) + 1.0, 65.0)
    return min(2.0 ** attempt, 60.0)


_RATE_LIMIT_PHRASES = ("rate_limit", "rate limit", "tokens per minute", "request too large")


def _is_rate_limited(exc: Exception) -> bool:
    """True for 429s and Groq's 413 tokens-per-minute rate-limit (both clear by waiting)."""
    if type(exc).__name__ == "RateLimitError":
        return True
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    # Groq returns 413 for TPM caps; match the several phrasings providers use.
    return code == 413 and any(p in text for p in _RATE_LIMIT_PHRASES)


def _tool_use_failed_text(exc: Exception) -> str | None:
    """Groq 400 tool_use_failed: model wrote prose instead of tool_calls.

    The rejected answer is usually in error.failed_generation — recover it so
    the agent loop can finish instead of crashing the eval run.
    """
    if getattr(exc, "status_code", None) != 400:
        return None

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if err.get("code") == "tool_use_failed":
            text = err.get("failed_generation")
            if isinstance(text, str) and text.strip():
                return text.strip()

    msg = str(exc)
    if "tool_use_failed" not in msg:
        return None

    # OpenAI SDK often embeds the provider payload in the exception string.
    for parser in (json.loads, ast.literal_eval):
        start = msg.find("{")
        if start < 0:
            continue
        try:
            payload = parser(msg[start:])
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            err = payload.get("error") or {}
            if err.get("code") == "tool_use_failed":
                text = err.get("failed_generation")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    m = re.search(r"failed_generation['\"]:\s*['\"](.+?)['\"]\s*[,}\]]", msg, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


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

    try:
        resp = _create_with_backoff(kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised unless Groq tool_use_failed
        recovered = _tool_use_failed_text(exc)
        if recovered is not None:
            return LLMResponse(content=recovered, tool_calls=[], usage=None)
        raise

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
