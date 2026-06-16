"""Langfuse tracing + scoring adapter.

Strictly optional: when LANGFUSE_ENABLED is false (or keys are missing) every
function here is a no-op, so the agent runs identically with no observability
dependency. Tracing is async/batched by the SDK, so it stays off the request's
critical path; scores are posted out-of-band by trace_id.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from typing import Any, Iterator

from blackbox_qa.config import settings


def enabled() -> bool:
    return settings.langfuse_enabled and bool(settings.langfuse_public_key)


@cache
def _client():
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host or "http://localhost:3000",
    )


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Open a trace observation (span/generation/tool/agent). Yields None when disabled."""
    if not enabled():
        yield None
        return
    with _client().start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def current_trace_id() -> str | None:
    """The active trace id, for attaching scores later. None when disabled."""
    if not enabled():
        return None
    return _client().get_current_trace_id()


def set_trace_io(input: Any = None, output: Any = None) -> None:
    if not enabled():
        return
    _client().set_current_trace_io(input=input, output=output)


def score(
    name: str,
    value: float | str,
    *,
    trace_id: str,
    data_type: str = "NUMERIC",
    comment: str | None = None,
) -> None:
    """Attach a score to an existing trace by id (out-of-band)."""
    if not enabled():
        return
    _client().create_score(
        name=name, value=value, trace_id=trace_id, data_type=data_type, comment=comment
    )


def flush() -> None:
    """Force the background exporter to send buffered events (call before exit)."""
    if not enabled():
        return
    _client().flush()
