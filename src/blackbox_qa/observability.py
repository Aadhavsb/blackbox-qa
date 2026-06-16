"""Langfuse tracing + scoring adapter.

Strictly optional: when LANGFUSE_ENABLED is false (or keys are missing) every
function here is a no-op, so the agent runs identically with no observability
dependency. Tracing is async/batched by the SDK, so it stays off the request's
critical path; scores are posted out-of-band by trace_id.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import Any, Iterator

from blackbox_qa.config import settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    # Require BOTH keys: a public key alone lets create_score / SDK calls fail later.
    return (
        settings.langfuse_enabled
        and bool(settings.langfuse_public_key)
        and bool(settings.langfuse_secret_key)
    )


@cache
def _client():
    """Build the Langfuse client. Returns None (logged) if construction fails, so
    observability never takes the agent down."""
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host or "http://localhost:3000",
        )
    except Exception:  # noqa: BLE001 - tracing must never break the app
        logger.warning("Langfuse client init failed; observability disabled", exc_info=True)
        return None


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Open a trace observation (span/generation/tool/agent). Yields None when
    disabled OR when the SDK errors — the caller's logic is identical either way."""
    client = _client() if enabled() else None
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
            yield obs
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse observation failed; continuing untraced", exc_info=True)
        yield None


def current_trace_id() -> str | None:
    """The active trace id, for attaching scores later. None when disabled/errored."""
    client = _client() if enabled() else None
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse get_current_trace_id failed", exc_info=True)
        return None


def set_trace_io(input: Any = None, output: Any = None) -> None:
    client = _client() if enabled() else None
    if client is None:
        return
    try:
        client.set_current_trace_io(input=input, output=output)
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse set_current_trace_io failed", exc_info=True)


def score(
    name: str,
    value: float | str,
    *,
    trace_id: str,
    data_type: str = "NUMERIC",
    comment: str | None = None,
) -> None:
    """Attach a score to an existing trace by id (out-of-band)."""
    client = _client() if enabled() else None
    if client is None:
        return
    try:
        client.create_score(
            name=name, value=value, trace_id=trace_id, data_type=data_type, comment=comment
        )
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse create_score failed", exc_info=True)


def flush() -> None:
    """Force the background exporter to send buffered events (call before exit)."""
    client = _client() if enabled() else None
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse flush failed", exc_info=True)
