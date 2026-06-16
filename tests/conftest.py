"""Shared test fixtures.

Disable Langfuse observability for every test by default so unit tests are
deterministic and never touch the network, regardless of a local .env that
enables it. Tests that exercise the enabled path re-enable it via monkeypatch.
"""

from __future__ import annotations

import pytest

from blackbox_qa.config import settings


@pytest.fixture(autouse=True)
def _disable_observability(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)
