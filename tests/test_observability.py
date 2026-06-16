"""When Langfuse is disabled (the default), every observability call is a no-op
and never imports/needs the SDK."""

from blackbox_qa import observability as obs


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(obs.settings, "langfuse_enabled", False)
    assert obs.enabled() is False


def test_observation_yields_none_when_disabled(monkeypatch):
    monkeypatch.setattr(obs.settings, "langfuse_enabled", False)
    with obs.observation("x", as_type="span", input={"a": 1}) as span:
        assert span is None


def test_score_and_flush_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(obs.settings, "langfuse_enabled", False)
    # Must not raise and must not need a client / network.
    obs.score("correctness", 1, trace_id="t1", data_type="BOOLEAN")
    obs.set_trace_io(output="x")
    obs.flush()
    assert obs.current_trace_id() is None


def test_enabled_requires_public_key(monkeypatch):
    monkeypatch.setattr(obs.settings, "langfuse_enabled", True)
    monkeypatch.setattr(obs.settings, "langfuse_public_key", "")
    monkeypatch.setattr(obs.settings, "langfuse_secret_key", "sk")
    assert obs.enabled() is False


def test_enabled_requires_secret_key(monkeypatch):
    monkeypatch.setattr(obs.settings, "langfuse_enabled", True)
    monkeypatch.setattr(obs.settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(obs.settings, "langfuse_secret_key", "")
    assert obs.enabled() is False


def _enable(monkeypatch):
    monkeypatch.setattr(obs.settings, "langfuse_enabled", True)
    monkeypatch.setattr(obs.settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(obs.settings, "langfuse_secret_key", "sk")


def test_failsafe_when_client_init_returns_none(monkeypatch):
    # Even when enabled, a failed client init (None) must make every call a no-op.
    _enable(monkeypatch)
    monkeypatch.setattr(obs, "_client", lambda: None)
    with obs.observation("x", as_type="span") as span:
        assert span is None
    assert obs.current_trace_id() is None
    obs.score("s", 1, trace_id="t")
    obs.set_trace_io(output="o")
    obs.flush()


class _BrokenClient:
    def start_as_current_observation(self, **kw):
        raise RuntimeError("boom")

    def get_current_trace_id(self):
        raise RuntimeError("boom")

    def set_current_trace_io(self, **kw):
        raise RuntimeError("boom")

    def create_score(self, **kw):
        raise RuntimeError("boom")

    def flush(self):
        raise RuntimeError("boom")


def test_failsafe_when_sdk_calls_raise(monkeypatch):
    # An enabled client whose SDK methods raise must never propagate into callers.
    _enable(monkeypatch)
    monkeypatch.setattr(obs, "_client", lambda: _BrokenClient())
    with obs.observation("x", as_type="generation") as span:
        assert span is None
    assert obs.current_trace_id() is None
    obs.score("s", 1, trace_id="t")
    obs.set_trace_io(output="o")
    obs.flush()


def test_caller_exception_propagates_through_observation(monkeypatch):
    # The guard must not swallow a real error raised inside the caller's block.
    _enable(monkeypatch)
    monkeypatch.setattr(obs, "_client", lambda: None)
    import pytest

    with pytest.raises(ValueError):
        with obs.observation("x"):
            raise ValueError("real agent error")
