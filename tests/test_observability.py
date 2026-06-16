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
    assert obs.enabled() is False
