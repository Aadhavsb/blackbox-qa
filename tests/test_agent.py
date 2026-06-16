"""Agent-loop tests. The LLM and tools are mocked, so these run with no network,
no API key, and zero tokens — the loop's real control logic is what's exercised.
"""

from blackbox_qa import agent, llm


def _text(content: str) -> llm.LLMResponse:
    return llm.LLMResponse(content=content, tool_calls=[])


def _call(name: str, arguments: str, call_id: str = "c1") -> llm.LLMResponse:
    return llm.LLMResponse(content=None, tool_calls=[llm.ToolCall(call_id, name, arguments)])


def _script(monkeypatch, responses):
    """Make llm.chat return the scripted responses in order, recording tool_choice."""
    seq = iter(responses)
    seen_tool_choice: list[str] = []

    def fake_chat(messages, tools=None, tool_choice="auto", **kw):
        seen_tool_choice.append(tool_choice)
        return next(seq)

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    return seen_tool_choice


def test_single_shot_answer(monkeypatch):
    _script(monkeypatch, [_text("Engine failure. CONFIDENCE: high")])
    result = agent.run("what happened?")
    assert result.turns == 1
    assert result.confidence == "high"
    assert "CONFIDENCE" not in result.answer


def test_tool_then_answer(monkeypatch):
    monkeypatch.setattr(agent.tools, "dispatch", lambda name, args: "ev_id=20080101X00001 ...")
    _script(
        monkeypatch,
        [
            _call("hybrid_search", '{"query": "engine failure"}'),
            _text("Report 20080101X00001 cites engine failure. CONFIDENCE: high"),
        ],
    )
    result = agent.run("why did it crash?")
    assert result.turns == 2
    assert result.citations == ["20080101X00001"]
    assert any(e.get("ok") for e in result.tool_log)


def test_invalid_json_args_self_correct(monkeypatch):
    # Model emits malformed JSON args; loop must feed an error back, not crash.
    _script(
        monkeypatch,
        [
            _call("hybrid_search", "{not valid json"),
            _text("Recovered. CONFIDENCE: high"),
        ],
    )
    result = agent.run("q")
    assert result.turns == 2
    assert any("error" in e for e in result.tool_log)


def test_confidence_retry_then_success(monkeypatch):
    chats = _script(
        monkeypatch,
        [
            _text("Not sure. CONFIDENCE: low"),
            _text("Found it: 20080101X00001. CONFIDENCE: high"),
        ],
    )
    result = agent.run("obscure question")
    assert result.confidence == "high"
    assert result.turns == 2
    assert len(chats) == 2  # retry consumed exactly one extra turn


def test_turn_cap_forces_tool_choice_none(monkeypatch):
    # Model never stops asking for tools; loop must still terminate at the cap
    # with tool_choice="none" forced on the final turn.
    always_call = [_call("hybrid_search", '{"query": "x"}') for _ in range(10)]
    monkeypatch.setattr(agent.tools, "dispatch", lambda name, args: "stuff")
    seen = _script(monkeypatch, always_call)
    result = agent.run("loops forever?", max_turns=3)
    assert result.turns == 3
    assert seen[-1] == "none"


def test_unexpected_tool_exception_does_not_crash(monkeypatch):
    # A non-ToolError (e.g. a DB failure) must be caught and fed back as an
    # error string so the loop self-corrects instead of crashing.
    def boom(name, args):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(agent.tools, "dispatch", boom)
    _script(
        monkeypatch,
        [
            _call("sql_query", '{"sql": "select 1"}'),
            _text("Recovered without that tool. CONFIDENCE: high"),
        ],
    )
    result = agent.run("q")
    assert result.turns == 2
    assert any("error" in e for e in result.tool_log)
    assert "RuntimeError" in str(result.tool_log)


def test_tool_call_budget_is_capped(monkeypatch):
    monkeypatch.setattr(agent, "MAX_TOOL_CALLS", 2)
    calls = {"n": 0}

    def counting_dispatch(name, args):
        calls["n"] += 1
        return "ok"

    monkeypatch.setattr(agent.tools, "dispatch", counting_dispatch)
    three_calls = llm.LLMResponse(
        content=None,
        tool_calls=[llm.ToolCall(f"c{i}", "hybrid_search", '{"query": "x"}') for i in range(3)],
    )
    _script(monkeypatch, [three_calls, _text("Done. CONFIDENCE: high")])
    result = agent.run("q")
    assert calls["n"] == 2  # third call was over budget, not executed
    assert any(e.get("error") == "tool budget exhausted" for e in result.tool_log)


def test_empty_final_turn_is_repaired(monkeypatch):
    # On the final turn the model still emits a tool call (empty content); the
    # loop must force one plain-text answer rather than returning "".
    final_tool_call = _call("hybrid_search", '{"query": "x"}')
    seen = _script(
        monkeypatch,
        [final_tool_call, _text("Best-effort answer. CONFIDENCE: low")],
    )
    result = agent.run("q", max_turns=1)
    assert result.answer == "Best-effort answer."
    assert seen == ["none", "none"]  # final turn + repair, both tool-free
