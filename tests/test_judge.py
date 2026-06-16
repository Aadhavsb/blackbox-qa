from blackbox_qa import judge


def test_parse_clean_json():
    score, reason = judge._parse('{"score": 0.8, "reason": "grounded, cites ev_id"}')
    assert score == 0.8
    assert "grounded" in reason


def test_parse_json_with_surrounding_text():
    score, _ = judge._parse('Sure!\n{"score": 1.0, "reason": "great"}\nThanks')
    assert score == 1.0


def test_parse_clamps_out_of_range():
    assert judge._parse('{"score": 5}')[0] == 1.0
    assert judge._parse('{"score": -2}')[0] == 0.0


def test_parse_unparseable():
    score, reason = judge._parse("no json here")
    assert score == 0.0
    assert reason == "unparseable judge output"


def test_judge_answer_uses_llm(monkeypatch):
    from blackbox_qa import llm

    captured = {}

    def fake_chat(messages, model=None, temperature=0.0, **kw):
        captured["model"] = model
        captured["temperature"] = temperature
        return llm.LLMResponse(content='{"score": 0.7, "reason": "ok"}', tool_calls=[])

    monkeypatch.setattr(judge.llm, "chat", fake_chat)
    score, reason = judge.judge_answer("q?", "an answer")
    assert score == 0.7
    assert captured["temperature"] == 0.0
