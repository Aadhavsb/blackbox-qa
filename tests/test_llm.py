"""Rate-limit detection + backoff timing. Pure functions, no network."""

from blackbox_qa import llm


class _Err(Exception):
    def __init__(self, msg: str, status_code: int | None = None):
        super().__init__(msg)
        self.status_code = status_code


class RateLimitError(Exception):
    """Stand-in matching the openai SDK class name _is_rate_limited keys on."""


def test_rate_limited_by_class_name():
    assert llm._is_rate_limited(RateLimitError("slow down")) is True


def test_rate_limited_by_429_code():
    assert llm._is_rate_limited(_Err("too many requests", status_code=429)) is True


def test_rate_limited_413_tpm_phrasings():
    for msg in [
        "rate_limit exceeded",
        "rate limit reached",
        "tokens per minute (TPM) exceeded",
        "Request too large for model",
    ]:
        assert llm._is_rate_limited(_Err(msg, status_code=413)) is True


def test_not_rate_limited():
    assert llm._is_rate_limited(_Err("bad request", status_code=400)) is False
    assert llm._is_rate_limited(_Err("413 unrelated payload issue", status_code=413)) is False


def test_retry_delay_prefers_server_hint():
    assert llm._retry_delay(Exception("Please retry in 7.5s."), attempt=0) == 8.5


def test_retry_delay_exponential_backoff():
    assert llm._retry_delay(Exception("no hint"), attempt=3) == 8.0


def test_tool_use_failed_from_body():
    exc = _Err("bad tool call", status_code=400)
    exc.body = {
        "error": {
            "code": "tool_use_failed",
            "failed_generation": "Answer citing ev_id=20080122X00081.",
        }
    }
    assert llm._tool_use_failed_text(exc) == "Answer citing ev_id=20080122X00081."


def test_tool_use_failed_from_string_payload():
    exc = _Err(
        """Error code: 400 - {'error': {'message': 'Failed to call a function.', 'code': 'tool_use_failed', 'failed_generation': 'The Airbus A319 rolled.'}}""",
        status_code=400,
    )
    assert llm._tool_use_failed_text(exc) == "The Airbus A319 rolled."


def test_tool_use_failed_returns_none_for_other_errors():
    assert llm._tool_use_failed_text(_Err("bad request", status_code=400)) is None
