"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://blackbox:blackbox@localhost:5432/blackbox"

    # OpenAI-compatible LLM provider (point base_url at Gemini / Ollama / etc.)
    openai_api_key: str = ""
    openai_base_url: str | None = None
    llm_model: str = ""
    judge_model: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Confidence gate: the agent retries (reformulate + re-search) when the top
    # cross-encoder rerank score of its evidence is below this threshold. A
    # better-calibrated signal than the model's self-reported CONFIDENCE line.
    # Calibrated via `python -m evals.run --mode calibrate`: on the n=12 gold set
    # failed retrievals scored <= 3.21 and successes >= 5.80 (cleanly separable),
    # so 4.5 sits in the gap (TPR 1.0, FPR 0.0). See evals/confidence_calibration.json.
    confidence_score_threshold: float = 4.5

    langfuse_enabled: bool = False
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


settings = Settings()
