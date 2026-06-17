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
    # Calibrated against the gold set via `python -m evals.run --mode calibrate`.
    confidence_score_threshold: float = 0.0

    langfuse_enabled: bool = False
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


settings = Settings()
