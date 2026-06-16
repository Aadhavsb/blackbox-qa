"""FastAPI HTTP interface — a thin wrapper over agent.run(), same as the CLI.

Run: poetry run uvicorn blackbox_qa.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from blackbox_qa import agent

app = FastAPI(title="blackbox-qa", version="0.0.1")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    turns: int
    confidence: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = agent.run(req.question)
    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        turns=result.turns,
        confidence=result.confidence,
    )
