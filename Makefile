.PHONY: help install db-up db-down ingest serve eval fmt lint test

help:
	@echo "Targets:"
	@echo "  install  Install deps into a poetry venv"
	@echo "  db-up    Start Postgres (pgvector) via docker compose"
	@echo "  db-down  Stop and remove the local stack"
	@echo "  ingest   Download NTSB data, convert .mdb, load + embed + index"
	@echo "  serve    Run the FastAPI app"
	@echo "  eval     Run the local eval pipeline (retrieval mode)"
	@echo "  fmt      Format + autofix with ruff"
	@echo "  lint     Lint with ruff"
	@echo "  test     Run pytest"

install:
	poetry install

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

# Phase 1: implemented in src/blackbox_qa/ingest.py
ingest:
	poetry run python -m blackbox_qa.ingest

serve:
	poetry run uvicorn blackbox_qa.app:app --reload --port 8000

eval:
	poetry run python -m evals.run --mode retrieval

fmt:
	poetry run ruff format . && poetry run ruff check --fix .

lint:
	poetry run ruff check .

test:
	poetry run pytest -q
