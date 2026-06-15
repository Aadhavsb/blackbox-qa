.PHONY: help db-up db-down ingest serve eval fmt lint test

help:
	@echo "Targets:"
	@echo "  db-up    Start Postgres (pgvector) via docker compose"
	@echo "  db-down  Stop and remove the local stack"
	@echo "  ingest   Download NTSB data, convert .mdb, load + embed + index"
	@echo "  serve    Run the FastAPI app"
	@echo "  eval     Run the local eval pipeline"
	@echo "  fmt      Format + autofix with ruff"
	@echo "  lint     Lint with ruff"
	@echo "  test     Run pytest"

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

# Phase 1: implemented in src/blackbox_qa/ingest.py
ingest:
	python -m blackbox_qa.ingest

serve:
	uvicorn blackbox_qa.app:app --reload --port 8000

eval:
	python -m evals.run --mode retrieval

fmt:
	ruff format . && ruff check --fix .

lint:
	ruff check .

test:
	pytest -q
