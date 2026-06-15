-- blackbox-qa schema (phase 1 fills this in).
-- Runs automatically on first Postgres start via docker-entrypoint-initdb.d.

CREATE EXTENSION IF NOT EXISTS vector;

-- Tables (reports, chunks with embeddings + tsvector, etc.) added in phase 1.
