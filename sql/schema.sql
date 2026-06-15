-- blackbox-qa schema.
-- Runs automatically on first Postgres start via docker-entrypoint-initdb.d,
-- and is idempotent so it can be re-applied by hand.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per NTSB event (joined to its primary aircraft at ingest time).
-- These are the structured fields the agent's `sql_query` tool aggregates over.
CREATE TABLE IF NOT EXISTS reports (
    ev_id              text PRIMARY KEY,
    ntsb_no            text,
    ev_date            date,
    ev_year            int,
    ev_month           int,
    ev_type            text,
    ev_city            text,
    ev_state           text,
    ev_country         text,
    latitude           double precision,
    longitude          double precision,
    ev_highest_injury  text,
    inj_tot_fatal      int,
    inj_tot_serious    int,
    inj_tot_minor      int,
    inj_tot_none       int,
    inj_tot_total      int,
    wx_cond_basic      text,
    light_cond         text,
    acft_make          text,
    acft_model         text,
    acft_series        text,
    acft_category      text,
    far_part           text,
    damage             text
);

CREATE INDEX IF NOT EXISTS reports_ev_year_idx   ON reports (ev_year);
CREATE INDEX IF NOT EXISTS reports_acft_make_idx ON reports (acft_make);
CREATE INDEX IF NOT EXISTS reports_state_idx     ON reports (ev_state);

-- Narrative text split into chunks. Each chunk carries a dense embedding
-- (HNSW-indexed) and a generated tsvector (GIN-indexed) so the same row
-- backs both arms of hybrid retrieval.
CREATE TABLE IF NOT EXISTS report_chunks (
    id          bigserial PRIMARY KEY,
    ev_id       text NOT NULL REFERENCES reports (ev_id) ON DELETE CASCADE,
    chunk_index int  NOT NULL,
    source      text NOT NULL,             -- 'factual' | 'cause'
    content     text NOT NULL,
    embedding   vector(384),               -- BAAI/bge-small-en-v1.5
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (ev_id, source, chunk_index)
);

-- Dense ANN index (cosine). Built after bulk load in practice; created here
-- so a fresh DB is query-ready. Tune m / ef_construction against the gold set.
CREATE INDEX IF NOT EXISTS report_chunks_embedding_idx
    ON report_chunks USING hnsw (embedding vector_cosine_ops);

-- Sparse / keyword index for BM25-style full-text search.
CREATE INDEX IF NOT EXISTS report_chunks_tsv_idx
    ON report_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS report_chunks_ev_id_idx
    ON report_chunks (ev_id);
