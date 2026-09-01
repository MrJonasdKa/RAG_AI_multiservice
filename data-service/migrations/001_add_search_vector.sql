-- Adds keyword/full-text search support to an already-initialized database
-- (init.sql only runs on a brand-new volume — this brings an existing one
-- up to date without losing ingested data).
--
-- Run: docker exec -i rag-db psql -U rag -d rag < data-service/migrations/001_add_search_vector.sql

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
    ON chunks USING gin (search_vector);
