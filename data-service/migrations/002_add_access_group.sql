-- Adds simple access-group scoping to documents on an already-initialized
-- database (init.sql only runs on a brand-new volume).
--
-- Run: docker exec -i rag-db psql -U rag -d rag < data-service/migrations/002_add_access_group.sql

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS access_group TEXT;
