-- Runs automatically on first container start (mounted into /docker-entrypoint-initdb.d)

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per document ingested into the knowledge base
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    source      TEXT,              -- e.g. url or file path it came from
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk of a document, with its embedding
-- 768 = gemini-embedding-001 output_dimensionality (see .env.example in ai-service)
CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    embedding     vector(768),
    chunk_index   INT NOT NULL,
    -- keyword/full-text search side of hybrid retrieval (paired with the
    -- embedding for vector search) — auto-maintained by Postgres, never
    -- written to directly
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
    ON chunks USING gin (search_vector);

-- Conversation history, for multi-turn follow-ups later
CREATE TABLE IF NOT EXISTS conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Feedback loop: thumbs up/down per assistant message
CREATE TABLE IF NOT EXISTS feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id  UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
