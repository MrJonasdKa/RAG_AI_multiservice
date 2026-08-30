import os

import asyncpg
from pgvector.asyncpg import register_vector

DATABASE_URL = os.environ["DATABASE_URL"]

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def init_pool() -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(DATABASE_URL, init=_init_connection)
    return _pool


async def close_pool() -> None:
    if _pool is not None:
        await _pool.close()


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized — is the app running via lifespan?")
    return _pool
