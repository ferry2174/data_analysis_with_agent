# redis_pool.py
from typing import Optional

import redis.asyncio as aioredis


class RedisPool:
    _client: Optional[aioredis.Redis] = None

    @classmethod
    async def init_pool(cls, url: str):
        if cls._client is None:
            cls._client = aioredis.from_url(url, decode_responses=True)

    @classmethod
    def get_client(cls) -> aioredis.Redis:
        if cls._client is None:
            raise RuntimeError("Redis pool not initialized")
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client:
            await cls._client.close()
