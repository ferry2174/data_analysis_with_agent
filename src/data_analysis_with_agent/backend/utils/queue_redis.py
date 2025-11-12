import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis  # ✅ 使用异步 Redis 客户端


logger = logging.getLogger(__name__)

class AsyncRedisQueue:
    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        decode_responses: bool = True
    ):
        self.key = name
        self.redis = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=decode_responses
        )

    async def put(self, item: Any) -> None:
        """入队：将任务压入队列左侧"""
        await self.redis.lpush(self.key, json.dumps(item))

    async def get(self, timeout: Optional[int] = None) -> Optional[Any]:
        """
        出队：阻塞直到有任务，或超时返回 None
        timeout=None 表示永久阻塞
        """
        result = await self.redis.brpop(self.key, timeout=timeout)
        if result:
            _, raw = result
            return json.loads(raw)
        return None

    async def get_nowait(self) -> Optional[Any]:
        """非阻塞出队：立即返回任务或 None"""
        raw = await self.redis.rpop(self.key)
        if raw:
            return json.loads(raw)
        return None

    async def qsize(self) -> int:
        """获取队列长度"""
        return await self.redis.llen(self.key)

    async def clear(self) -> None:
        """清空队列"""
        await self.redis.delete(self.key)

    async def deque(self, item: Any, count: int = 0) -> int:
        """
        从队列中删除指定元素。
        参数:
            item: 要删除的元素（会被序列化为 JSON 字符串）
            count: 删除的数量，默认删除第一个匹配项
                   count > 0: 从左到右删除最多 count 个匹配项
                   count < 0: 从右到左删除最多 count 个匹配项
                   count = 0: 删除所有匹配项
        返回:
            实际删除的元素数量
        """
        serialized = json.dumps(item)
        return await self.redis.lrem(self.key, count, serialized)

    async def close(self):
        """关闭连接池"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info(f"Redis queue {self.key} closed")
