import logging
import pickle
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis


logger = logging.getLogger(__name__)

class RedisHelper:
    _client: Optional[aioredis.Redis] = None

    @classmethod
    async def init_pool(cls, url: str, **kwargs):
        """初始化Redis连接池

        :param url: Redis连接URL
        :param kwargs: 其他连接参数
        """
        if cls._client is None:
            cls._client = aioredis.from_url(
                url,
                decode_responses=False,  # 设置为False以便处理二进制数据
                **kwargs
            )
            logger.info("Redis connection pool initialized")

    @classmethod
    def get_client(cls) -> aioredis.Redis:
        """获取Redis客户端实例"""
        if cls._client is None:
            raise RuntimeError("Redis pool not initialized")
        return cls._client

    @classmethod
    async def close(cls):
        """关闭连接池"""
        if cls._client:
            await cls._client.close()
            cls._client = None
            logger.info("Redis connection pool closed")

    @classmethod
    async def set(cls, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        设置键值对，支持任意可序列化的Python对象

        :param key: 键名
        :param value: 值，可以是任意Python对象
        :param expire: 过期时间(秒)
        :return: 是否设置成功
        """
        serialized = pickle.dumps(value)
        client = cls.get_client()
        if expire is not None:
            return await client.setex(key, expire, serialized)
        return await client.set(key, serialized)

    @classmethod
    async def get(
        cls,
        key: str,
        expire: Optional[int] = None
    ) -> Optional[Any]:
        """改进错误处理的get方法"""
        client = cls.get_client()
        try:
            serialized = await client.get(key)
            if serialized is None:
                return None

            if expire is not None:
                await client.expire(key, expire)

            try:
                return pickle.loads(serialized)
            except pickle.PickleError as e:
                logger.error(f"Failed to unpickle value for key {key}: {str(e)}")
                return None

        except aioredis.RedisError as e:
            logger.error(f"Redis operation failed for key {key}: {str(e)}")
            return None

    @classmethod
    async def delete(cls, *keys: str) -> int:
        """
        删除一个或多个键

        :param keys: 要删除的键名
        :return: 删除的键数量
        """
        client = cls.get_client()
        return await client.delete(*keys)

    @classmethod
    async def exists(cls, key: str) -> bool:
        """
        检查键是否存在

        :param key: 键名
        :return: 是否存在
        """
        client = cls.get_client()
        return await client.exists(key) == 1

    @classmethod
    async def expire(cls, key: str, seconds: int) -> bool:
        """
        设置键的过期时间

        :param key: 键名
        :param seconds: 过期时间(秒)
        :return: 是否设置成功
        """
        client = cls.get_client()
        return await client.expire(key, seconds)

    @classmethod
    async def ttl(cls, key: str) -> int:
        """
        获取键的剩余生存时间

        :param key: 键名
        :return: 剩余时间(秒)，-2表示键不存在，-1表示键没有设置过期时间
        """
        client = cls.get_client()
        return await client.ttl(key)

    @classmethod
    async def set_str(
        cls,
        key: str,
        value: str,
        expire: Optional[int] = None
    ) -> bool:
        """
        设置字符串值

        :param key: 键名
        :param value: 字符串值（自动编码为UTF-8）
        :param expire: 过期时间(秒)
        :return: 是否设置成功
        """
        client = cls.get_client()
        # 显式编码为bytes确保一致性
        value_bytes = value.encode('utf-8')
        if expire is not None:
            return await client.setex(key, expire, value_bytes)
        return await client.set(key, value_bytes)

    @classmethod
    async def get_str(
        cls,
        key: str,
        expire: Optional[int] = None
    ) -> Optional[str]:
        """
        获取字符串值并可选刷新过期时间

        :param key: 键名
        :param expire: 如果提供，则刷新过期时间为该值(秒)
        :return: UTF-8解码后的字符串或None
        """
        client = cls.get_client()
        value = await client.get(key)

        if value is None:
            return None

        if expire is not None:
            await client.expire(key, expire)
            logger.debug(f"Refreshed str cache TTL: {key}={expire}s")

        # 统一解码处理（即使decode_responses=False）
        return value.decode('utf-8') if isinstance(value, bytes) else str(value)

    @classmethod
    async def set_dict(
        cls,
        key: str,
        value: Dict[str, Any],
        expire: Optional[int] = None
    ) -> bool:
        """统一编码所有键为bytes"""
        client = cls.get_client()

        serialized = {
            k.encode('utf-8') if not isinstance(k, (bytes, bytearray)) else k:
            pickle.dumps(v) if not isinstance(v, (bytes, bytearray)) else v
            for k, v in value.items()
        }

        result = await client.hset(key, mapping=serialized)
        if expire is not None and result:
            await client.expire(key, expire)
        return result

    @classmethod
    async def get_dict(
        cls,
        key: str,
        expire: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        client = cls.get_client()
        result = await client.hgetall(key)

        if not result:
            return {}

        if expire is not None:
            await client.expire(key, expire)

        deserialized = {}
        for k, v in result.items():
            try:
                # 统一解码键为字符串
                str_key = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                # 反序列化值
                deserialized[str_key] = pickle.loads(v) if not isinstance(v, str) else v
            except (pickle.PickleError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to process dict item {k}: {str(e)}")
                deserialized[str(k)] = v.decode('utf-8') if isinstance(v, bytes) else v

        return deserialized

    @classmethod
    async def set_list(
        cls,
        key: str,
        values: List[Any],
        expire: Optional[int] = None
    ) -> bool:
        """
        设置列表值（自动序列化元素）

        :param key: 键名
        :param values: 列表值（元素会被自动序列化）
        :param expire: 过期时间(秒)
        :return: 是否设置成功
        """
        client = cls.get_client()

        # 使用pipeline保证原子性
        async with client.pipeline() as pipe:
            try:
                # 删除旧列表并添加新元素（原子操作）
                await pipe.delete(key)
                if values:
                    await pipe.rpush(key, *[pickle.dumps(v) for v in values])
                else:
                    await pipe.rpush(key)  # 空列表

                if expire is not None:
                    await pipe.expire(key, expire)

                results = await pipe.execute()
                return bool(results[1])  # rpush的结果

            except (pickle.PickleError, aioredis.RedisError) as e:
                logger.error(f"Failed to set list {key}: {str(e)}")
                return False

    @classmethod
    async def get_list(
        cls,
        key: str,
        expire: Optional[int] = None
    ) -> List[Any]:
        """
        获取列表值并可选刷新过期时间

        :param key: 键名
        :param expire: 如果提供，则刷新过期时间为该值(秒)
        :return: 反序列化后的列表
        """
        client = cls.get_client()

        try:
            # 使用pipeline保证原子性
            async with client.pipeline() as pipe:
                await pipe.lrange(key, 0, -1)
                if expire is not None:
                    await pipe.expire(key, expire)

                serialized = await pipe.execute()

                if not serialized and not serialized[0]:
                    return []

                # 反序列化处理
                result = []
                for item in serialized[0]:
                    try:
                        result.append(pickle.loads(item))
                    except pickle.PickleError:
                        logger.warning(f"Failed to unpickle item in list {key}")
                        result.append(item)  # 保留原始值

                logger.debug(f"Retrieved list {key} with {len(result)} items")
                return result

        except aioredis.RedisError as e:
            logger.error(f"Failed to get list {key}: {str(e)}")
            return []

    @classmethod
    async def set_set(
        cls,
        key: str,
        values: List[Any],
        expire: Optional[int] = None,
        clear: bool = True  # 新增参数控制是否清除旧集合
    ) -> int:
        """
        设置集合值（自动序列化元素）

        :param key: 键名
        :param values: 集合元素列表（会被自动序列化）
        :param expire: 过期时间(秒)
        :param clear: 是否清除旧集合
        :return: 实际添加的新元素数量
        """
        client = cls.get_client()

        try:
            async with client.pipeline() as pipe:
                if clear:
                    await pipe.delete(key)

                if values:
                    # 序列化所有元素
                    members = [pickle.dumps(v) for v in values]
                    await pipe.sadd(key, *members)
                else:
                    # 空集合处理（Redis不支持空集合创建，需要特殊处理）
                    if clear:
                        await pipe.sadd(key, b'')  # 添加临时元素
                        await pipe.srem(key, b'')  # 立即删除，创建空集合

                if expire is not None:
                    await pipe.expire(key, expire)

                results = await pipe.execute()
                return results[-2] if values else 0  # 返回sadd的结果

        except (pickle.PickleError, aioredis.RedisError) as e:
            logger.error(f"Failed to set set {key}: {str(e)}")
            return 0

    @classmethod
    async def get_set(
        cls,
        key: str,
        expire: Optional[int] = None
    ) -> List[Any]:
        """
        获取集合值并可选刷新过期时间

        :param key: 键名
        :param expire: 如果提供，则刷新过期时间为该值(秒)
        :return: 反序列化后的元素列表
        """
        client = cls.get_client()

        try:
            async with client.pipeline() as pipe:
                await pipe.smembers(key)
                if expire is not None:
                    await pipe.expire(key, expire)

                serialized, _ = await pipe.execute()

                if not serialized:
                    return []

                # 反序列化处理
                result = []
                for item in serialized:
                    try:
                        result.append(pickle.loads(item))
                    except pickle.PickleError:
                        logger.warning(f"Failed to unpickle item in set {key}")
                        result.append(item.decode() if isinstance(item, bytes) else item)

                logger.debug(f"Retrieved set {key} with {len(result)} items")
                return result

        except aioredis.RedisError as e:
            logger.error(f"Failed to get set {key}: {str(e)}")
            return []

    @classmethod
    async def incr(cls, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        """
        自增操作

        :param key: 键名
        :param amount: 自增值
        :param expire: 过期时间(秒)
        :return: 自增后的值
        """
        client = cls.get_client()
        result = await client.incrby(key, amount)
        if expire is not None:
            await client.expire(key, expire)
        return result

    @classmethod
    async def decr(cls, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        """
        自减操作

        :param key: 键名
        :param amount: 自减值
        :param expire: 过期时间(秒)
        :return: 自减后的值
        """
        client = cls.get_client()
        result = await client.decrby(key, amount)
        if expire is not None:
            await client.expire(key, expire)
        return result

    @classmethod
    async def keys(cls, pattern: str = "*") -> List[str]:
        """
        查找匹配模式的键

        :param pattern: 匹配模式
        :return: 匹配的键列表
        """
        client = cls.get_client()
        return await client.keys(pattern)

    @classmethod
    async def flush_db(cls) -> bool:
        """
        清空当前数据库

        :return: 是否成功
        """
        client = cls.get_client()
        return await client.flushdb()

    @classmethod
    async def pipeline(cls):
        """返回一个管道上下文管理器"""
        return PipelineContext(cls._client)

class PipelineContext:
    """Redis管道上下文管理器"""
    def __init__(self, client):
        self.client = client
        self.pipeline = None

    async def __aenter__(self):
        self.pipeline = self.client.pipeline()
        return self.pipeline

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.pipeline.execute()
        self.pipeline = None
