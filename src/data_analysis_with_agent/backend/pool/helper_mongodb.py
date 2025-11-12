import logging
from typing import Any, Dict, List, Optional, Union

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError


logger = logging.getLogger(__name__)

class MongoHelper:
    _client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
    _db = None

    @classmethod
    async def init_client(
        cls,
        hosts: Union[str, List[str]],
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        authSource: str = "admin",
        database: str = "",
        maxPoolSize: int = 100,
        minPoolSize: int = 0,
        retries: int = 3,
        **kwargs
    ):
        """
        初始化 MongoDB 异步客户端，支持多节点（副本集连接字符串）。
        hosts 可传单个host或列表。
        """
        if not isinstance(hosts, list):
            hosts = [hosts]

        # 拼接连接字符串
        host_str = ",".join(f"{h}:{port}" for h in hosts)
        auth_part = ""
        if username and password:
            auth_part = f"{username}:{password}@"

        uri = f"mongodb://{auth_part}{host_str}/{database}?authSource={authSource}"

        for attempt in range(retries):
            try:
                cls._client = motor.motor_asyncio.AsyncIOMotorClient(
                    uri,
                    maxPoolSize=maxPoolSize,
                    minPoolSize=minPoolSize,
                    **kwargs
                )
                cls._db = cls._client[database]
                # 测试连接
                await cls._db.command("ping")
                logger.info(f"Connected to MongoDB: {host_str}")
                return
            except PyMongoError as e:
                logger.warning(f"Failed to connect to MongoDB ({attempt+1}/{retries}): {str(e)}")
                if attempt == retries - 1:
                    raise

    @classmethod
    def get_db(cls):
        if cls._db is None:
            raise RuntimeError("MongoDB client is not initialized")
        return cls._db

    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None

    # ---------- CRUD ----------
    @classmethod
    async def insert_one(cls, collection: str, document: Dict[str, Any]) -> str:
        result = await cls.get_db()[collection].insert_one(document)
        return str(result.inserted_id)

    @classmethod
    async def insert_many(cls, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        if not documents:
            return []
        result = await cls.get_db()[collection].insert_many(documents)
        return [str(_id) for _id in result.inserted_ids]

    @classmethod
    async def find_one(cls, collection: str, filter: Dict[str, Any], projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        return await cls.get_db()[collection].find_one(filter, projection)

    @classmethod
    async def find_many(
        cls,
        collection: str,
        filter: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        cursor = cls.get_db()[collection].find(filter, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=limit or 0)

    @classmethod
    async def update_one(cls, collection: str, filter: Dict[str, Any], update: Dict[str, Any]) -> int:
        result = await cls.get_db()[collection].update_one(filter, update)
        return result.modified_count

    @classmethod
    async def update_many(cls, collection: str, filter: Dict[str, Any], update: Dict[str, Any]) -> int:
        result = await cls.get_db()[collection].update_many(filter, update)
        return result.modified_count

    @classmethod
    async def delete_one(cls, collection: str, filter: Dict[str, Any]) -> int:
        result = await cls.get_db()[collection].delete_one(filter)
        return result.deleted_count

    @classmethod
    async def delete_many(cls, collection: str, filter: Dict[str, Any]) -> int:
        result = await cls.get_db()[collection].delete_many(filter)
        return result.deleted_count

    @classmethod
    async def count(cls, collection: str, filter: Optional[Dict[str, Any]] = None) -> int:
        return await cls.get_db()[collection].count_documents(filter or {})

    # ---------- 分页 ----------
    @classmethod
    async def paginate(
        cls,
        collection: str,
        filter: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        sort: Optional[List[tuple]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        skip = (page - 1) * page_size
        cursor = cls.get_db()[collection].find(filter, projection)
        if sort:
            cursor = cursor.sort(sort)
        total = await cls.count(collection, filter)
        items = await cursor.skip(skip).limit(page_size).to_list(length=page_size)
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items
        }