# db_doris.py
from typing import Optional

import aiomysql


class DorisPool:
    _pool: Optional[aiomysql.Pool] = None

    @classmethod
    async def init_pool(cls, host, port, user, password, db, minsize=1, maxsize=10):
        if cls._pool is None:
            cls._pool = await aiomysql.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                db=db,
                minsize=minsize,
                maxsize=maxsize,
                autocommit=True
            )

    @classmethod
    def get_pool(cls) -> aiomysql.Pool:
        if cls._pool is None:
            raise RuntimeError("Doris pool not initialized")
        return cls._pool
