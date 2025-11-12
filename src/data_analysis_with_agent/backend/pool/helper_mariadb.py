import logging
from typing import Any, Dict, List, Optional, Union

import aiomysql


logger = logging.getLogger(__name__)


class MariaDBHelper:
    _instance = None
    _pool = None

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 9030,
                 user: str = 'root',
                 password: str = '',
                 database: str = '',
                 min_size: int = 1,
                 max_size: int = 5):
        """
        初始化DorisHelper

        :param host: 数据库主机地址
        :param port: 数据库端口
        :param user: 用户名
        :param password: 密码
        :param database: 默认数据库
        :param pool_size: 连接池大小
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.min_size = min_size
        self.max_size = max_size

    @classmethod
    async def init_pool(cls,
                       host: str = 'localhost',
                       port: int = 9030,
                       user: str = 'root',
                       password: str = '',
                       database: str = '',
                       min_size: int = 1,
                       max_size: int = 5):
        """初始化连接池"""
        if cls._instance is None:
            cls._instance = cls(host, port, user, password, database, min_size, max_size)
            cls._pool = await aiomysql.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                db=database,
                minsize=min_size,
                maxsize=max_size,
                autocommit=True
            )
        return cls._instance

    @classmethod
    def get_pool(cls):
        """获取连接池"""
        if cls._pool is None:
            raise RuntimeError("Connection pool is not initialized")
        return cls._pool

    @classmethod
    async def close(cls):
        """关闭连接池"""
        if cls._pool is not None:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None
            cls._instance = None
            logger.info("Mariadb connection pool closed.")

    @classmethod
    async def execute(cls, sql: str, args: Union[tuple, list, dict] = None) -> int:
        """
        执行SQL语句，返回影响的行数

        :param sql: SQL语句
        :param args: 参数
        :return: 影响的行数
        """
        async with cls.get_pool().acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, args)
                return cursor.rowcount

    @classmethod
    async def executemany(cls, sql: str, args: List[Union[tuple, list, dict]]) -> int:
        """
        批量执行SQL语句，返回影响的行数

        :param sql: SQL语句
        :param args: 参数列表
        :return: 影响的行数
        """
        async with cls.get_pool().acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(sql, args)
                return cursor.rowcount

    @classmethod
    async def fetchone(cls, sql: str, args: Union[tuple, list, dict] = None) -> Optional[Dict[str, Any]]:
        """
        查询单条记录

        :param sql: SQL语句
        :param args: 参数
        :return: 单条记录字典或None
        """
        async with cls.get_pool().acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, args)
                return await cursor.fetchone()

    @classmethod
    async def fetchall(cls, sql: str, args: Union[tuple, list, dict] = None) -> List[Dict[str, Any]]:
        """
        查询多条记录

        :param sql: SQL语句
        :param args: 参数
        :return: 记录列表
        """
        async with cls.get_pool().acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, args)
                return await cursor.fetchall()

    @classmethod
    async def insert(cls, table: str, data: Dict[str, Any]) -> int:
        """
        插入单条记录

        :param table: 表名
        :param data: 数据字典
        :return: 影响的行数
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return await cls.execute(sql, tuple(data.values()))

    @classmethod
    async def insert_many(cls, table: str, data: List[Dict[str, Any]]) -> int:
        """
        批量插入记录

        :param table: 表名
        :param data: 数据字典列表
        :return: 影响的行数
        """
        if not data:
            return 0

        columns = ', '.join(data[0].keys())
        placeholders = ', '.join(['%s'] * len(data[0]))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        args = [tuple(item.values()) for item in data]
        return await cls.executemany(sql, args)

    @classmethod
    async def update(cls, table: str, data: Dict[str, Any], condition: str, args: Union[tuple, list, dict] = None) -> int:
        """
        更新记录

        :param table: 表名
        :param data: 要更新的数据字典
        :param condition: WHERE条件语句
        :param args: WHERE条件参数
        :return: 影响的行数
        """
        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        params = tuple(data.values())
        if args:
            if isinstance(args, dict):
                params += tuple(args.values())
            else:
                params += tuple(args)
        return await cls.execute(sql, params)

    @classmethod
    async def delete(cls, table: str, condition: str, args: Union[tuple, list, dict] = None) -> int:
        """
        删除记录

        :param table: 表名
        :param condition: WHERE条件语句
        :param args: WHERE条件参数
        :return: 影响的行数
        """
        sql = f"DELETE FROM {table} WHERE {condition}"
        return await cls.execute(sql, args)

    @classmethod
    async def get_by_id(cls, table: str, id_value: Any, id_column: str = 'id') -> Optional[Dict[str, Any]]:
        """
        根据ID获取单条记录

        :param table: 表名
        :param id_value: ID值
        :param id_column: ID列名，默认为'id'
        :return: 单条记录字典或None
        """
        sql = f"SELECT * FROM {table} WHERE {id_column} = %s"
        return await cls.fetchone(sql, (id_value,))

    @classmethod
    async def query(cls, table: str, 
                   columns: Union[str, List[str]] = '*',
                   condition: Optional[str] = None,
                   args: Union[tuple, list, dict] = None,
                   order_by: Optional[str] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        通用查询方法

        :param table: 表名
        :param columns: 要查询的列，可以是字符串或列表
        :param condition: WHERE条件语句
        :param args: WHERE条件参数
        :param order_by: 排序条件
        :param limit: 限制条数
        :param offset: 偏移量
        :return: 记录列表
        """
        if isinstance(columns, list):
            columns = ', '.join(columns)

        sql = f"SELECT {columns} FROM {table}"

        if condition:
            sql += f" WHERE {condition}"

        if order_by:
            sql += f" ORDER BY {order_by}"

        if limit is not None:
            sql += f" LIMIT {limit}"
            if offset is not None:
                sql += f" OFFSET {offset}"

        return await cls.fetchall(sql, args)

    @classmethod
    async def count(cls, table: str, condition: Optional[str] = None, args: Union[tuple, list, dict] = None) -> int:
        """
        统计记录数

        :param table: 表名
        :param condition: WHERE条件语句
        :param args: WHERE条件参数
        :return: 记录数
        """
        sql = f"SELECT COUNT(*) AS count FROM {table}"
        if condition:
            sql += f" WHERE {condition}"

        result = await cls.fetchone(sql, args)
        return result['count'] if result else 0

    @classmethod
    def transaction(cls):
        """返回一个事务上下文管理器"""
        return TransactionContext(cls._pool)

class TransactionContext:
    """事务上下文管理器"""
    def __init__(self, pool):
        self.pool = pool
        self.conn = None
        self.cursor = None

    async def __aenter__(self):
        self.conn = await self.pool.acquire()
        await self.conn.begin()
        self.cursor = await self.conn.cursor(aiomysql.DictCursor)
        return self.cursor

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                await self.conn.commit()
            else:
                await self.conn.rollback()
        finally:
            await self.cursor.close()
            self.conn.close()
