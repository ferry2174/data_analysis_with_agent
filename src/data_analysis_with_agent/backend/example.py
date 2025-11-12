import logging

from fastapi import APIRouter

from data_analysis_with_agent.backend.constants import (
    RESPONSE_CODE_SERVICE_UNAVAILABLE,
    RESPONSE_CODE_SERVICE_UNAVAILABLE_MSG,
)
from data_analysis_with_agent.backend.models import Response
from data_analysis_with_agent.backend.pool.helper_doris import DorisHelper
from data_analysis_with_agent.backend.pool.helper_mariadb import MariaDBHelper
from data_analysis_with_agent.backend.pool.pool_kafka import KafkaPool
from data_analysis_with_agent.backend.pool.pool_redis import RedisPool


logger = logging.getLogger(__name__)

# 创建路由实例
router = APIRouter(
    prefix="/example",  # 所有路由都会自动添加此前缀
    tags=["example"],      # 在Swagger文档中分组显示
    responses={404: {"description": "Not found"}},
    include_in_schema=False,  # 控制是否在Swagger文档中显示
)

@router.get("/mariadb")
async def query_mariadb():
    try:
        MariaDBHelper.get_pool()
    except RuntimeError:
        return Response(status=RESPONSE_CODE_SERVICE_UNAVAILABLE, message=RESPONSE_CODE_SERVICE_UNAVAILABLE_MSG)
    return Response(data = await MariaDBHelper.fetchone("SELECT NOW();"))

@router.get("/doris")
async def query_doris():
    try:
        DorisHelper.get_pool()
    except RuntimeError:
        return Response(status=RESPONSE_CODE_SERVICE_UNAVAILABLE, message=RESPONSE_CODE_SERVICE_UNAVAILABLE_MSG)
    return Response(data = await DorisHelper.fetchone("SELECT NOW();"))

@router.get("/redis")
async def redis_example():
    try:
        RedisPool.get_client()
    except RuntimeError:
        return Response(status=RESPONSE_CODE_SERVICE_UNAVAILABLE, message=RESPONSE_CODE_SERVICE_UNAVAILABLE_MSG)
    redis = RedisPool.get_client()
    await redis.set("greeting", "hello redis")
    value = await redis.get("greeting")
    return {"redis_value": value}

@router.get("/kafka")
async def kafka_example():
    try:
        KafkaPool.get_producer()
    except RuntimeError:
        return Response(status=RESPONSE_CODE_SERVICE_UNAVAILABLE, message=RESPONSE_CODE_SERVICE_UNAVAILABLE_MSG)
    producer = KafkaPool.get_producer()
    await producer.send_and_wait("test-topic", b"hello kafka")
    return {"kafka_status": "message sent"}
