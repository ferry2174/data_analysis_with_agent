# kafka_pool.py
from typing import Optional

from aiokafka import AIOKafkaProducer


class KafkaPool:
    _producer: Optional[AIOKafkaProducer] = None

    @classmethod
    async def init_pool(cls, bootstrap_servers: str):
        if cls._producer is None:
            cls._producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
            await cls._producer.start()

    @classmethod
    def get_producer(cls) -> AIOKafkaProducer:
        if cls._producer is None:
            raise RuntimeError("Kafka producer not initialized")
        return cls._producer

    @classmethod
    async def close(cls):
        if cls._producer:
            await cls._producer.stop()
