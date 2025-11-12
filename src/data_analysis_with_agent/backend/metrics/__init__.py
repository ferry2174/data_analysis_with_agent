import logging
import os
from http.server import HTTPServer

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary, multiprocess
from prometheus_client.exposition import MetricsHandler

from data_analysis_with_agent.backend.config import ConfigManager
from data_analysis_with_agent.backend.constants import ENV_KEY_IN_OSENV


logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()

_registered_collectors = {}

def create_collector(type, name, description, attributes, buckets=None):
    if name in _registered_collectors:
        REGISTRY.unregister(_registered_collectors[name])

    if type == "Counter":
        collector = Counter(name, description, attributes, registry=REGISTRY)
    elif type == "Histogram":
        collector = Histogram(name, description, attributes, buckets=buckets, registry=REGISTRY)
    elif type == "Gauge":
        collector = Gauge(name, description, attributes, registry=REGISTRY)
    elif type == "Summary":
        collector = Summary(name, description, attributes, registry=REGISTRY)
    else:
        raise ValueError(f"Unsupported type: {type}")

    _registered_collectors[name] = collector
    return collector

def start_metrics_server():
    REGISTRY = CollectorRegistry()
    try:
        multiprocess.MultiProcessCollector(REGISTRY)
        logger.info(f"Prometheus multiprocess collector registered.Metrics:{', '.join([metric.name for metric in REGISTRY.collect()])}")
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # 已注册，忽略
            logger.info(f"Prometheus multiprocess collector already registered, skipping.Metrics:{', '.join([metric.name for metric in REGISTRY.collect()])}")
        else:
            raise
    CustomHandler = MetricsHandler.factory(registry=REGISTRY)
    config = ConfigManager.init_config(env=os.environ.get(ENV_KEY_IN_OSENV, "dev"))
    httpd = HTTPServer(('0.0.0.0', config.get("metrics_server_port", 8091)), CustomHandler)
    logger.info(f"Serving metrics on http://0.0.0.0:{config.get('metrics_server_port', 8091)}/metrics")
    httpd.serve_forever()
