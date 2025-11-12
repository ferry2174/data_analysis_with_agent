import logging
import re
import time
from typing import Awaitable, Callable, Dict, Optional, Tuple

from fastapi import Request

from data_analysis_with_agent.backend.metrics import create_collector


logger = logging.getLogger(__name__)


"""通用请求指标"""
REQUEST_COUNT = create_collector("Counter",
    'fastapi_request_count',
    'Total number of requests to Euler data endpoints',
    ['system', 'method', 'path_template', 'status_code', 'country', 'business_type', 'sub_type', 'version']
)
REQUEST_LATENCY = create_collector("Histogram",
    'fastapi_request_latency_seconds',
    'Request latency to Euler data endpoints in seconds',
    ['system', 'method', 'path_template', 'status_code', 'country', 'business_type', 'sub_type', 'version'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
)

# 定义要监控的路径前缀
MONITORED_PREFIXS = ["/data_analysis_with_agent"]

# 通用路径模式
GENERIC_PATH_PATTERN = re.compile(
    r'^/(?P<system>[a-z_]+)/'        # 系统分类
    r'(?P<country>[a-z_]+)/'         # 国家代码
    r'(?P<business_type>[a-z_]+)/'   # 业务类型
    r'(?P<sub_type>[^/]+)/'          # 子类型
    r'(?P<version>[^/]+)'            # 版本
    r'(?:/(?P<id>[^/]+))?/?$'        # ID（可选）
)

def should_monitor(path: str) -> bool:
    """检查路径是否是需要监控的路径"""
    return any(path.startswith(prefix) for prefix in MONITORED_PREFIXS)

def parse_and_normalize_path(path: str) -> Optional[Tuple[str, Dict]]:
    """
    解析并规范化路径
    返回: (规范化路径模板, 提取的参数字典)
    """
    match = GENERIC_PATH_PATTERN.match(path)
    if not match:
        return None

    params = match.groupdict()

    # 基础模板（不含 id）
    template_parts = [
        "/{system}",
        "{country}",
        "{business_type}",
        "{sub_type}",
        "{version}"
    ]

    # 如果 id 存在，则加上占位符
    if params.get("id"):
        template_parts.append(f"{{{params['business_type']}_id}}")

    # 拼接成路径模板
    template = "/".join(template_parts)

    return template, params


async def monitor_requests_middleware(request: Request, call_next: Callable[[Request], Awaitable]):
    """监控请求的中间件函数"""
    original_path = request.url.path
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    if should_monitor(original_path):
        result = parse_and_normalize_path(original_path)
        if result:
            path_template, params = result

            REQUEST_COUNT.labels(
                system=params['system'],
                method=request.method,
                path_template=path_template,
                status_code=response.status_code,
                country=params['country'],
                business_type=params['business_type'],
                sub_type=params['sub_type'],
                version=params['version']
            ).inc()

            REQUEST_LATENCY.labels(
                system=params['system'],
                method=request.method,
                path_template=path_template,
                status_code=response.status_code,
                country=params['country'],
                business_type=params['business_type'],
                sub_type=params['sub_type'],
                version=params['version']
            ).observe(process_time)

    return response
