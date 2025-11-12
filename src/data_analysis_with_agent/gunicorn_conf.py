import multiprocessing
import os

from data_analysis_with_agent.backend.config import ConfigManager
from data_analysis_with_agent.backend.constants import ENV_KEY_IN_OSENV
from data_analysis_with_agent.backend.logger import get_gunicorn_log_config
from data_analysis_with_agent.backend.metrics import start_metrics_server


env = os.environ.get(ENV_KEY_IN_OSENV, "dev")
print(f"env: {env}")

env_config = ConfigManager.init_config(env=env)
# 基础配置
bind = f"0.0.0.0:{env_config.get('uvicorn.port', 8000)}"
workers = env_config.get('uvicorn.workers', 1)
worker_class = "uvicorn.workers.UvicornWorker"
threads = env_config.get('uvicorn.threads', 4)
# worker 在收到重启信号时，允许的优雅退出时间
graceful_timeout = env_config.get('uvicorn.graceful_timeout', 30)   # 等价于 --graceful-timeout 30
timeout = env_config.get('uvicorn.timeout', 60)            # 等价于 --timeout 60
pidfile = os.path.expanduser(env_config.get('pip_file_path', '~/Program/data_analysis_with_agent/.app_pid'))   # 等价于 --pid $PID_FILE
daemon = env_config.get('uvicorn.daemon', False)            # 等价于 --daemon
backlog = env_config.get('uvicorn.backlog', 2048)
worker_connections = env_config.get('uvicorn.worker_connections', 1000)
max_requests = env_config.get('uvicorn.limit_max_requests', 0)
max_requests_jitter = env_config.get('uvicorn.max_requests_jitter', 0)
keep_alive = env_config.get('uvicorn.timeout_keep_alive', 5)
preload_app = env_config.get('uvicorn.preload_app', False)
# 限制和安全
limit_request_line = env_config.get('uvicorn.limit_request_line', 4094)  # 限制 HTTP 请求行长度
limit_request_fields = env_config.get('uvicorn.limit_request_fields', 100)  # 限制 HTTP 请求头数量
limit_request_field_size = env_config.get('uvicorn.limit_request_field_size', 8190)  # 限制单个请求头大小

# 在配置文件加载时应用 logging 配置
logconfig_dict = get_gunicorn_log_config(env_config.get('uvicorn.log_level', "INFO").upper())

def on_starting(server):
    # 在新程中启动 Prometheus HTTP 服务
    metrics_process = multiprocessing.Process(
        target=start_metrics_server,
        daemon=True
    )
    metrics_process.start()

