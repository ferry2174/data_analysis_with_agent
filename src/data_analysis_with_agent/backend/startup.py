import logging
import os
import signal
import threading

import psutil
import uvicorn

from data_analysis_with_agent import APP_PATH, get_root_path
from data_analysis_with_agent.backend.config import ConfigManager
from data_analysis_with_agent.backend.constants import ENV_KEY_IN_OSENV
from data_analysis_with_agent.backend.logger import (
    get_uvicorn_log_config,
)
from data_analysis_with_agent.backend.metrics import start_metrics_server


logger = logging.getLogger(__name__)

pid_file_name = ".app_pid"

def start_data_analysis_with_agent(env: str = "dev"):
#    parser = argparse.ArgumentParser(description="EUL Python Project Template.")
#    parser.add_argument("--port", type=int, default=8000, help="App startup port")
#    parser.add_argument("--env", type=str, default="dev", help="run environment")
#    args = parser.parse_args()

    metrics_dir = os.path.join(APP_PATH, "metrics_dir")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
    def clear_metrics_dir(metrics_dir):
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        if not os.path.isdir(metrics_dir):
            raise ValueError(f"{metrics_dir} is not a valid directory")

        for filename in os.listdir(metrics_dir):
            file_path = os.path.join(metrics_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    # 调用函数清理目录
    clear_metrics_dir(metrics_dir)

    if env not in ["dev", "test", "prod"]:
        raise ValueError(f"Env value must in ['dev', 'test', 'prod'], current env value is {env}")
    os.environ[ENV_KEY_IN_OSENV] = env   # 全局全局域

    # 检查 PID 文件是否存在且不为空
    # 如果进程已存在且正在运行，判断当前进程是否为子进程，如果是，则继续执行，否则抛出异常
    pid_file_path = os.path.join(APP_PATH, pid_file_name)
    is_child_process = False
    if os.path.exists(pid_file_path) and os.path.getsize(pid_file_path) > 0:
        current_process = psutil.Process(os.getpid())
        parent_process = current_process.parent()
        with open(pid_file_path, "r") as pid_file:
            existing_pid = int(pid_file.read().strip())
            is_child_process = existing_pid == parent_process.pid   # 子进程，不检查，也不记录pid
            if not is_child_process and psutil.pid_exists(existing_pid):
                raise ValueError(
                    f"Another instance of data_analysis_with_agent is already running with PID {existing_pid}."
                )
    # 记录当前进程的 PID
    if not is_child_process:
        with open(pid_file_path, "w") as pid_file:
            logger.info(f"Starting data_analysis_with_agent app server with PID {os.getpid()}")
            pid_file.write(str(os.getpid()))

        # 设置启动参数, 如果是子进程，则不需要再启动uvicorn了
        #    ssl_keyfile = os.path.join(get_root_path(), "assets", "cert", "192.168.1.5+2-key.pem"),
        #    ssl_certfile = os.path.join(get_root_path(), "assets", "cert", "192.168.1.5+2.pem"),
        #    headers =
        """初始化全局配置句柄"""
        config = ConfigManager.init_config(env=env)
        # 启动服务
        try:
            # 启动 prometheuse 指标采集服务线程
            metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
            metrics_thread.start()

            uvicorn.run(
                "data_analysis_with_agent.backend.main:app",
                host="0.0.0.0",
                #            ssl_keyfile=ssl_keyfile,
                #            ssl_certfile=ssl_certfile,
                port=config.get("uvicorn.port"),
                access_log=config.get("uvicorn.access_log"),
                reload=config.get("uvicorn.reload", True),
                workers=min(config.get("uvicorn.workers"), os.cpu_count() + 1),
                log_config=get_uvicorn_log_config(config.get("uvicorn.log_config")),  # 使用自定义日志配置
                log_level=config.get("uvicorn.log_level"),
                timeout_keep_alive=config.get("uvicorn.timeout_keep_alive"),
                limit_max_requests=config.get("uvicorn.limit_max_requests"),
                limit_concurrency=config.get("uvicorn.limit_concurrency"),
                proxy_headers=config.get("uvicorn.proxy_headers"),
                reload_dirs=[get_root_path()]
            )
        except KeyboardInterrupt:
            logger.error("Server stopped")


def stop_data_analysis_with_agent():
    with open(os.path.join(APP_PATH, pid_file_name), "r") as pid_file:
        app_pid = int(pid_file.read().strip())
    logger.info(f"Stopping data_analysis_with_agent app with PID {app_pid}")
    os.kill(app_pid, signal.SIGTERM)

if __name__ == "__main__":
    start_data_analysis_with_agent("dev")
