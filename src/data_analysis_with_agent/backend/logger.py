import logging
import os
import sys

from concurrent_log_handler import ConcurrentTimedRotatingFileHandler

from data_analysis_with_agent.backend.constants import (
    DATE_FORMAT,
    LOG_FORMAT,
    LOG_FORMAT_METRICS,
    LOG_FORMAT_UVICORN,
    LOG_FORMAT_UVICORN_ACCESS,
    LOG_PATH,
    LOG_PATH_APP,
    LOG_PATH_GUNICORN,
    LOG_PATH_METRICS,
    LOG_PATH_UVICORN,
    LOG_PATH_UVICORN_ACCESS,
    LOG_REMAIN_DAYS,
)


def setup_logging(
    log_level,
    log_dir=LOG_PATH,
    log_format=LOG_FORMAT,
    date_format=DATE_FORMAT,
    backup_count=LOG_REMAIN_DAYS,
    log_path=LOG_PATH_APP,
):
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(log_format, date_format)
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    # 文件 handler（按天滚动）
    file_handler = ConcurrentTimedRotatingFileHandler(
        filename=log_path, when="midnight", interval=1, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    # 业务 logger
    buzi_logger = logging.getLogger("data_analysis_with_agent")
    buzi_logger.setLevel(log_level)
    buzi_logger.handlers.clear()
    buzi_logger.addHandler(console_handler)
    buzi_logger.addHandler(file_handler)


def get_uvicorn_log_config(
    log_level,
    log_dir=LOG_PATH,
    log_format=LOG_FORMAT,
    log_format_access=LOG_FORMAT_UVICORN_ACCESS,
    log_format_error=LOG_FORMAT_UVICORN,
    log_format_metrics=LOG_FORMAT_METRICS,
    log_file_buzi=os.path.basename(LOG_PATH_APP),
    log_file_metrics=os.path.basename(LOG_PATH_METRICS),
    log_file_uvicorn=os.path.basename(LOG_PATH_UVICORN),
    log_file_uvicorn_access=os.path.basename(LOG_PATH_UVICORN_ACCESS),
    log_remain_days=LOG_REMAIN_DAYS,
):
    os.makedirs(log_dir, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": log_format,
                "use_colors": False,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": log_format_access,
            },
            "error": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": log_format_error,
            },
            "metrics": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": log_format_metrics,
            },
        },
        "handlers": {
            #            "default": {
            #                "formatter": "default",
            #                "class": "logging.handlers.TimedRotatingFileHandler",
            #                "filename": f"{log_dir}/{log_file_uvicorn}",
            #                "when": "midnight",  # 每天午夜切换
            #                "interval": 1,       # 每天
            #                "backupCount": log_remain_days,    # 保留5天的日志
            #                "encoding": "utf8",
            #            },
            "access": {
                "formatter": "access",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_uvicorn_access}",
                "when": "midnight",
                "interval": 1,
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "error": {
                "formatter": "error",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_uvicorn}",
                "when": "midnight",
                "interval": 1,
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "buzi": {
                "formatter": "default",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_buzi}",
                "when": "midnight",
                "interval": 1,
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "metrics": {
                "formatter": "metrics",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_metrics}",
                "when": "midnight",
                "interval": 1,
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "console_access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "console_error": {
                "formatter": "error",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "console_buzi": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            #            "uvicorn": {"handlers": ["default", "console"], "level": log_level.upper(), "propagate": False},
            "uvicorn.error": {"handlers": ["error", "console_error"], "level": log_level.upper(), "propagate": False},
            "uvicorn.access": {
                "handlers": ["access", "console_access"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "data_analysis_with_agent": {
                "handlers": ["buzi", "console_buzi"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "metrics": {"handlers": ["metrics"], "level": log_level.upper(), "propagate": False},
        },
    }

def get_gunicorn_log_config(
    log_level,
    log_dir=LOG_PATH,
    log_format=LOG_FORMAT,
    log_format_access=LOG_FORMAT_UVICORN_ACCESS,
    log_format_error=LOG_FORMAT_UVICORN,
    log_format_metrics=LOG_FORMAT_METRICS,
    log_file_buzi=os.path.basename(LOG_PATH_APP),
    log_file_metrics=os.path.basename(LOG_PATH_METRICS),
    log_file_gunicorn=os.path.basename(LOG_PATH_GUNICORN),
    log_file_gunicorn_access=os.path.basename(LOG_PATH_UVICORN_ACCESS),
    log_remain_days=LOG_REMAIN_DAYS,
):
    os.makedirs(log_dir, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": log_format,
                "use_colors": False,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": log_format_access,
            },
            "error": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": log_format_error,
            },
            "metrics": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": log_format_metrics,
            },
        },
        "handlers": {
            "console" : {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "console_access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "console_error": {
                "formatter": "error",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "console_buzi": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "default": {
                "formatter": "default",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_gunicorn}",
                "when": "midnight",  # 每天午夜切换
                "backupCount": log_remain_days,    # 保留5天的日志
                "encoding": "utf8",
            },
            "access": {
                "formatter": "access",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_gunicorn_access}",
                "when": "midnight",
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "error": {
                "formatter": "error",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_gunicorn}",
                "when": "midnight",
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "buzi": {
                "formatter": "default",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_buzi}",
                "when": "midnight",
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
            "metrics": {
                "formatter": "metrics",
                "class": "concurrent_log_handler.ConcurrentTimedRotatingFileHandler",
                "filename": f"{log_dir}/{log_file_metrics}",
                "when": "midnight",
                "backupCount": log_remain_days,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "gunicorn": {"handlers": ["default", "console"], "level": log_level.upper(), "propagate": False},
            "gunicorn.error": {"handlers": ["error", "console_error"], "level": log_level.upper(), "propagate": False},
            "gunicorn.access": {
                "handlers": ["access", "console_access"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "data_analysis_with_agent": {
                "handlers": ["buzi", "console_buzi"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "metrics": {"handlers": ["metrics"], "level": log_level.upper(), "propagate": False},
        },
    }
