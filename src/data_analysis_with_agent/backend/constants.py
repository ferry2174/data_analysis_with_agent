import os

from data_analysis_with_agent import APP_PATH, get_root_path


ENV_KEY_IN_OSENV = "APP_ENV"

FILE_FONT_STKAITI = os.path.abspath(os.path.join(get_root_path(), "assets/font/STKAITI.TTF"))

API_VERSION = "v1"
API_VERSION_PREFIX_V1 = f"/api/{API_VERSION}"

EXCEPTION_OBJECT_NOT_FOUND = 404

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_PATH = os.path.join(APP_PATH, "logs")
LOG_PATH_APP = os.path.join(LOG_PATH, "data_analysis_with_agent.log")
LOG_PATH_UVICORN = os.path.join(LOG_PATH, "uvicorn.log")
LOG_PATH_GUNICORN = os.path.join(LOG_PATH, "gunicorn.log")
LOG_PATH_UVICORN_ACCESS = os.path.join(LOG_PATH, "access.log")
LOG_PATH_METRICS = os.path.join(LOG_PATH, "metrics.log")
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | pid=%(process)d | tid=%(thread)d | %(name)s.%(funcName)s:%(lineno)d | %(message)s"
)
LOG_FORMAT_UVICORN = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_UVICORN_ACCESS = (
    '%(asctime)s | %(levelname)-8s | pid=%(process)d | %(client_addr)s - "%(request_line)s" %(status_code)s'
)
LOG_FORMAT_METRICS = "%(message)s"
LOG_REMAIN_DAYS = 99
LOG_LEVEL_DEBUG = "debug"
LOG_LEVEL_INFO = "info"
LOG_LEVEL_ERROR = "error"

RESPONSE_CODE_SUCCESS = 200
RESPONSE_CODE_SUCCESS_MSG = "OK"
RESPONSE_CODE_DATA_NOT_FOUND = 404
RESPONSE_CODE_DATA_NOT_FOUND_MSG = "Data not found"
RESPONSE_CODE_SERVICE_UNAVAILABLE = 503
RESPONSE_CODE_SERVICE_UNAVAILABLE_MSG = "Service unavailable"
RESPONSE_CODE_SERVICE_ERROR = 500
RESPONSE_CODE_SERVICE_ERROR_MSG = "Service internal error"

HTTP_HEADER_JSON = {"Content-Type": "application/json"}
