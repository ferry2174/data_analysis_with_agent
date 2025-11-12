import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import Request, status
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse

from data_analysis_with_agent.backend.constants import RESPONSE_CODE_SERVICE_ERROR


logger = logging.getLogger(__name__)

class EnhancedResponseValidationHandler:
    def __init__(self, log_level: str = "ERROR"):
        self.log_level = log_level

    async def get_enriched_request_info(self, request: Request) -> Dict[str, Any]:
        """获取增强的请求信息"""
        try:
            # 基础请求信息
            request_info = {
                "method": request.method,
                "url": str(request.url),
                "client": f"{request.client.host}:{request.client.port}" if request.client else "unknown",
                "timestamp": datetime.now().isoformat(),
                "path_parameters": dict(request.path_params),
                "query_parameters": dict(request.query_params.multi_items()),
            }

            # 安全的请求头信息（过滤敏感信息）
            sensitive_headers = {'authorization', 'cookie', 'set-cookie', 'token', 'password'}
            request_info["headers"] = {
                k: v for k, v in request.headers.items()
                if k.lower() not in sensitive_headers
            }

            # 请求体信息（异步获取）
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    # 尝试获取JSON体
                    body = await request.json()
                    request_info["body"] = body
                except json.JSONDecodeError:
                    try:
                        # 如果不是JSON，获取原始体
                        body = await request.body()
                        if body:
                            request_info["body"] = body.decode('utf-8', errors='ignore')[:1000] + "..." if len(body) > 1000 else body.decode('utf-8', errors='ignore')
                    except Exception as e:
                        request_info["body_error"] = f"Failed to read body: {e}"
                except Exception as e:
                    request_info["body_error"] = f"Failed to parse body: {e}"

            return request_info

        except Exception as e:
            return {"error": f"Failed to collect request info: {e}"}

    def format_validation_errors(self, errors: List[Dict]) -> List[Dict]:
        """格式化验证错误信息"""
        formatted_errors = []
        for error in errors:
            formatted_error = {
                "field": " → ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input_type": type(error["input"]).__name__,
                "input_preview": str(error["input"])[:200] + "..." if len(str(error["input"])) > 200 else str(error["input"])
            }
            formatted_errors.append(formatted_error)
        return formatted_errors

    async def __call__(self, request: Request, exc: ResponseValidationError):
        """处理响应验证错误"""
        # 获取请求信息
        request_info = await self.get_enriched_request_info(request)

        # 格式化错误信息
        formatted_errors = self.format_validation_errors(exc.errors())

        # 构建错误ID用于追踪
        error_id = f"val_err_{int(time.time())}_{hash(str(request_info)) % 10000}"

        # 记录详细错误日志
        log_data = {
            "error_id": error_id,
            "request": request_info,
            "validation_errors": formatted_errors,
            "exception_type": type(exc).__name__,
            "timestamp": datetime.now().isoformat()
        }

        if self.log_level == "ERROR":
            logger.error(json.dumps(log_data, indent=2, default=str))
        else:
            logger.warning(json.dumps(log_data, indent=2, default=str))

        # 返回用户友好的错误响应
        error_response_data = {
            "status": RESPONSE_CODE_SERVICE_ERROR,
            "message": "Internal server error: Response validation failed",
            "data": {
                "error_id": error_id,
                "request_summary": {
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "timestamp": datetime.now().isoformat()
                },
                "validation_issues": formatted_errors,
                "suggestion": "This may be due to data type mismatches in the API response. Please contact support with the error_id."
            }
        }

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response_data
        )
