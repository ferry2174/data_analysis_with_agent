from typing import Any, Generic, Literal, Optional, Sequence, TypeVar

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

from data_analysis_with_agent import __version__
from data_analysis_with_agent.backend.constants import (
    RESPONSE_CODE_SUCCESS,
    RESPONSE_CODE_SUCCESS_MSG,
)


T = TypeVar("T")

class Response(GenericModel, Generic[T]):
    class Config:
        arbitrary_types_allowed = True

    status: Literal[200, 404, 500, 503, 422, 403] = Field(
        RESPONSE_CODE_SUCCESS,
        description="业务状态码：200:表示成功;404:数据不存在;500:服务器内部错误;503:服务不可用",
        examples=[200, 404, 500, 503, 422, 403],
    )
    message: str = Field(
        RESPONSE_CODE_SUCCESS_MSG,
        description="返回的提示信息",
        min_length=1,
        max_length=999999999,
        examples=["OK", "Data not found"],
    )
    data: Optional[T] = Field(
        None,
        description="返回的数据载荷，依赖具体接口类型",
    )

GLOBAL_RESPONSES = {
    "404": {
        "description": "Data not found",
        "content": {
            "application/json": {
                "example": {"status": 404, "message": "Data not found", "data": None}
            }
        },
    },
    "500": {
        "description": "Service internal error",
        "content": {
            "application/json": {
                "example": {"status": 500, "message": "Service internal error", "data": None}
            }
        },
    },
    "503": {
        "description": "Service unavailable",
        "content": {
            "application/json": {
                "example": {"status": 503, "message": "Service unavailable", "data": None}
            }
        },
    },
    "422": {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "example": {"status": 422, "message": "参数验证失败", "data": {"errors":["错误信息"], "body": {"参数": "值"}}}
            }
        },
    },
}

class ValidationErrorProperties(BaseModel):
    """参数验证失败反馈信息属性"""
    errors: Optional[Sequence[Any]] = Field([], description="参数验证失败信息")
    body: Optional[Any] = Field(None, description="请求参数")

def custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title="data_analysis_with_agent API",
        version=__version__,
        description="data_analysis_with_agent API",
        routes=app.routes,
    )

    # 手动注入 ValidationErrorProperties 到 $defs
    if "$defs" not in schema:
        schema["$defs"] = {}
    schema["$defs"]["ValidationErrorProperties"] = ValidationErrorProperties.model_json_schema()

    for _, methods in schema["paths"].items():
        for _, meta in methods.items():
            for code, resp in GLOBAL_RESPONSES.items():
                meta["responses"][code] = resp

    app.openapi_schema = schema
    return schema
