import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from data_analysis_with_agent.backend.constants import ENV_KEY_IN_OSENV


class ConfigManager:
    _instance = None
    _initialized = False
    _config = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_data: Dict[str, Any] = None):
        if not self._initialized:
            self._config = config_data or {}
            self._initialized = True

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init_config(cls, env: str = None):
        instance = cls.get_instance()
        instance.from_yaml_file(cls._get_config_path_by_env(env))
        return instance

    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> 'ConfigManager':
        """从YAML文件加载配置并初始化单例"""
        config_data = cls._load_yaml_config(file_path)
        instance = cls.get_instance()
        instance._config = config_data
        instance._initialized = True
        return instance

    @staticmethod
    def _load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _get_config_path_by_env(env: str) -> Path:
        """根据环境获取配置文件路径"""
        if env is None:
            env = os.environ.get(ENV_KEY_IN_OSENV, "dev")
        return Path(__file__).parent.parent / "config" / f"config_{env}.yaml"

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取配置值，支持字典和列表的嵌套访问"""
        keys = key.lower().split(".")
        value = self._config

        try:
            for k in keys:
                if isinstance(value, list):
                    # 如果是列表，尝试将 k 转换为整数索引
                    index = int(k)
                    value = value[index]
                else:
                    value = value[k]
            return value
        except (KeyError, TypeError, ValueError, IndexError):
            return default

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问，如果 key 不存在则抛出 KeyError"""
        result = self.get(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found in config")
        return result

    def __contains__(self, key: str) -> bool:
        """支持 in 操作符检查 key 是否存在"""
        return self.get(key) is not None

#    def reload(self, config_data: Dict[str, Any]) -> None:
#        """重新加载配置"""
#        self._config = config_data

    def reload_from_file(self, file_path: Union[str, Path]) -> None:
        """从文件重新加载配置"""
        self._config = self._load_yaml_config(file_path)
