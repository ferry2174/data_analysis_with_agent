# AI Agent Powered Data Analysis Chatbot

## 项目初始化

Python版本使用3.11.18

```shell
conda create -n {env_name} python=3.11.18
```

## 将本地包发布至Devpi

devpi login euler_risk --password=qYBlUs7hKzknbz4xuxMY4oFVRLoZkkrK
devpi use euler_risk/test
执行 devpi upload 之前，必须git提交所有文件。因为devpi会使用git命令将文件copy到临时文件夹

## 用开发模式安装程序包（适用本地开发）

```shell
cd ${your_workspace_dir}/${your_project_name}
pip install -e .
```

## 查看日志

日志文件在 ~/data_analysis_with_agent/logs/ 目录下
data_analysis_with_agent.log
access.log
metrics.log
gunicorn.log
uvicorn.log

## 配置接口返回参数为空时，是否返回默认值

fastapi接口装饰器中，加参数 response_model_exclude_none=True
需要返回默认值的加，会返回所有字段为默认值的完整对象
如果不需要，则会返回空对象，空对象中没有字段

## 本地调试

在开发工具中Debug方式执行 src/data_analysis_with_agent/backend/startup.sh
默认会使用 src/data_analysis_with_agent/config/config_dev.yaml配置参数启动

## 生产环境部署

* 配置devpi作为pip源

* 安装 Python 包
pip install --upgrade data_analysis_with_agent

* 优雅重启（优雅重启后打印日志有问题，暂不用）
* 停止服务
  * stop_data_analysis_with_agent.sh
* 启动服务
  * start_data_analysis_with_agent.sh test

## 配置文件路径

dev 环境在 ${your_project_dir}/config_dev
test & prod 环境在 ~/${your_project_name}/config_${env}.yaml

## 代码中使用配置项

from data_analysis_with_agent.backend.config import ConfigManager
ConfigManager.get_instance().get("your_key", default_value)

## 数据库帮助类（线程池）

## API文档路径

api返回的对象要做成Pydantic数据模型，并写好description。一方面可以自动生成文档，更重要的是将来可以可以在规则引擎中读取元数据，形成提示性列表。
/data_analysis_with_agent/docs

## 核心Pydantic数据对象

接口的返回对象，使用Pydantic数据模型对象，不能使用其他方式，用途：

1. 自动生成api docs
2. 自动生成元数据服务，供规则引擎等外部系统使用

### 在fastapi上配置安全访问

1. 安装mkcert

```shell
sudo apt update
sudo apt install libnss3-tools
sudo apt install mkcert
```

2. 安装根证书

```shell
mkcert -install
```

3. 生成证书

```shell
mkcert 192.168.1.5 127.0.0.1 localhost
```

4. 应用证书

```python
uvicorn.run(
    .., .., .., .., 
    ssl_keyfile=os.path.join(get_root_path(), "assets", "cert", ".."),
    ssl_certfile=os.path.join(get_root_path(), "assets", "cert", ".."),
)
```
