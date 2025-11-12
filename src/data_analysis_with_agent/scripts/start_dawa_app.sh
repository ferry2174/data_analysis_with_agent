#!/bin/bash

# 设置变量
PYTHON_CMD="python3.11"
SCRIPT_PATH="import data_analysis_with_agent.backend.app.demo_main"
PID_FILE="$HOME/Program/data_analysis_with_agent/.gradio_app_pid"
MAX_LOG_SIZE=10M  # 单个日志文件最大大小
BACKUP_COUNT=5    # 保留的旧日志文件数量
SERVER_NAME="0.0.0.0"    # 服务器启动IP
KNOWLEDGE_GRAPH_SERVER_IP="http://127.0.0.1:8090/data_analysis_with_agent/graph/"    # 知识图谱服务url

# 检查Python和脚本是否存在
check_dependencies() {
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "错误：未找到 $PYTHON_CMD 命令" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# 主执行函数（后台）
run_script() {
    echo "=== 启动执行 $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # 后台执行脚本，输出追加到日志
    $PYTHON_CMD -c "$SCRIPT_PATH" --server_name $SERVER_NAME --knowledge_graph_server_ip $KNOWLEDGE_GRAPH_SERVER_IP 2>&1 &
    #nohup $PYTHON_CMD -c "$SCRIPT_PATH" --server_name $SERVER_NAME --knowledge_graph_server_ip $KNOWLEDGE_GRAPH_SERVER_IP 2>&1 &

    GRADIO_APP_PID=$!
    echo $GRADIO_APP_PID > $PID_FILE
    echo "脚本已在后台启动，PID: $!"
    echo "=== 执行后台开始 $(date '+%Y-%m-%d %H:%M:%S') ==="
}

# 主流程
main() {
    check_dependencies
    run_script
}

# 执行主函数
main